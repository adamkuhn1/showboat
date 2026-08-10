// The frozen measured-shot gate: N real opponent decisions, every one of them
// checked against what its physics actually did.
//
//   npx tsx qa/measured-gate.ts [decisions] [outDir] [--mix]
//
// Two fixture sources. Without `--mix` the fixtures are `eval/harness.ts`'s,
// which is what the frozen 100-decision run at seed 90210773 used. With `--mix`
// they come from `qa/boards.ts`: five stated board families — open, crowded,
// late, ball-in-hand, snookered — in a round-robin, because the harness set
// keeps only states with six or more generated candidates and therefore cannot
// reach the positions where no trick exists at all.
//
// Each decision runs the SHIPPED path — `neuralTrickOnlyBrain` over the real
// trained ranker and the real WASM physics, then `executeAiShot` — and is then
// re-examined from outside:
//
//   * the selected shot's route is classified a second time, from the EXECUTED
//     simulation's own event log, by the same classifier the policy used on the
//     pre-shot rollout. The two must agree; a disagreement means the route the
//     interface showed is not the route the balls took.
//   * the wording the interface would print is generated from the trace with
//     the interface's own functions and matched against the measurement.
//   * the previous selection ladder is replayed over the identical search
//     output, so the cost of requiring measured evidence is a paired number
//     rather than an impression.
//
// Nothing here is allowed to change a decision. The harness reads.

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { type GameState } from "../src/game/state";
import { CUE_ID } from "../src/game/rack";
import type { ShotEvent } from "../src/physics/engine";
import { simulateShotWasm } from "../src/physics/wasm-bridge";
import { neuralTrickOnlyBrain, classicalTrickOnlyBrain } from "../src/ai/brain";
import { defaultConfig, TRICK_RELIABILITY_THRESHOLD } from "../src/ai/shotSearch";
import { isTrickCandidate } from "../src/ai/policy/trickOnly";
import { executeAiShot } from "../src/ai/policy/execute";
import { extractExecutedMotion, withExecutedMotion } from "../src/ai/trace/executed";
import { classifyMeasuredShot, measuredRouteLabel } from "../src/ai/measure/classify";
import { plannedVsMeasured, rungText, shotSentence } from "../src/ui/shotSentence";
import { shotOutcomeLine } from "../src/ui/shotOutcome";
import { bootstrap, makeFixtures, APP_ROOT } from "../eval/harness";
import { makeBoardMix, type BoardType } from "./boards";

/** Frozen before the run, and not reused from any other evaluation in the repo. */
const SEED = 90210773;

const ARGS = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const MIX = process.argv.includes("--mix");
const COUNT = Number(ARGS[0] ?? 100);
const OUT = ARGS[1] ?? "/tmp/showboat-measured-gate";

/** Cushions each generated kind plans for, i.e. `Candidate.banks` by kind. */
const PLANNED_RAILS: Record<string, number> = {
  direct: 0,
  bank: 1,
  "double-bank": 2,
  combo: 0,
  "rail-combo": 1,
  "safety-kick": 1,
};

const asState = (balls: GameState["balls"]): GameState => ({
  balls,
  turn: 0,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 3,
});

/** One fixture, from either source, in the shape the loop below needs. */
interface GateFixture {
  id: string;
  boardType: BoardType | "harness";
  state: GameState;
  ballInHandApplied: boolean;
  metrics: Record<string, number> | null;
}

async function main() {
  mkdirSync(OUT, { recursive: true });
  const { table, evaluator } = await bootstrap();
  void APP_ROOT;
  const fixtures: GateFixture[] = MIX
    ? makeBoardMix(table, COUNT, SEED).map((f) => ({
        id: f.id,
        boardType: f.boardType,
        state: f.state,
        ballInHandApplied: f.ballInHandApplied,
        metrics: f.metrics as unknown as Record<string, number>,
      }))
    : makeFixtures(table, COUNT, SEED).map((f) => ({
        id: f.id,
        boardType: "harness" as const,
        state: asState(f.balls),
        ballInHandApplied: false,
        metrics: null,
      }));
  const brain = neuralTrickOnlyBrain(evaluator);
  const classical = classicalTrickOnlyBrain();

  const rows: Record<string, unknown>[] = [];
  const violations: { fixture: string; check: string; detail: string }[] = [];
  const note = (fixture: string, check: string, detail: string) =>
    violations.push({ fixture, check, detail });

  for (const f of fixtures) {
    const state = f.state;
    // Half the decisions run without the model, so the "no neural claim during
    // classical fallback" check has classical decisions to look at.
    const useNeural = rows.length % 2 === 0;
    const res = await (useNeural ? brain : classical).plan(state, table, 0, {
      ...defaultConfig,
      seedTimeoutMs: Infinity,
      searchTimeoutMs: Infinity,
    });

    const trace0 = res.decision;
    if (trace0.selected === null || res.shot === null) {
      rows.push({
        fixture: f.id,
        boardType: f.boardType,
        useNeural,
        selected: null,
        reason: "no shot could be constructed",
      });
      continue;
    }

    // Execute it exactly as the app does, then republish with measured motion.
    const report = executeAiShot(state, table, res.shot, simulateShotWasm);
    const trace = withExecutedMotion(trace0, extractExecutedMotion(report.sim));
    // Read the republished trace, not the search's copy: `executed` only exists
    // on the one the shot has been attached to, which is also the one the page
    // renders from.
    const sel = trace.selected!;
    const chosen =
      sel.candidateIndex === null
        ? null
        : (trace.candidates.find((c) => c.index === sel.candidateIndex) ?? null);

    // Independent second classification, from the run that was actually played.
    const executedMeasured = classifyMeasuredShot(report.sim, {
      target: chosen?.target ?? null,
      potId: chosen?.potId ?? null,
      legalTargets: trace.turn.legalTargets,
    });

    // The interface's own strings.
    const sentence = shotSentence(trace)?.text ?? null;
    const reconcile = plannedVsMeasured(trace);
    const rung = rungText(trace);
    const outcome = shotOutcomeLine(trace, report.outcome);

    // The previous ladder, replayed over the identical search output.
    const priorLadder = replayPreviousLadder(res);

    // Does the wording a visitor reads agree with the measurement? The noun in
    // the plan sentence must be the measured route's own label, and the result
    // line must not name a shape the measurement does not support.
    //
    // `shotOutcomeLine` returns a `ShotOutcomeLine`, not a string. This clause
    // used to hand the object itself to `RegExp.test`, which stringifies it to
    // "[object Object]"; the pattern never matched, the disjunction was always
    // satisfied, and the result-line half of the wording check ran on every
    // decision without being able to fail. It reads `.text` now.
    const outcomeText = outcome?.text ?? null;
    const measuredLabel = sel.measured === null ? null : measuredRouteLabel(sel.measured);
    const namesAShape = outcomeText !== null && /^[A-Z][a-z-]*( [a-z-]+)* made\./.test(outcomeText);
    const wordingMatchesTrace =
      sel.kind === "safety-kick"
        ? sentence !== null && /safety off the cushion/.test(sentence)
        : sentence !== null &&
          measuredLabel !== null &&
          sentence.startsWith(`Playing a ${measuredLabel} on the `) &&
          (!namesAShape ||
            outcomeText!.toLowerCase().startsWith(measuredLabel.toLowerCase() + " made"));

    const audit = auditEvents(report.sim.events, executedMeasured);

    const row = {
      fixture: f.id,
      boardType: f.boardType,
      ballInHandApplied: f.ballInHandApplied,
      boardMetrics: f.metrics,
      useNeural,
      mode: trace.mode,
      modelId: trace.model?.artifact ?? null,
      plannedKind: sel.plannedKind,
      plannedRails: PLANNED_RAILS[sel.plannedKind] ?? null,
      displayedKind: sel.kind,
      measuredClass: sel.measured?.classification ?? null,
      measuredRails: sel.measured?.rails ?? null,
      contactChain: sel.measured?.contactChain ?? [],
      executedClass: executedMeasured.classification,
      executedRails: executedMeasured.rails,
      executedChain: executedMeasured.contactChain,
      pottedBall: executedMeasured.pottedBall,
      potResult: report.outcome.pocketedThisShot.filter((b) => b !== CUE_ID),
      foul: report.outcome.foul,
      foulReason: report.outcome.foulReason,
      rung: sel.rung,
      straightPotRejected: sel.passedOverDirectIndex !== null,
      passedTrickValidation: sel.measured?.trickVerified ?? false,
      nominalTricksRejectedByMeasurement: trace.candidates.filter(
        (c) => c.rejection === "planned-trick-not-measured",
      ).length,
      previousLadderRung: priorLadder.rung,
      previousLadderKind: priorLadder.kind,
      sentence,
      reconcile,
      rung_text: rung,
      outcome: outcome?.text ?? null,
      wordingMatchesTrace,
      executedRoutePublished: sel.executed !== null,
      ...audit,
    };
    rows.push(row);

    // ---- the required results, checked one decision at a time -------------
    if (sel.kind === "safety-kick") {
      if (sentence && /\bbank|combination\b/.test(sentence)) {
        note(f.id, "labelled-without-evidence", `safety described as: ${sentence}`);
      }
    } else {
      const m = sel.measured;
      if (m === null || !m.trickVerified) {
        note(f.id, "labelled-without-evidence", `kind=${sel.kind} with measured=${JSON.stringify(m)}`);
      } else if (measuredRouteLabel(m) !== labelFor(sel.kind, m.rails)) {
        note(f.id, "displayed-type-not-measured", `${sel.kind} vs ${measuredRouteLabel(m)}`);
      }
      if (m && m.classification === "direct") {
        note(f.id, "direct-pot-played", `selected route measured as a direct pot`);
      }
    }

    // The pre-shot measurement and the executed one are the same run's answer.
    if (sel.measured !== null) {
      if (sel.measured.classification !== executedMeasured.classification) {
        note(
          f.id,
          "shown-but-not-measured",
          `pre-shot ${sel.measured.classification} vs executed ${executedMeasured.classification}`,
        );
      }
      if (sel.measured.rails !== executedMeasured.rails) {
        note(f.id, "shown-but-not-measured", `rails ${sel.measured.rails} vs ${executedMeasured.rails}`);
      }
    }
    if (sel.executed === null) {
      note(f.id, "no-measured-route-published", "executed motion missing after the shot ran");
    }

    // A classical decision may never carry a model identity or neural priors.
    const claimsNeural = trace.mode === "neural-hybrid" && trace.fallback === null;
    if (!claimsNeural && trace.model !== null) {
      note(f.id, "neural-claim-in-classical", `model=${trace.model.artifact}`);
    }
    if (!claimsNeural && trace.candidates.some((c) => c.neural !== null)) {
      note(f.id, "neural-claim-in-classical", "candidate rows carry a neural prior");
    }

    if (outcome === null) {
      note(f.id, "no-post-shot-result", "shotOutcomeLine returned null for a played shot");
    }
    if (!wordingMatchesTrace) {
      note(f.id, "wording-does-not-match-trace", `${sentence} / ${outcome?.text ?? "no result line"}`);
    }
  }

  const played = rows.filter((r) => r.measuredClass !== undefined && r.selected !== null);
  const summary = summarise(rows, violations);
  writeFileSync(join(OUT, "decisions.json"), JSON.stringify({ seed: SEED, rows }, null, 2));
  writeFileSync(join(OUT, "summary.json"), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
  console.log(`\n${played.length} decisions recorded -> ${OUT}`);
  if (violations.length > 0) process.exitCode = 1;
}

/**
 * The four structural questions the previous run did not ask, answered from the
 * executed shot's own event log.
 *
 * None of these is a second classifier. Each one measures a quantity the
 * classifier's two rules are supposed to remove, so the report can say how often
 * the rules had something to remove rather than only that they exist:
 *
 *  * `railsAllEvents` counts every cushion in the whole log, and
 *    `railsPrePotAnyBall` every cushion before the decisive pot regardless of
 *    who took it. The published `rails` is what survives both rules, so the
 *    three numbers together are the truncation and the causal filter, per shot.
 *  * `pairChatterMax` is the largest number of contacts one unordered pair of
 *    balls made before the pot, and `chatterWindowMs` the tightest gap between
 *    two of them. A pair that touches four times in six milliseconds is the
 *    chatter case.
 *  * `restrikesOnChain` counts contacts in which an off-chain ball struck a
 *    chain ball after that chain ball was already moving — the case where the
 *    recorded driver is not the ball that last pushed it.
 *  * `bothMovingContacts` counts contacts where the log had already seen both
 *    balls set in motion, which is the ambiguity itself.
 */
function auditEvents(
  events: readonly ShotEvent[],
  m: { potEventIndex: number; contactChain: number[]; rails: number },
): Record<string, unknown> {
  const potAt = m.potEventIndex;
  const pre = potAt >= 0 ? events.slice(0, potAt) : events.slice();
  const post = potAt >= 0 ? events.slice(potAt + 1) : [];

  const railsAllEvents = events.filter((e) => e.kind === "ball-cushion").length;
  const railsPrePotAnyBall = pre.filter((e) => e.kind === "ball-cushion").length;
  const railsPostPot = post.filter((e) => e.kind === "ball-cushion").length;
  const chain = new Set(m.contactChain.filter((id) => id !== CUE_ID));
  const railsPrePotOffChain = pre.filter(
    (e) => e.kind === "ball-cushion" && e.balls[0] !== CUE_ID && !chain.has(e.balls[0]),
  ).length;
  const railsPostPotOnChain = post.filter(
    (e) => e.kind === "ball-cushion" && chain.has(e.balls[0]),
  ).length;

  // What the rail count WOULD have been with each of the classifier's two rules
  // switched off, so "the rule matters" is a number per shot and not a claim.
  // Both alternatives use the classifier's own credit conditions otherwise.
  const railsWithoutTruncation = m.rails + railsPostPotOnChain;
  const railsWithoutCausalFilter =
    pre.filter((e) => e.kind === "ball-cushion" && e.balls[0] !== CUE_ID).length;

  // Chatter: the same ball meeting the same cushion twice inside CHATTER_MS is
  // one rattle, physically. Counted here only to report how often the published
  // rail count contains one — the classifier counts cushion events, and a
  // cushion event is a cushion event.
  const CHATTER_MS = 50;
  let railsChatterCollapsed = 0;
  const lastTouch = new Map<string, number>();
  for (const e of pre) {
    if (e.kind !== "ball-cushion") continue;
    const id = e.balls[0];
    if (id === CUE_ID || !chain.has(id)) continue;
    const key = `${id}:${e.cushion ?? ""}`;
    const prev = lastTouch.get(key);
    lastTouch.set(key, e.time);
    if (prev !== undefined && (e.time - prev) * 1000 <= CHATTER_MS) railsChatterCollapsed++;
  }

  // Chatter: contacts per unordered pair, and the tightest interval within one.
  const perPair = new Map<string, number[]>();
  for (const e of pre) {
    if (e.kind !== "ball-ball" || e.balls.length < 2) continue;
    const key = [e.balls[0], e.balls[1]].sort((a, b) => a - b).join("-");
    const times = perPair.get(key) ?? [];
    times.push(e.time);
    perPair.set(key, times);
  }
  let pairChatterMax = 0;
  let chatterWindowMs = Infinity;
  for (const times of perPair.values()) {
    pairChatterMax = Math.max(pairChatterMax, times.length);
    for (let i = 1; i < times.length; i++) {
      chatterWindowMs = Math.min(chatterWindowMs, (times[i] - times[i - 1]) * 1000);
    }
  }

  // Repeated cushion contacts by the same ball on the same rail, which is the
  // shape cushion chatter would take if it existed.
  const perBallRail = new Map<string, number>();
  for (const e of pre) {
    if (e.kind !== "ball-cushion") continue;
    const key = `${e.balls[0]}:${e.cushion ?? ""}`;
    perBallRail.set(key, (perBallRail.get(key) ?? 0) + 1);
  }
  const sameRailRepeatMax = Math.max(0, ...perBallRail.values());

  // Driver ambiguity, replaying the classifier's own motion bookkeeping.
  const moving = new Set<number>([CUE_ID]);
  let bothMovingContacts = 0;
  let restrikesOnChain = 0;
  for (const e of pre) {
    if (e.kind !== "ball-ball" || e.balls.length < 2) continue;
    const [a, b] = e.balls;
    const am = moving.has(a);
    const bm = moving.has(b);
    if (am && bm) {
      bothMovingContacts++;
      // One of the two is on the chain and the other is not: the chain ball has
      // just been struck by something the chain does not name.
      if (chain.has(a) !== chain.has(b)) restrikesOnChain++;
    }
    if (am) moving.add(b);
    if (bm) moving.add(a);
  }

  // The sharp version of the same question. The chain credits the potted ball
  // to whichever ball first set it moving; the ball that last touched it before
  // it dropped is the one that actually sent it in. When those differ, the
  // published chain names the wrong driver — and if the last toucher is the cue
  // ball, a route the cue finished is being described as a combination.
  const potted = m.contactChain[m.contactChain.length - 1] ?? null;
  const credited = m.contactChain.length >= 2 ? m.contactChain[m.contactChain.length - 2] : null;
  let lastToucher: number | null = null;
  if (potted !== null) {
    for (const e of pre) {
      if (e.kind !== "ball-ball" || e.balls.length < 2) continue;
      if (e.balls[0] === potted) lastToucher = e.balls[1];
      else if (e.balls[1] === potted) lastToucher = e.balls[0];
    }
  }
  const lastToucherIsCredited =
    potted === null || lastToucher === null || credited === null || lastToucher === credited;

  return {
    // The ordered log, compactly: `a-b` a ball-ball contact, `id|cushion` a
    // cushion, `POT id@pocket` a drop, and `>>>` marking the decisive pot the
    // classification is truncated at.
    orderedContacts: events
      .map((e, i) => {
        const t = `@${e.time.toFixed(3)}`;
        const mark = i === potAt ? ">>>" : "";
        if (e.kind === "ball-ball") return `${mark}${e.balls[0]}-${e.balls[1]}${t}`;
        if (e.kind === "ball-cushion") return `${mark}${e.balls[0]}|${e.cushion ?? "?"}${t}`;
        if (e.kind === "pocket") return `${mark}POT ${e.balls[0]}@${e.pocket ?? "?"}${t}`;
        return "";
      })
      .filter(Boolean),
    events: events.length,
    eventLogTruncated: events.length >= 200,
    railsAllEvents,
    railsPrePotAnyBall,
    railsPostPot,
    railsPrePotOffChain,
    railsPostPotOnChain,
    railsPublished: m.rails,
    railsWithoutTruncation,
    railsWithoutCausalFilter,
    railsChatterCollapsed,
    pairChatterMax,
    chatterWindowMs: isFinite(chatterWindowMs) ? Math.round(chatterWindowMs * 100) / 100 : -1,
    sameRailRepeatMax,
    bothMovingContacts,
    restrikesOnChain,
    lastToucherIsCredited,
    lastToucherWasTheCue: lastToucher === CUE_ID && credited !== CUE_ID,
  };
}

/** `TrickKind` -> the words the measured label uses, for the agreement check. */
function labelFor(kind: string, rails: number): string {
  const railWords = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight"];
  switch (kind) {
    case "bank":
      return "bank";
    case "double-bank":
      return `${railWords[rails] ?? rails}-rail bank`;
    case "combo":
      return "combination";
    case "rail-combo":
      return rails <= 1 ? "rail combination" : `${railWords[rails] ?? rails}-rail combination`;
    default:
      return kind;
  }
}

/**
 * What the ladder BEFORE measured validation would have picked, over the exact
 * same `allStats`/`verifications` this decision produced. Read-only: it makes
 * no physics calls and changes nothing.
 */
function replayPreviousLadder(res: {
  allStats: readonly { candidate: { kind: string }; potsTarget: boolean; strength: number }[];
  verifications: readonly ({ legalFirstContact: boolean; scratched: boolean } | null)[];
}): { rung: string; kind: string | null } {
  const tricks = res.allStats
    .map((s, i) => ({ s, v: res.verifications[i] ?? null }))
    .filter((t) => isTrickCandidate(t.s.candidate as never));
  const qualifying = tricks.filter(
    (t) => t.s.potsTarget && t.s.strength >= TRICK_RELIABILITY_THRESHOLD,
  );
  if (qualifying.length > 0) return { rung: "trick-qualified", kind: qualifying[0].s.candidate.kind };
  const potting = tricks.filter((t) => t.s.potsTarget);
  if (potting.length > 0) return { rung: "trick-below-threshold", kind: potting[0].s.candidate.kind };
  const attempts = tricks.filter((t) => t.v !== null && t.v.legalFirstContact && !t.v.scratched);
  if (attempts.length > 0) {
    return { rung: "trick-attempt-no-verified-pot", kind: attempts[0].s.candidate.kind };
  }
  return { rung: "safety", kind: "safety-kick" };
}

function summarise(rows: Record<string, unknown>[], violations: unknown[]) {
  const played = rows.filter((r) => r.rung !== undefined);
  const count = (pred: (r: Record<string, unknown>) => boolean) => played.filter(pred).length;
  const byRung: Record<string, number> = {};
  const byMeasured: Record<string, number> = {};
  const byPrevious: Record<string, number> = {};
  const byBoard: Record<string, number> = {};
  const boardRung: Record<string, Record<string, number>> = {};
  for (const r of rows) byBoard[String(r.boardType)] = (byBoard[String(r.boardType)] ?? 0) + 1;
  for (const r of played) {
    byRung[String(r.rung)] = (byRung[String(r.rung)] ?? 0) + 1;
    byMeasured[String(r.measuredClass)] = (byMeasured[String(r.measuredClass)] ?? 0) + 1;
    byPrevious[String(r.previousLadderRung)] = (byPrevious[String(r.previousLadderRung)] ?? 0) + 1;
    const bt = String(r.boardType);
    boardRung[bt] = boardRung[bt] ?? {};
    boardRung[bt][String(r.rung)] = (boardRung[bt][String(r.rung)] ?? 0) + 1;
  }
  const sum = (k: string) => played.reduce((a, r) => a + Number(r[k] ?? 0), 0);
  const max = (k: string) => played.reduce((a, r) => Math.max(a, Number(r[k] ?? 0)), 0);
  return {
    decisions: played.length,
    noShotConstructed: rows.length - played.length,
    boardTypes: byBoard,
    rungByBoardType: boardRung,
    /** A board on which the ladder found no playable trick at all. */
    noTrickPositions: count(
      (r) => r.rung === "non-direct-safety" || r.rung === "forced-legal-contact",
    ),
    ballInHandDecisions: count((r) => r.ballInHandApplied === true),
    selectionRung: byRung,
    measuredClassOfSelectedShot: byMeasured,
    previousLadderRung: byPrevious,
    reclassified: count((r) => r.plannedKind !== r.displayedKind && r.displayedKind !== "safety-kick"),
    nominalTricksRejectedByMeasurement: played.reduce(
      (a, r) => a + Number(r.nominalTricksRejectedByMeasurement ?? 0),
      0,
    ),
    decisionsWithAtLeastOneMeasurementRejection: count(
      (r) => Number(r.nominalTricksRejectedByMeasurement ?? 0) > 0,
    ),
    potsMade: count((r) => Array.isArray(r.potResult) && (r.potResult as unknown[]).length > 0),
    fouls: count((r) => r.foul === true),
    directPotsPlayed: count((r) => r.measuredClass === "direct"),
    straightPotsRejected: count((r) => r.straightPotRejected === true),
    everyShotHasOutcome: played.every((r) => typeof r.outcome === "string" && r.outcome.length > 0),
    everyShotWordedFromItsMeasurement: played.every((r) => r.wordingMatchesTrace === true),
    everyShotPublishedItsMeasuredRoute: played.every((r) => r.executedRoutePublished === true),
    audits: {
      // 1 and 2: how much the two rules actually removed across the run.
      cushionsInWholeLog: sum("railsAllEvents"),
      cushionsBeforeThePot: sum("railsPrePotAnyBall"),
      cushionsAfterThePot: sum("railsPostPot"),
      cushionsBeforeThePotByOffChainBalls: sum("railsPrePotOffChain"),
      cushionsPublishedAsRails: sum("railsPublished"),
      decisionsWithAPostPotCushion: count((r) => Number(r.railsPostPot ?? 0) > 0),
      decisionsWithAnOffChainCushion: count((r) => Number(r.railsPrePotOffChain ?? 0) > 0),
      /** Decisions where dropping the truncation rule would raise the rail count. */
      decisionsTruncationSaved: count(
        (r) => Number(r.railsWithoutTruncation ?? 0) > Number(r.railsPublished ?? 0),
      ),
      /** Decisions where dropping the causal filter would raise the rail count. */
      decisionsCausalFilterSaved: count(
        (r) => Number(r.railsWithoutCausalFilter ?? 0) > Number(r.railsPublished ?? 0),
      ),
      /** Published rails that are a second contact on the same rail within 50 ms. */
      publishedRailsThatAreARattle: sum("railsChatterCollapsed"),
      decisionsWithARattleInTheRailCount: count((r) => Number(r.railsChatterCollapsed ?? 0) > 0),
      // 3: chatter.
      maxContactsBetweenOnePair: max("pairChatterMax"),
      decisionsWithARepeatedPairContact: count((r) => Number(r.pairChatterMax ?? 0) > 1),
      tightestRepeatIntervalMs: played
        .map((r) => Number(r.chatterWindowMs ?? -1))
        .filter((v) => v >= 0)
        .reduce((a, b) => Math.min(a, b), Number.MAX_SAFE_INTEGER),
      maxRepeatsOnOneBallOneRail: max("sameRailRepeatMax"),
      // 4: already-moving-ball ambiguity.
      contactsWhereBothBallsWereMoving: sum("bothMovingContacts"),
      decisionsWhereAnOffChainBallRestruckAChainBall: count(
        (r) => Number(r.restrikesOnChain ?? 0) > 0,
      ),
      /** The chain's last edge names a ball that is not the one that last touched the potted ball. */
      decisionsWhereTheCreditedDriverIsNotTheLastToucher: count(
        (r) => r.lastToucherIsCredited === false,
      ),
      /** …and the last toucher was the cue ball, i.e. the cue finished a route called a combination. */
      decisionsWhereTheCueFinishedAnAttributedCombination: count(
        (r) => r.lastToucherWasTheCue === true,
      ),
      eventLogsHittingTheParseCap: count((r) => r.eventLogTruncated === true),
    },
    violations,
  };
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
