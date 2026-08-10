// The frozen measured-shot gate: N real opponent decisions, every one of them
// checked against what its physics actually did.
//
//   npx tsx qa/measured-gate.ts [decisions] [outDir]
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

/** Frozen before the run, and not reused from any other evaluation in the repo. */
const SEED = 90210773;

const COUNT = Number(process.argv[2] ?? 100);
const OUT = process.argv[3] ?? "/tmp/showboat-measured-gate";

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

async function main() {
  mkdirSync(OUT, { recursive: true });
  const { table, evaluator } = await bootstrap();
  void APP_ROOT;
  const fixtures = makeFixtures(table, COUNT, SEED);
  const brain = neuralTrickOnlyBrain(evaluator);
  const classical = classicalTrickOnlyBrain();

  const rows: Record<string, unknown>[] = [];
  const violations: { fixture: string; check: string; detail: string }[] = [];
  const note = (fixture: string, check: string, detail: string) =>
    violations.push({ fixture, check, detail });

  for (const f of fixtures) {
    const state = asState(f.balls);
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
      rows.push({ fixture: f.id, useNeural, selected: null, reason: "no legal target" });
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
    const measuredLabel = sel.measured === null ? null : measuredRouteLabel(sel.measured);
    const wordingMatchesTrace =
      sel.kind === "safety-kick"
        ? sentence !== null && /safety off the cushion/.test(sentence)
        : sentence !== null &&
          measuredLabel !== null &&
          sentence.startsWith(`Playing a ${measuredLabel} on the `) &&
          (outcome === null || !/^Bank made|^Combination made/.test(outcome) ||
            outcome.toLowerCase().startsWith(measuredLabel.toLowerCase() + " made"));

    const row = {
      fixture: f.id,
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
  for (const r of played) {
    byRung[String(r.rung)] = (byRung[String(r.rung)] ?? 0) + 1;
    byMeasured[String(r.measuredClass)] = (byMeasured[String(r.measuredClass)] ?? 0) + 1;
    byPrevious[String(r.previousLadderRung)] = (byPrevious[String(r.previousLadderRung)] ?? 0) + 1;
  }
  return {
    decisions: played.length,
    noLegalTarget: rows.length - played.length,
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
    violations,
  };
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
