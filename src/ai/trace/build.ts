// Internal search state -> the published `DecisionTraceV1`.
//
// This indirection is deliberate. `contract.ts` is a binding interface another
// team consumes and cannot change; `shotSearch.ts` is free to be refactored.
// This file is the only place the two meet, so a search refactor breaks a
// compile here rather than silently changing what a renderer draws.
//
// The one rule that matters: **a non-null `physics` block is emitted only from
// a real `CandidateVerification`,** which the search creates only immediately
// after a real `simulateShotWasm` call. There is no other code path that can
// produce one. `contract.test.ts` re-checks that against `verifiedIndices`.

import { CUE_ID } from "../../game/rack";
import { type GameState, type PlayerId } from "../../game/state";
import { type ShotEvent } from "../../physics/engine";
import {
  type CandidateStat,
  type CandidateVerification,
  type DecisionTrace,
  type SearchOutcome,
  TRICK_RELIABILITY_THRESHOLD,
} from "../shotSearch";
import { isTrickCandidate, type TrickOnlyDecision } from "../policy/trickOnly";
import { classifyMeasuredShot, toMeasuredRoute } from "../measure/classify";
import { isPathClear, type Candidate } from "../candidates";
import { type Ball } from "../../physics/ball";
import { type ProgressCandidate } from "../search/progress";
import {
  DECISION_TRACE_VERSION,
  type ContactEvent,
  type DecisionTiming,
  type DecisionTraceV1,
  type FallbackTrace,
  type MeasuredRoute,
  type ModelIdentity,
  type PhysicsVerification,
  type RailContact,
  type RejectionReason,
  type TracedCandidate,
  type TracedKind,
  type Vec2Trace,
} from "./contract";

const v2 = (p: { x: number; y: number }): Vec2Trace => ({ x: p.x, y: p.y });

/**
 * Candidate geometry for the LIVE progress stream, mapped here rather than in
 * `search/progress.ts` so that the live route and the recorded route are built
 * by one file and cannot drift into disagreeing about where a line goes. The
 * fields are the same ones `TracedCandidate` carries, and the index is the same
 * generation-order index, so a consumer can address a candidate identically
 * whether it is watching the search or reading the record afterwards.
 */
export function progressCandidates(
  candidates: Candidate[],
  cueBall: Vec2Trace,
): ProgressCandidate[] {
  return candidates.map((c, index) => ({
    index,
    kind: c.kind as TracedKind,
    eligible: isTrickCandidate(c),
    target: c.target,
    potId: c.potId,
    pocket: c.pocket,
    aimPoint: v2(c.aimPoint),
    cuePath: [cueBall, v2(c.aimPoint)],
    path: c.path.map(v2),
  }));
}

const contactEvents = (events: ShotEvent[]): ContactEvent[] =>
  events
    .filter((e) => e.kind === "ball-ball" || e.kind === "ball-cushion" || e.kind === "pocket")
    .map((e) => ({
      kind: e.kind as ContactEvent["kind"],
      timeSec: e.time,
      balls: [...e.balls],
      cushion: e.cushion ?? null,
      pocket: e.pocket ?? null,
    }));

const railContacts = (events: ShotEvent[]): RailContact[] =>
  events
    .filter((e) => e.kind === "ball-cushion" && e.cushion !== undefined)
    .map((e) => ({ ballId: e.balls[0], cushion: e.cushion as string, timeSec: e.time }));

const physicsOf = (stat: CandidateStat, v: CandidateVerification): PhysicsVerification => ({
  firstContact: v.firstContact,
  legalFirstContact: v.legalFirstContact,
  scratched: v.scratched,
  legalPot: v.legalPot,
  pocketed: [...v.pocketed],
  railContacts: railContacts(v.events),
  railsBeforePot: v.railsBeforePot,
  contactSequence: contactEvents(v.events),
  strength: stat.strength,
  value: stat.value,
  visits: stat.visits,
  styleScore: stat.styleScore,
});

/**
 * Why this candidate is not the shot being played — the FIRST reason that
 * applied, in the order the policy applied them. Order is the whole point: a
 * direct that also scratched is rejected as a direct, because that is the
 * decision the policy actually made.
 */
const rejectionOf = (
  index: number,
  stat: CandidateStat,
  verification: CandidateVerification | null,
  measured: MeasuredRoute | null,
  selectedIndex: number | null,
  considered: Set<number>,
  usedPrior: boolean,
  seedTimedOut: boolean,
): RejectionReason | null => {
  if (selectedIndex !== null && index === selectedIndex) return null;
  if (!isTrickCandidate(stat.candidate)) return "direct-excluded-by-policy";
  if (verification === null) {
    if (!considered.has(index)) return usedPrior ? "pruned-by-prior" : "budget-exhausted";
    return seedTimedOut ? "seed-timeout" : "budget-exhausted";
  }
  if (verification.scratched) return "scratched-in-simulation";
  if (!verification.legalFirstContact) return "illegal-first-contact";
  if (!verification.legalPot) return "did-not-pot";
  // It potted the ball it was for, and the route it took is not the structure
  // it was generated as, nor any other supported one. This is the rejection the
  // measured classifier added, and it is placed exactly where the policy makes
  // it: after "did it pot", before "was it reliable".
  if (measured === null || !measured.trickVerified) return "planned-trick-not-measured";
  if (stat.strength < TRICK_RELIABILITY_THRESHOLD) return "below-reliability-threshold";
  return "lower-utility-than-selected";
};

/**
 * The straight pot the trick-only policy turned down, or null if none was on.
 *
 * `generateCandidates` emits a `direct` once the cue can reach the ghost-ball
 * contact point unobstructed at a makeable cut angle. It never checks the other
 * half of the shot — whether the object ball can actually reach the pocket —
 * because for every other kind the physics search settles that question, and a
 * direct is never simulated: the policy excludes it before the search spends a
 * unit on it. So "a direct was enumerated" is weaker than "a straight pot was
 * on", and the panel's claim needs the stronger one.
 *
 * This closes exactly that gap, with the generator's own clearance primitive
 * over the generator's own object-ball leg (`path[0]` is the object ball,
 * `path[1]` the pocket for a direct). The cue ball is skipped because the
 * ghost-ball construction puts it behind the object ball relative to the
 * pocket, and the target is skipped because it is the ball travelling.
 *
 * Among the directs that pass, the shortest object-ball run is recorded: the
 * easiest of them. Which one is named only decides how strong the example is —
 * the claim a renderer makes from it is true of every member of the set.
 */
function passedOverDirectIndex(candidates: TracedCandidate[], balls: Ball[]): number | null {
  const live = balls.filter((b) => !b.pocketed);
  let bestIndex: number | null = null;
  let bestRun = Infinity;
  for (const c of candidates) {
    if (c.kind !== "direct" || c.path.length < 2) continue;
    const from = c.path[0];
    const to = c.path[c.path.length - 1];
    if (!isPathClear(from, to, live, new Set([CUE_ID, c.target]))) continue;
    const run = Math.hypot(to.x - from.x, to.y - from.y);
    if (run < bestRun) {
      bestRun = run;
      bestIndex = c.index;
    }
  }
  return bestIndex;
}

export interface BuildTraceInput {
  outcome: SearchOutcome;
  decision: TrickOnlyDecision;
  state: GameState;
  player: PlayerId;
  targets: number[];
  physicsUnitsAllowed: number;
  model: ModelIdentity | null;
  fallback: FallbackTrace | null;
  timing: DecisionTiming;
}

export function buildDecisionTrace(input: BuildTraceInput): DecisionTraceV1 {
  const { outcome, decision, state, player, targets } = input;
  const internal: DecisionTrace | undefined = outcome.trace;
  const usedPrior = internal?.mode === "neural-hybrid";
  const considered = new Set(internal?.consideredIndices ?? []);
  const seedTimedOut = internal?.seedTimedOut ?? false;
  const selectedIndex = decision.shot?.candidateIndex ?? null;
  const cue = state.balls.find((b) => b.id === CUE_ID);
  const cueBall = v2(cue ? cue.pos : { x: 0, y: 0 });

  const candidates: TracedCandidate[] = outcome.allStats.map((stat, index) => {
    const c = stat.candidate;
    const verification = outcome.verifications[index] ?? null;
    // One classification per real rollout, from the same event log the policy
    // classified. `selectTrickOnly` runs the identical call on the identical
    // input, so the trace cannot disagree with the decision about what a route
    // did — and a candidate with no rollout gets no measurement at all.
    const measured: MeasuredRoute | null =
      verification === null
        ? null
        : toMeasuredRoute(
            classifyMeasuredShot(
              { events: verification.events },
              { target: c.target, potId: c.potId, legalTargets: targets },
            ),
          );
    return {
      index,
      kind: c.kind as TracedKind,
      eligible: isTrickCandidate(c),
      target: c.target,
      potId: c.potId,
      pocket: c.pocket,
      aimPoint: v2(c.aimPoint),
      cuePath: [cueBall, v2(c.aimPoint)],
      path: c.path.map(v2),
      action: { ...c.action },
      neural:
        stat.priorScore === undefined || stat.priorRank === undefined
          ? null
          : {
              score: stat.priorScore,
              logit: stat.priorLogit ?? null,
              rank: stat.priorRank,
            },
      physics: verification === null ? null : physicsOf(stat, verification),
      measured,
      rejection: rejectionOf(
        index,
        stat,
        verification,
        measured,
        selectedIndex,
        considered,
        usedPrior,
        seedTimedOut,
      ),
    };
  });

  const shot = decision.shot;
  return {
    version: DECISION_TRACE_VERSION,
    policy: "trick-only",
    mode: usedPrior ? "neural-hybrid" : "classical-trick-only",
    turn: {
      player: player as 0 | 1,
      shotIndex: state.shotCount,
      legalTargets: [...targets],
      cueBall,
    },
    model: input.model,
    budget: {
      physicsUnitsAllowed: input.physicsUnitsAllowed,
      physicsUnitsSpent: outcome.simulations,
      safetySimsSpent: decision.safetySimsSpent,
      candidatesGenerated: outcome.allStats.length,
      candidatesEligible: outcome.allStats.filter((s) => isTrickCandidate(s.candidate)).length,
      candidatesConsidered: considered.size,
      physicsVerified: internal?.physicsVerified ?? 0,
      prunedByPrior: internal?.prunedByPrior ?? 0,
      reservePromotions: internal?.reservePromotions ?? 0,
      seedTimedOut,
    },
    timing: input.timing,
    candidates,
    selected:
      shot === null || decision.rung === null
        ? null
        : {
            candidateIndex: shot.candidateIndex,
            kind: shot.kind,
            plannedKind: shot.plannedKind,
            measured: shot.measured,
            rung: decision.rung,
            action: { ...shot.action },
            cuePath: shot.cuePath.map(v2),
            path: shot.path.map(v2),
            // The shot has not been played yet — `decide()` runs before
            // `executeAiShot`. `ui/planner/plan.ts` publishes the measured
            // motion onto the trace the moment there is one, through
            // `withExecutedMotion`. Null here is the truth, not a placeholder.
            executed: null,
            passedOverDirectIndex: passedOverDirectIndex(candidates, state.balls),
            utility: decision.utility,
            reliabilityThreshold: TRICK_RELIABILITY_THRESHOLD,
            qualifyingTricks: decision.qualifyingTricks,
            // `pickSafety`'s own verdict, carried rather than dropped. `none`
            // never reaches here: it means no kick was constructed, which is
            // the `shot === null` branch above.
            safetyQuality:
              decision.safetyQuality === null || decision.safetyQuality === "none"
                ? null
                : decision.safetyQuality,
          },
    fallback: input.fallback,
  };
}
