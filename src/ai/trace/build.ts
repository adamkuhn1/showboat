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
import {
  DECISION_TRACE_VERSION,
  type ContactEvent,
  type DecisionTiming,
  type DecisionTraceV1,
  type FallbackTrace,
  type ModelIdentity,
  type PhysicsVerification,
  type RailContact,
  type RejectionReason,
  type TracedCandidate,
  type TracedKind,
  type Vec2Trace,
} from "./contract";

const v2 = (p: { x: number; y: number }): Vec2Trace => ({ x: p.x, y: p.y });

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
  if (stat.strength < TRICK_RELIABILITY_THRESHOLD) return "below-reliability-threshold";
  return "lower-utility-than-selected";
};

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
      rejection: rejectionOf(
        index,
        stat,
        verification,
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
            rung: decision.rung,
            action: { ...shot.action },
            cuePath: shot.cuePath.map(v2),
            path: shot.path.map(v2),
            // The shot has not been played yet — `decide()` runs before
            // `executeAiShot`. `ui/planner/plan.ts` publishes the measured
            // motion onto the trace the moment there is one, through
            // `withExecutedMotion`. Null here is the truth, not a placeholder.
            executed: null,
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
