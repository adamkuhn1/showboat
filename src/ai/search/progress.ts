// The live search progress protocol.
//
// WHAT THIS IS, AND WHY IT IS NOT THE DECISION TRACE
//
// `trace/contract.ts` publishes a COMPLETED decision: it exists only once the
// search has finished, and every field in it is final. This protocol publishes
// the same search WHILE IT RUNS. The two are not redundant and neither replaces
// the other — the trace is the record, this is the observation.
//
// RULES THIS FILE ENFORCES BY CONSTRUCTION
//
//  1. **Every event is emitted from the production search path**, at the moment
//     the thing it describes actually happened. There is no producer of these
//     events other than `shotSearch.ts` and `brain.ts`; in particular no
//     renderer can construct one. `progress.test.ts` drives a real search and
//     checks the stream against the search's own `DecisionTrace`.
//  2. **Monotonic.** `seq` increases by exactly 1 per event and `atMs` never
//     decreases. `validateProgressStream` is the check, and it is run against
//     real streams rather than asserted in a comment.
//  3. **Causally ordered.** A candidate's geometry is published before anything
//     refers to that candidate; `candidate-simulating` precedes the
//     `candidate-verified` for the same index; `search-completed` is last.
//     These are the orderings that make "a route may not appear before it
//     exists" and "a verification may not appear before its simulation
//     completes" mechanical rather than aspirational.
//  4. **No progress percentage, ever.** There is no field here from which one
//     could be computed honestly: the search does not know how many candidates
//     it will reach before the budget or the clock stops it. A consumer that
//     wants to show motion shows the events themselves.
//  5. **Structured-cloneable plain data**, so the stream crosses the worker
//     boundary unchanged. `null`, never `undefined`.
//
// COST
//
// A sink is called at most once per candidate per phase, plus a fixed handful
// per search. `NO_PROGRESS` is the default and its methods are empty, so a
// search with no observer builds no payloads at all — the cost of publication
// when nobody is listening is one call to an empty function per event site.
// The measured overhead of a real sink is in the sprint report.

import type { RejectionReason, TracedKind, Vec2Trace } from "../trace/contract";

export const SEARCH_PROGRESS_VERSION = "showboat-search-progress/1" as const;

/** Enough geometry to draw a candidate route, published before it is referenced. */
export interface ProgressCandidate {
  index: number;
  kind: TracedKind;
  eligible: boolean;
  target: number;
  potId: number;
  pocket: string;
  aimPoint: Vec2Trace;
  /** [cue ball, ghost-ball contact point] — the cue's own intended travel. */
  cuePath: Vec2Trace[];
  /** The intended object-ball route, starting at the object ball. */
  path: Vec2Trace[];
}

/** One candidate's place in the model's ordering. Straight from model output. */
export interface ProgressRank {
  index: number;
  /** 1-based global rank by calibrated score. */
  rank: number;
  score: number;
  logit: number | null;
}

/** The real outcome of one real simulation. Mirrors `CandidateVerification`. */
export interface ProgressPhysics {
  firstContact: number | null;
  legalFirstContact: boolean;
  scratched: boolean;
  legalPot: boolean;
  pocketed: number[];
  railsBeforePot: number;
}

interface Stamp {
  /** Increases by exactly 1 across the stream. */
  seq: number;
  /** ms since the search started. Never decreases. */
  atMs: number;
}

export type SearchProgressEvent = Stamp &
  (
    | { kind: "search-started"; mode: "classical" | "neural-hybrid"; budgetUnits: number }
    | { kind: "candidates-generated"; candidates: ProgressCandidate[] }
    /** The model's actual output, in its own order. Absent in classical mode. */
    | { kind: "neural-scored"; modelId: string; inferenceMs: number; ranks: ProgressRank[] }
    /** Which candidates the prior dropped before any physics ran, and why. */
    | { kind: "prior-pruned"; indices: number[]; keptTop: number; reservePromotions: number }
    /** This candidate is being handed to the simulator right now. */
    | { kind: "candidate-simulating"; index: number }
    /** That simulation returned. Emitted only after `simulateShotWasm` returns. */
    | { kind: "candidate-verified"; index: number; physics: ProgressPhysics }
    /** Eliminated, with the reason the search itself applied. */
    | { kind: "candidate-rejected"; index: number; reason: RejectionReason }
    /** Survived physics. `strength` is the search's own bounded value, not a %. */
    | { kind: "candidate-retained"; index: number; strength: number; visits: number }
    | { kind: "fallback"; cause: string; detail: string }
    | {
        kind: "selected";
        /** Null for a generated safety kick, which has no candidate row. */
        index: number | null;
        shotKind: string;
        rung: string;
      }
    | {
        kind: "search-completed";
        physicsUnitsSpent: number;
        physicsMs: number;
        physicsVerified: number;
        seedTimedOut: boolean;
        searchTimedOut: boolean;
      }
  );

/**
 * What the search calls. Methods take primitives so that the no-op
 * implementation constructs nothing — the point of publication being cheap is
 * that it can be left on in production without the search getting slower.
 */
export interface SearchProgressSink {
  started(mode: "classical" | "neural-hybrid", budgetUnits: number): void;
  candidatesGenerated(candidates: ProgressCandidate[]): void;
  neuralScored(modelId: string, inferenceMs: number, ranks: ProgressRank[]): void;
  priorPruned(indices: number[], keptTop: number, reservePromotions: number): void;
  simulating(index: number): void;
  verified(index: number, physics: ProgressPhysics): void;
  rejected(index: number, reason: RejectionReason): void;
  retained(index: number, strength: number, visits: number): void;
  fallback(cause: string, detail: string): void;
  selected(index: number | null, shotKind: string, rung: string): void;
  completed(
    physicsUnitsSpent: number,
    physicsMs: number,
    physicsVerified: number,
    seedTimedOut: boolean,
    searchTimedOut: boolean,
  ): void;
}

/**
 * The sink installed when nothing is observing. Every method is empty, so a
 * search with no observer pays one empty call per event site and allocates
 * nothing. This is the default everywhere, including the evaluation harness and
 * the unit suite, which is what keeps their timings comparable to before.
 */
export const NO_PROGRESS: SearchProgressSink = {
  started() {},
  candidatesGenerated() {},
  neuralScored() {},
  priorPruned() {},
  simulating() {},
  verified() {},
  rejected() {},
  retained() {},
  fallback() {},
  selected() {},
  completed() {},
};

/**
 * Build a real sink over `publish`.
 *
 * `seq` and `atMs` are assigned HERE and nowhere else, which is what makes
 * monotonicity a property of the emitter rather than a discipline expected of
 * eleven call sites. `now` is injectable so a test can drive a deterministic
 * clock without stubbing `performance`.
 */
export function createProgressSink(
  publish: (e: SearchProgressEvent) => void,
  now: () => number = () => performance.now(),
): SearchProgressSink {
  const t0 = now();
  let seq = 0;
  // Clamped to the previous value: a clock that goes backwards (which
  // `performance.now()` should not do, but a stubbed one can) must not be able
  // to emit a stream that violates the protocol's own ordering rule.
  let lastAt = 0;
  const stamp = (): Stamp => {
    const at = Math.max(lastAt, now() - t0);
    lastAt = at;
    return { seq: seq++, atMs: at };
  };
  return {
    started: (mode, budgetUnits) => publish({ ...stamp(), kind: "search-started", mode, budgetUnits }),
    candidatesGenerated: (candidates) =>
      publish({ ...stamp(), kind: "candidates-generated", candidates }),
    neuralScored: (modelId, inferenceMs, ranks) =>
      publish({ ...stamp(), kind: "neural-scored", modelId, inferenceMs, ranks }),
    priorPruned: (indices, keptTop, reservePromotions) =>
      publish({ ...stamp(), kind: "prior-pruned", indices, keptTop, reservePromotions }),
    simulating: (index) => publish({ ...stamp(), kind: "candidate-simulating", index }),
    verified: (index, physics) => publish({ ...stamp(), kind: "candidate-verified", index, physics }),
    rejected: (index, reason) => publish({ ...stamp(), kind: "candidate-rejected", index, reason }),
    retained: (index, strength, visits) =>
      publish({ ...stamp(), kind: "candidate-retained", index, strength, visits }),
    fallback: (cause, detail) => publish({ ...stamp(), kind: "fallback", cause, detail }),
    selected: (index, shotKind, rung) =>
      publish({ ...stamp(), kind: "selected", index, shotKind, rung }),
    completed: (physicsUnitsSpent, physicsMs, physicsVerified, seedTimedOut, searchTimedOut) =>
      publish({
        ...stamp(),
        kind: "search-completed",
        physicsUnitsSpent,
        physicsMs,
        physicsVerified,
        seedTimedOut,
        searchTimedOut,
      }),
  };
}

/** What `validateProgressStream` found wrong, or nothing. */
export interface ProgressViolation {
  at: number;
  problem: string;
}

/**
 * Check a stream against the four ordering rules above. Used by the tests that
 * drive a real search, and by nothing in production — this is a check on the
 * producer, not a filter on the consumer.
 */
export function validateProgressStream(events: SearchProgressEvent[]): ProgressViolation[] {
  const bad: ProgressViolation[] = [];
  const known = new Set<number>();
  const simulating = new Set<number>();
  let lastSeq = -1;
  let lastAt = -1;
  let completedAt = -1;

  events.forEach((e, i) => {
    if (e.seq !== lastSeq + 1) bad.push({ at: i, problem: `seq ${e.seq} follows ${lastSeq}` });
    lastSeq = e.seq;
    if (e.atMs < lastAt) bad.push({ at: i, problem: `atMs ${e.atMs} follows ${lastAt}` });
    lastAt = e.atMs;
    if (completedAt >= 0) bad.push({ at: i, problem: `${e.kind} after search-completed` });

    switch (e.kind) {
      case "candidates-generated":
        for (const c of e.candidates) known.add(c.index);
        break;
      case "candidate-simulating":
        if (!known.has(e.index)) bad.push({ at: i, problem: `simulating unknown candidate ${e.index}` });
        simulating.add(e.index);
        break;
      case "candidate-verified":
        if (!simulating.has(e.index)) {
          bad.push({ at: i, problem: `verified ${e.index} with no preceding simulation` });
        }
        break;
      case "candidate-rejected":
      case "candidate-retained":
        if (!known.has(e.index)) bad.push({ at: i, problem: `resolved unknown candidate ${e.index}` });
        break;
      case "neural-scored":
        for (const r of e.ranks) {
          if (!known.has(r.index)) bad.push({ at: i, problem: `ranked unknown candidate ${r.index}` });
        }
        break;
      case "selected":
        if (e.index !== null && !known.has(e.index)) {
          bad.push({ at: i, problem: `selected unknown candidate ${e.index}` });
        }
        break;
      case "search-completed":
        completedAt = i;
        break;
      default:
        break;
    }
  });
  return bad;
}
