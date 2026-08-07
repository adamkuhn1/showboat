// The decision-trace contract: everything a renderer is allowed to read about
// an AI decision, and nothing else.
//
// RULES THIS FILE ENFORCES BY CONSTRUCTION
//
//  1. **Types and constants only. Zero imports.** Nothing here can drag search,
//     physics or model code into a rendering bundle, and nothing here can be
//     broken by a refactor of any of them. `src/ai/trace/build.ts` is the only
//     module that maps internal search state onto these types; it can be
//     rewritten freely without changing what the renderer sees.
//  2. **JSON-serializable, `null` never `undefined`.** Absence is spelled
//     `null` at every field. No class instances, no functions, no `Map`/`Set`,
//     no typed arrays. `contract.test.ts` round-trips a real trace through
//     `JSON.parse(JSON.stringify(x))` and asserts deep equality, which fails
//     loudly if an optional property ever creeps back in.
//  3. **Every field is measured.** There is no derived-for-display quantity
//     here and no intermediate "event" that did not occur. A renderer may
//     animate *over* this data; it must not invent steps between the points.
//     In particular a non-null `physics` block means one real WASM simulation
//     ran on that candidate — `contract.test.ts` checks each such index against
//     the search's own `verifiedIndices`.
//  4. **Version-stamped**, so a future change is detectable rather than silent.
//
// THREE THINGS THIS CONTRACT KEEPS APART, AND WHY
//
// A shot has three descriptions and they are not interchangeable:
//
//   - the **candidate plan**: `TracedCandidate.cuePath` / `.path` and
//     `SelectedShotTrace.cuePath` / `.path`. Mirror geometry from the candidate
//     generator. An intention, computed before any physics ran;
//   - the **physics verification**: `TracedCandidate.physics`. What a real
//     simulation of that candidate did — legality, pot, rails, the event log;
//   - the **executed motion**: `SelectedShotTrace.executed`. The authoritative
//     run whose outcome is committed to the game state, as measured geometry.
//
// /1 published only the first two, so the only geometry a renderer could draw
// was the plan — and the overlay drew it as though it were the shot being
// played. /2 adds the third. A renderer may draw a plan as a plan; it may not
// draw a plan as the route the balls take.

export const DECISION_TRACE_VERSION = "showboat-decision-trace/2" as const;

/**
 * The four shot kinds Showboat is allowed to play. This is the product
 * decision, in the type system: bank and multi-wall combination routes only —
 * no jump shots and no massé exist in the action space at all, and a plain
 * `direct` pot is never selected in live play.
 */
export const TRICK_KINDS = ["bank", "double-bank", "combo", "rail-combo"] as const;
export type TrickKind = (typeof TRICK_KINDS)[number];

/** Everything that can appear in a trace, including the directs we reject. */
export type TracedKind = TrickKind | "direct" | "safety-kick";

/**
 * Which rung of the selection ladder produced the shot. Ordered strongest to
 * weakest; none of them can return a direct.
 */
export type SelectionRung =
  | "trick-qualified"
  | "trick-below-threshold"
  | "trick-attempt-no-verified-pot"
  | "non-direct-safety"
  | "forced-legal-contact";

/**
 * Why a candidate is not the shot being played. Exactly one reason per
 * candidate — the *first* one that applied, in the order the policy applied
 * them, so the string is the real cause and not a plausible-sounding one.
 */
export type RejectionReason =
  | "direct-excluded-by-policy"
  | "pruned-by-prior"
  | "budget-exhausted"
  | "seed-timeout"
  | "scratched-in-simulation"
  | "illegal-first-contact"
  | "did-not-pot"
  | "below-reliability-threshold"
  | "lower-utility-than-selected";

export interface Vec2Trace {
  x: number;
  y: number;
}

export interface CueActionTrace {
  phi: number;
  power: number;
  sideSpin: number;
  topSpin: number;
}

/** One entry of the simulator's own event log, in emission order. */
export interface ContactEvent {
  kind: "ball-ball" | "ball-cushion" | "pocket";
  timeSec: number;
  balls: number[];
  /** Set only for `ball-cushion`; `null` otherwise. */
  cushion: string | null;
  /** Set only for `pocket`; `null` otherwise. */
  pocket: string | null;
}

export interface RailContact {
  ballId: number;
  cushion: string;
  timeSec: number;
}

/**
 * The result of running ONE real WASM simulation on this candidate. Non-null
 * iff the search actually spent a physics call on it.
 */
export interface PhysicsVerification {
  /** First object ball the cue struck, or null if it struck nothing. */
  firstContact: number | null;
  /** Did that first contact match the candidate's intended target ball? */
  legalFirstContact: boolean;
  scratched: boolean;
  /** `isLegalPot()` on the real simulation, not a geometric estimate. */
  legalPot: boolean;
  /** Real ball ids, in the order they were pocketed. */
  pocketed: number[];
  railContacts: RailContact[];
  railsBeforePot: number;
  contactSequence: ContactEvent[];
  /**
   * Bounded monotonic transform of the rollout value, `1 - exp(-value)`.
   * **NOT a probability.** Nothing may render it with a `%`.
   */
  strength: number;
  /** Mean balls pocketed across the rollouts, the raw search value. */
  value: number;
  /** UCB visits. 0 when the candidate was simulated but never rolled out. */
  visits: number;
  /** Rails before the pot, plus 1 for combo/double-bank/rail-combo routes. */
  styleScore: number;
}

export interface NeuralPrior {
  /** Calibrated make-estimate from the ranker, in [0,1]. */
  score: number;
  /** Raw pre-calibration logit, or null if the runtime did not surface one. */
  logit: number | null;
  /** 1-based global rank the model gave this candidate, before any reserve. */
  rank: number;
}

export interface TracedCandidate {
  /** Stable id: position in `generateCandidates`'s output order. */
  index: number;
  kind: TracedKind;
  /** False for every `direct` while `policy === "trick-only"`. */
  eligible: boolean;
  /** Ball the cue must legally strike first. */
  target: number;
  /** Ball intended to drop — differs from `target` for combo/rail-combo. */
  potId: number;
  pocket: string;
  /** Ghost-ball contact point the cue is aimed through. */
  aimPoint: Vec2Trace;
  /**
   * The intended CUE-ball route, starting at the cue ball's actual position.
   * Always at least two points. Draw the aiming line from here — `path` starts
   * at the object ball and drawing from it misrepresents what the cue does.
   */
  cuePath: Vec2Trace[];
  /** The intended OBJECT-ball route, starting at the object ball. */
  path: Vec2Trace[];
  action: CueActionTrace;
  /** Null in classical mode, or when the prior did not cover this candidate. */
  neural: NeuralPrior | null;
  /** Null iff no simulation was ever run on this candidate. */
  physics: PhysicsVerification | null;
  /** Null iff this candidate is the one being played. */
  rejection: RejectionReason | null;
}

/** What interrupted a stretch of a ball's measured motion. Each is a real event. */
export type TrajectoryBreakKind = "ball-contact" | "cushion" | "pocket";

export interface TrajectoryBreak {
  /** Index into the trajectory's `points`. */
  at: number;
  kind: TrajectoryBreakKind;
  timeSec: number;
  /** The other ball, for `ball-contact`. Null otherwise. */
  withBall: number | null;
  /** Cushion side, for `cushion`. Null otherwise. */
  cushion: string | null;
  /** Pocket id, for `pocket`. Null otherwise. */
  pocket: string | null;
}

/**
 * What a ball DID in the executed shot, read off the event log.
 *
 * Never the candidate's intent. A shot planned as a combination that struck the
 * wrong ball has to read as what happened, so `first-contact` is the ball the
 * cue really hit and `combination` is a ball really set moving by another
 * object ball. A ball can hold several of these at once.
 */
export type TrajectoryRole = "cue" | "first-contact" | "combination" | "potted";

/** One ball's measured route through one executed simulation. */
export interface ExecutedTrajectory {
  ballId: number;
  roles: TrajectoryRole[];
  /** Position among the balls that moved, ordered by when each started. */
  order: number;
  /**
   * Recorded positions, in simulation-time order, at least two. Straight runs
   * are thinned within `ExecutedMotion.simplifyToleranceM`; no point is moved,
   * invented, or interpolated, and no vertex at a `break` is ever dropped.
   *
   * A potted ball's route ENDS at its last measured position — at the pocket
   * lip, not in the pocket. The simulator stops recording a position for a ball
   * once it is captured, and drawing on to the pocket centre would be geometry
   * nothing measured.
   */
  points: Vec2Trace[];
  /** `timesSec[i]` is the simulation time of `points[i]`. Same length as `points`. */
  timesSec: number[];
  breaks: TrajectoryBreak[];
  startSec: number;
  endSec: number;
  pocketed: boolean;
}

/**
 * The motion the authoritative simulation actually produced.
 *
 * This is the run whose result is committed to the game state and whose
 * waypoints the animation replays — so a route drawn from here is the route the
 * balls take, not a route they were meant to take.
 */
export interface ExecutedMotion {
  /** Simulated seconds the shot took. */
  durationSec: number;
  /** Every ball that moved, in order of first motion. Cue ball first. */
  trajectories: ExecutedTrajectory[];
  /** The simulator's own contact log for this run, in emission order. */
  contactSequence: ContactEvent[];
  /** The straight-line thinning bound that was applied, in metres. */
  simplifyToleranceM: number;
  /**
   * The largest deviation the thinning actually introduced, in metres —
   * measured while simplifying, not asserted. Always <= the tolerance.
   */
  maxDeviationM: number;
}

export interface SelectedShotTrace {
  /**
   * Index into `DecisionTraceV1.candidates` — i.e. generation order. Null only
   * for a generated safety kick, which is not a member of the candidate list.
   * A renderer highlighting the shot being played must address it by THIS,
   * never by position in a re-sorted list.
   */
  candidateIndex: number | null;
  kind: TrickKind | "safety-kick";
  rung: SelectionRung;
  action: CueActionTrace;
  /**
   * INTENDED cue-ball route, from the cue ball's real position. At least two
   * points. This is plan geometry — see `executed` for what the cue ball did.
   */
  cuePath: Vec2Trace[];
  /** INTENDED object-ball route. Empty for a safety kick, which plans no pot. */
  path: Vec2Trace[];
  /**
   * What the table actually did, from the authoritative simulation.
   *
   * Null in two cases and no others: the trace was published by the search,
   * which finishes BEFORE the shot is executed (so a trace read straight off
   * `brain.plan` carries null here, and `ui/planner/plan.ts` fills it in once
   * the shot has been run); or that simulation captured no waypoints, which is
   * true of the TS reference engine and of every search rollout.
   *
   * When it is null there is nothing measured to draw and a renderer must draw
   * nothing — `cuePath`/`path` are not a stand-in for it.
   */
  executed: ExecutedMotion | null;
  /** `strength + STYLE_WEIGHT * styleScore`, or null for a safety kick. */
  utility: number | null;
  /** The bar a trick must clear on rung 1. A physics-derived threshold. */
  reliabilityThreshold: number;
  /** Verified tricks that pot AND clear that threshold. */
  qualifyingTricks: number;
  /**
   * How far the safety rung's own verification got, straight from
   * `SafetyResult.quality`. Null for every non-safety shot.
   *
   *   `foul-free`          a real simulation showed `applyShotRules().foul === false`
   *   `legal-contact-only` fouled, but the cue did strike a legal target
   *   `unverified`         nothing survived simulation; the shortest kick is
   *                        being played at minimum power and may well foul
   *
   * This field exists because it was being thrown away. `unverified` and
   * `legal-contact-only` both mapped onto the rung `forced-legal-contact`, and
   * the panel described that rung as "the shortest legal contact" — an
   * assertion nothing had established, on a shot that scratched 94 % of the
   * time in an adversarial sample. The rung says which branch fired; this says
   * what the physics actually knew when it fired.
   */
  safetyQuality: "foul-free" | "legal-contact-only" | "unverified" | null;
}

export interface SearchBudget {
  physicsUnitsAllowed: number;
  physicsUnitsSpent: number;
  /** Extra real simulations spent verifying safety kicks. Never netted. */
  safetySimsSpent: number;
  candidatesGenerated: number;
  /** Candidates the policy allows to be selected at all (non-directs). */
  candidatesEligible: number;
  /** Candidates handed to the physics search after prior pruning. */
  candidatesConsidered: number;
  physicsVerified: number;
  prunedByPrior: number;
  reservePromotions: number;
  seedTimedOut: boolean;
}

export interface DecisionTiming {
  totalMs: number;
  /** Null in classical mode. */
  neuralEncodeMs: number | null;
  neuralRunMs: number | null;
  physicsMs: number;
  selectionMs: number;
}

export interface ModelIdentity {
  artifact: string;
  sha256: string;
  schema: string;
  /** Did the runtime actually recompute and match the sha256 this session? */
  hashVerified: boolean;
}

export interface FallbackTrace {
  /**
   * What was expected to produce the decision. `planning-worker` covers the one
   * failure the worker cannot report on its own behalf: it stopped answering.
   */
  from: "neural-hybrid" | "planning-worker";
  /** What produced it instead. */
  to: "classical-trick-only" | "main-thread-classical";
  /**
   * Why. `no-candidates` is deliberately separate from `no-scores`: both end in
   * the same classical trick-only search, but one is "there was nothing to
   * rank" and the other is "the model was asked and gave nothing back".
   * Collapsing them blames the model for an empty board.
   *
   * The three timeouts are separate for the same reason. `inference-timeout` is
   * a `session.run()` that did not come back; `model-load-timeout` is a session
   * that was never built (a stalled fetch, a stalled `InferenceSession.create`);
   * `planner-timeout` is the worker going silent, which is not a model failure
   * at all. Every member of this union is emitted from exactly one branch, and
   * `brain.fallback.test.ts` drives each of those branches.
   */
  cause:
    | "model-absent"
    | "model-invalid"
    | "model-load-timeout"
    | "inference-error"
    | "inference-timeout"
    | "planner-timeout"
    | "no-candidates"
    | "no-scores";
  /** Human-readable, non-empty. The UI is required to be able to show it. */
  detail: string;
}

export interface TurnContext {
  player: 0 | 1;
  shotIndex: number;
  legalTargets: number[];
  /** Cue-ball position at decision time. The origin of every `cuePath`. */
  cueBall: Vec2Trace;
}

/**
 * THE contract. Everything the renderer may read about one AI decision.
 *
 * `candidates` is in **generation order with directs included**. Directs carry
 * `eligible: false` and `rejection: "direct-excluded-by-policy"`: they are
 * truthfully-labelled rejected comparisons, which is the only role a direct
 * has in live play.
 */
export interface DecisionTraceV1 {
  version: typeof DECISION_TRACE_VERSION;
  /** Invariant: always `"trick-only"` for a live decision. */
  policy: "trick-only";
  /** Derived from whether a prior was actually applied, never from intent. */
  mode: "neural-hybrid" | "classical-trick-only";
  turn: TurnContext;
  /** Non-null iff `mode === "neural-hybrid"`. */
  model: ModelIdentity | null;
  budget: SearchBudget;
  timing: DecisionTiming;
  candidates: TracedCandidate[];
  /** Null iff there was no legal target at all — no shot could be taken. */
  selected: SelectedShotTrace | null;
  /** Non-null iff `mode === "classical-trick-only"` after a neural request. */
  fallback: FallbackTrace | null;
}
