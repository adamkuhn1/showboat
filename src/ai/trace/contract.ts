// ===========================================================================
// !! PLACEHOLDER — THIS FILE BELONGS TO THE ML TEAM (T2) !!
//
// The decision-trace contract is owned and published by T2 and consumed
// read-only by the rendering team (T3). T3 built the presentation against the
// contract as specified in
//   docs/repair/personal-authorship-sprint/recon/B_showboat_policy.md §5
// before T2's implementation landed, and needed the types to exist in order to
// typecheck.
//
// AT MERGE: delete this copy wholesale and keep T2's. It is transcribed from
// the spec, types only, zero runtime values other than the version constant —
// so if T2's version differs, T2's is right by definition.
// ===========================================================================

export const DECISION_TRACE_VERSION = "showboat-decision-trace/1" as const;

export type TrickKind = "bank" | "double-bank" | "combo" | "rail-combo";
export type TracedKind = TrickKind | "direct" | "safety-kick";

export type SelectionRung =
  | "trick-qualified"
  | "trick-below-threshold"
  | "trick-attempt-no-verified-pot"
  | "non-direct-safety"
  | "forced-legal-contact";

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

export interface ContactEvent {
  /** Ordered exactly as the simulator emitted it. */
  kind: "ball-ball" | "ball-cushion" | "pocket";
  timeSec: number;
  balls: number[];
  cushion: string | null; // set only for ball-cushion
  pocket: string | null; // set only for pocket
}

export interface RailContact {
  ballId: number;
  cushion: string;
  timeSec: number;
}

/** Result of running ONE real WASM simulation on this candidate. */
export interface PhysicsVerification {
  firstContact: number | null;
  legalFirstContact: boolean;
  scratched: boolean;
  legalPot: boolean; // isLegalPot() on the real sim
  pocketed: number[]; // real ids, in pocketing order
  railContacts: RailContact[];
  railsBeforePot: number;
  contactSequence: ContactEvent[];
  /** Bounded monotonic transform of the rollout value. NOT a probability. */
  strength: number;
  value: number;
  visits: number;
  styleScore: number;
}

export interface NeuralPrior {
  /** Calibrated make-estimate from the ranker, [0,1]. */
  score: number;
  /** Raw pre-calibration logit. */
  logit: number;
  /** 1-based global rank the model gave this candidate, before any reserve. */
  rank: number;
}

export interface TracedCandidate {
  index: number; // stable id = index in generation order
  kind: TracedKind;
  /** False for every "direct" while policy is "trick-only". */
  eligible: boolean;
  target: number;
  potId: number;
  pocket: string;
  aimPoint: Vec2Trace;
  path: Vec2Trace[];
  action: CueActionTrace;
  neural: NeuralPrior | null; // null in classical mode
  physics: PhysicsVerification | null; // null iff never simulated
  rejection: RejectionReason | null; // null iff this is the selected candidate
}

export interface SelectedShotTrace {
  /** null only when the shot is a generated safety, which has no candidate. */
  candidateIndex: number | null;
  kind: TrickKind | "safety-kick";
  rung: SelectionRung;
  action: CueActionTrace;
  path: Vec2Trace[];
  /** strength + STYLE_WEIGHT * styleScore, or null for a safety. */
  utility: number | null;
  reliabilityThreshold: number;
  qualifyingTricks: number;
}

export interface SearchBudget {
  physicsUnitsAllowed: number;
  physicsUnitsSpent: number;
  safetySimsSpent: number;
  candidatesGenerated: number;
  candidatesEligible: number;
  candidatesConsidered: number;
  physicsVerified: number;
  prunedByPrior: number;
  reservePromotions: number;
  seedTimedOut: boolean;
}

export interface DecisionTiming {
  totalMs: number;
  neuralEncodeMs: number | null;
  neuralRunMs: number | null;
  physicsMs: number;
  selectionMs: number;
}

export interface ModelIdentity {
  artifact: string;
  sha256: string;
  schema: string;
  hashVerified: boolean;
}

export interface FallbackTrace {
  from: "neural-hybrid";
  to: "classical-trick-only";
  cause: "model-absent" | "model-invalid" | "inference-error" | "inference-timeout" | "no-scores";
  detail: string;
}

export interface TurnContext {
  player: 0 | 1;
  shotIndex: number;
  legalTargets: number[];
}

/** THE contract. Everything the renderer may read, and nothing else. */
export interface DecisionTraceV1 {
  version: typeof DECISION_TRACE_VERSION;
  policy: "trick-only";
  mode: "neural-hybrid" | "classical-trick-only";
  turn: TurnContext;
  model: ModelIdentity | null;
  budget: SearchBudget;
  timing: DecisionTiming;
  /** Every generated candidate, directs included, in generation order. */
  candidates: TracedCandidate[];
  selected: SelectedShotTrace | null; // null iff no legal target existed
  fallback: FallbackTrace | null;
}
