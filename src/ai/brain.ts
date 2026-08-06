import { type GameState, type PlayerId } from "../game/state";
import { type Table } from "../physics/table";
import { simulateShotWasm } from "../physics/wasm-bridge";
import {
  type SearchResult,
  type SearchConfig,
  type SearchOutcome,
  type SelectionReason,
  defaultConfig,
  searchCandidates,
  DEFAULT_PRIOR_KEEP_TOP,
} from "./shotSearch";
import { generateCandidates, type CandidateKind } from "./candidates";
import { legalTargets } from "./turn";
import { CUE_ID } from "../game/rack";
import { neuralEvaluator, type NeuralCandidateEvaluator } from "./neural/evaluator";
import {
  isTrickCandidate,
  selectTrickOnly,
  TrickOnlyInvariantError,
  type PlayableShot,
  type TrickOnlyDecision,
} from "./policy/trickOnly";
import { buildDecisionTrace } from "./trace/build";
import {
  type DecisionTraceV1,
  type FallbackTrace,
  type ModelIdentity,
  type SelectionRung,
} from "./trace/contract";
import { rankerHashWasVerified } from "./onnx";

/**
 * Two brains, ONE policy.
 *
 * - `neural-hybrid`: `generateCandidates` -> one batched inference through the
 *   trained Deep Sets ranker -> the authoritative physics search, visited in
 *   the model's order and pruned to its top-K -> `selectTrickOnly`.
 * - `classical-trick-only`: the same, minus the prior. This is also the
 *   fallback whenever the model is absent, invalid, throws, times out, or
 *   returns nothing usable.
 *
 * There is no brain, mode, toggle or failure path in which a different
 * selection function runs. Both call `selectTrickOnly`, and there is exactly
 * one `selectTrickOnly` — which is why "Showboat never plays a direct" is a
 * property of the code rather than a claim about it.
 *
 * The model still cannot reach a decision the physics didn't sanction: it
 * supplies a prior over which candidates get simulated, nothing more. Legality
 * (`isLegalPot` on a real simulation), value (real rollouts) and the trick
 * reliability threshold are all downstream of it.
 */
export type BrainKind = "neural-hybrid" | "classical-trick-only";

/**
 * What an AI turn produces.
 *
 * `shot` is the only playable thing here and it is branded — see
 * `policy/trickOnly.ts`. `best` and `stats` are retained for the existing
 * overlay renderer; `best` is the selected candidate's stat, or null when the
 * shot is a generated safety kick that has no candidate. `decision` is the
 * published trace contract and is what new rendering code should read.
 */
export interface AiDecision extends SearchResult {
  shot: PlayableShot | null;
  decision: DecisionTraceV1;
}

export interface Brain {
  kind: BrainKind;
  /** Human-facing description of what is actually running. Never aspirational. */
  label: string;
  plan: (
    state: GameState,
    table: Table,
    player: PlayerId,
    config?: SearchConfig,
  ) => Promise<AiDecision>;
}

/**
 * Deadline on neural inference.
 *
 * Not a guess: the corrected gate measured this artifact's inference at a
 * median of 1.63 ms and a p95 of 4.60 ms over 400 fixtures
 * (docs/repair/visual-authorship/showboat/CORRECTED_GATE_RESULT.md section 5).
 * 250 ms is ~54x that p95 — it cannot fire on a slow-but-working machine, and
 * it guarantees a stalled onnxruntime call can never wedge an AI turn. The race
 * loser is treated exactly like "the model returned nothing": classical
 * trick-only, labelled `inference-timeout`.
 */
export const NEURAL_SCORE_DEADLINE_MS = 250;

const withDeadline = async <T,>(
  work: Promise<T>,
  ms: number,
): Promise<{ ok: true; value: T } | { ok: false }> => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<{ ok: false }>((resolve) => {
    timer = setTimeout(() => resolve({ ok: false }), ms);
  });
  try {
    return await Promise.race([work.then((value) => ({ ok: true as const, value })), deadline]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
};

const emptySearch = (): SearchOutcome => ({
  stats: [],
  allStats: [],
  verifications: [],
  simulations: 0,
  trace: {
    mode: "classical",
    candidatesGenerated: 0,
    candidatesConsidered: 0,
    prunedByPrior: 0,
    physicsVerified: 0,
    verifiedIndices: [],
    consideredIndices: [],
    legalPots: 0,
    scratched: 0,
    physicsCalls: 0,
    reservePromotions: 0,
    ineligible: 0,
    seedTimedOut: false,
    physicsMs: 0,
  },
});

const emptyDecision = (): TrickOnlyDecision => ({
  shot: null,
  rung: null,
  qualifyingTricks: 0,
  utility: null,
  excludedIndices: [],
  safetySimsSpent: 0,
  safety: null,
});

const legacyReasonFor = (rung: SelectionRung | null): SelectionReason => {
  switch (rung) {
    case "trick-qualified":
      return "trick-qualified";
    case "trick-below-threshold":
      return "no-trick-qualified";
    case "trick-attempt-no-verified-pot":
      return "no-verified-pot";
    default:
      return "none";
  }
};

/**
 * Assemble the turn result: the branded shot, the published trace, and the
 * legacy `best`/`stats` shape the current overlay renderer still reads.
 *
 * `best` is looked up from the selected candidate's generation index, so the
 * highlighted candidate is by construction the shot being played. It is null
 * for a safety kick, which is not a member of the candidate list — a renderer
 * must read `decision.selected` for that case rather than infer one.
 */
const assemble = (
  outcome: SearchOutcome,
  decision: TrickOnlyDecision,
  state: GameState,
  player: PlayerId,
  targets: number[],
  physicsUnitsAllowed: number,
  model: ModelIdentity | null,
  fallback: FallbackTrace | null,
  timing: { totalMs: number; neuralEncodeMs: number | null; neuralRunMs: number | null; selectionMs: number },
): AiDecision => {
  const idx = decision.shot?.candidateIndex ?? null;
  return {
    ...outcome,
    best: idx === null ? null : outcome.allStats[idx] ?? null,
    // Compatibility shim for the CURRENT overlay panel, which renders a
    // sentence keyed off the legacy `SelectionReason`. The mapping is lossy but
    // not untrue: rung 2 really is "no trick cleared the reliability bar", rung
    // 3 really is "nothing potted in simulation", and rungs 4-5 really are "no
    // trick candidate survived physics verification". New rendering should read
    // `decision.selected.rung`, which has all five values; this line goes away
    // when it does.
    trace: outcome.trace && { ...outcome.trace, selectionReason: legacyReasonFor(decision.rung) },
    shot: decision.shot,
    decision: buildDecisionTrace({
      outcome,
      decision,
      state,
      player,
      targets,
      physicsUnitsAllowed,
      model,
      fallback,
      timing: {
        totalMs: timing.totalMs,
        neuralEncodeMs: timing.neuralEncodeMs,
        neuralRunMs: timing.neuralRunMs,
        physicsMs: outcome.trace?.physicsMs ?? 0,
        selectionMs: timing.selectionMs,
      },
    }),
  };
};

/**
 * Run the search and the policy. Shared by both brains, so the fallback path
 * is not a second implementation that could drift from the primary one.
 */
const decide = (
  state: GameState,
  table: Table,
  player: PlayerId,
  targets: number[],
  cfg: SearchConfig,
  model: ModelIdentity | null,
  fallback: FallbackTrace | null,
  prior: SearchConfig["prior"],
  neuralEncodeMs: number | null,
  neuralRunMs: number | null,
): AiDecision => {
  const t0 = performance.now();
  const candidates = generateCandidates(state.balls, table, targets);
  // The eligibility filter is a BUDGET decision, not the exclusion mechanism:
  // it stops the search paying for simulations of shots the policy could never
  // select. Directs still appear in the trace, with `physics: null` and an
  // honest rejection reason. `trickOnly.test.ts` A7 proves selection alone is
  // sufficient by running with this filter off.
  const outcome = searchCandidates(candidates, state.balls, targets, {
    ...cfg,
    prior,
    eligible: isTrickCandidate,
  });

  const tSel = performance.now();
  let decision: TrickOnlyDecision;
  try {
    decision = selectTrickOnly(outcome.allStats, outcome.verifications, {
      // `turn: player` explicitly: the safety rung verifies its kicks through
      // the real `applyShotRules`, which decides first-contact legality from
      // `pre.turn`'s group. Deriving the shooter from the argument rather than
      // from the state means a caller that plans for a player who is not to
      // move cannot silently get the other player's legality rules.
      state: { ...state, turn: player },
      table,
      targets,
      simulate: simulateShotWasm,
    });
  } catch (e) {
    // Layer 3. Unreachable unless the type or the partition has been broken by
    // a later edit. Degrade to the safety rung — a worse shot, never a direct,
    // and never a rejected promise that would hang the turn.
    if (!(e instanceof TrickOnlyInvariantError)) throw e;
    console.error(`[showboat] trick-only invariant violated: ${e.message}`);
    decision = selectTrickOnly([], [], {
      state: { ...state, turn: player },
      table,
      targets,
      simulate: simulateShotWasm,
    });
  }
  const selectionMs = performance.now() - tSel;

  return assemble(outcome, decision, state, player, targets, cfg.simulations, model, fallback, {
    totalMs: performance.now() - t0,
    neuralEncodeMs,
    neuralRunMs,
    selectionMs,
  });
};

const noLegalTarget = (
  state: GameState,
  player: PlayerId,
  cfg: SearchConfig,
  model: ModelIdentity | null,
  fallback: FallbackTrace | null,
): AiDecision =>
  assemble(emptySearch(), emptyDecision(), state, player, [], cfg.simulations, model, fallback, {
    totalMs: 0,
    neuralEncodeMs: null,
    neuralRunMs: null,
    selectionMs: 0,
  });

export const classicalTrickOnlyBrain = (): Brain => ({
  kind: "classical-trick-only",
  label: "Physics search",
  plan: async (state, table, player, config) => {
    const cfg = config ?? defaultConfig;
    const targets = legalTargets(state, player);
    const cue = state.balls.find((b) => b.id === CUE_ID);
    if (!cue || targets.length === 0) return noLegalTarget(state, player, cfg, null, null);
    return decide(state, table, player, targets, cfg, null, null, undefined, null, null);
  },
});

export const neuralTrickOnlyBrain = (
  evaluator: NeuralCandidateEvaluator = neuralEvaluator,
  keepTop: number = DEFAULT_PRIOR_KEEP_TOP,
  // Trick-only reserves nothing for directs: reserving physics slots for a kind
  // that can never be selected would be spending the budget on nothing. This is
  // part of arm B of the pre-registered evaluation, stated rather than slipped
  // in. Correctness does not depend on it.
  reserve: Partial<Record<CandidateKind, number>> = {},
): Brain => ({
  kind: "neural-hybrid",
  label: "Neural evaluator + physics search",
  plan: async (state, table, player, config) => {
    const cfg = config ?? defaultConfig;
    const targets = legalTargets(state, player);
    const cue = state.balls.find((b) => b.id === CUE_ID);
    if (!cue || targets.length === 0) return noLegalTarget(state, player, cfg, null, null);

    const candidates = generateCandidates(state.balls, table, targets);

    // Every way inference can fail resolves to the SAME classical trick-only
    // search, with an accurate `fallback.cause`. None of them can hang: the
    // evaluator never throws (it catches ORT errors internally) and the call is
    // additionally deadline-raced.
    const raced = await withDeadline(
      evaluator.score(state.balls, table, candidates),
      NEURAL_SCORE_DEADLINE_MS,
    );
    const scored = raced.ok ? raced.value : null;

    if (!scored) {
      const evalState = evaluator.getState();
      const cause: FallbackTrace["cause"] = !raced.ok
        ? "inference-timeout"
        : evalState.status === "ready"
          ? "no-scores"
          : evalState.status === "invalid"
            ? "inference-error"
            : "model-absent";
      const detail = !raced.ok
        ? `inference exceeded ${NEURAL_SCORE_DEADLINE_MS} ms`
        : evalState.status === "ready"
          ? "model loaded but returned no scores for this candidate set"
          : `model unavailable: ${evalState.reason}`;
      const res = decide(state, table, player, targets, cfg, null, { from: "neural-hybrid", to: "classical-trick-only", cause, detail }, undefined, null, null);
      return {
        ...res,
        trace: res.trace && { ...res.trace, fallbackReason: detail },
      };
    }

    const manifest = evaluator.getManifest()!;
    const model: ModelIdentity = {
      artifact: manifest.artifact,
      sha256: manifest.onnx_sha256,
      schema: manifest.schema_version,
      hashVerified: rankerHashWasVerified(),
    };
    return decide(
      state,
      table,
      player,
      targets,
      cfg,
      model,
      null,
      {
        scores: scored.scores,
        logits: scored.logits,
        keepTop,
        reserve,
        source: manifest.artifact.replace(/\.onnx$/, ""),
        inferenceMs: scored.inferenceMs,
      },
      scored.encodeMs,
      scored.runMs,
    );
  },
});

/**
 * Pick the brain for this turn. `preferNeural` is the user-facing toggle (the
 * model-disabled comparison mode); the evaluator's own validated readiness is
 * the hard gate. Asking for neural when no valid artifact loaded gets the
 * classical trick-only brain, honestly labelled — never a neural label over a
 * classical decision, and never a policy change either way.
 */
export const getBrain = (
  preferNeural: boolean,
  evaluator: NeuralCandidateEvaluator = neuralEvaluator,
): Brain =>
  preferNeural && evaluator.isReady() ? neuralTrickOnlyBrain(evaluator) : classicalTrickOnlyBrain();

/**
 * The opponent description. Derived from validated model state.
 *
 * It names which *evaluator* is running, not which policy: the policy is
 * always trick-only, in every mode, so putting it in this string would say
 * nothing. `src/ui/opponentClaims.test.ts` pins these exact strings.
 */
export const brainLabel = (
  preferNeural: boolean,
  evaluator: NeuralCandidateEvaluator = neuralEvaluator,
): string =>
  preferNeural && evaluator.isReady() ? "the neural + physics opponent" : "the physics-search opponent";
