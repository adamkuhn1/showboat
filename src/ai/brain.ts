import { type GameState, type PlayerId } from "../game/state";
import { type Table } from "../physics/table";
import {
  type SearchResult,
  type SearchConfig,
  type DecisionTrace,
  defaultConfig,
  searchCandidates,
  DEFAULT_PRIOR_KEEP_TOP,
  DEFAULT_PRIOR_RESERVE,
} from "./shotSearch";
import { generateCandidates } from "./candidates";
import { legalTargets } from "./turn";
import { CUE_ID } from "../game/rack";
import { neuralEvaluator, type NeuralCandidateEvaluator } from "./neural/evaluator";

/**
 * Two brains, one search.
 *
 * - `classical`: `generateCandidates` -> the authoritative UCB physics search,
 *   in raw generation order. No model involved at any point.
 * - `neural-hybrid`: `generateCandidates` -> one batched inference through the
 *   trained Phase 2D ranker -> the SAME authoritative physics search, over the
 *   same candidates, with the same total physics budget, but visited in the
 *   model's order and pruned to its top-K.
 *
 * The hybrid brain cannot reach a decision the physics didn't sanction: it
 * hands `searchCandidates` a prior and nothing else. Legality (`isLegalPot` on
 * a real simulation), value (real rollouts), and the trick-reliability
 * threshold are all downstream of the model and unaffected by it.
 */
export interface Brain {
  kind: "classical" | "neural-hybrid";
  /** Human-facing description of what is actually running. Never aspirational. */
  label: string;
  plan: (
    state: GameState,
    table: Table,
    player: PlayerId,
    config?: SearchConfig,
  ) => Promise<SearchResult>;
}

const emptyResult = (mode: DecisionTrace["mode"], reason: string): SearchResult => ({
  best: null,
  stats: [],
  simulations: 0,
  trace: {
    mode,
    candidatesGenerated: 0,
    candidatesConsidered: 0,
    prunedByPrior: 0,
    physicsVerified: 0,
    verifiedIndices: [],
    legalPots: 0,
    scratched: 0,
    physicsCalls: 0,
    reservePromotions: 0,
    fallbackReason: reason,
  },
});

export const classicalBrain = (): Brain => ({
  kind: "classical",
  label: "Physics search",
  plan: async (state, table, player, config) => {
    const targets = legalTargets(state, player);
    const cue = state.balls.find((b) => b.id === CUE_ID);
    if (!cue || targets.length === 0) return emptyResult("classical", "no legal target");
    return searchCandidates(
      generateCandidates(state.balls, table, targets),
      state.balls,
      targets,
      config ?? defaultConfig,
    );
  },
});

export const neuralHybridBrain = (
  evaluator: NeuralCandidateEvaluator = neuralEvaluator,
  keepTop: number = DEFAULT_PRIOR_KEEP_TOP,
  reserve = DEFAULT_PRIOR_RESERVE,
): Brain => ({
  kind: "neural-hybrid",
  label: "Neural evaluator + physics search",
  plan: async (state, table, player, config) => {
    const targets = legalTargets(state, player);
    const cue = state.balls.find((b) => b.id === CUE_ID);
    if (!cue || targets.length === 0) return emptyResult("neural-hybrid", "no legal target");

    const candidates = generateCandidates(state.balls, table, targets);
    const cfg = config ?? defaultConfig;
    const scored = await evaluator.score(state.balls, table, candidates);

    if (!scored) {
      // The model was expected but produced no usable scores. Run the
      // identical classical search and label the result "classical" in the
      // trace, with the reason attached — so no UI surface can present this
      // decision as a neural one.
      const evalState = evaluator.getState();
      const res = searchCandidates(candidates, state.balls, targets, cfg);
      return {
        ...res,
        trace: res.trace && {
          ...res.trace,
          fallbackReason:
            evalState.status === "ready"
              ? "model loaded but returned no scores for this candidate set"
              : `model unavailable: ${evalState.reason}`,
        },
      };
    }

    const manifest = evaluator.getManifest()!;
    return searchCandidates(candidates, state.balls, targets, {
      ...cfg,
      prior: {
        scores: scored.scores,
        keepTop,
        reserve,
        source: manifest.artifact.replace(/\.onnx$/, ""),
        inferenceMs: scored.inferenceMs,
      },
    });
  },
});

/**
 * Pick the brain for this turn. `preferNeural` is the user-facing toggle (the
 * model-disabled comparison mode Phase 2F requires); the evaluator's own
 * validated readiness is the hard gate. Asking for neural when no valid
 * artifact loaded gets the classical brain, honestly labelled — never a neural
 * label over a classical decision.
 */
export const getBrain = (
  preferNeural: boolean,
  evaluator: NeuralCandidateEvaluator = neuralEvaluator,
): Brain => (preferNeural && evaluator.isReady() ? neuralHybridBrain(evaluator) : classicalBrain());

/** The opponent description shown in the header. Derived from validated state. */
export const brainLabel = (
  preferNeural: boolean,
  evaluator: NeuralCandidateEvaluator = neuralEvaluator,
): string =>
  preferNeural && evaluator.isReady() ? "the neural + physics opponent" : "the physics-search opponent";
