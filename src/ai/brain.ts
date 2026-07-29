import { type GameState } from "../game/state";
import { type Table } from "../physics/table";
import { type SearchResult, type SearchConfig } from "./mcts";
import { hasTrainedModel, trainedModelStatus } from "./onnx";

// The "brain" is the seam between the two possible decision-makers:
//
//   1. BASELINE — pure-search MCTS with uniform priors + physics rollout value
//      (the Rust hot loop). No learned network. This is what ships until the
//      Colab training run lands.
//   2. TRAINED — the same MCTS but with priors + leaf value supplied by the
//      ONNX policy/value net exported from the LightZero self-play training.
//
// HONESTY RULE (PLAN.md §5 / CLAUDE.md #2): the UI must always be truthful about
// which brain is actually playing. `brainLabel()` reflects the real state — it
// says "search baseline" until a trained model file is present and loaded, and
// only then "trained net". We never present the baseline as the trained AI.

export type PlanFn = (
  state: GameState,
  table: Table,
  runSearch: (s: GameState, t: Table, cfg?: SearchConfig) => SearchResult,
) => SearchResult;

export interface Brain {
  kind: "baseline" | "trained";
  plan: (
    state: GameState,
    table: Table,
    runSearch: (s: GameState, t: Table, cfg?: SearchConfig) => SearchResult,
  ) => SearchResult;
}

// The baseline brain just runs the provided search (which already uses uniform
// priors + rollout value). When the ONNX net is loaded, the search config /
// value source is swapped inside runSearch; here we keep the plumbing simple.
export const getBrain = (): Brain => {
  if (hasTrainedModel()) {
    return {
      kind: "trained",
      plan: (state, table, runSearch) => runSearch(state, table),
    };
  }
  return {
    kind: "baseline",
    plan: (state, table, runSearch) => runSearch(state, table),
  };
};

// Truthful label for the UI. Reflects whether a real trained model is loaded.
export const brainLabel = (): string => {
  const status = trainedModelStatus();
  if (status === "loaded") return "trained net (ONNX)";
  return "search baseline";
};
