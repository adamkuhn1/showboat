import { type GameState } from "../game/state";
import { type Table } from "../physics/table";
import { type SearchResult, type SearchConfig, defaultConfig } from "./shotSearch";
import { hasTrainedModel, trainedModelStatus, evaluate, encodeObservation } from "./onnx";

// Two modes: flat UCB candidate search (default, no network needed) and trained net
// (same search but the seeding phase uses the ONNX value estimate instead of
// per-candidate rollouts, freeing the budget for UCB refinement).

export interface Brain {
  kind: "baseline" | "trained";
  plan: (
    state: GameState,
    table: Table,
    runSearch: (s: GameState, t: Table, cfg?: SearchConfig) => SearchResult,
  ) => Promise<SearchResult>;
}

export const getBrain = (): Brain => {
  if (hasTrainedModel()) {
    return {
      kind: "trained",
      plan: async (state, table, runSearch) => {
        const obs = encodeObservation(
          state.balls.map((b) => ({ id: b.id, pos: b.pos, pocketed: b.pocketed })),
          table.length / 2,
          table.width / 2,
        );
        const netOut = await evaluate(obs);
        if (netOut !== null) {
          return runSearch(state, table, { ...defaultConfig, netSeedValue: netOut.value });
        }
        return runSearch(state, table);
      },
    };
  }
  return {
    kind: "baseline",
    plan: async (state, table, runSearch) => runSearch(state, table),
  };
};

export const brainLabel = (): string => {
  const status = trainedModelStatus();
  if (status === "loaded") return "trained AI";
  return "the AI";
};
