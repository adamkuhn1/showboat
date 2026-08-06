// The one message channel between the page and the planning worker.
// Everything crossing it is structured-cloneable plain data: `GameState`,
// `Table` and `DecisionTraceV1` are all plain objects by construction.

import type { GameState, PlayerId } from "../../game/state";
import type { Table } from "../../physics/table";
import type { PlannedTurn } from "./plan";

export interface PlanRequest {
  type: "plan";
  id: number;
  state: GameState;
  table: Table;
  player: PlayerId;
  useNeural: boolean;
  /** Absolute; see `PlanInput.modelDir` for why it cannot be relative here. */
  modelDir: string;
}

/** Build the ranker session before the first turn needs it. */
export interface WarmRequest {
  type: "warm";
  modelDir: string;
}

export type WorkerRequest = PlanRequest | WarmRequest;

export type PlanResponse =
  /** The model session is being created — the first neural turn pays for it. */
  | { type: "model-loading"; id: number }
  | { type: "model-status"; id: number; ok: boolean; reason: string | null; hashVerified: boolean }
  /**
   * The finished turn, exactly as `planTurnTraced` returned it — including the
   * `no-legal-shot` case, which crosses the boundary as data rather than as a
   * missing field. In the `shot` case `report` carries the authoritative
   * simulation: its waypoints are what the page replays, and its event trace is
   * where the cushion/contact marks come from. All plain data: `Ball`,
   * `SimResult`, `ShotOutcome`, `GameState` and `DecisionTraceV1` are
   * structured-cloneable by construction (`Motion` is a string enum), so
   * nothing needs a transfer list.
   */
  | { type: "done"; id: number; planned: PlannedTurn }
  | { type: "error"; id: number; message: string };
