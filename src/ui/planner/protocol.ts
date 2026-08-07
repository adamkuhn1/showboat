// The one message channel between the page and the planning worker.
// Everything crossing it is structured-cloneable plain data: `GameState`,
// `Table` and `DecisionTraceV1` are all plain objects by construction.

import type { GameState, PlayerId } from "../../game/state";
import type { Table } from "../../physics/table";
import type { SearchProgressEvent } from "../../ai/search/progress";
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
  /**
   * One live search event, posted AS IT HAPPENS rather than collected and sent
   * at the end. This is the message that makes the reasoning display an
   * observation of the search instead of a reconstruction of its result.
   *
   * Each one also re-arms the host's silence watchdog, which is a second,
   * unplanned benefit: a search that is working can no longer be mistaken for a
   * worker that has died, because it is now audibly working.
   */
  | { type: "progress"; id: number; event: SearchProgressEvent }
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
