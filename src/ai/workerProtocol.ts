import { type Vec2 } from "../physics/vec";
import { type Table } from "../physics/table";
import { type GameState } from "../game/state";
import { aiTakeTurn, type AiMove, type EvaluatedCandidate } from "./agent";
import { type CandidateKind } from "./candidates";
import { makeRanker, type RankerName } from "./ranker";
import { makeRng } from "./rollout";

// The message contract between the page and the AI worker (ai.worker.ts).
//
// The worker runs aiTakeTurn unchanged; each of its hooks becomes a message
// to the page. Everything that crosses is plain data (GameState, Table,
// ShotReport, EvaluatedCandidate), so structured clone carries it as-is. The
// ranker holds a function, so only its name crosses and the worker rebuilds
// it with makeRanker.
//
// Every message carries the turn id it belongs to, so a late message from an
// abandoned turn can be told apart from the current one.

export type ToWorker =
  | {
      type: "start";
      id: number;
      state: GameState;
      table: Table;
      ranker: RankerName;
      delayMs: number;
      seed?: number; // deterministic jitter (tests); Math.random otherwise
    }
  // The page finished showing the ball-in-hand placement; carry on.
  | { type: "placed-ack"; id: number };

export type FromWorker =
  | { type: "phase"; id: number; phase: string }
  | { type: "placed"; id: number; placed: GameState; spot: Vec2 }
  | {
      type: "generated";
      id: number;
      count: number;
      byKind: Record<CandidateKind, number>;
    }
  | {
      type: "candidate";
      id: number;
      // Progress only: the verification report (with every frame) is left
      // out, since the panel shows the measurements. The final `done`
      // decision keeps the full reports.
      ec: EvaluatedCandidate;
      index: number;
      total: number;
    }
  | { type: "done"; id: number; move: AiMove | null }
  | { type: "error"; id: number; message: string };

// The subset of a worker scope / MessagePort that both sides use. A real
// worker `self` and a Node or browser MessagePort all fit.
export interface Port<In, Out> {
  postMessage(msg: Out): void;
  addEventListener(type: "message", fn: (e: MessageEvent<In>) => void): void;
  start?(): void;
}

// Worker side: answer every `start` with one AI turn.
export const serveAiTurn = (port: Port<ToWorker, FromWorker>): void => {
  const acks = new Map<number, () => void>();

  const run = async (msg: Extract<ToWorker, { type: "start" }>): Promise<void> => {
    const { id } = msg;
    const post = (m: FromWorker): void => port.postMessage(m);
    try {
      const move = await aiTakeTurn(msg.state, msg.table, makeRanker(msg.ranker), {
        delayMs: msg.delayMs,
        rng: msg.seed !== undefined ? makeRng(msg.seed) : undefined,
        onPhase: (phase) => post({ type: "phase", id, phase }),
        onPlaced: (placed, spot) =>
          new Promise<void>((resolve) => {
            acks.set(id, resolve);
            post({ type: "placed", id, placed, spot });
          }),
        onGenerated: (count, byKind) => post({ type: "generated", id, count, byKind }),
        onCandidate: (ec, index, total) =>
          post({ type: "candidate", id, ec: { ...ec, report: undefined }, index, total }),
      });
      post({ type: "done", id, move });
    } catch (err) {
      post({ type: "error", id, message: err instanceof Error ? err.message : String(err) });
    } finally {
      acks.delete(id);
    }
  };

  port.addEventListener("message", (e) => {
    const msg = e.data;
    if (msg.type === "start") {
      void run(msg);
    } else if (msg.type === "placed-ack") {
      acks.get(msg.id)?.();
      acks.delete(msg.id);
    }
  });
  port.start?.();
};
