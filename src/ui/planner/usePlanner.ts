import { useCallback, useEffect, useRef } from "react";
import type { GameState, PlayerId } from "../../game/state";
import type { Table } from "../../physics/table";
import type { PlanRequest, PlanResponse } from "./protocol";
import { planTurnTraced, type ModelStatus, type PlannedTurn } from "./plan";

export interface PlanCallbacks {
  onModelLoading?: () => void;
  onModelStatus?: (s: ModelStatus) => void;
}

export interface Planner {
  plan: (
    state: GameState,
    table: Table,
    player: PlayerId,
    useNeural: boolean,
    cb?: PlanCallbacks,
  ) => Promise<PlannedTurn>;
  /** False when the worker could not be constructed and planning is inline. */
  offMainThread: () => boolean;
}

/**
 * Owns the planning worker.
 *
 * The worker is the path that ships. The inline fallback exists because a
 * module worker can fail to construct in environments the game still has to
 * run in — a `file://` page, a hardened CSP, a test runner with no `Worker` —
 * and a game that will not take its opponent's turn is worse than a game that
 * stutters for a few seconds. `planTurnTraced` is the same function in both
 * cases, so the fallback cannot drift from the worker.
 */
export function usePlanner(): Planner {
  const workerRef = useRef<Worker | null>(null);
  const failedRef = useRef(false);
  const nextId = useRef(1);

  const ensureWorker = useCallback((): Worker | null => {
    if (workerRef.current) return workerRef.current;
    if (failedRef.current || typeof Worker === "undefined") return null;
    try {
      // `new URL(..., import.meta.url)` is the form Vite statically analyses,
      // so the worker is code-split and hashed like any other entry.
      workerRef.current = new Worker(new URL("./planWorker.ts", import.meta.url), {
        type: "module",
      });
      return workerRef.current;
    } catch (err) {
      // Loud, once. A silent downgrade here would hide the freeze coming back.
      console.error(
        "[showboat] planning worker unavailable; the opponent will think on the main " +
          "thread and the page will stall during its turn. Reason:",
        err,
      );
      failedRef.current = true;
      return null;
    }
  }, []);

  useEffect(
    () => () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    },
    [],
  );

  const plan = useCallback<Planner["plan"]>(
    (state, table, player, useNeural, cb) => {
      const worker = ensureWorker();
      if (!worker) {
        return planTurnTraced(
          { state, table, player, useNeural },
          { onModelLoadStart: cb?.onModelLoading, onModelStatus: cb?.onModelStatus },
        );
      }

      const id = nextId.current++;
      return new Promise<PlannedTurn>((resolve, reject) => {
        const onMessage = (e: MessageEvent<PlanResponse>) => {
          const msg = e.data;
          if (!msg || msg.id !== id) return;
          if (msg.type === "model-loading") return cb?.onModelLoading?.();
          if (msg.type === "model-status") {
            return cb?.onModelStatus?.({
              status: msg.ok ? "ready" : "failed",
              reason: msg.reason,
              hashVerified: msg.hashVerified,
            });
          }
          worker.removeEventListener("message", onMessage);
          worker.removeEventListener("error", onError);
          if (msg.type === "done") {
            resolve({ trace: msg.trace, action: msg.action, report: msg.report });
          } else reject(new Error(msg.message));
        };
        const onError = (e: ErrorEvent) => {
          worker.removeEventListener("message", onMessage);
          worker.removeEventListener("error", onError);
          reject(new Error(e.message || "planning worker failed"));
        };
        worker.addEventListener("message", onMessage);
        worker.addEventListener("error", onError);
        const req: PlanRequest = { type: "plan", id, state, table, player, useNeural };
        worker.postMessage(req);
      });
    },
    [ensureWorker],
  );

  const offMainThread = useCallback(() => workerRef.current !== null, []);
  return { plan, offMainThread };
}
