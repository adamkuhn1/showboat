import { useCallback, useEffect, useRef } from "react";
import type { GameState, PlayerId } from "../../game/state";
import type { Table } from "../../physics/table";
import type { WarmRequest } from "./protocol";
import { planTurnTraced, type ModelStatus, type PlannedTurn } from "./plan";
import { planViaWorker } from "./workerPlan";

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
  /** Build the ranker session while the page is idle. */
  warm: () => void;
  /** False when the worker could not be constructed and planning is inline. */
  offMainThread: () => boolean;
}

/**
 * Absolute URL of the model directory, resolved against the document. It has
 * to be absolute: inside the worker a relative path resolves against the
 * worker's own `/assets/planWorker-*.js` URL, not the page.
 */
const modelDirUrl = (): string => new URL("model/ranker", document.baseURI).href;

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
      // The request, its watchdog and the main-thread rescue all live in
      // `planViaWorker` — no React in it, so `workerPlan.test.ts` can drive the
      // silent-worker case directly with a stub channel and fake timers.
      return planViaWorker({
        worker,
        id: nextId.current++,
        state,
        table,
        player,
        useNeural,
        modelDir: modelDirUrl(),
        onModelLoading: cb?.onModelLoading,
        onModelStatus: cb?.onModelStatus,
        // A worker declared dead must not be reused; the next turn builds a
        // fresh one through `ensureWorker`.
        onWorkerDeclaredDead: () => {
          if (workerRef.current === worker) workerRef.current = null;
        },
      });
    },
    [ensureWorker],
  );

  const warm = useCallback(() => {
    const worker = ensureWorker();
    if (!worker) return;
    const req: WarmRequest = { type: "warm", modelDir: modelDirUrl() };
    worker.postMessage(req);
  }, [ensureWorker]);

  const offMainThread = useCallback(() => workerRef.current !== null, []);
  return { plan, warm, offMainThread };
}
