// One plan request, over the worker channel, with a watchdog on it.
//
// Lifted out of `usePlanner` so it can be driven directly by a test with a
// stub worker and fake timers. It has no React in it and it owns no state
// beyond the request it is running.

import type { GameState, PlayerId } from "../../game/state";
import type { Table } from "../../physics/table";
import type { PlanRequest, PlanResponse } from "./protocol";
import { planTurnTraced, type ModelStatus, type PlannedTurn } from "./plan";
import { WORKER_SILENCE_MS } from "../../ai/deadline";
import type { SearchProgressEvent } from "../../ai/search/progress";

/** The subset of `Worker` this function uses. Lets a test supply a stub. */
export interface PlanChannel {
  addEventListener: Worker["addEventListener"];
  removeEventListener: Worker["removeEventListener"];
  postMessage: (msg: PlanRequest) => void;
  terminate: () => void;
}

export interface WorkerPlanInput {
  worker: PlanChannel;
  id: number;
  state: GameState;
  table: Table;
  player: PlayerId;
  useNeural: boolean;
  modelDir: string;
  onModelLoading?: () => void;
  onModelStatus?: (s: ModelStatus) => void;
  /** One live search event, forwarded as the worker posts it. */
  onProgress?: (e: SearchProgressEvent) => void;
  /**
   * Called when the watchdog fires, before the main-thread rescue plan starts,
   * so the host can drop its reference to a worker it must not reuse.
   */
  onWorkerDeclaredDead?: () => void;
  /** Injected for the test; production always uses the real planner. */
  replanInline?: typeof planTurnTraced;
  /** Injected for the test. */
  silenceMs?: number;
}

/**
 * Send one plan request and resolve with the turn.
 *
 * THE WATCHDOG. It is a *silence* timer, not a total-time timer.
 *
 * The worker bounds its own work — `plan.ts` gives the model load and the
 * decision a deadline each — but it cannot report the one failure that matters
 * here: itself going away. A module worker can be reclaimed under memory
 * pressure (the ~27 MB onnxruntime WASM makes that not theoretical), and the
 * observable result is a live `Worker` object that never posts again and never
 * fires `error`. Awaiting it is awaiting a promise that will not settle, which
 * is what wedged the reasoning panel at "searching…" indefinitely — Shoot
 * button gone from the DOM, no recovery short of a page reload.
 *
 * Every message for this request re-arms the timer, so a worker that is making
 * progress may take as long as its own deadlines allow. A worker that has
 * stopped talking is declared dead after `WORKER_SILENCE_MS`, terminated so it
 * cannot answer late into a turn that has moved on, and the turn is replanned
 * on the main thread without the model — the one plan with no dependency on the
 * thing that just failed. The page stutters for a few seconds. It does not
 * hang, and the trace says exactly what happened.
 */
export function planViaWorker(input: WorkerPlanInput): Promise<PlannedTurn> {
  const {
    worker,
    id,
    state,
    table,
    player,
    useNeural,
    modelDir,
    replanInline = planTurnTraced,
    silenceMs = WORKER_SILENCE_MS,
  } = input;

  return new Promise<PlannedTurn>((resolve, reject) => {
    let settled = false;
    let silence: ReturnType<typeof setTimeout> | undefined;

    const detach = () => {
      if (silence !== undefined) clearTimeout(silence);
      worker.removeEventListener("message", onMessage as EventListener);
      worker.removeEventListener("error", onError as EventListener);
    };

    const onSilence = () => {
      if (settled) return;
      settled = true;
      detach();
      console.error(
        `[showboat] the planning worker went silent for ${silenceMs} ms; terminating it ` +
          `and replanning on the main thread without the model.`,
      );
      worker.terminate();
      input.onWorkerDeclaredDead?.();
      resolve(
        replanInline(
          {
            state,
            table,
            player,
            // Deliberately `false`, whatever the toggle says: the model lives in
            // the worker that just failed, and pulling a second copy of the
            // ~27 MB runtime onto the main thread to rescue one turn is worse
            // than the turn being classical and saying so.
            useNeural: false,
            neuralUnavailable: {
              from: "planning-worker",
              to: "main-thread-classical",
              cause: "planner-timeout",
              detail: `the planning worker stopped responding for ${silenceMs} ms`,
            },
          },
          // The rescue search runs on the main thread, so its events all land
          // in one blocking burst rather than spread over the search. That is
          // what actually happens, and the host renders it as what it is.
          { onModelStatus: input.onModelStatus, onProgress: input.onProgress },
        ),
      );
    };

    const arm = () => {
      if (silence !== undefined) clearTimeout(silence);
      silence = setTimeout(onSilence, silenceMs);
    };

    const onMessage = (e: MessageEvent<PlanResponse>) => {
      const msg = e.data;
      if (!msg || msg.id !== id || settled) return;
      arm();
      if (msg.type === "progress") return input.onProgress?.(msg.event);
      if (msg.type === "model-loading") return input.onModelLoading?.();
      if (msg.type === "model-status") {
        return input.onModelStatus?.({
          status: msg.ok ? "ready" : "failed",
          reason: msg.reason,
          hashVerified: msg.hashVerified,
        });
      }
      settled = true;
      detach();
      if (msg.type === "done") resolve(msg.planned);
      else reject(new Error(msg.message));
    };

    const onError = (e: ErrorEvent) => {
      if (settled) return;
      settled = true;
      detach();
      reject(new Error(e.message || "planning worker failed"));
    };

    worker.addEventListener("message", onMessage as EventListener);
    worker.addEventListener("error", onError as EventListener);
    const req: PlanRequest = { type: "plan", id, state, table, player, useNeural, modelDir };
    worker.postMessage(req);
    arm();
  });
}
