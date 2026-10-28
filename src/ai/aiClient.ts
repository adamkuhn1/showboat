import { type Table } from "../physics/table";
import { type GameState } from "../game/state";
import { aiTakeTurn, type AiHooks, type AiMove } from "./agent";
import { makeRanker, type RankerName } from "./ranker";
import { makeRng } from "./rollout";
import { type FromWorker, type ToWorker } from "./workerProtocol";

// Page side of the AI worker. `run` takes the same hooks aiTakeTurn does and
// fires them as the worker reports progress, so the caller can't tell the
// difference except that the page never freezes.
//
// One worker is reused across turns. Cancelling terminates it, which stops a
// turn at once, even in the middle of a simulation; the next turn starts a
// fresh one. Where workers aren't available the turn runs on the main thread
// exactly as it did before.

export type TurnHooks = Pick<AiHooks, "onPhase" | "onPlaced" | "onGenerated" | "onCandidate">;

export interface TurnOptions {
  delayMs: number;
  seed?: number;
}

export interface TurnHandle {
  move: Promise<AiMove | null>;
  cancel: () => void;
}

// What the runner needs from a Worker. A real Worker fits; tests pass a
// MessagePort wrapper.
export interface WorkerLike {
  postMessage(msg: ToWorker): void;
  addEventListener(type: "message", fn: (e: MessageEvent<FromWorker>) => void): void;
  addEventListener(type: "error", fn: (e: Event) => void): void;
  terminate(): void;
}

export const browserWorker = (): WorkerLike | null =>
  typeof Worker === "undefined"
    ? null
    : (new Worker(new URL("./ai.worker.ts", import.meta.url), {
        type: "module",
      }) as unknown as WorkerLike);

export class AiRunner {
  private worker: WorkerLike | null = null;
  private nextId = 1;
  // The turn in flight, so a message or worker error can reach it.
  private active: {
    id: number;
    onMessage: (m: FromWorker) => void;
    onCrash: () => void;
  } | null = null;
  private cancelCurrent: (() => void) | null = null;

  constructor(private readonly makeWorker: () => WorkerLike | null = browserWorker) {}

  run(
    state: GameState,
    table: Table,
    ranker: RankerName,
    hooks: TurnHooks,
    opts: TurnOptions,
  ): TurnHandle {
    this.cancel();
    const worker = this.ensureWorker();
    if (!worker) return this.runInThread(state, table, ranker, hooks, opts);

    const id = this.nextId++;
    let settle!: (m: AiMove | null) => void;
    let fail!: (e: Error) => void;
    const move = new Promise<AiMove | null>((res, rej) => {
      settle = res;
      fail = rej;
    });
    const finish = (): void => {
      if (this.active?.id === id) this.active = null;
      if (this.cancelCurrent === cancel) this.cancelCurrent = null;
    };

    const onMessage = (m: FromWorker): void => {
      switch (m.type) {
        case "phase":
          hooks.onPhase?.(m.phase);
          break;
        case "placed":
          void Promise.resolve(hooks.onPlaced?.(m.placed, m.spot)).then(() => {
            if (this.active?.id === id) worker.postMessage({ type: "placed-ack", id });
          });
          break;
        case "generated":
          hooks.onGenerated?.(m.count, m.byKind);
          break;
        case "candidate":
          hooks.onCandidate?.(m.ec, m.index, m.total);
          break;
        case "done":
          finish();
          settle(m.move);
          break;
        case "error":
          finish();
          fail(new Error(`AI turn failed: ${m.message}`));
          break;
      }
    };

    // The worker itself died (failed to load, out of memory). Play this turn
    // on the main thread rather than leaving the game stuck.
    let fallbackCancel: (() => void) | null = null;
    const onCrash = (): void => {
      console.warn("[showboat] AI worker failed; running this turn on the main thread.");
      this.dropWorker();
      const fallback = this.runInThread(state, table, ranker, hooks, opts);
      fallbackCancel = fallback.cancel;
      this.cancelCurrent = cancel;
      fallback.move.then(settle, fail).finally(finish);
    };

    const cancel = (): void => {
      if (this.active?.id === id) this.dropWorker();
      fallbackCancel?.();
      finish();
      settle(null);
    };

    this.active = { id, onMessage, onCrash };
    this.cancelCurrent = cancel;
    worker.postMessage({ type: "start", id, state, table, ranker, ...opts });
    return { move, cancel };
  }

  // Stop whatever turn is running (no-op if none).
  cancel(): void {
    this.cancelCurrent?.();
    this.cancelCurrent = null;
  }

  dispose(): void {
    this.cancel();
    this.dropWorker();
  }

  private ensureWorker(): WorkerLike | null {
    if (this.worker) return this.worker;
    const w = this.makeWorker();
    if (!w) return null;
    w.addEventListener("message", (e) => {
      if (this.worker === w && this.active?.id === e.data.id) this.active.onMessage(e.data);
    });
    w.addEventListener("error", (e) => {
      e.preventDefault?.();
      if (this.worker === w) this.active?.onCrash();
    });
    this.worker = w;
    return w;
  }

  private dropWorker(): void {
    this.worker?.terminate();
    this.worker = null;
    this.active = null;
  }

  private runInThread(
    state: GameState,
    table: Table,
    ranker: RankerName,
    hooks: TurnHooks,
    opts: TurnOptions,
  ): TurnHandle {
    const signal = { cancelled: false };
    const move = aiTakeTurn(state, table, makeRanker(ranker), {
      ...hooks,
      delayMs: opts.delayMs,
      rng: opts.seed !== undefined ? makeRng(opts.seed) : undefined,
      signal,
    });
    const cancel = (): void => {
      signal.cancelled = true;
    };
    this.cancelCurrent = cancel;
    return { move, cancel };
  }
}
