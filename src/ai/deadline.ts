// ONE clock for one AI turn.
//
// Every neural failure that *returns* was already handled. A failure that
// *stalls* was not: there was no `AbortSignal` on any fetch, no deadline around
// `evaluator.load()`, and no watchdog on the worker round-trip. A stall inside
// `InferenceSession.create()` — or a worker that is reclaimed under memory
// pressure and simply never answers — wedged the turn at "searching…" with no
// recovery short of a page reload.
//
// The fix is deliberately ONE object rather than three scattered races. A
// `Deadline` is created once, at the outermost seam of a turn
// (`planTurnTraced`), and threaded down. Every place that can block asks the
// same clock how much time is left:
//
//   * `evaluator.load()`  — raced (a promise that never settles).
//   * `evaluator.score()` — raced (ditto).
//   * the physics search  — polled, because a synchronous WASM call cannot be
//                           aborted; the loops check the clock between calls.
//
// A race cannot cancel the work it lost to, and this class does not pretend to:
// the losing promise is abandoned, not killed. That is safe here because the
// only thing downstream of it is a cached evaluator state, which a later turn
// picks up if the stalled load eventually completes.

/** Wall-clock budget for one AI turn, shared by everything the turn does. */
export class Deadline {
  private constructor(
    private readonly startedAt: number,
    private readonly budgetMs: number,
  ) {}

  /** A budget of `ms` starting now. */
  static in(ms: number): Deadline {
    return new Deadline(performance.now(), ms);
  }

  /**
   * No budget at all. Used by the evaluation harness and by unit tests, so
   * results depend on the physics budget and not on how loaded the machine is.
   */
  static none(): Deadline {
    return new Deadline(performance.now(), Number.POSITIVE_INFINITY);
  }

  /** Milliseconds left, floored at 0. `Infinity` for `Deadline.none()`. */
  remainingMs(): number {
    if (this.budgetMs === Number.POSITIVE_INFINITY) return Number.POSITIVE_INFINITY;
    return Math.max(0, this.budgetMs - (performance.now() - this.startedAt));
  }

  expired(): boolean {
    return this.remainingMs() <= 0;
  }

  /**
   * Race `work` against the smaller of this deadline's remaining time and
   * `capMs`. `{ ok: false }` means the clock won — the caller must treat that
   * exactly as it treats a failure that returned, never as a value.
   */
  async race<T>(work: Promise<T>, capMs = Number.POSITIVE_INFINITY): Promise<Raced<T>> {
    const ms = Math.min(this.remainingMs(), capMs);
    if (ms === Number.POSITIVE_INFINITY) return { ok: true, value: await work };
    if (ms <= 0) return { ok: false, waitedMs: 0 };
    let timer: ReturnType<typeof setTimeout> | undefined;
    const expiry = new Promise<Raced<T>>((resolve) => {
      timer = setTimeout(() => resolve({ ok: false, waitedMs: ms }), ms);
    });
    try {
      return await Promise.race([
        work.then((value) => ({ ok: true as const, value })),
        expiry,
      ]);
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  }
}

export type Raced<T> = { ok: true; value: T } | { ok: false; waitedMs: number };

/**
 * Bounded window for `evaluator.load()` — the manifest fetch, the artifact
 * fetch, the sha256 check and `InferenceSession.create()`.
 *
 * Generous on purpose. `usePlanner.warm()` normally pays this cost while the
 * page is idle, so a turn finds the session already built; when it does not,
 * a cold load on a slow connection legitimately takes seconds and killing it at
 * a typical-case value would turn a working model into a fallback. 6 s is well
 * past any load this artifact (177 KB) plus onnxruntime-web has been observed
 * to need locally, and it is short enough that a stalled load reads as a pause
 * rather than as a broken game.
 */
export const MODEL_LOAD_DEADLINE_MS = 6000;

/**
 * Bounded window for the decision itself: inference, the physics search, the
 * selection ladder and the safety rung. Starts after the model load resolves,
 * so a slow first load does not eat the search's budget.
 */
export const DECISION_DEADLINE_MS = 5000;

/**
 * How long the page waits for ANY message from the planning worker before
 * declaring it dead.
 *
 * This is a silence timer, not a total-time timer: it is re-armed by every
 * message the worker sends for the request in flight. The longest legitimate
 * silence is one of the two sub-budgets above (6 s across a model load, 5 s
 * across a search), so 8 s cannot fire on a worker that is making progress —
 * and a worker that has genuinely gone away is detected in 8 s rather than
 * never.
 */
export const WORKER_SILENCE_MS = 8000;
