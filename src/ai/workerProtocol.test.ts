import { afterEach, describe, expect, it } from "vitest";
import { makeGame } from "../game/game";
import { makeBall } from "../physics/ball";
import { type GameState } from "../game/state";
import { aiTakeTurn, type AiMove } from "./agent";
import { AiRunner, type TurnHooks, type WorkerLike } from "./aiClient";
import { makeRanker } from "./ranker";
import { makeRng } from "./rollout";
import { serveAiTurn, type FromWorker, type Port, type ToWorker } from "./workerProtocol";

// The worker path is exercised over a real MessageChannel, so every message
// goes through structured clone exactly as it would between page and worker.
// The worker side runs in this thread; what's under test is the protocol and
// the runner, not the browser's threading.

const { table } = makeGame();

// Ball in hand behind the head string with a few balls left: exercises the
// placement round trip as well as generation and verification, while staying
// quick (a full rack takes seconds per turn, which is the point of the worker).
const ballInHandState = (): GameState => ({
  balls: [
    makeBall(0, -0.8, 0),
    makeBall(3, 0.3, 0.2),
    makeBall(8, 0.6, -0.25),
    makeBall(10, 0.1, -0.3),
    makeBall(12, 0.75, 0.3),
  ],
  turn: 1,
  groups: { 0: "solids", 1: "stripes" },
  ballInHand: "behind-head",
  winner: null,
  broken: true,
  shotCount: 9,
});

const channels: MessageChannel[] = [];
afterEach(() => {
  for (const ch of channels.splice(0)) {
    ch.port1.close();
    ch.port2.close();
  }
});

// A WorkerLike whose far end is served by serveAiTurn.
const channelWorker = (counter?: { terminated: number }): WorkerLike => {
  const ch = new MessageChannel();
  channels.push(ch);
  serveAiTurn(ch.port2 as unknown as Port<ToWorker, FromWorker>);
  ch.port1.start();
  return {
    postMessage: (m) => ch.port1.postMessage(m),
    addEventListener: (type: string, fn: (e: never) => void) => {
      if (type === "message") ch.port1.addEventListener("message", fn as never);
    },
    terminate: () => {
      ch.port1.close();
      if (counter) counter.terminated++;
    },
  };
};

// Records the order hooks fire in, and acts like the page for onPlaced.
const recordingHooks = (log: string[]): TurnHooks => ({
  onPhase: (p) => log.push(`phase:${p}`),
  onPlaced: async (_placed, spot) => {
    log.push(`placed:${spot.x.toFixed(4)},${spot.y.toFixed(4)}`);
    await new Promise((r) => setTimeout(r, 5)); // the roll-in animation
  },
  onGenerated: (n) => log.push(`generated:${n}`),
  onCandidate: (ec, i, total) =>
    log.push(`candidate:${i}/${total}:${ec.cand.id}:${ec.success ?? "?"}:${ec.report ? "frames" : "no-frames"}`),
});

describe("AI turn through the worker protocol", () => {
  it("returns the same move, with the same hook sequence, as an in-thread turn", async () => {
    const state = ballInHandState();

    const directLog: string[] = [];
    const direct = (await aiTakeTurn(state, table, makeRanker("classical"), {
      ...recordingHooks(directLog),
      rng: makeRng(7),
    }))!;

    const workerLog: string[] = [];
    const runner = new AiRunner(() => channelWorker());
    const viaWorker = (await runner.run(state, table, "classical", recordingHooks(workerLog), {
      delayMs: 0,
      seed: 7,
    }).move)!;

    expect(viaWorker).toEqual(direct);
    expect(viaWorker.decision.cuePlacedAt).not.toBeNull();
    // Progress messages carry no frames; everything else matches in order.
    expect(workerLog).toEqual(directLog.map((l) => l.replace(/:frames$/, ":no-frames")));
    // The played-back simulation still has every frame.
    expect(viaWorker.report.sim.frames?.length).toBeGreaterThan(1);
    runner.dispose();
  });

  it("cancel stops the turn, terminates the worker, and the next turn gets a fresh one", async () => {
    const counter = { terminated: 0 };
    let made = 0;
    const runner = new AiRunner(() => {
      made++;
      return channelWorker(counter);
    });
    const log: string[] = [];
    let firstCandidate!: () => void;
    const sawCandidate = new Promise<void>((r) => (firstCandidate = r));
    const turn = runner.run(
      ballInHandState(),
      table,
      "classical",
      {
        ...recordingHooks(log),
        onCandidate: () => {
          log.push("candidate");
          firstCandidate();
        },
      },
      { delayMs: 50, seed: 1 },
    );
    await sawCandidate;
    turn.cancel();
    expect(await turn.move).toBeNull();
    expect(counter.terminated).toBe(1);

    const after = log.length;
    await new Promise((r) => setTimeout(r, 150));
    expect(log.length).toBe(after); // nothing reaches the page after cancel

    const next = await runner.run(ballInHandState(), table, "classical", {}, { delayMs: 0, seed: 1 })
      .move;
    expect(next).not.toBeNull();
    expect(made).toBe(2);
    runner.dispose();
  });

  it("runs on the main thread when workers aren't available", async () => {
    const state = ballInHandState();
    const direct = await aiTakeTurn(state, table, makeRanker("classical"), { rng: makeRng(3) });
    const runner = new AiRunner(() => null);
    const move: AiMove | null = await runner.run(state, table, "classical", {}, {
      delayMs: 0,
      seed: 3,
    }).move;
    expect(move).toEqual(direct);
  });

  it("falls back to the main thread if the worker crashes", async () => {
    let onError: ((e: Event) => void) | undefined;
    let terminated = false;
    const broken: WorkerLike = {
      postMessage: () => queueMicrotask(() => onError?.(new Event("error"))),
      addEventListener: (type: string, fn: (e: never) => void) => {
        if (type === "error") onError = fn as never;
      },
      terminate: () => {
        terminated = true;
      },
    };
    const state = ballInHandState();
    const direct = await aiTakeTurn(state, table, makeRanker("classical"), { rng: makeRng(5) });
    const warn = console.warn;
    console.warn = () => {};
    try {
      const runner = new AiRunner(() => broken);
      const move = await runner.run(state, table, "classical", {}, { delayMs: 0, seed: 5 }).move;
      expect(move).toEqual(direct);
      expect(terminated).toBe(true);
    } finally {
      console.warn = warn;
    }
  });
});
