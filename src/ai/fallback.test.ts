// Every fallback cause, driven.
//
// `FallbackTrace.cause` is an eight-member union and, before this suite, not
// one member had a test. Two of the eight were also unreachable: `model-invalid`
// appeared in no branch of `brain.ts` at all, and `model-absent` could not fire
// because `getBrain` routed a non-ready evaluator to a classical brain that
// passed `fallback: null` — so on a genuinely missing or corrupt artifact the
// published trace recorded no fallback at all. The visitor was told the truth by
// the badge; the trace was not.
//
// The rule this suite enforces: every member of the union fires under exactly
// the condition its name describes, and the `detail` string it carries says
// something true about that condition. If a cause is added and left unreachable,
// the exhaustiveness check at the bottom fails.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../physics/table";
import { makeBall, type Ball } from "../physics/ball";
import { CUE_ID } from "../game/rack";
import type { GameState } from "../game/state";
import { initPhysics } from "../physics/wasm-bridge";
import { defaultConfig } from "./shotSearch";
import { getBrain, neuralTrickOnlyBrain, NEURAL_SCORE_DEADLINE_MS } from "./brain";
import type { NeuralCandidateEvaluator } from "./neural/evaluator";
import type { FallbackTrace } from "./trace/contract";
import { Deadline } from "./deadline";
import { planTurnTraced } from "../ui/planner/plan";
import { planViaWorker, type PlanChannel } from "../ui/planner/workerPlan";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../..");
const WASM = readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm"));
const table = makeTable();

beforeAll(async () => {
  await initPhysics(WASM);
}, 60_000);

const asState = (balls: Ball[]): GameState => ({
  balls,
  turn: 0,
  groups: { 0: "solids", 1: "stripes" },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 1,
});

/** An ordinary open board: candidates exist, so `no-candidates` cannot fire. */
const OPEN: Ball[] = [
  makeBall(CUE_ID, -0.6, -0.1),
  makeBall(1, 0.3, 0.02),
  makeBall(2, 0.38, 0.1),
  makeBall(4, 0.1, -0.25),
];
/**
 * The cue screened off its only legal target by the 8: `generateCandidates`
 * returns nothing at all here (asserted in `safety.test.ts` B1), which is the
 * one board on which "the model gave nothing back" would be a lie.
 */
const NO_CANDIDATES: Ball[] = [
  makeBall(CUE_ID, -0.6, 0.05),
  makeBall(8, -0.3, 0.05),
  makeBall(1, 0.55, 0.05),
];

const manifest = { artifact: "x.onnx", onnx_sha256: "deadbeef", schema_version: "v2" };

type EvalStub = Partial<NeuralCandidateEvaluator>;
const evaluator = (stub: EvalStub): NeuralCandidateEvaluator =>
  ({ getManifest: () => manifest, ...stub }) as unknown as NeuralCandidateEvaluator;

const cfg = { ...defaultConfig, seed: 20260806, seedTimeoutMs: Infinity, searchTimeoutMs: Infinity };

/** Run a brain and hand back the fallback its published trace carries. */
const fallbackOf = async (
  brain: ReturnType<typeof getBrain>,
  balls: Ball[] = OPEN,
): Promise<FallbackTrace | null> => {
  const res = await brain.plan(asState(balls), table, 0, cfg, Deadline.none());
  return res.decision.fallback;
};

const seen = new Set<FallbackTrace["cause"]>();
const record = (f: FallbackTrace | null): FallbackTrace => {
  expect(f, "a fallback was expected and none was published").not.toBeNull();
  seen.add(f!.cause);
  expect(f!.detail.length, "every fallback must carry a non-empty detail").toBeGreaterThan(0);
  return f!;
};

describe("every fallback cause fires under the condition it names", () => {
  it("no-candidates: the board offered the model nothing to rank", async () => {
    const brain = neuralTrickOnlyBrain(
      evaluator({ isReady: () => true, getState: () => ({ status: "ready" }) as never, score: async () => null }),
    );
    const f = record(await fallbackOf(brain, NO_CANDIDATES));
    expect(f.cause).toBe("no-candidates");
    // The distinction this cause exists for: it must not blame the model.
    expect(f.detail).toBe("no shots to rank from this position");
    expect(f.detail).not.toMatch(/model/i);
  }, 60_000);

  it("no-scores: the model was asked, on a board with candidates, and gave nothing", async () => {
    const brain = neuralTrickOnlyBrain(
      evaluator({ isReady: () => true, getState: () => ({ status: "ready" }) as never, score: async () => null }),
    );
    const f = record(await fallbackOf(brain));
    expect(f.cause).toBe("no-scores");
    expect(f.detail).toBe("the model scored nothing for this set of shots");
  }, 60_000);

  it("inference-timeout: `score()` never settles", async () => {
    const brain = neuralTrickOnlyBrain(
      evaluator({
        isReady: () => true,
        getState: () => ({ status: "ready" }) as never,
        score: () => new Promise(() => {}),
      }),
    );
    const t0 = performance.now();
    const f = record(await fallbackOf(brain));
    const elapsed = performance.now() - t0;
    expect(f.cause).toBe("inference-timeout");
    // The number in the sentence is the deadline that actually applied.
    expect(f.detail).toMatch(/^inference exceeded \d+ ms$/);
    // And it really did bound the turn rather than merely describing one.
    expect(elapsed).toBeGreaterThanOrEqual(NEURAL_SCORE_DEADLINE_MS - 20);
    expect(elapsed).toBeLessThan(NEURAL_SCORE_DEADLINE_MS + 5000);
  }, 60_000);

  it("inference-error: the session broke during this call", async () => {
    // What `evaluator.score()` really does when onnxruntime throws mid-run: it
    // downgrades itself to `invalid` and returns null. Ready before, invalid
    // after — that transition is what distinguishes this from `model-invalid`.
    let status = "ready";
    const brain = neuralTrickOnlyBrain(
      evaluator({
        isReady: () => true,
        getState: () => ({ status, reason: "onnx runtime error during inference: boom" }) as never,
        score: async () => {
          status = "invalid";
          return null;
        },
      }),
    );
    const f = record(await fallbackOf(brain));
    expect(f.cause).toBe("inference-error");
    expect(f.detail).toContain("onnx runtime error");
  }, 60_000);

  it("model-invalid: the artifact loaded and failed validation", async () => {
    const brain = getBrain(
      true,
      evaluator({
        isReady: () => false,
        getState: () => ({ status: "invalid", reason: "sha256 mismatch" }) as never,
      }),
    );
    // The point of the fix: this is the CLASSICAL brain, and it still reports.
    expect(brain.kind).toBe("classical-trick-only");
    const f = record(await fallbackOf(brain));
    expect(f.cause).toBe("model-invalid");
    expect(f.detail).toContain("sha256 mismatch");
  }, 60_000);

  it("model-absent: no artifact to load", async () => {
    const brain = getBrain(
      true,
      evaluator({
        isReady: () => false,
        getState: () => ({ status: "absent", reason: "HTTP 404 fetching model/ranker/manifest.json" }) as never,
      }),
    );
    expect(brain.kind).toBe("classical-trick-only");
    const f = record(await fallbackOf(brain));
    expect(f.cause).toBe("model-absent");
    expect(f.detail).toContain("404");
  }, 60_000);

  it("model-load-timeout: `load()` never settles, and the turn still happens", async () => {
    // The blocker, in a test. `planTurnTraced` is the function both the worker
    // and the inline fallback call, and it is where the turn's clock starts.
    // Before the fix this awaited forever.
    const stalled = evaluator({
      isReady: () => false,
      getState: () => ({ status: "absent", reason: "not loaded yet" }) as never,
      load: () => new Promise(() => {}),
    });
    const t0 = performance.now();
    const planned = await planTurnTraced({
      state: asState(OPEN),
      table,
      player: 0,
      useNeural: true,
      wasmSource: WASM,
      // Injected rather than reached through the module singleton, so this test
      // cannot leave a stalled evaluator behind for another suite.
      evaluatorOverride: stalled,
    });
    const elapsed = performance.now() - t0;
    // A shot, not a hang and not a rejection.
    expect(planned.kind).toBe("shot");
    const f = record(planned.trace.fallback);
    expect(f.cause).toBe("model-load-timeout");
    expect(f.detail).toMatch(/did not finish loading within \d+ ms/);
    expect(elapsed).toBeLessThan(30_000);
  }, 60_000);

  it("planner-timeout: the worker goes silent and the turn is rescued on the main thread", async () => {
    // A channel that accepts the request and then says nothing, ever — exactly
    // the observable state of a worker reclaimed under memory pressure, and
    // exactly what the cold review reproduced in the browser by swallowing the
    // worker's `done` reply.
    let terminated = false;
    const silent: PlanChannel = {
      addEventListener: () => {},
      removeEventListener: () => {},
      postMessage: () => {},
      terminate: () => {
        terminated = true;
      },
    } as unknown as PlanChannel;

    let deadNotified = false;
    const planned = await planViaWorker({
      worker: silent,
      id: 1,
      state: asState(OPEN),
      table,
      player: 0,
      useNeural: true,
      modelDir: "model/ranker",
      onWorkerDeclaredDead: () => {
        deadNotified = true;
      },
      // 50 ms rather than the shipped 8 s: the behaviour under test is the
      // watchdog firing and the rescue landing, not how long the wait is.
      silenceMs: 50,
      replanInline: (input, hooks) => planTurnTraced({ ...input, wasmSource: WASM }, hooks),
    });

    expect(terminated, "a worker declared dead must be terminated").toBe(true);
    expect(deadNotified, "the host must be told not to reuse it").toBe(true);
    expect(planned.kind).toBe("shot");
    const f = record(planned.trace.fallback);
    expect(f.cause).toBe("planner-timeout");
    expect(f.from).toBe("planning-worker");
    expect(f.to).toBe("main-thread-classical");
    // The rescue is genuinely classical — never a neural label over a decision
    // the model had no part in.
    expect(planned.trace.mode).toBe("classical-trick-only");
    expect(planned.trace.model).toBeNull();
  }, 60_000);

  it("all eight causes in the contract were exercised above", () => {
    // Written out rather than derived: the union is a type, so the only way to
    // hold this file to it is to restate it and let a compile error catch a
    // member that is added here without being driven.
    const ALL: FallbackTrace["cause"][] = [
      "model-absent",
      "model-invalid",
      "model-load-timeout",
      "inference-error",
      "inference-timeout",
      "planner-timeout",
      "no-candidates",
      "no-scores",
    ];
    expect([...seen].sort()).toEqual([...ALL].sort());
  });
});
