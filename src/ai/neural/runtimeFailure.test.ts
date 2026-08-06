// Regression test for a real gap found during the ML-truth review track
// (docs/repair/release-candidate/reviews/ml-truth.md): every existing
// fallback test exercised a *load-time* failure (missing/corrupt manifest or
// artifact). None exercised onnxruntime-web failing *after* a successful
// load — e.g. `session.run()` throwing on a pathological input or a WASM
// runtime error. Before this fix, `NeuralCandidateEvaluator.score()` awaited
// `evaluateCandidateRows()` with no try/catch, so that exception propagated
// straight out of `neuralHybridBrain.plan()` — which `App.tsx` awaits inside
// a bare `setTimeout(async () => { ... })` with no surrounding try/catch —
// leaving the AI's turn hung in "searching…" forever instead of falling back
// to classical.
//
// This test uses the real committed model and a real load (same as
// productionModel.test.ts), then makes only the inference call itself throw,
// to prove the failure is caught at exactly the boundary a genuine ORT
// runtime error would cross.

import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { NeuralCandidateEvaluator } from "./evaluator";
import { makeFileFetch } from "./fileFetch";
import { neuralTrickOnlyBrain, getBrain, brainLabel } from "../brain";
import { _resetRankerForTests } from "../onnx";
import { initPhysics } from "../../physics/wasm-bridge";
import { makeTable } from "../../physics/table";
import { makeBall, type Ball } from "../../physics/ball";
import { CUE_ID } from "../../game/rack";
import { generateCandidates } from "../candidates";
import { defaultConfig, type SearchConfig } from "../shotSearch";
import { type GameState } from "../../game/state";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../../..");
const table = makeTable();

const BOARD: Ball[] = [
  makeBall(CUE_ID, -0.35, 0.05),
  makeBall(1, 0.25, 0.15),
  makeBall(3, -0.05, -0.28),
  makeBall(9, 0.5, -0.2),
];

const asState = (balls: Ball[]): GameState => ({
  balls,
  turn: 0,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 1,
});

const config: SearchConfig = { ...defaultConfig, simulations: 30, seed: 20260805 };

afterEach(() => {
  // Deliberately NOT vi.resetModules(): evaluator.ts's own static import of
  // "../onnx" is bound once, at this test file's load time. Resetting the
  // module registry between tests would hand a dynamically re-imported
  // `import("../onnx")` a *different* module instance than the one
  // evaluator.ts actually calls, silently decoupling the spy from the real
  // call site instead of restoring a clean one.
  vi.restoreAllMocks();
});

describe("neural hybrid: a runtime ONNX failure (not just a load failure) still falls back", () => {
  it("session.run() throwing after a successful load is caught, not propagated", async () => {
    _resetRankerForTests();
    await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));

    const evaluator = new NeuralCandidateEvaluator("model/ranker");
    const state = await evaluator.load(makeFileFetch(join(APP_ROOT, "public")));
    expect(state.status).toBe("ready");
    expect(evaluator.isReady()).toBe(true);

    // Force the ONNX runtime call itself to throw, exactly as a genuine
    // session.run() failure would, without touching load/preflight or any
    // physics/search code.
    const onnxModule = await import("../onnx");
    const spy = vi
      .spyOn(onnxModule, "evaluateCandidateRows")
      .mockRejectedValue(new Error("simulated ORT runtime failure (e.g. WASM OOM)"));

    const candidates = generateCandidates(BOARD, table, [1, 3, 9]);
    expect(candidates.length).toBeGreaterThan(0);
    const scores = await evaluator.score(BOARD, table, candidates);
    expect(scores).toBeNull();
    // The evaluator must not keep pretending to be ready and re-attempting a
    // session that just proved broken on every subsequent turn.
    expect(evaluator.isReady()).toBe(false);
    expect(evaluator.getState().status).toBe("invalid");
    expect((evaluator.getState() as { reason: string }).reason).toMatch(/onnx runtime error/i);

    spy.mockRestore();
  });

  it("the hybrid brain still produces a real classical decision instead of hanging", async () => {
    _resetRankerForTests();
    await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));

    const evaluator = new NeuralCandidateEvaluator("model/ranker");
    await evaluator.load(makeFileFetch(join(APP_ROOT, "public")));
    expect(evaluator.isReady()).toBe(true);

    const onnxModule = await import("../onnx");
    vi.spyOn(onnxModule, "evaluateCandidateRows").mockRejectedValue(
      new Error("simulated ORT runtime failure"),
    );

    const brain = neuralTrickOnlyBrain(evaluator);
    // This must resolve — not hang, and not throw — exactly like the
    // existing "unloadable model" fallback test, but for a failure that
    // happens after a successful load instead of before one.
    const result = await brain.plan(asState(BOARD), table, 0, config);
    expect(result.best).not.toBeNull();
    expect(result.simulations).toBeGreaterThan(0);
    expect(result.trace!.fallbackReason).toMatch(/onnx runtime error/i);

    // And the user-facing brain selection honestly reports classical once
    // the evaluator has downgraded itself — never a neural label over a
    // decision the model didn't actually produce.
    expect(getBrain(true, evaluator).kind).toBe("classical-trick-only");
    expect(brainLabel(true, evaluator)).toBe("the physics-search opponent");
  });
});
