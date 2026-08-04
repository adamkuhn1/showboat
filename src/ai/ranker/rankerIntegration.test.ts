// Phase 2A acceptance test: proves the trained candidate-ranker model
// (a) produces genuinely different scores per candidate (not the old flat
// scalar every candidate shared — see docs/repair/showboat-ml/01-current-ml-audit.md
// §6/§2), and (b) that plugging those per-candidate scores into mcts.ts's
// seeding loop changes which candidate seeds highest, compared to the old
// whole-board-scalar approach, on a deterministic fixture.
//
// Uses the real exported artifact from training/ranker/artifacts/run1/ — not
// a mock, not a random-weights smoke model. If that artifact is regenerated
// (see training/ranker/README.md), this test re-runs against whatever a real
// training run produced.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../../physics/table";
import { makeBall, type Ball } from "../../physics/ball";
import { CUE_ID } from "../../game/rack";
import { generateCandidates } from "../candidates";
import { defaultConfig, searchBaseline } from "../mcts";
import { initPhysics } from "../../physics/wasm-bridge";
import { encodeRow, TOTAL_DIM } from "./encode";

const __dirname = dirname(fileURLToPath(import.meta.url));
const MODEL_PATH = join(
  __dirname,
  "../../../training/ranker/artifacts/run1/showboat-ranker.onnx",
);
const PHYSICS_WASM_PATH = join(__dirname, "../../wasm/showboat_physics_bg.wasm");

const table = makeTable();

/** A fixed, hand-placed mid-game-like board with several viable candidate kinds. */
function fixtureBoard(): Ball[] {
  return [
    makeBall(CUE_ID, -0.35, 0.05),
    makeBall(1, 0.25, 0.15), // near a straightforward direct pot
    makeBall(3, -0.05, -0.28), // sets up a bank angle
    makeBall(9, 0.5, -0.2), // stripe, further away
    makeBall(2, 0.55, 0.3),
  ];
}

describe("Phase 2A: trained ranker changes candidate ordering (deterministic fixture)", () => {
  let session: import("onnxruntime-web").InferenceSession;
  let ort: typeof import("onnxruntime-web");

  beforeAll(async () => {
    await initPhysics(readFileSync(PHYSICS_WASM_PATH));
    ort = await import("onnxruntime-web");
    const buf = readFileSync(MODEL_PATH);
    session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
  });

  it("scores each candidate distinctly, not one flat scalar for all", async () => {
    const balls = fixtureBoard();
    const targets = balls.filter((b) => b.id !== CUE_ID).map((b) => b.id);
    const candidates = generateCandidates(balls, table, targets);
    expect(candidates.length).toBeGreaterThan(1);

    const rows = new Float32Array(candidates.length * TOTAL_DIM);
    candidates.forEach((c, i) => rows.set(encodeRow(balls, table, c), i * TOTAL_DIM));

    const input = new ort.Tensor("float32", rows, [candidates.length, TOTAL_DIM]);
    const out = await session.run({ [session.inputNames[0]]: input });
    const scores = Array.from(out[session.outputNames[0]].data as Float32Array);

    expect(scores.length).toBe(candidates.length);
    const distinct = new Set(scores.map((s) => s.toFixed(5)));
    // The old bug (mcts.ts pre-fix): every candidate got IDENTICAL netSeedValue.
    // A real per-candidate model must not reproduce that.
    expect(distinct.size).toBeGreaterThan(1);

    // ---- Feed real per-candidate scores into the actual search seeding loop ----
    // simulations: 0 isolates pure seeding behavior (no UCB refinement rounds
    // afterward, which would selectively perturb a few candidates' values via
    // real rollouts and muddy this specific before/after comparison — UCB
    // refinement itself is already covered by the unmodified "no-model
    // fallback" test below).
    const withScores = searchBaseline(balls, table, targets, {
      ...defaultConfig,
      simulations: 0,
      netSeedScores: scores,
    });
    // Compare against the OLD behavior: one flat scalar (mean of the real
    // scores) applied to every candidate, exactly as `netSeedValue` did before
    // this phase's fix.
    const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    const withFlatScalar = searchBaseline(balls, table, targets, {
      ...defaultConfig,
      simulations: 0,
      netSeedValue: meanScore,
    });

    expect(withScores.stats.length).toBeGreaterThan(0);
    expect(withFlatScalar.stats.length).toBeGreaterThan(0);

    // Under the flat scalar, every seeded candidate has the identical value —
    // assert that directly, so this test would fail if mcts.ts's fix regressed.
    const flatValues = new Set(withFlatScalar.stats.map((s) => s.value.toFixed(5)));
    expect(flatValues.size).toBe(1);

    // Under real per-candidate scores, seeded values must differ across candidates.
    const scoredValues = new Set(withScores.stats.map((s) => s.value.toFixed(5)));
    expect(scoredValues.size).toBeGreaterThan(1);

    // The top-seeded candidate (highest initial value, before UCB refinement
    // reallocates visits) must be genuinely selectable by the model, i.e. its
    // seeded value equals the raw top score the model actually produced.
    const topSeeded = [...withScores.stats].sort((a, b) => b.value - a.value)[0];
    expect(topSeeded.value).toBeCloseTo(Math.max(...scores), 5);
  });

  it("no-model fallback (classical search) still works unchanged", () => {
    const balls = fixtureBoard();
    const targets = balls.filter((b) => b.id !== CUE_ID).map((b) => b.id);
    const result = searchBaseline(balls, table, targets, defaultConfig);
    expect(result.best).not.toBeNull();
    expect(result.simulations).toBeGreaterThan(0);
  });
});
