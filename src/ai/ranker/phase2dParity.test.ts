// Phase 2D Stage B.9 acceptance test: onnxruntime-web (the browser runtime,
// same import path evaluateCandidateRows() uses) must reproduce the exported
// ONNX graph's own predictions within the numeric tolerance already verified
// against the PyTorch checkpoint by training/ranker/phase2d/parity_fixture.py.
// Also measures single-batch inference latency for a typical decision-sized
// candidate set, per EVALUATION_PROTOCOL.md's "documented latency budget,
// measured not assumed" requirement.
//
// Fixture rows are drawn from the VALIDATION split only (parity_fixture.py),
// not test -- this is a pure export/runtime-correctness check, independent
// of the one-time test-set evaluation.
//
// Unlike Phase 2A's rankerIntegration.test.ts, this ONNX artifact is NOT
// committed: training/ranker/.gitignore's blanket *.onnx rule explicitly
// notes its Phase 2A exception is "not a precedent for committing real-scale
// weights -- Phase 2C+ artifacts follow the root policy" (regenerate on
// demand instead). So this suite skips itself with a clear message when the
// artifact/fixture aren't present locally, rather than failing CI on a fresh
// clone. Reproduce with (from apps/showboat/training/ranker/phase2d/):
//   python export_onnx.py --results results --out results/artifact
//   python parity_fixture.py --dataset-dir ../phase2c/data/baseline --artifact results/artifact --out results/artifact/parity_fixture.json

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { TOTAL_DIM } from "./encode";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ARTIFACT_DIR = join(__dirname, "../../../training/ranker/phase2d/results/artifact");
const MODEL_PATH = join(ARTIFACT_DIR, "showboat_ranker_phase2d.onnx");
const FIXTURE_PATH = join(ARTIFACT_DIR, "parity_fixture.json");

const TOLERANCE = 1e-4; // same tolerance parity_fixture.py checked PyTorch-vs-ONNX(Python) against
const ARTIFACT_PRESENT = existsSync(MODEL_PATH) && existsSync(FIXTURE_PATH);

interface FixtureRow {
  kind: string;
  features: number[];
  pytorch_logit: number;
  onnx_python_logit: number;
}
interface Fixture {
  n: number;
  tolerance: number;
  pytorch_vs_onnx_python: { max_abs_diff: number; mean_abs_diff: number; passes: boolean };
  rows: FixtureRow[];
}

if (!ARTIFACT_PRESENT) {
  // eslint-disable-next-line no-console
  console.warn(
    `[phase2dParity.test.ts] SKIPPED: Phase 2D artifact/fixture not present locally (this is expected -- ` +
      `they are not committed, see the file header for reproduction commands). Expected ${MODEL_PATH} and ${FIXTURE_PATH}.`,
  );
}

describe.skipIf(!ARTIFACT_PRESENT)("Phase 2D: onnxruntime-web reproduces the exported ONNX graph's predictions", () => {
  let ort: typeof import("onnxruntime-web");
  let session: import("onnxruntime-web").InferenceSession;
  let fixture: Fixture;

  beforeAll(async () => {
    if (!existsSync(MODEL_PATH) || !existsSync(FIXTURE_PATH)) {
      throw new Error(
        `Phase 2D artifact/fixture missing -- run export_onnx.py and parity_fixture.py first. ` +
          `Expected ${MODEL_PATH} and ${FIXTURE_PATH}.`,
      );
    }
    ort = await import("onnxruntime-web");
    const buf = readFileSync(MODEL_PATH);
    session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
    fixture = JSON.parse(readFileSync(FIXTURE_PATH, "utf-8"));
  });

  it("fixture covers every candidate kind with >=100 total examples", () => {
    expect(fixture.n).toBeGreaterThanOrEqual(100);
    const kinds = new Set(fixture.rows.map((r) => r.kind));
    expect(kinds).toEqual(new Set(["direct", "bank", "double-bank", "combo", "rail-combo"]));
  });

  it("PyTorch-vs-ONNX(Python) parity (already checked by parity_fixture.py) is recorded as a pass", () => {
    // Not re-derived here (that's a Python-side numeric check) -- just asserts
    // the fixture this test depends on actually passed that check when generated.
    expect(fixture.pytorch_vs_onnx_python.passes).toBe(true);
    expect(fixture.pytorch_vs_onnx_python.max_abs_diff).toBeLessThan(TOLERANCE);
  });

  it("onnxruntime-web matches the ONNX(Python) reference logits within tolerance, batched", async () => {
    const rows = fixture.rows;
    const flat = new Float32Array(rows.length * TOTAL_DIM);
    rows.forEach((r, i) => flat.set(r.features, i * TOTAL_DIM));

    const input = new ort.Tensor("float32", flat, [rows.length, TOTAL_DIM]);
    const out = await session.run({ [session.inputNames[0]]: input });
    const webLogits = out[session.outputNames[0]].data as Float32Array;

    expect(webLogits.length).toBe(rows.length);
    let maxDiff = 0;
    let sumDiff = 0;
    for (let i = 0; i < rows.length; i++) {
      const diff = Math.abs(webLogits[i] - rows[i].onnx_python_logit);
      maxDiff = Math.max(maxDiff, diff);
      sumDiff += diff;
    }
    const meanDiff = sumDiff / rows.length;
    console.log(
      `[phase2d parity] onnxruntime-web vs ONNX(Python): max_abs_diff=${maxDiff.toFixed(8)} ` +
        `mean_abs_diff=${meanDiff.toFixed(8)} (tolerance=${TOLERANCE})`,
    );
    expect(maxDiff).toBeLessThan(TOLERANCE);
  });

  it("inference latency for a typical decision-sized batch is within budget (measured)", async () => {
    // A typical Showboat decision generates on the order of 10-40 candidates
    // (see generateCandidates/shotSearch.ts). 24 is a representative midpoint;
    // reuse the fixture's real feature vectors (cycled) rather than zeros, so
    // the runtime isn't handed a degenerate all-zero input.
    const BATCH = 24;
    const flat = new Float32Array(BATCH * TOTAL_DIM);
    for (let i = 0; i < BATCH; i++) {
      flat.set(fixture.rows[i % fixture.rows.length].features, i * TOTAL_DIM);
    }
    const input = new ort.Tensor("float32", flat, [BATCH, TOTAL_DIM]);

    // Warm up (first call pays one-time graph-optimization/allocator cost).
    await session.run({ [session.inputNames[0]]: input });

    const N_TRIALS = 30;
    const timings: number[] = [];
    for (let i = 0; i < N_TRIALS; i++) {
      const start = performance.now();
      await session.run({ [session.inputNames[0]]: input });
      timings.push(performance.now() - start);
    }
    timings.sort((a, b) => a - b);
    const median = timings[Math.floor(N_TRIALS / 2)];
    const p95 = timings[Math.floor(N_TRIALS * 0.95)];
    console.log(
      `[phase2d latency] batch=${BATCH} candidates, N=${N_TRIALS} trials: ` +
        `median=${median.toFixed(3)}ms p95=${p95.toFixed(3)}ms`,
    );

    // Budget: 50ms for a full 24-candidate batch in one session.run() call.
    // Showboat's shot search runs many times per AI turn (UCB refinement
    // rounds), and the classical search's own per-turn budget is already on
    // the order of tens of milliseconds -- a single-digit-to-low-double-digit
    // millisecond ranker call keeps this from becoming the bottleneck. This
    // is a documented budget, not a previously-measured one: no live
    // integration exists yet to compare against (Phase 2F, out of scope).
    expect(median).toBeLessThan(50);
  });
});
