// Phase 2E browser-runtime parity + latency for the relational (Deep Sets) model.
//
// Mirrors src/ai/ranker/phase2dParity.test.ts exactly in method — same
// onnxruntime-web package, same "wasm" execution provider, same 1e-4 tolerance,
// same 24-candidate latency batch — but lives here in training/ rather than in
// the app's test suite, because this phase is research and must not touch the
// live application. If the relational model is adopted, THIS is the check that
// gets ported into a vitest file alongside the Phase 2D one.
//
// onnxruntime-web is resolved from an existing node_modules tree (read-only);
// pass its path with --ort so this script has no install step of its own.
//
// Usage:
//   node parity_web.mjs \
//     --model results/artifact/showboat_ranker_phase2e_deepsets.onnx \
//     --fixture results/artifact/parity_fixture_deepsets.json \
//     --ort /path/to/node_modules/onnxruntime-web/dist/ort.node.min.mjs \
//     --out results/artifact/parity_web_deepsets.json

import { readFileSync, writeFileSync } from "node:fs";

const argv = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};

const MODEL = arg("model");
const FIXTURE = arg("fixture");
const ORT = arg("ort");
const OUT = arg("out");
const TOLERANCE = 1e-4;
const TOTAL_DIM = 68;

const ort = await import(ORT);
const fixture = JSON.parse(readFileSync(FIXTURE, "utf-8"));
const session = await ort.InferenceSession.create(readFileSync(MODEL), {
  executionProviders: ["wasm"],
});

// --- correctness: onnxruntime-web vs the PyTorch reference logits ------------
const rows = fixture.inputs;
const flat = new Float32Array(rows.length * TOTAL_DIM);
rows.forEach((r, i) => flat.set(r, i * TOTAL_DIM));
const out = await session.run({
  [session.inputNames[0]]: new ort.Tensor("float32", flat, [rows.length, TOTAL_DIM]),
});
const web = out[session.outputNames[0]].data;

let maxDiff = 0;
let sumDiff = 0;
for (let i = 0; i < rows.length; i++) {
  const d = Math.abs(web[i] - fixture.expected_logits[i]);
  maxDiff = Math.max(maxDiff, d);
  sumDiff += d;
}

// --- kind coverage (same assertion the Phase 2D suite makes) ----------------
const kinds = [...new Set(fixture.kinds)].sort();

// --- latency: 24-candidate batch, 1 warmup + 30 trials ----------------------
const BATCH = 24;
const lflat = new Float32Array(BATCH * TOTAL_DIM);
for (let i = 0; i < BATCH; i++) lflat.set(rows[i % rows.length], i * TOTAL_DIM);
const linput = new ort.Tensor("float32", lflat, [BATCH, TOTAL_DIM]);
await session.run({ [session.inputNames[0]]: linput });
const timings = [];
for (let i = 0; i < 30; i++) {
  const t0 = performance.now();
  await session.run({ [session.inputNames[0]]: linput });
  timings.push(performance.now() - t0);
}
timings.sort((a, b) => a - b);

const result = {
  model: MODEL,
  n_rows: rows.length,
  kinds_covered: kinds,
  tolerance: TOLERANCE,
  onnxruntime_web_vs_pytorch: {
    max_abs_diff: maxDiff,
    mean_abs_diff: sumDiff / rows.length,
    pass: maxDiff < TOLERANCE,
  },
  latency_batch24_ms: {
    median: timings[Math.floor(timings.length / 2)],
    p95: timings[Math.floor(timings.length * 0.95)],
    min: timings[0],
    max: timings[timings.length - 1],
  },
  ort_version: ort.env?.versions?.common ?? "unknown",
};
console.log(JSON.stringify(result, null, 2));
if (OUT) writeFileSync(OUT, JSON.stringify(result, null, 2));
if (!result.onnxruntime_web_vs_pytorch.pass) process.exit(1);
