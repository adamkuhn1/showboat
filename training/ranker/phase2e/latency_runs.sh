#!/usr/bin/env bash
# Repeated onnxruntime-web latency/parity runs.
#
# Phase 2D's review found single-run sub-millisecond `performance.now()` timings
# to be measurement-noise-dominated (2-3x spread across runs), so both models
# are measured over several independent process launches here and reported as a
# range, not a point estimate. Same harness, same machine, same session for
# both — the only fair way to compare them.
#
# Usage:  bash latency_runs.sh <path-to>/node_modules/onnxruntime-web/dist/ort.node.min.mjs [n_runs]
set -euo pipefail
ORT="$1"
RUNS="${2:-5}"
cd "$(dirname "$0")"
for v in mlp_matched deepsets; do
  for i in $(seq 1 "$RUNS"); do
    node parity_web.mjs \
      --model "results/artifact/showboat_ranker_phase2e_${v}.onnx" \
      --fixture "results/artifact/parity_fixture_${v}.json" \
      --ort "$ORT" \
      --out "results/artifact/parity_web_${v}_run${i}.json" > /dev/null
  done
done
python3 - <<'PY'
import json, glob, statistics
for v in ["mlp_matched", "deepsets"]:
    med, p95, mx = [], [], []
    for f in sorted(glob.glob(f"results/artifact/parity_web_{v}_run*.json")):
        d = json.load(open(f))
        assert d["onnxruntime_web_vs_pytorch"]["pass"], f
        med.append(d["latency_batch24_ms"]["median"])
        p95.append(d["latency_batch24_ms"]["p95"])
        mx.append(d["latency_batch24_ms"]["max"])
    print(f"{v:16s} runs={len(med)}  median {min(med):.3f}-{max(med):.3f} ms  "
          f"p95 {min(p95):.3f}-{max(p95):.3f} ms  worst-single-call {max(mx):.3f} ms")
PY
