"""Phase 2D Stage B.9: PyTorch vs ONNX (Python) numeric parity + fixture export.

Builds a >=100-example fixture stratified across all 5 candidate kinds,
drawn from the VALIDATION split only (not test -- this is a pure numerics/
export-correctness check, unrelated to the one-time test evaluation, and
deliberately kept off the test split so it never has to be justified against
the "touch test once" policy). For each example, computes the raw logit from
(a) the exact PyTorch checkpoint used for export and (b) the exported ONNX
graph via onnxruntime (Python), and reports the max/mean absolute
difference. Writes a fixture JSON (features + kind + both logits) for the
browser-side (onnxruntime-web) half of the parity check.

Usage:
    python parity_fixture.py --dataset-dir ../phase2c/data/baseline --artifact results/artifact --out results/artifact/parity_fixture.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2c"))
from diagnostic import TinyNet  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import TOTAL_DIM  # noqa: E402

from train_baseline import load_rows, KINDS  # noqa: E402

SEED = 20260804
PER_KIND = 30
TOLERANCE = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = load_rows(args.dataset_dir)
    val_rows = [r for r in rows if r["split"] == "val"]

    rng = np.random.default_rng(SEED)
    sampled = []
    for kind in KINDS:
        kind_rows = [r for r in val_rows if r["candidate_kind"] == kind]
        n = min(PER_KIND, len(kind_rows))
        idx = rng.choice(len(kind_rows), size=n, replace=False)
        sampled.extend(kind_rows[i] for i in idx)
    print(f"[parity] sampled {len(sampled)} val rows across {len(KINDS)} kinds (target {PER_KIND}/kind)")
    assert len(sampled) >= 100, f"fixture too small ({len(sampled)} < 100)"

    X = np.array([r["features"] for r in sampled], dtype=np.float32)

    ckpt_path = args.artifact.parent / "checkpoints" / f"main_seed{SEED}.pt"
    model = TinyNet(TOTAL_DIM, hidden=32)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        pytorch_logits = model(torch.from_numpy(X)).numpy()

    onnx_path = args.artifact / "showboat_ranker_phase2d.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(None, {sess.get_inputs()[0].name: X})[0].reshape(-1)

    diff = np.abs(pytorch_logits - onnx_logits)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    passes = max_diff < TOLERANCE
    print(f"[parity] PyTorch vs ONNX(Python): max_abs_diff={max_diff:.8f} mean_abs_diff={mean_diff:.8f} "
          f"tolerance={TOLERANCE} -> {'PASS' if passes else 'FAIL'}")

    fixture = {
        "n": len(sampled),
        "seed": SEED,
        "tolerance": TOLERANCE,
        "pytorch_vs_onnx_python": {"max_abs_diff": max_diff, "mean_abs_diff": mean_diff, "passes": passes},
        "rows": [
            {
                "kind": r["candidate_kind"],
                "features": r["features"],
                "pytorch_logit": float(pytorch_logits[i]),
                "onnx_python_logit": float(onnx_logits[i]),
            }
            for i, r in enumerate(sampled)
        ],
    }
    with open(args.out, "w") as f:
        json.dump(fixture, f)
    print(f"[parity] wrote {args.out}")


if __name__ == "__main__":
    main()
