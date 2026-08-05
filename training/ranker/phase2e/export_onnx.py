"""Phase 2E ONNX export + Python-side parity, for the relational model.

Adopting an architecture is only a real recommendation if it actually ships, so
this is run BEFORE recommending anything — not asserted as "should be feasible".

Two things are proved here:

  1. The relational model exports to a single ONNX graph with the SAME I/O
     contract Phase 2D established and the live game already speaks:
     `candidate_features [N, 68] float32  ->  logit [N, 1] float32`,
     dynamic batch axis. The entity construction (reshape into 16 balls,
     relational geometry, masked pooling) is inside the graph, so no change is
     required in `encode.ts`, `onnx.ts`, or anything Team A owns.

  2. onnxruntime (Python) reproduces the PyTorch checkpoint's logits on a
     stratified 150-row fixture within 1e-4 — the same tolerance and the same
     fixture design Phase 2D used (`parity_fixture.py`), sampled from the
     VALIDATION split, never test.

Model selection uses Phase 2D's rule verbatim: the seed at the cross-seed median
validation BCE, computed mechanically from the results JSON before any test
number exists.

Usage:
    python export_onnx.py --variant deepsets --out results/artifact
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data import load
from metrics import KINDS
from models import DeepSetsAttn, DeepSetsRanker, FlatEntityMLP, TinyNet, count_params

BUILDERS = {
    "deepsets": lambda: DeepSetsRanker(hidden=64, phi_hidden=64),
    "deepsets_small": lambda: DeepSetsRanker(hidden=32, phi_hidden=32),
    "deepsets_attn": lambda: DeepSetsAttn(hidden=64, phi_hidden=64, heads=4),
    "flat_entity_mlp": lambda: FlatEntityMLP(hidden=64),
    "mlp_matched": lambda: TinyNet(68, 32),
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def select_median_seed(results: dict) -> int:
    """Phase 2D's rule: median validation BCE across seeds. Deterministic, no
    test data involved, no hand-picking."""
    pairs = sorted((s["val"]["bce"], s["seed"]) for s in results["per_seed"])
    assert len(pairs) >= 3, f"expected >=3 seeds, got {len(pairs)}"
    return pairs[len(pairs) // 2][1]


def build_fixture(va, per_kind: int = 30, seed: int = 20260804):
    rng = np.random.default_rng(seed)
    idx = []
    for k in range(len(KINDS)):
        pool = np.flatnonzero(va.kind == k)
        idx.append(rng.choice(pool, size=min(per_kind, len(pool)), replace=False))
    return np.concatenate(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("results/dataset.npz"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--variant", default="deepsets")
    ap.add_argument("--out", type=Path, default=Path("results/artifact"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    res = json.load(open(args.results / f"variant_{args.variant}.json"))
    seed = select_median_seed(res)
    print(f"[export] {args.variant}: median-val-BCE seed = {seed}")

    model = BUILDERS[args.variant]()
    ckpt = args.results / "checkpoints" / f"{args.variant}_seed{seed}.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    splits = load(args.cache)
    va = splits["val"]
    fidx = build_fixture(va)
    Xf = va.X[fidx]

    class Wrapped(torch.nn.Module):
        """[N,68] -> [N,1], matching the Phase 2D artifact's output rank."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, candidate_features):
            return self.m(candidate_features).unsqueeze(-1)

    wrapped = Wrapped(model).eval()
    onnx_path = args.out / f"showboat_ranker_phase2e_{args.variant}.onnx"
    torch.onnx.export(
        wrapped,
        (torch.from_numpy(Xf[:8]),),
        str(onnx_path),
        input_names=["candidate_features"],
        output_names=["logit"],
        dynamic_axes={"candidate_features": {0: "N"}, "logit": {0: "N"}},
        opset_version=17,
        do_constant_folding=True,
    )

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(onnx_path)))

    with torch.no_grad():
        torch_logits = wrapped(torch.from_numpy(Xf)).numpy().reshape(-1)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_logits = sess.run(["logit"], {"candidate_features": Xf})[0].reshape(-1)
    diff = np.abs(torch_logits - ort_logits)

    # Batch-size independence: the graph must give the same answer for N=1,
    # N=7 and N=150 (dynamic axis really dynamic, no baked-in batch).
    single = np.concatenate(
        [sess.run(["logit"], {"candidate_features": Xf[i : i + 1]})[0].reshape(-1) for i in range(5)]
    )
    batch_diff = float(np.max(np.abs(single - ort_logits[:5])))

    fixture = {
        "variant": args.variant,
        "seed": seed,
        "split": "val",
        "n_rows": int(len(fidx)),
        "kinds": [KINDS[int(k)] for k in va.kind[fidx]],
        "inputs": Xf.astype(np.float32).tolist(),
        "expected_logits": torch_logits.astype(float).tolist(),
        "tolerance": 1e-4,
    }
    (args.out / f"parity_fixture_{args.variant}.json").write_text(json.dumps(fixture))

    manifest = {
        "phase": "2E",
        "variant": args.variant,
        "architecture": type(model).__name__,
        "n_params": count_params(model),
        "selected_seed": seed,
        "selection_rule": "cross-seed median validation BCE (Phase 2D rule, computed from variant JSON)",
        "source_checkpoint": str(ckpt),
        "source_checkpoint_sha256": sha256(ckpt),
        "artifact": str(onnx_path),
        "artifact_sha256": sha256(onnx_path),
        "artifact_bytes": onnx_path.stat().st_size,
        "opset": 17,
        "io_contract": {
            "input": {"name": "candidate_features", "shape": ["N", 68], "dtype": "float32"},
            "output": {"name": "logit", "shape": ["N", 1], "dtype": "float32"},
            "note": "identical to the Phase 2D artifact contract; entity construction happens inside the graph",
        },
        "platt": {
            "a": res["per_seed"][[s["seed"] for s in res["per_seed"]].index(seed)]["platt_a"],
            "b": res["per_seed"][[s["seed"] for s in res["per_seed"]].index(seed)]["platt_b"],
            "fit_on": "validation only",
        },
        "parity_pytorch_vs_onnxruntime_python": {
            "n_rows": int(len(fidx)),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "tolerance": 1e-4,
            "pass": bool(diff.max() < 1e-4),
        },
        "dynamic_batch_check": {
            "max_abs_diff_batch1_vs_batch150": batch_diff,
            "pass": bool(batch_diff < 1e-4),
        },
    }
    (args.out / f"MANIFEST_{args.variant}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(
        {k: manifest[k] for k in
         ["artifact_bytes", "n_params", "parity_pytorch_vs_onnxruntime_python", "dynamic_batch_check"]},
        indent=2))


if __name__ == "__main__":
    main()
