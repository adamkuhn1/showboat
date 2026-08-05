"""Phase 2D artifact export (Stage B.8).

Exports the single main-model checkpoint selected by the protocol's
model-selection rule -- the seed closest to the cross-seed median
validation BCE, chosen from validation results alone, before any test
number influenced the choice -- to ONNX, plus a manifest documenting
exactly what the artifact is, how it was chosen, its known limitation
(double-bank kind), and a content hash for integrity checking.

This does NOT wire the model into the live game (Phase 2F, out of scope).
It produces a standalone artifact + manifest for Stage B.9's parity and
latency testing.

Usage:
    python export_onnx.py --results results --out results/artifact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2c"))
from diagnostic import TinyNet  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import TOTAL_DIM  # noqa: E402


def select_median_seed(per_seed: list[dict]) -> dict:
    assert len(per_seed) == 5, f"expected exactly 5 seeds per protocol, got {len(per_seed)}"
    ranked = sorted(per_seed, key=lambda s: s["val_bce"])
    return ranked[len(ranked) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.results / "phase2d_results.json") as f:
        results = json.load(f)
    with open(args.results / "phase2d_test_results.json") as f:
        test_results = json.load(f)

    per_seed = results["candidate_conditioned"]["per_seed"]
    chosen = select_median_seed(per_seed)
    seed = chosen["seed"]
    print(f"[export] model-selection rule (median val BCE, chosen before looking at test): seed={seed} "
          f"(val_bce={chosen['val_bce']:.4f}, cross-seed val_bce values were "
          f"{sorted(round(s['val_bce'], 4) for s in per_seed)})")

    ckpt_path = args.results / "checkpoints" / f"main_seed{seed}.pt"
    model = TinyNet(TOTAL_DIM, hidden=32)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    onnx_path = args.out / "showboat_ranker_phase2d.onnx"
    dummy = torch.zeros(1, TOTAL_DIM, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["candidate_features"],
        output_names=["logit"],
        dynamic_axes={"candidate_features": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )
    onnx_hash = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    ckpt_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    # The manifest documents THIS specific exported checkpoint, so its
    # known-limitations text must cite that seed's own per-kind numbers, not
    # the 5-seed ensemble mean (which can look better than any single
    # deployed seed, including the one actually being shipped here).
    chosen_val_double_bank = chosen["per_kind"]["double-bank"]["group_aware_spearman"]
    chosen_test_seed_entry = next(
        s for s in test_results["candidate_conditioned"]["per_seed"] if s["seed"] == seed
    )
    chosen_test_double_bank = chosen_test_seed_entry["per_kind"]["double-bank"]["group_aware_spearman"]

    manifest = {
        "artifact": "showboat_ranker_phase2d.onnx",
        "onnx_sha256": onnx_hash,
        "source_checkpoint": str(ckpt_path.relative_to(args.results.parent) if args.results in ckpt_path.parents else ckpt_path),
        "source_checkpoint_sha256": ckpt_hash,
        "selected_seed": seed,
        "selection_rule": "cross-seed median validation BCE, chosen from validation results only, before any test evaluation",
        "architecture": "TinyNet(input_dim=68, hidden=32), 2 hidden layers, ReLU, single logit output (apply sigmoid for probability)",
        "input_contract": {
            "name": "candidate_features",
            "shape": ["N", TOTAL_DIM],
            "dtype": "float32",
            "encoding": "48-dim board block + 20-dim candidate block, per encode.ts's encodeState/encodeCandidateFeatures -- see EVALUATION_PROTOCOL.md for offsets. Batch dim N is dynamic (exported with dynamic_axes) -- N=1 for a single candidate, N>1 for a batched session.run() call as evaluateCandidateRows() in onnx.ts does.",
        },
        "output_contract": {
            "name": "logit",
            "shape": ["N", 1],
            "dtype": "float32",
            "note": "raw logit per row, NOT a probability -- apply sigmoid(a*logit + b) with this seed's Platt parameters below for a calibrated probability",
        },
        "platt_calibration": {"a": chosen["platt_a"], "b": chosen["platt_b"], "fit_on": "validation split only"},
        "target": "legal_pot_rate (empirical fraction of perturbed rollouts where the candidate legally pots its intended ball)",
        "validation_metrics": {
            "val_bce": chosen["val_bce"],
            "val_group_aware_spearman": chosen["val_group_aware_spearman"],
        },
        "test_metrics_all_seeds_mean": {
            "test_bce_mean": test_results["candidate_conditioned"]["test_bce_mean"],
            "test_bce_std": test_results["candidate_conditioned"]["test_bce_std"],
            "test_group_aware_spearman_mean": test_results["candidate_conditioned"]["test_group_aware_spearman_mean"],
            "test_group_aware_spearman_std": test_results["candidate_conditioned"]["test_group_aware_spearman_std"],
            "note": "reported for the 5-seed ensemble, not this specific exported seed alone -- see TRAINING_REPORT.md",
        },
        "known_limitations": [
            f"double-bank candidates: this exported seed's ({seed}) own group-aware Spearman is "
            f"{chosen_val_double_bank:.4f} (val) / {chosen_test_double_bank:.4f} (test) -- far below "
            f"the pooled figure and, on validation, over the protocol's predeclared catastrophic-"
            f"kind threshold (see ABLATIONS.md; the 5-seed-ensemble mean is a materially different, "
            f"less alarming number and is NOT what this specific artifact was measured at). Do not "
            f"rely on this artifact's output to rank double-bank candidates with confidence; fall "
            f"back to the candidate-kind-mean baseline or the transparent heuristic for that kind.",
            "No hyperparameter search was performed (frozen architecture, per protocol) -- this is a "
            "first defensible baseline, not a tuned model.",
            "Not wired into live gameplay -- this is a standalone artifact for Stage B.9 parity/latency "
            "testing only, per the protocol's explicit scope exclusion of Phase 2F.",
        ],
        "protocol": "docs/repair/showboat-ml/phase-2d/EVALUATION_PROTOCOL.md",
    }
    with open(args.out / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[export] wrote {onnx_path} (sha256={onnx_hash[:16]}...)")
    print(f"[export] wrote {args.out / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
