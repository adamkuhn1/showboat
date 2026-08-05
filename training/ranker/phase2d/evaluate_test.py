"""Phase 2D: the ONE test-set evaluation, per the frozen protocol.

Run exactly once, after train_baseline.py's results have been reviewed and
the configuration confirmed final. Loads the EXACT persisted checkpoints
(not retrained copies) for the main model and the state-only ablation,
applies each seed's own recorded Platt-scaling parameters (fit on
validation, never re-fit here), and evaluates against the test split.

Usage:
    python evaluate_test.py --dataset-dir ../phase2c/data/baseline --results results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2c"))
from diagnostic import TinyNet, pooled_spearman, group_aware_spearman  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import TOTAL_DIM  # noqa: E402

from train_baseline import (  # noqa: E402
    load_rows,
    to_arrays,
    brier,
    ece,
    apply_platt,
    per_kind_metrics,
    KINDS,
    BOARD_DIM,
    KIND_ONEHOT_OFFSET,
    BANKS_OFFSET,
    CLEARANCE_OFFSET,
)


def load_checkpoint_predict(ckpt_path: Path, X: np.ndarray) -> np.ndarray:
    model = TinyNet(TOTAL_DIM, hidden=32)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X)).numpy()
    return logits


def evaluate_variant(key: str, mask: np.ndarray, Xtest: np.ndarray, ytest: np.ndarray, ktest: np.ndarray, stest: np.ndarray, per_seed_platt: list[dict], ckpt_dir: Path) -> dict:
    Xtest_v = Xtest * mask
    per_seed = []
    for seed_info in per_seed_platt:
        seed = seed_info["seed"]
        logits = load_checkpoint_predict(ckpt_dir / f"{key}_seed{seed}.pt", Xtest_v)
        pred = 1.0 / (1.0 + np.exp(-logits))
        pred_calibrated = apply_platt(logits, seed_info["platt_a"], seed_info["platt_b"])
        bce = float(np.mean(-(ytest * np.log(np.clip(pred, 1e-7, 1)) + (1 - ytest) * np.log(np.clip(1 - pred, 1e-7, 1)))))
        per_seed.append({
            "seed": seed,
            "test_bce": bce,
            "test_brier": brier(pred, ytest),
            "test_brier_calibrated": brier(pred_calibrated, ytest),
            "test_ece": ece(pred, ytest),
            "test_ece_calibrated": ece(pred_calibrated, ytest),
            "test_pooled_spearman": pooled_spearman(pred, ytest),
            "test_group_aware_spearman": group_aware_spearman(pred, ytest, stest)[0],
            "per_kind": per_kind_metrics(pred, ytest, ktest, stest),
        })
    bces = [s["test_bce"] for s in per_seed]
    spearmans = [s["test_group_aware_spearman"] for s in per_seed]
    return {
        "per_seed": per_seed,
        "test_bce_mean": float(np.mean(bces)),
        "test_bce_std": float(np.std(bces)),
        "test_group_aware_spearman_mean": float(np.mean(spearmans)),
        "test_group_aware_spearman_std": float(np.std(spearmans)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path, help="Directory containing phase2d_results.json + checkpoints/ from train_baseline.py")
    args = ap.parse_args()

    with open(args.results / "phase2d_results.json") as f:
        val_results = json.load(f)

    rows = load_rows(args.dataset_dir)
    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows = [r for r in rows if r["split"] == "train"]
    Xtest, ytest, ktest, stest = to_arrays(test_rows)
    Xtr, ytr, ktr, _ = to_arrays(train_rows)
    print(f"[evaluate_test] THE ONE TEST EVALUATION -- {len(test_rows)} rows, {len(set(stest.tolist()))} states")

    ckpt_dir = args.results / "checkpoints"
    main_mask = np.array(val_results["candidate_conditioned"]["mask"], dtype=np.float32)
    state_only_mask = np.array(val_results["ablation_state_only"]["mask"], dtype=np.float32)

    test_results = {
        "candidate_conditioned": evaluate_variant("main", main_mask, Xtest, ytest, ktest, stest, val_results["candidate_conditioned"]["per_seed"], ckpt_dir),
        "ablation_state_only": evaluate_variant("state_only", state_only_mask, Xtest, ytest, ktest, stest, val_results["ablation_state_only"]["per_seed"], ckpt_dir),
    }

    # Baselines re-evaluated on test using TRAIN-fit statistics only (never test-fit).
    const_p = val_results["baseline_constant_mean"]["p"]
    const_test_bce = float(np.mean(-(ytest * np.log(const_p) + (1 - ytest) * np.log(1 - const_p))))
    kind_means = val_results["baseline_candidate_kind_mean"]["kind_means"]
    pred_kind_mean_test = np.array([kind_means[k] for k in ktest], dtype=np.float32)
    kind_mean_test_bce = float(np.mean(-(ytest * np.log(np.clip(pred_kind_mean_test, 1e-7, 1)) + (1 - ytest) * np.log(np.clip(1 - pred_kind_mean_test, 1e-7, 1)))))

    test_results["baseline_constant_mean_test_bce"] = const_test_bce
    test_results["baseline_candidate_kind_mean_test_bce"] = kind_mean_test_bce

    # Transparent-heuristic baseline, reconstructed from its persisted (rounded)
    # coefficients -- a one-time reporting comparison, not a retrain. The
    # heuristic was never gated on test in train_baseline.py; the protocol's
    # minimum-evidence checklist asks for it here.
    h_coefs = val_results["baseline_heuristic"]["coefficients"]
    h_idx = list(range(BOARD_DIM + KIND_ONEHOT_OFFSET, BOARD_DIM + KIND_ONEHOT_OFFSET + 5)) + [
        BOARD_DIM + BANKS_OFFSET,
        BOARD_DIM + CLEARANCE_OFFSET,
    ]
    h_w = np.array([h_coefs[k] for k in KINDS + ["banks", "clearance"]], dtype=np.float32)
    h_b = np.float32(h_coefs["intercept"])
    h_logits_test = Xtest[:, h_idx] @ h_w + h_b
    pred_heuristic_test = 1.0 / (1.0 + np.exp(-h_logits_test))
    test_results["baseline_heuristic_test"] = {
        "test_bce": float(np.mean(-(ytest * np.log(np.clip(pred_heuristic_test, 1e-7, 1)) + (1 - ytest) * np.log(np.clip(1 - pred_heuristic_test, 1e-7, 1))))),
        "test_group_aware_spearman": group_aware_spearman(pred_heuristic_test, ytest, stest)[0],
    }

    main_bce = test_results["candidate_conditioned"]["test_bce_mean"]
    checklist = {
        "beats_constant_mean": main_bce < const_test_bce,
        "beats_candidate_kind_mean": main_bce < kind_mean_test_bce,
        "beats_state_only": main_bce < test_results["ablation_state_only"]["test_bce_mean"],
        "beats_heuristic_on_group_aware_spearman": test_results["candidate_conditioned"]["test_group_aware_spearman_mean"] > test_results["baseline_heuristic_test"]["test_group_aware_spearman"],
    }
    state_only_test_by_seed = {s["seed"]: s["test_bce"] for s in test_results["ablation_state_only"]["per_seed"]}
    heuristic_gas_test = test_results["baseline_heuristic_test"]["test_group_aware_spearman"]
    checklist["consistent_direction_across_seeds"] = all(
        s["test_bce"] < const_test_bce
        and s["test_bce"] < kind_mean_test_bce
        and s["test_bce"] < state_only_test_by_seed[s["seed"]]
        and s["test_group_aware_spearman"] > heuristic_gas_test
        for s in test_results["candidate_conditioned"]["per_seed"]
    )
    per_kind_spearman_test = {
        k: float(np.mean([s["per_kind"].get(k, {}).get("group_aware_spearman", np.nan) for s in test_results["candidate_conditioned"]["per_seed"]]))
        for k in KINDS
    }
    pooled_gas_test = test_results["candidate_conditioned"]["test_group_aware_spearman_mean"]
    checklist["no_kind_catastrophically_worse"] = all(
        (np.isnan(v) or v >= pooled_gas_test - 0.3) for v in per_kind_spearman_test.values()
    )
    test_results["minimum_evidence_checklist_test"] = checklist
    test_results["per_kind_group_aware_spearman_test"] = {k: (None if np.isnan(v) else v) for k, v in per_kind_spearman_test.items()}
    test_results["val_vs_test_consistency"] = {
        "main_val_bce": val_results["candidate_conditioned"]["val_bce_mean"],
        "main_test_bce": main_bce,
        "main_val_group_aware_spearman": val_results["candidate_conditioned"]["val_group_aware_spearman_mean"],
        "main_test_group_aware_spearman": pooled_gas_test,
    }

    print(f"[evaluate_test] main model: test_bce={main_bce:.4f}+-{test_results['candidate_conditioned']['test_bce_std']:.4f} "
          f"(val was {val_results['candidate_conditioned']['val_bce_mean']:.4f}) "
          f"group_aware_spearman={pooled_gas_test:.4f} (val was {val_results['candidate_conditioned']['val_group_aware_spearman_mean']:.4f})")
    print(f"[evaluate_test] minimum-evidence checklist (TEST): {checklist}")

    with open(args.results / "phase2d_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"[evaluate_test] wrote {args.results / 'phase2d_test_results.json'}")


if __name__ == "__main__":
    main()
