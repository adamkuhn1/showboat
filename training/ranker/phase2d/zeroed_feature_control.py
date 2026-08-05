"""Phase 2D: the literal zeroed-feature control, at full baseline scale.

Closes the outstanding gap `REVIEW.md` flagged ("Should-fix: zeroed-feature
control never run at baseline scale") and `TRAINING_REPORT.md` records as a
real, unclosed item. Phase 2C's `memorization_test.py` ran this control at
n=64; the protocol's minimum-evidence checklist asks for it "re-run at
baseline scale with the frozen architecture."

What it does: trains the SAME frozen architecture and optimizer as
`train_baseline.py` (TinyNet(68, hidden=32), Adam lr=1e-2, soft-label BCE,
early stopping patience=30 on validation BCE, max 500 epochs, the same 5
seeds) on the FULL baseline train split with **all 68 input dimensions
zeroed** in both train and val. With no input information whatsoever, a
feedforward net can only learn a single constant (the bias chain), so its
held-out BCE must land on the constant-training-mean baseline. If it lands
materially *below* that baseline, information is reaching the model through
something other than its inputs -- row order, sample weighting, the split,
the metric, or calibration -- and that is a leak that invalidates the
phase's headline result.

This is a strictly stronger check than the state-only/candidate-only
ablations, which only zero half the input and answer a feature-importance
question, not a leak-detection one.

Train/val only. Never reads the test split for anything.

Usage:
    python zeroed_feature_control.py --dataset-dir ../phase2c/data/baseline --out results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2c"))
from diagnostic import train_model_early_stopping, constant_mean_bce  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import TOTAL_DIM  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import load_rows, to_arrays, brier, SEEDS, MAX_EPOCHS  # noqa: E402

# Same tolerance the shuffled-label control uses, carried over deliberately so
# the two controls are directly comparable. Noted in TRAINING_REPORT.md as
# loose relative to the model's ~0.044 BCE effect size; reported alongside the
# raw gap so the reader can apply a stricter bar themselves.
TOLERANCE = 0.02


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--seeds",
        type=int,
        default=len(SEEDS),
        help="how many of the frozen protocol seeds to run (default: all 5)",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.dataset_dir)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    Xtr, ytr, _, _ = to_arrays(train_rows)
    Xval, yval, _, sval = to_arrays(val_rows)

    print(
        f"[zero-control] rows: train={len(train_rows)} val={len(val_rows)} "
        f"test={len(test_rows)} (test loaded for the count only, never used)"
    )

    baseline_bce = constant_mean_bce(ytr, yval)
    train_mean = float(ytr.mean())

    # The control: every input dimension zeroed, train AND val, so the model
    # sees literally no information about which row it is looking at.
    Xtr_z = np.zeros_like(Xtr)
    Xval_z = np.zeros_like(Xval)
    assert Xtr_z.shape[1] == TOTAL_DIM and not Xtr_z.any()
    assert Xval_z.shape[1] == TOTAL_DIM and not Xval_z.any()

    seeds = SEEDS[: args.seeds]
    per_seed = []
    for seed in seeds:
        model, val_bce, epoch = train_model_early_stopping(
            Xtr_z, ytr, Xval_z, yval, TOTAL_DIM, max_epochs=MAX_EPOCHS, seed=seed
        )
        with torch.no_grad():
            logits_val = model(torch.from_numpy(Xval_z)).numpy()
        pred_val = 1.0 / (1.0 + np.exp(-logits_val))
        gap = baseline_bce - val_bce  # positive = zero-input model BEAT the baseline
        per_seed.append(
            {
                "seed": seed,
                "best_epoch": epoch,
                "val_bce": float(val_bce),
                "val_brier": brier(pred_val, yval),
                "gap_below_constant_mean_baseline": float(gap),
                # A zero-input feedforward net is mathematically incapable of
                # producing different outputs per row. Asserting it empirically
                # rather than assuming it catches an accidental input leak
                # (e.g. a mask that didn't apply) that BCE alone might not.
                "prediction_std": float(np.std(pred_val)),
                "prediction_mean": float(np.mean(pred_val)),
                "collapses_to_baseline": bool(val_bce >= baseline_bce - TOLERANCE),
            }
        )
        print(
            f"[zero-control] seed={seed}: val_bce={val_bce:.6f} "
            f"(baseline {baseline_bce:.6f}, gap {gap:+.6f}), "
            f"pred_mean={np.mean(pred_val):.6f} pred_std={np.std(pred_val):.3e}, "
            f"best_epoch={epoch}"
        )

    bces = [s["val_bce"] for s in per_seed]
    result = {
        "control": "zeroed-feature (all 68 input dims zeroed, train and val)",
        "scale": "full baseline split (not the n=64 memorization_test.py pilot)",
        "protocol_match": {
            "architecture": "TinyNet(input_dim=68, hidden=32)",
            "optimizer": "Adam lr=1e-2",
            "loss": "soft-label BCE",
            "early_stopping": "patience=30 on val BCE, max 500 epochs, best checkpoint",
            "seeds": seeds,
            "source": "identical call path to train_baseline.py's run_variant()",
        },
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "train_label_mean": train_mean,
        "constant_mean_baseline_val_bce": baseline_bce,
        "per_seed": per_seed,
        "val_bce_mean": float(np.mean(bces)),
        "val_bce_std": float(np.std(bces)),
        "max_gap_below_baseline": float(baseline_bce - min(bces)),
        "tolerance": TOLERANCE,
        "collapses_to_baseline_all_seeds": bool(all(s["collapses_to_baseline"] for s in per_seed)),
        "predictions_constant_all_seeds": bool(all(s["prediction_std"] < 1e-6 for s in per_seed)),
        "test_split_touched": False,
    }

    with open(args.out / "phase2d_zeroed_feature_control.json", "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[zero-control] mean val_bce={result['val_bce_mean']:.6f} +- {result['val_bce_std']:.6f} "
        f"vs constant-mean {baseline_bce:.6f}; max gap below baseline "
        f"{result['max_gap_below_baseline']:+.6f} (tolerance {TOLERANCE}); "
        f"collapses={result['collapses_to_baseline_all_seeds']}, "
        f"constant_predictions={result['predictions_constant_all_seeds']}"
    )
    print(f"[zero-control] wrote {args.out / 'phase2d_zeroed_feature_control.json'}")


if __name__ == "__main__":
    main()
