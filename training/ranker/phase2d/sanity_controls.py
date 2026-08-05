"""Phase 2D: shuffled-label control, re-run at full baseline scale.

Phase 2C's diagnostic.py already ran this control on the small pilot
dataset. The evaluation protocol's minimum-evidence checklist requires it
re-confirmed at baseline scale with the frozen architecture -- a shuffled-
label model that still beats/matches the constant-mean baseline on
120k-row data would indicate a leak that the smaller pilot run could have
missed. Train/val only -- never touches test.

Usage:
    python sanity_controls.py --dataset-dir ../phase2c/data/baseline --out results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2c"))
from diagnostic import TinyNet, train_model_early_stopping, constant_mean_bce  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import TOTAL_DIM  # noqa: E402

from train_baseline import load_rows, to_arrays, brier  # noqa: E402

SEED = 20260804
MAX_EPOCHS = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = load_rows(args.dataset_dir)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    Xtr, ytr, _, _ = to_arrays(train_rows)
    Xval, yval, _, _ = to_arrays(val_rows)

    baseline_bce = constant_mean_bce(ytr, yval)

    rng = np.random.default_rng(SEED)
    y_shuffled = ytr.copy()
    rng.shuffle(y_shuffled)
    model, shuffled_val_bce, epoch = train_model_early_stopping(
        Xtr, y_shuffled, Xval, yval, TOTAL_DIM, max_epochs=MAX_EPOCHS, seed=SEED
    )
    with np.errstate(all="ignore"):
        import torch

        with torch.no_grad():
            pred_val = 1.0 / (1.0 + np.exp(-model(torch.from_numpy(Xval)).numpy()))
    collapses = shuffled_val_bce >= baseline_bce - 0.02

    result = {
        "seed": SEED,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "constant_mean_baseline_val_bce": baseline_bce,
        "shuffled_label_val_bce": shuffled_val_bce,
        "shuffled_label_val_brier": brier(pred_val, yval),
        "best_epoch": epoch,
        "collapses_to_baseline_or_worse": bool(collapses),
        "note": (
            "Re-run of diagnostic.py's shuffled-label control at full baseline "
            "dataset scale (120k rows vs. the pilot's much smaller set). "
            "collapses_to_baseline_or_worse=True means training on shuffled "
            "labels could not beat the constant-mean baseline by more than the "
            "0.02 BCE tolerance -- i.e. no leak lets the model cheat."
        ),
    }
    print(f"[sanity] shuffled-label control @ baseline scale: val_bce={shuffled_val_bce:.4f} "
          f"vs constant-mean={baseline_bce:.4f} (collapses: {collapses})")

    with open(args.out / "phase2d_sanity_controls.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[sanity] wrote {args.out / 'phase2d_sanity_controls.json'}")


if __name__ == "__main__":
    main()
