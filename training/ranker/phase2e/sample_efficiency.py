"""Sample-efficiency curve: how much training data does each architecture need?

One of the predeclared dimensions on which a new architecture may be adopted.
Sub-samples the TRAIN split by *state* (never by row — splitting a state's
candidates across the in/out sets would leak within-state structure and
flatter the curve), retrains from scratch at each fraction, and evaluates on
the untouched validation split. Test is never loaded.

3 seeds per point (the brief's minimum) rather than 5, since this is a
supporting curve, not a headline comparison.

Usage:  python sample_efficiency.py --out results/sample_efficiency.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import metrics
from data import load
from models import DeepSetsRanker, TinyNet
from train import predict, train_one

FRACTIONS = [0.05, 0.1, 0.25, 0.5, 1.0]
SEEDS = [20260804, 20260805, 20260806]
ARCHS = {
    "mlp_matched": lambda: TinyNet(68, 32),
    "deepsets": lambda: DeepSetsRanker(64, 64),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("results/dataset.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/sample_efficiency.json"))
    args = ap.parse_args()

    splits = load(args.cache)
    tr, va = splits["train"], splits["val"]
    Xval, yval = torch.from_numpy(va.X), torch.from_numpy(va.y)
    states = np.unique(tr.state)

    out = {"fractions": FRACTIONS, "seeds": SEEDS, "split_used": "val", "curves": {}}
    for name, build in ARCHS.items():
        curve = []
        for frac in FRACTIONS:
            rng = np.random.default_rng(20260804)
            keep = rng.choice(states, size=max(1, int(round(len(states) * frac))), replace=False)
            m = np.isin(tr.state, keep)
            Xtr, ytr = torch.from_numpy(tr.X[m]), torch.from_numpy(tr.y[m])
            bces, gases = [], []
            for seed in SEEDS:
                torch.manual_seed(seed)
                model, bce, _, _ = train_one(build(), Xtr, ytr, Xval, yval, 3e-3, seed)
                pred = 1.0 / (1.0 + np.exp(-predict(model, Xval)))
                bces.append(bce)
                gases.append(metrics.group_aware_spearman(pred, va.y, va.state)[0])
            curve.append({
                "fraction": frac,
                "n_train_states": int(len(keep)),
                "n_train_rows": int(m.sum()),
                "val_bce_mean": float(np.mean(bces)),
                "val_bce_std": float(np.std(bces)),
                "val_group_aware_spearman_mean": float(np.mean(gases)),
                "val_group_aware_spearman_std": float(np.std(gases)),
            })
            print(f"{name:14s} frac={frac:<5} rows={m.sum():6d}  "
                  f"val BCE {np.mean(bces):.4f}  GAS {np.mean(gases):.4f}")
        out["curves"][name] = curve
        args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
