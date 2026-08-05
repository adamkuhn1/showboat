"""Phase 2E bounded configuration search — VALIDATION ONLY.

Phase 2D deliberately froze its architecture with no search at all, to remove
any avenue for tuning pressure. This phase has to choose *some* configuration
for a new architecture, so the search is (a) tiny, (b) declared here in full
including every configuration tried, and (c) run on ONE seed against the
validation split only. The test split is not loaded by this file.

Crucially the MLP baseline gets the *same* courtesy search over width and
learning rate, so the relational arm is never compared against a deliberately
under-tuned MLP.

Every configuration attempted is written to `results/search/search.json`,
including the ones that lost — no quiet dropping of unfavourable configs.

Usage:  python search.py --out results/search
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import load
from models import DeepSetsRanker, TinyNet
from train import Variant, run_variant

SEARCH_SEED = 20260804

GRID = [
    # (name, builder, lr)
    ("mlp_h32_lr1e-3", lambda: TinyNet(68, 32), 1e-3),
    ("mlp_h32_lr3e-3", lambda: TinyNet(68, 32), 3e-3),
    ("mlp_h64_lr3e-3", lambda: TinyNet(68, 64), 3e-3),
    ("mlp_h128_lr3e-3", lambda: TinyNet(68, 128), 3e-3),
    ("mlp_h128_lr1e-3", lambda: TinyNet(68, 128), 1e-3),
    ("ds_h32_lr3e-3", lambda: DeepSetsRanker(32, 32), 3e-3),
    ("ds_h64_lr1e-3", lambda: DeepSetsRanker(64, 64), 1e-3),
    ("ds_h64_lr3e-3", lambda: DeepSetsRanker(64, 64), 3e-3),
    ("ds_h128_lr3e-3", lambda: DeepSetsRanker(128, 128), 3e-3),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("results/dataset.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/search"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits = load(args.cache)
    rows = []
    for name, build, lr in GRID:
        v = Variant(name, build, lr=lr, seeds=[SEARCH_SEED])
        s = run_variant(v, splits, args.out, save_ckpt=False)
        rows.append({
            "config": name,
            "lr": lr,
            "n_params": s["n_params"],
            "val_bce": s["val_bce"]["mean"],
            "val_group_aware_spearman": s["val_group_aware_spearman"]["mean"],
            "val_double_bank_gas": s["per_kind_group_aware_spearman"]["double-bank"]["mean"],
            "train_seconds": s["train_seconds"]["mean"],
            "best_epoch": s["per_seed"][0]["best_epoch"],
        })
        with open(args.out / "search.json", "w") as f:
            json.dump({"seed": SEARCH_SEED, "split_used": "val", "grid": rows}, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
