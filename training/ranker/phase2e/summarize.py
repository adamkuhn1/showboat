"""Print the Phase 2E comparison tables from the per-variant result JSONs.

Reads only what `train.py` / `evaluate_test.py` wrote; computes nothing new, so
the report's tables and the raw artifacts cannot drift apart.

Usage:  python summarize.py [--split val|test]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ORDER = [
    "mlp_matched", "mlp_wide", "flat_entity_mlp",
    "deepsets_small", "deepsets", "deepsets_attn", "deepsets_pot_id",
    "deepsets_no_relations", "deepsets_no_kind",
    "deepsets_state_only", "deepsets_candidate_only",
    "zero_input_mlp", "zero_input_deepsets",
]
KINDS = ["direct", "bank", "double-bank", "combo", "rail-combo"]


def f(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  n/a "
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()

    print(f"{'variant':24s} {'params':>7s} {'in':>4s} {'seeds':>5s} "
          f"{'valBCE':>16s} {'GAS':>16s} {'NDCG@3':>8s} {'top1':>7s} {'regret':>7s} "
          f"{'ECE':>7s} {'train s':>8s} {'lat ms':>7s}")
    for name in ORDER:
        p = args.results / f"variant_{name}.json"
        if not p.exists():
            continue
        s = json.load(open(p))
        bce = f"{f(s['val_bce']['mean'])}+-{f(s['val_bce']['std'])}"
        gas = f"{f(s['val_group_aware_spearman']['mean'])}+-{f(s['val_group_aware_spearman']['std'])}"
        print(f"{name:24s} {s['n_params']:7d} {s['input_dim']:4d} {len(s['seeds']):5d} "
              f"{bce:>16s} {gas:>16s} {f(s['val_ndcg@3']['mean']):>8s} "
              f"{f(s['val_top1_hit_rate']['mean']):>7s} {f(s['val_mean_regret']['mean']):>7s} "
              f"{f(s['val_ece']['mean']):>7s} {s['train_seconds']['mean']:8.1f} "
              f"{f(s['latency_median_ms']['mean'], 3):>7s}")

    print()
    print("per-kind group-aware Spearman (validation, mean +- std across seeds)")
    print(f"{'variant':24s} " + " ".join(f"{k:>16s}" for k in KINDS))
    for name in ORDER:
        p = args.results / f"variant_{name}.json"
        if not p.exists():
            continue
        s = json.load(open(p))
        cells = []
        for k in KINDS:
            d = s["per_kind_group_aware_spearman"].get(k)
            cells.append("n/a".rjust(16) if d is None else f"{f(d['mean'])}+-{f(d['std'])}".rjust(16))
        print(f"{name:24s} " + " ".join(cells))


if __name__ == "__main__":
    main()
