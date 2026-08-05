"""Print the Phase 2E test-split tables from `results/phase2e_test_results.json`.

Reads only what `evaluate_test.py` wrote; computes nothing.
Usage:  python summarize_test.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KINDS = ["direct", "bank", "double-bank", "combo", "rail-combo"]


def f(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  n/a "
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/phase2e_test_results.json"))
    args = ap.parse_args()
    d = json.load(open(args.results))

    print(f"{'variant':24s} {'testBCE':>16s} {'GAS':>16s} {'NDCG@3':>8s} {'top1':>7s} "
          f"{'regret':>7s} {'ECE':>7s} {'ECEcal':>7s} {'Brier':>7s}")
    for name, v in d["variants"].items():
        bce = f"{f(v['test_bce']['mean'])}+-{f(v['test_bce']['std'])}"
        g = v["test_group_aware_spearman"]
        gas = "undefined".rjust(16) if g["mean"] is None else f"{f(g['mean'])}+-{f(g['std'])}".rjust(16)
        print(f"{name:24s} {bce:>16s} {gas} {f(v['test_ndcg@3']['mean']):>8s} "
              f"{f(v['test_top1_hit_rate']['mean']):>7s} {f(v['test_mean_regret']['mean']):>7s} "
              f"{f(v['test_ece']['mean']):>7s} {f(v['test_ece_calibrated']['mean']):>7s} "
              f"{f(v['test_brier']['mean']):>7s}")

    print()
    print("per-kind group-aware Spearman (test, mean +- std across seeds)")
    print(f"{'variant':24s} " + " ".join(f"{k:>16s}" for k in KINDS))
    for name, v in d["variants"].items():
        cells = []
        for k in KINDS:
            c = v["per_kind_group_aware_spearman"].get(k)
            cells.append("n/a".rjust(16) if c is None else f"{f(c['mean'])}+-{f(c['std'])}".rjust(16))
        print(f"{name:24s} " + " ".join(cells))

    print()
    print("per-seed test group-aware Spearman (consistency of direction)")
    for name in ["mlp_matched", "deepsets"]:
        v = d["variants"][name]
        print(f"{name:24s} " + " ".join(f"{p['group_aware_spearman']:.4f}" for p in v["per_seed"]))
    for name in ["mlp_matched", "deepsets"]:
        v = d["variants"][name]
        print(f"{name:24s} BCE " + " ".join(f"{p['bce']:.4f}" for p in v["per_seed"]))


if __name__ == "__main__":
    main()
