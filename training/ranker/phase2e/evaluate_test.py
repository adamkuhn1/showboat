"""Phase 2E one-shot test-split evaluation.

This is the ONLY file in Phase 2E that reads the test split. It refuses to run
unless `frozen_config.json` exists, and it evaluates exactly the checkpoints
and calibration parameters that file names — nothing is chosen, tuned, or
re-fit here. Platt (a, b) come from each seed's validation fit, applied once.

Honest caveat, recorded here and in the report rather than glossed: this phase's
comparison is NOT perfectly blind, because Phase 2D's test numbers for the MLP
were already public in `docs/repair/showboat-ml/phase-2d/` before this phase
started. What is preserved is the part that actually matters — no architecture,
width, learning rate, seed, calibration, ablation, or recommendation was chosen
using test data, and the recommendation was written into `frozen_config.json`
before this script ran for the first time.

Usage:
    python evaluate_test.py --frozen frozen_config.json --out results/phase2e_test_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import metrics
from data import load
from models import DeepSetsAttn, DeepSetsRanker, FlatEntityMLP, TinyNet
from train import identity, with_pot, zero_all, zero_board, zero_candidate, zero_kind

BUILDERS = {
    "mlp_matched": lambda: TinyNet(68, 32),
    "mlp_wide": lambda: TinyNet(68, 128),
    "flat_entity_mlp": lambda: FlatEntityMLP(hidden=64),
    "deepsets_small": lambda: DeepSetsRanker(32, 32),
    "deepsets": lambda: DeepSetsRanker(64, 64),
    "deepsets_attn": lambda: DeepSetsAttn(64, 64, 4),
    "deepsets_pot_id": lambda: DeepSetsRanker(64, 64, use_pot_id=True),
    "deepsets_no_relations": lambda: DeepSetsRanker(64, 64, zero_relations=True),
    "deepsets_no_kind": lambda: DeepSetsRanker(64, 64),
    "deepsets_state_only": lambda: DeepSetsRanker(64, 64),
    "deepsets_candidate_only": lambda: DeepSetsRanker(64, 64),
    "zero_input_mlp": lambda: TinyNet(68, 32),
    "zero_input_deepsets": lambda: DeepSetsRanker(64, 64),
}
TRANSFORMS = {
    "deepsets_pot_id": with_pot,
    "deepsets_no_kind": zero_kind,
    "deepsets_state_only": zero_candidate,
    "deepsets_candidate_only": zero_board,
    "zero_input_mlp": zero_all,
    "zero_input_deepsets": zero_all,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("results/dataset.npz"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--frozen", type=Path, default=Path("frozen_config.json"))
    ap.add_argument("--out", type=Path, default=Path("results/phase2e_test_results.json"))
    args = ap.parse_args()

    if not args.frozen.exists():
        raise SystemExit(
            f"{args.frozen} missing — the frozen configuration and the recommendation "
            "must be committed before the test split is read even once."
        )
    frozen = json.loads(args.frozen.read_text())

    splits = load(args.cache)
    te = splits["test"]
    out = {
        "frozen_config": frozen,
        "n_test_rows": int(len(te.y)),
        "variants": {},
        "policy_note": (
            "Test split read once, by this script, using checkpoints and validation-fit "
            "Platt parameters frozen beforehand. No selection or tuning happens here."
        ),
    }

    for name in frozen["evaluate_on_test"]:
        res = json.load(open(args.results / f"variant_{name}.json"))
        tf = TRANSFORMS.get(name, identity)
        X = torch.from_numpy(tf(te.X, te.pot))
        per_seed = []
        for s in res["per_seed"]:
            ck = args.results / "checkpoints" / f"{name}_seed{s['seed']}.pt"
            m = BUILDERS[name]()
            m.load_state_dict(torch.load(ck, map_location="cpu"))
            m.eval()
            with torch.no_grad():
                logits = np.concatenate(
                    [m(X[i : i + 16384]).numpy() for i in range(0, len(X), 16384)]
                )
            pred = 1.0 / (1.0 + np.exp(-logits))
            pred_cal = 1.0 / (1.0 + np.exp(-(s["platt_a"] * logits + s["platt_b"])))
            rep = metrics.full_report(pred, te.y, te.state, te.kind)
            rep["ece_calibrated"] = metrics.ece(pred_cal, te.y)
            rep["brier_calibrated"] = metrics.brier(pred_cal, te.y)
            rep["seed"] = s["seed"]
            per_seed.append(rep)

        def agg(key):
            v = [p[key] for p in per_seed]
            v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
            return {"mean": float(np.mean(v)), "std": float(np.std(v))} if v else {"mean": None, "std": None}

        out["variants"][name] = {
            "seeds": [p["seed"] for p in per_seed],
            "test_bce": agg("bce"),
            "test_brier": agg("brier"),
            "test_ece": agg("ece"),
            "test_ece_calibrated": agg("ece_calibrated"),
            "test_group_aware_spearman": agg("group_aware_spearman"),
            "test_pooled_spearman": agg("pooled_spearman"),
            "test_ndcg@3": agg("ndcg@3"),
            "test_top1_hit_rate": agg("top1_hit_rate"),
            "test_mean_regret": agg("mean_regret"),
            "per_kind_group_aware_spearman": {
                k: {
                    "mean": float(np.nanmean([p["per_kind"][k]["group_aware_spearman"] for p in per_seed])),
                    "std": float(np.nanstd([p["per_kind"][k]["group_aware_spearman"] for p in per_seed])),
                    "n": per_seed[0]["per_kind"][k]["n"],
                }
                for k in metrics.KINDS
                if k in per_seed[0]["per_kind"]
            },
            "per_seed": per_seed,
        }
        v = out["variants"][name]
        print(f"{name:24s} test BCE {v['test_bce']['mean']:.4f}+-{v['test_bce']['std']:.4f}  "
              f"GAS {v['test_group_aware_spearman']['mean'] if v['test_group_aware_spearman']['mean'] is not None else float('nan'):.4f}  "
              f"dbank {v['per_kind_group_aware_spearman'].get('double-bank', {}).get('mean', float('nan')):.4f}")

    args.out.write_text(json.dumps(out, indent=2))
    print(f"[phase2e] wrote {args.out}")


if __name__ == "__main__":
    main()
