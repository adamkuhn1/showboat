"""Physics-budget proxy: how many candidates would you have to simulate?

The live AI's real cost is physics rollouts, not neural inference. A ranker earns
its keep by letting the search simulate fewer candidates for the same decision
quality. `recall_of_best@k` measures exactly that: over states with >=k+1
candidates, the fraction where a *truly best* candidate (one attaining the
state's maximum `legal_pot_rate`) appears in the model's top-k.

If the relational model reaches at k=3 the recall the MLP needs k=8 for, the
physics budget for equal quality is ~2.6x smaller. That is one of the
predeclared dimensions on which adopting a new architecture is allowed.

Validation split only. Uses the already-trained checkpoints; trains nothing.

Usage:  python budget.py --out results/budget.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import load
from models import DeepSetsRanker, FlatEntityMLP, TinyNet

BUILDERS = {
    "mlp_matched": lambda: TinyNet(68, 32),
    "mlp_wide": lambda: TinyNet(68, 128),
    "flat_entity_mlp": lambda: FlatEntityMLP(hidden=64),
    "deepsets_small": lambda: DeepSetsRanker(32, 32),
    "deepsets": lambda: DeepSetsRanker(64, 64),
}
KS = [1, 2, 3, 5, 8, 12]


def recall_of_best_at_k(pred, target, state, k):
    hits, n = 0, 0
    order = np.argsort(state, kind="stable")
    for idx in np.split(order, np.flatnonzero(np.diff(state[order])) + 1):
        if len(idx) <= k:
            continue
        t = target[idx]
        if t.max() <= 0:
            continue
        n += 1
        top = np.argsort(-pred[idx], kind="stable")[:k]
        if t[top].max() >= t.max() - 1e-9:
            hits += 1
    return (hits / n if n else float("nan")), n


def mean_best_rate_at_k(pred, target, state, k):
    """Expected legal_pot_rate of the best candidate among the model's top k —
    i.e. the quality the search would end up with if it only simulated k."""
    vals = []
    order = np.argsort(state, kind="stable")
    for idx in np.split(order, np.flatnonzero(np.diff(state[order])) + 1):
        if len(idx) <= k:
            continue
        t = target[idx]
        top = np.argsort(-pred[idx], kind="stable")[:k]
        vals.append(float(t[top].max()))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("results/dataset.npz"))
    ap.add_argument("--ckpt", type=Path, default=Path("results/checkpoints"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/budget.json"))
    args = ap.parse_args()

    va = load(args.cache)["val"]
    X = torch.from_numpy(va.X)
    out = {"split": "val", "ks": KS, "models": {}}

    # random ordering, as the "no ranker at all" reference point
    rng = np.random.default_rng(20260804)
    rand = rng.random(len(va.y)).astype(np.float32)
    out["models"]["random_order"] = {
        "recall_of_best@k": {str(k): recall_of_best_at_k(rand, va.y, va.state, k)[0] for k in KS},
        "mean_best_rate@k": {str(k): mean_best_rate_at_k(rand, va.y, va.state, k) for k in KS},
    }

    for name, build in BUILDERS.items():
        res_path = args.results / f"variant_{name}.json"
        if not res_path.exists():
            continue
        res = json.load(open(res_path))
        per_seed = {}
        for s in res["per_seed"]:
            ck = args.ckpt / f"{name}_seed{s['seed']}.pt"
            if not ck.exists():
                continue
            m = build()
            m.load_state_dict(torch.load(ck, map_location="cpu"))
            m.eval()
            with torch.no_grad():
                logits = np.concatenate([m(X[i: i + 16384]).numpy() for i in range(0, len(X), 16384)])
            per_seed[str(s["seed"])] = {
                "recall_of_best@k": {str(k): recall_of_best_at_k(logits, va.y, va.state, k)[0] for k in KS},
                "mean_best_rate@k": {str(k): mean_best_rate_at_k(logits, va.y, va.state, k) for k in KS},
            }
        if not per_seed:
            continue
        out["models"][name] = {
            "per_seed": per_seed,
            "recall_of_best@k": {
                str(k): float(np.mean([v["recall_of_best@k"][str(k)] for v in per_seed.values()])) for k in KS
            },
            "mean_best_rate@k": {
                str(k): float(np.mean([v["mean_best_rate@k"][str(k)] for v in per_seed.values()])) for k in KS
            },
        }
    _, n_states = recall_of_best_at_k(rand, va.y, va.state, 1)
    out["n_states_at_k1"] = n_states
    args.out.write_text(json.dumps(out, indent=2))

    print(f"{'model':24s} " + " ".join(f"k={k:<6d}" for k in KS))
    for name, d in out["models"].items():
        print(f"{name:24s} " + " ".join(f"{d['recall_of_best@k'][str(k)]:.4f}  " for k in KS))


if __name__ == "__main__":
    main()
