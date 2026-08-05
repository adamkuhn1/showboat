"""Phase 2E structural controls.

These are not accuracy measurements — they check that the relational model has
the properties it is being adopted *for*. A permutation-invariant architecture
whose invariance is never tested is decoration; this file is what makes the
claim real.

Controls implemented:

  1. entity-order permutation invariance
     Permute the order in which the 16 entities are presented (each entity's own
     feature vector unchanged) and require a bit-identical-to-float-noise output.
     This is the literal definition of the property Deep Sets buys.

  2. same-group relabelling invariance (the physically meaningful version)
     Swap the board positions of two balls in the SAME group (both stripes, say)
     where neither is the target nor the intended pot. Physically nothing about
     the shot has changed. Because no ball-number feature exists in the entity
     encoding, the relational model must return exactly the same score. Run the
     identical perturbation through the flat MLPs to show they do not.

  3. masked / missing-ball behaviour
     (a) Pocketed balls must have zero influence: overwrite a pocketed ball's
         x,y with garbage (leaving the pocketed flag set) — the relational
         model's output must not move at all, because the mask removes it from
         both pools. The 68-dim MLP has no such guarantee and its output does
         move.
     (b) Pocketing an irrelevant live ball must change the score in a bounded,
         sane way rather than producing NaN/inf or a wild swing.

  4. candidate-kind ablation and relationship-feature ablation are trained
     variants and live in `train.py`; the full-scale all-zero-input control is
     the `zero_input_*` variants there.

Usage:
    python controls.py --ckpt results/checkpoints --out results/controls.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import load
from entities import BOARD_DIM
from models import DeepSetsRanker, FlatEntityMLP, TinyNet

SEED = 20260804
N = 512


def _load(cls, path: Path, **kw):
    m = cls(**kw)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def permutation_invariance(model, X: torch.Tensor, rng) -> dict:
    """Control 1. Requires the model to expose forward_from_entities."""
    with torch.no_grad():
        ent, mask, ctx = model.enc(X)
        base = model.forward_from_entities(ent, mask, ctx).numpy()
        worst = 0.0
        for _ in range(20):
            p = torch.from_numpy(rng.permutation(ent.shape[1]))
            out = model.forward_from_entities(ent[:, p, :], mask[:, p], ctx).numpy()
            worst = max(worst, float(np.max(np.abs(out - base))))
    return {
        "max_abs_logit_change_over_20_permutations": worst,
        "n_rows": int(X.shape[0]),
        "passes_1e-5": bool(worst < 1e-5),
    }


def _same_group_swap(X: np.ndarray, target: np.ndarray, pot: np.ndarray, rng):
    """Build a copy of X with two same-group, non-target, non-pot LIVE balls'
    positions swapped. Returns (X_swapped, n_rows_actually_swapped)."""
    Y = X.copy()
    groups = [list(range(1, 8)), list(range(9, 16))]  # solids, stripes
    n_swapped = 0
    for r in range(len(X)):
        cands = []
        for g in groups:
            live = [
                b for b in g
                if X[r, 3 * b + 2] < 0.5 and b != int(target[r]) and b != int(pot[r])
            ]
            if len(live) >= 2:
                cands.append(live)
        if not cands:
            continue
        g = cands[rng.integers(len(cands))]
        i, j = rng.choice(len(g), size=2, replace=False)
        a, b = g[i], g[j]
        Y[r, 3 * a : 3 * a + 2], Y[r, 3 * b : 3 * b + 2] = (
            X[r, 3 * b : 3 * b + 2].copy(),
            X[r, 3 * a : 3 * a + 2].copy(),
        )
        n_swapped += 1
    return Y, n_swapped


def relabel_invariance(models: dict, X: np.ndarray, target, pot, rng) -> dict:
    Y, n = _same_group_swap(X, target, pot, rng)
    out = {"n_rows": int(len(X)), "n_rows_swapped": n}
    with torch.no_grad():
        for name, m in models.items():
            a = m(torch.from_numpy(X)).numpy()
            b = m(torch.from_numpy(Y)).numpy()
            d = float(np.max(np.abs(a - b)))
            out[name] = {
                "max_abs_logit_change": d,
                "mean_abs_logit_change": float(np.mean(np.abs(a - b))),
                "invariant_1e-5": bool(d < 1e-5),
            }
    return out


def pocketed_ball_garbage(models: dict, X: np.ndarray, rng) -> dict:
    """Control 3a: scramble the coordinates of already-pocketed balls."""
    Y = X.copy()
    n = 0
    for r in range(len(X)):
        poc = [b for b in range(16) if X[r, 3 * b + 2] > 0.5]
        if not poc:
            continue
        n += 1
        for b in poc:
            Y[r, 3 * b] = rng.uniform(-1, 1)
            Y[r, 3 * b + 1] = rng.uniform(-1, 1)
    out = {"n_rows": int(len(X)), "n_rows_with_pocketed_balls": n}
    with torch.no_grad():
        for name, m in models.items():
            a = m(torch.from_numpy(X)).numpy()
            b = m(torch.from_numpy(Y)).numpy()
            d = float(np.max(np.abs(a - b)))
            out[name] = {
                "max_abs_logit_change": d,
                "mean_abs_logit_change": float(np.mean(np.abs(a - b))),
                "unaffected_1e-5": bool(d < 1e-5),
            }
    return out


def remove_live_ball(models: dict, X: np.ndarray, target, pot, rng) -> dict:
    """Control 3b: pocket one irrelevant live ball; check the response is finite
    and bounded, not NaN and not a wild swing."""
    Y = X.copy()
    n = 0
    for r in range(len(X)):
        live = [
            b for b in range(1, 16)
            if X[r, 3 * b + 2] < 0.5 and b != int(target[r]) and b != int(pot[r]) and b != 8
        ]
        if not live:
            continue
        b = live[rng.integers(len(live))]
        Y[r, 3 * b] = 0.0
        Y[r, 3 * b + 1] = 0.0
        Y[r, 3 * b + 2] = 1.0
        n += 1
    out = {"n_rows": int(len(X)), "n_rows_modified": n}
    with torch.no_grad():
        for name, m in models.items():
            a = m(torch.from_numpy(X)).numpy()
            b = m(torch.from_numpy(Y)).numpy()
            d = np.abs(a - b)
            out[name] = {
                "finite": bool(np.isfinite(b).all()),
                "mean_abs_logit_change": float(d.mean()),
                "p99_abs_logit_change": float(np.percentile(d, 99)),
                "max_abs_logit_change": float(d.max()),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("results/dataset.npz"))
    ap.add_argument("--ckpt", type=Path, default=Path("results/checkpoints"))
    ap.add_argument("--out", type=Path, default=Path("results/controls.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    splits = load(args.cache)
    va = splits["val"]
    idx = rng.choice(len(va), size=N, replace=False)
    X = va.X[idx]
    target, pot = va.target[idx], va.pot[idx]

    ds = _load(DeepSetsRanker, args.ckpt / f"deepsets_seed{SEED}.pt", hidden=64, phi_hidden=64)
    mlp = _load(TinyNet, args.ckpt / f"mlp_matched_seed{SEED}.pt", input_dim=68, hidden=32)
    flat = _load(FlatEntityMLP, args.ckpt / f"flat_entity_mlp_seed{SEED}.pt", hidden=64)
    models = {"deepsets": ds, "mlp_matched": mlp, "flat_entity_mlp": flat}

    res = {
        "seed": SEED,
        "n_eval_rows": N,
        "split_used": "val",
        "permutation_invariance_deepsets": permutation_invariance(ds, torch.from_numpy(X), rng),
        "permutation_invariance_flat_entity_mlp": permutation_invariance(
            flat_as_setmodel(flat), torch.from_numpy(X), rng
        ),
        "same_group_relabel_invariance": relabel_invariance(models, X, target, pot, rng),
        "pocketed_ball_coordinate_garbage": pocketed_ball_garbage(models, X, rng),
        "remove_one_irrelevant_live_ball": remove_live_ball(models, X, target, pot, rng),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


class _FlatAsSet(torch.nn.Module):
    """Adapter so the flat MLP can be run through the identical
    permutation-invariance harness — it consumes the same entities, just
    concatenated in a fixed order. Expected to FAIL the control; that failure
    is the point of including it."""

    def __init__(self, flat: FlatEntityMLP):
        super().__init__()
        self.flat = flat
        self.enc = flat.enc

    def forward_from_entities(self, ent, mask, ctx):
        e = ent * mask.unsqueeze(-1)
        return self.flat.net(torch.cat([e.reshape(e.shape[0], -1), ctx], dim=-1)).squeeze(-1)


def flat_as_setmodel(flat: FlatEntityMLP) -> _FlatAsSet:
    return _FlatAsSet(flat)


if __name__ == "__main__":
    main()
