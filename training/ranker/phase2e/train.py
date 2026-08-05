"""Phase 2E training/evaluation driver.

Every decision this script makes — architecture, width, learning rate, batch
size, early stopping, calibration — is made against TRAIN and VALIDATION only.
The test split is never loaded here; `evaluate_test.py` touches it once, after
configs are frozen in `frozen_config.json`.

Protocol (fixed for every architecture so the comparison is fair):
  - optimizer Adam, minibatch (Phase 2D used full-batch GD, which is fine for a
    3.3k-parameter MLP but would take hours for the relational models; the MLP
    is therefore ALSO re-trained under this protocol — see the
    `mlp_matched` variant — so no comparison is MLP-full-batch vs
    relational-minibatch)
  - loss: soft-label BCE against legal_pot_rate, identical target to Phase 2D
  - early stopping on validation BCE, patience 20 epochs, max 200, best
    checkpoint kept (not last)
  - seeds: the same 5 Phase 2D used, 20260804..20260808
  - Platt scaling (a, b) fit on validation predictions only

Usage:
    python train.py --variants mlp_matched deepsets --out results
    python train.py --list
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

import metrics
from data import load
from entities import BOARD_DIM, TOTAL_DIM
from models import DeepSetsAttn, DeepSetsRanker, FlatEntityMLP, TinyNet, count_params

SEEDS = [20260804, 20260805, 20260806, 20260807, 20260808]
MAX_EPOCHS = 200
PATIENCE = 20
BATCH = 4096
DEFAULT_LR = 3e-3

torch.set_num_threads(4)


# ---------------------------------------------------------------------------
# input transforms (controls/ablations operate on the 68-dim row, exactly like
# Phase 2D's zero-masks, so "same information removed" means the same thing)
# ---------------------------------------------------------------------------
def identity(X: np.ndarray, pot: np.ndarray) -> np.ndarray:
    return X


def zero_all(X, pot):
    return np.zeros_like(X)


def zero_board(X, pot):
    Y = X.copy()
    Y[:, :BOARD_DIM] = 0.0
    return Y


def zero_candidate(X, pot):
    Y = X.copy()
    Y[:, BOARD_DIM:] = 0.0
    return Y


def zero_kind(X, pot):
    Y = X.copy()
    Y[:, 53:58] = 0.0
    return Y


def with_pot(X, pot):
    return np.concatenate([X, (pot.astype(np.float32) / 15.0)[:, None]], axis=1)


@dataclass
class Variant:
    name: str
    build: Callable[[], nn.Module]
    transform: Callable = identity
    lr: float = DEFAULT_LR
    seeds: list[int] = field(default_factory=lambda: list(SEEDS))
    note: str = ""


def variants() -> dict[str, Variant]:
    v: dict[str, Variant] = {}

    # --- baselines / MLP arms -------------------------------------------------
    v["mlp_matched"] = Variant(
        "mlp_matched", lambda: TinyNet(TOTAL_DIM, 32),
        note="Phase 2D's exact architecture, retrained under this phase's minibatch protocol.",
    )
    v["mlp_wide"] = Variant(
        "mlp_wide", lambda: TinyNet(TOTAL_DIM, 128),
        note="Capacity-matched-ish MLP control so the baseline is not handicapped by width.",
    )

    # --- relational arms ------------------------------------------------------
    v["deepsets"] = Variant(
        "deepsets", lambda: DeepSetsRanker(hidden=64, phi_hidden=64),
        note="Candidate-conditioned Deep Sets, identical [N,68] input contract.",
    )
    v["deepsets_small"] = Variant(
        "deepsets_small", lambda: DeepSetsRanker(hidden=32, phi_hidden=32),
    )
    v["deepsets_attn"] = Variant(
        "deepsets_attn", lambda: DeepSetsAttn(hidden=64, phi_hidden=64, heads=4),
        seeds=SEEDS[:3],
        note="GNN arm: one masked self-attention layer over the 16-ball set. 3 seeds (cost).",
    )
    v["flat_entity_mlp"] = Variant(
        "flat_entity_mlp", lambda: FlatEntityMLP(hidden=64),
        note="Same entity features, flattened — isolates features from permutation invariance.",
    )

    # --- ablations (relational) ----------------------------------------------
    v["deepsets_no_relations"] = Variant(
        "deepsets_no_relations", lambda: DeepSetsRanker(hidden=64, phi_hidden=64, zero_relations=True),
        note="Relationship-feature ablation: path/bank relation dims zeroed, everything else kept.",
    )
    v["deepsets_no_kind"] = Variant(
        "deepsets_no_kind", lambda: DeepSetsRanker(hidden=64, phi_hidden=64), transform=zero_kind,
        note="Candidate-kind one-hot zeroed (Phase 2D ablation 3, relational arm).",
    )
    v["deepsets_state_only"] = Variant(
        "deepsets_state_only", lambda: DeepSetsRanker(hidden=64, phi_hidden=64), transform=zero_candidate,
        note="Candidate block zeroed (Phase 2D ablation 1, relational arm).",
    )
    v["deepsets_candidate_only"] = Variant(
        "deepsets_candidate_only", lambda: DeepSetsRanker(hidden=64, phi_hidden=64), transform=zero_board,
        note="Board block zeroed (Phase 2D ablation 2, relational arm).",
    )
    v["deepsets_pot_id"] = Variant(
        "deepsets_pot_id", lambda: DeepSetsRanker(hidden=64, phi_hidden=64, use_pot_id=True),
        transform=with_pot,
        note="CONTRACT CHANGE: 69-dim input adding the intended pot ball id.",
    )

    # --- full-scale all-zero-input controls (the gap Phase 2D flagged) --------
    v["zero_input_mlp"] = Variant(
        "zero_input_mlp", lambda: TinyNet(TOTAL_DIM, 32), transform=zero_all,
        note="Full-scale all-68-dims-zero leak control for the MLP.",
    )
    v["zero_input_deepsets"] = Variant(
        "zero_input_deepsets", lambda: DeepSetsRanker(hidden=64, phi_hidden=64), transform=zero_all,
        note="Full-scale all-68-dims-zero leak control for the relational model.",
    )
    return v


# ---------------------------------------------------------------------------
def fit_platt(logits: np.ndarray, target: np.ndarray, epochs: int = 500, lr: float = 0.05):
    """Same two-parameter Platt scaling Phase 2D used (`train_baseline.py`)."""
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    z, t = torch.from_numpy(logits), torch.from_numpy(target)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(a * z + b, t).backward()
        opt.step()
    return float(a.detach()), float(b.detach())


def predict(model: nn.Module, X: torch.Tensor, batch: int = 16384) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(X[i : i + batch]).numpy())
    return np.concatenate(out)


def train_one(model, Xtr, ytr, Xval, yval, lr, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = len(ytr)
    best, best_state, best_epoch, stale = float("inf"), None, -1, 0
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(Xval), 16384):
                vp.append(model(Xval[i : i + 16384]))
            vloss = nn.functional.binary_cross_entropy_with_logits(torch.cat(vp), yval).item()
        if vloss < best - 1e-6:
            best, best_epoch, stale = vloss, epoch, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best, best_epoch, time.time() - t0


def infer_latency_ms(model: nn.Module, X: torch.Tensor, batch: int = 24, trials: int = 50):
    model.eval()
    x = X[:batch]
    with torch.no_grad():
        for _ in range(5):
            model(x)
        ts = []
        for _ in range(trials):
            t = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t) * 1000.0)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}


def run_variant(var: Variant, splits, out_dir: Path, save_ckpt: bool) -> dict:
    tr, va = splits["train"], splits["val"]
    Xtr_np = var.transform(tr.X, tr.pot)
    Xval_np = var.transform(va.X, va.pot)
    Xtr, ytr = torch.from_numpy(Xtr_np), torch.from_numpy(tr.y)
    Xval, yval = torch.from_numpy(Xval_np), torch.from_numpy(va.y)

    per_seed = []
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for seed in var.seeds:
        torch.manual_seed(seed)
        model = var.build()
        n_params = count_params(model)
        model, val_bce, epoch, secs = train_one(Xtr=Xtr, ytr=ytr, Xval=Xval, yval=yval,
                                                model=model, lr=var.lr, seed=seed)
        logits = predict(model, Xval)
        pred = 1.0 / (1.0 + np.exp(-logits))
        a, b = fit_platt(logits, va.y)
        pred_cal = 1.0 / (1.0 + np.exp(-(a * logits + b)))
        rep = metrics.full_report(pred, va.y, va.state, va.kind)
        rep_cal_ece = metrics.ece(pred_cal, va.y)
        if save_ckpt:
            torch.save(model.state_dict(), ckpt_dir / f"{var.name}_seed{seed}.pt")
        per_seed.append({
            "seed": seed,
            "n_params": n_params,
            "best_epoch": epoch,
            "train_seconds": secs,
            "val": rep,
            "val_brier_calibrated": metrics.brier(pred_cal, va.y),
            "val_ece_calibrated": rep_cal_ece,
            "platt_a": a,
            "platt_b": b,
            "latency": infer_latency_ms(model, Xval),
        })
        print(f"  [{var.name}] seed {seed}: val BCE {rep['bce']:.4f}  "
              f"GAS {rep['group_aware_spearman']:.4f}  "
              f"dbank {rep['per_kind'].get('double-bank', {}).get('group_aware_spearman', float('nan')):.4f}  "
              f"({secs:.0f}s, {epoch + 1} epochs, {n_params} params)")

    def agg(path):
        vals = []
        for s in per_seed:
            d = s
            for p in path:
                d = d[p] if isinstance(d, dict) else d
            vals.append(d)
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            return {"mean": None, "std": None}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {
        "name": var.name,
        "note": var.note,
        "seeds": var.seeds,
        "n_params": per_seed[0]["n_params"],
        "input_dim": int(Xtr.shape[1]),
        "train_seconds": agg(["train_seconds"]),
        "val_bce": agg(["val", "bce"]),
        "val_group_aware_spearman": agg(["val", "group_aware_spearman"]),
        "val_pooled_spearman": agg(["val", "pooled_spearman"]),
        "val_brier": agg(["val", "brier"]),
        "val_ece": agg(["val", "ece"]),
        "val_ece_calibrated": agg(["val_ece_calibrated"]),
        "val_ndcg@3": agg(["val", "ndcg@3"]),
        "val_top1_hit_rate": agg(["val", "top1_hit_rate"]),
        "val_mean_regret": agg(["val", "mean_regret"]),
        "latency_median_ms": agg(["latency", "median_ms"]),
        "per_kind_group_aware_spearman": {
            k: {
                "mean": float(np.nanmean([s["val"]["per_kind"][k]["group_aware_spearman"] for s in per_seed])),
                "std": float(np.nanstd([s["val"]["per_kind"][k]["group_aware_spearman"] for s in per_seed])),
                "n": per_seed[0]["val"]["per_kind"][k]["n"],
            }
            for k in metrics.KINDS
            if k in per_seed[0]["val"]["per_kind"]
        },
        "per_seed": per_seed,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("results/dataset.npz"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="override the seed list (used for the bounded val-only config search)")
    ap.add_argument("--lr", type=float, default=None, help="override lr (config search only)")
    ap.add_argument("--tag", type=str, default="", help="suffix for the output filename")
    args = ap.parse_args()

    vs = variants()
    if args.seeds:
        for v in vs.values():
            v.seeds = args.seeds
    if args.lr:
        for v in vs.values():
            v.lr = args.lr
    if args.list:
        for k, v in vs.items():
            print(f"{k:26s} {v.note}")
        return

    splits = load(args.cache)
    names = args.variants or list(vs)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in vs:
            raise SystemExit(f"unknown variant {name}; --list to see all")
        print(f"[phase2e] === {name} ===")
        s = run_variant(vs[name], splits, args.out, save_ckpt=not args.no_ckpt)
        s["env"] = {"torch": torch.__version__, "platform": platform.platform()}
        with open(args.out / f"variant_{name}{args.tag}.json", "w") as f:
            json.dump(s, f, indent=2)
        # A control whose predictions are constant (the all-zero-input arm) has
        # NO defined ranking metric — group_aware_spearman aggregates to None.
        # Print that as "undefined" rather than crashing or, worse, coercing it
        # to a number; the whole point of Phase 2D's tie-handling fix was that
        # an undefined ordering must not be silently invented.
        gas = s["val_group_aware_spearman"]
        gas_s = "undefined (constant predictions)" if gas["mean"] is None else f"{gas['mean']:.4f}+-{gas['std']:.4f}"
        print(f"[phase2e] {name}: val BCE {s['val_bce']['mean']:.4f}+-{s['val_bce']['std']:.4f}  GAS {gas_s}")


if __name__ == "__main__":
    main()
