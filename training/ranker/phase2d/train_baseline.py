"""Phase 2D: first defensible candidate-conditioned baseline.

Implements docs/repair/showboat-ml/phase-2d/EVALUATION_PROTOCOL.md exactly
— frozen architecture (no hyperparameter search), 5 seeds, early stopping
on validation BCE only, calibration fit on validation only. Never touches
the test split: it's loaded only to print its row count, never used for
training/evaluation. Saves checkpoints for the main model and state-only
ablation so evaluate_test.py can later touch the test split exactly once
against these exact frozen weights, not retrained copies.

Usage:
    python train_baseline.py --dataset-dir ../phase2c/data/baseline --out results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Reuse Phase 2C's reviewed, working pieces rather than reimplementing them.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2c"))
from diagnostic import (  # noqa: E402
    TinyNet,
    soft_bce,
    train_model_early_stopping,
    constant_mean_bce,
    pooled_spearman,
    group_aware_spearman,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # training/ranker/, for schema.py
from schema import BOARD_DIM, TOTAL_DIM  # noqa: E402

SEEDS = [20260804, 20260805, 20260806, 20260807, 20260808]
KINDS = ["direct", "bank", "double-bank", "combo", "rail-combo"]
MAX_EPOCHS = 500

# Candidate-block feature offsets within the 68-dim row, per encode.ts's
# encodeCandidateFeatures (verified against the TS source directly, not
# guessed): [sin(phi), cos(phi), power, sideSpin, topSpin, *5 kind one-hot,
# banks/4, target/(OBS_BALLS-1), *6 pocket one-hot, pathLength/diag,
# clearance/8]. Absolute index = BOARD_DIM + offset.
KIND_ONEHOT_OFFSET = 5  # 5 dims
BANKS_OFFSET = 10
CLEARANCE_OFFSET = 19  # last candidate feature


def load_rows(dataset_dir: Path) -> list[dict]:
    rows = []
    for shard in sorted(dataset_dir.glob("*.ndjson")):
        with open(shard) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def to_arrays(rows: list[dict]):
    X = np.array([r["features"] for r in rows], dtype=np.float32)
    y = np.array([r["raw_counts"]["legal_pot"] / r["n_perturbations"] for r in rows], dtype=np.float32)
    kind = np.array([r["candidate_kind"] for r in rows])
    state_id = np.array([r["state_id"] for r in rows])
    return X, y, kind, state_id


def brier(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def ece(pred: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(pred)
    err = 0.0
    for i in range(n_bins):
        mask = (pred >= bins[i]) & (pred < bins[i + 1] if i < n_bins - 1 else pred <= bins[i + 1])
        if mask.sum() == 0:
            continue
        conf = pred[mask].mean()
        acc = target[mask].mean()
        err += (mask.sum() / total) * abs(conf - acc)
    return float(err)


def fit_platt(logits: np.ndarray, target: np.ndarray, epochs: int = 500, lr: float = 0.05) -> tuple[float, float]:
    """1-parameter-pair Platt scaling: sigmoid(a*logit + b), fit by gradient descent on soft-label BCE."""
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    z = torch.from_numpy(logits)
    t = torch.from_numpy(target)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(a * z + b, t)
        loss.backward()
        opt.step()
    return float(a.detach()), float(b.detach())


def apply_platt(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * logits + b)))


def train_heuristic(Xtr, ytr, Xval, yval, seed: int):
    """Transparent baseline: logistic regression on 8 interpretable geometric
    features (5 kind one-hot + banks count + clearance), not the full 68-dim
    board+candidate vector. A linear model over these 8 named coefficients is
    directly inspectable, unlike the 32-hidden-unit MLP."""
    idx = list(range(BOARD_DIM + KIND_ONEHOT_OFFSET, BOARD_DIM + KIND_ONEHOT_OFFSET + 5)) + [
        BOARD_DIM + BANKS_OFFSET,
        BOARD_DIM + CLEARANCE_OFFSET,
    ]
    torch.manual_seed(seed)
    Xtr_h = Xtr[:, idx]
    Xval_h = Xval[:, idx]
    w = torch.zeros(len(idx), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    Xt, yt = torch.from_numpy(Xtr_h), torch.from_numpy(ytr)
    Xv, yv = torch.from_numpy(Xval_h), torch.from_numpy(yval)
    opt = torch.optim.Adam([w, b], lr=0.05)
    best_val, best_w, best_b, epochs_since = float("inf"), w.detach().clone(), b.detach().clone(), 0
    for epoch in range(200):
        opt.zero_grad()
        logits = Xt @ w + b
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            vloss = nn.functional.binary_cross_entropy_with_logits(Xv @ w + b, yv).item()
        if vloss < best_val:
            best_val, best_w, best_b, epochs_since = vloss, w.detach().clone(), b.detach().clone(), 0
        else:
            epochs_since += 1
            if epochs_since >= 30:
                break
    coefficients = {name: round(float(best_w[i]), 4) for i, name in enumerate(KINDS + ["banks", "clearance"])}
    coefficients["intercept"] = round(float(best_b[0]), 4)
    return best_w, best_b, idx, best_val, coefficients


def eval_heuristic(X, w, b, idx) -> np.ndarray:
    Xh = torch.from_numpy(X[:, idx])
    with torch.no_grad():
        return torch.sigmoid(Xh @ w + b).numpy()


def per_kind_metrics(pred: np.ndarray, target: np.ndarray, kind: np.ndarray, state_id: np.ndarray) -> dict:
    out = {}
    for k in KINDS:
        mask = kind == k
        if mask.sum() == 0:
            continue
        out[k] = {
            "n": int(mask.sum()),
            "bce": float(nn.functional.binary_cross_entropy_with_logits(
                torch.logit(torch.clamp(torch.from_numpy(pred[mask]), 1e-6, 1 - 1e-6)),
                torch.from_numpy(target[mask]),
            ).item()) if mask.sum() > 0 else None,
            "brier": brier(pred[mask], target[mask]),
            "group_aware_spearman": group_aware_spearman(pred[mask], target[mask], state_id[mask])[0],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.dataset_dir)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"[phase2d] rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    Xtr, ytr, ktr, str_ = to_arrays(train_rows)
    Xval, yval, kval, sval = to_arrays(val_rows)

    results: dict = {"n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows), "seeds": SEEDS}

    # --- Baseline 1: constant-mean ---
    const_p = float(np.clip(ytr.mean(), 1e-6, 1 - 1e-6))
    const_val_bce = constant_mean_bce(ytr, yval)
    results["baseline_constant_mean"] = {"p": const_p, "val_bce": const_val_bce, "val_brier": brier(np.full_like(yval, const_p), yval)}
    print(f"[phase2d] baseline constant-mean: p={const_p:.4f} val_bce={const_val_bce:.4f}")

    # --- Baseline 2: candidate-kind mean ---
    kind_means = {k: float(np.clip(ytr[ktr == k].mean(), 1e-6, 1 - 1e-6)) if (ktr == k).sum() else const_p for k in KINDS}
    pred_kind_mean_val = np.array([kind_means[k] for k in kval], dtype=np.float32)
    kind_mean_val_bce = float(nn.functional.binary_cross_entropy(torch.from_numpy(pred_kind_mean_val), torch.from_numpy(yval)).item())
    results["baseline_candidate_kind_mean"] = {"kind_means": kind_means, "val_bce": kind_mean_val_bce, "val_brier": brier(pred_kind_mean_val, yval)}
    print(f"[phase2d] baseline candidate-kind-mean: val_bce={kind_mean_val_bce:.4f}")

    # --- Baseline 3: transparent heuristic (single seed sufficient -- it's a 8-coefficient linear model, not sensitive to init) ---
    hw, hb, hidx, h_val_bce, h_coefs = train_heuristic(Xtr, ytr, Xval, yval, SEEDS[0])
    pred_heuristic_val = eval_heuristic(Xval, hw, hb, hidx)
    results["baseline_heuristic"] = {
        "coefficients": h_coefs,
        "val_bce": h_val_bce,
        "val_brier": brier(pred_heuristic_val, yval),
        "val_group_aware_spearman": group_aware_spearman(pred_heuristic_val, yval, sval)[0],
    }
    print(f"[phase2d] baseline heuristic: val_bce={h_val_bce:.4f}, coefficients={h_coefs}")

    # --- Main model + ablations, across all seeds. Checkpoints for the two
    # variants Stage B.5 needs on the untouched test split (main model,
    # state-only) are persisted to disk so evaluate_test.py can load the
    # EXACT frozen weights rather than retraining -- retraining with the
    # "same" seed is not guaranteed bit-identical across machines/torch
    # versions, and the whole point of freezing is that test-set evaluation
    # must run against the precise artifact the protocol was frozen against.
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def zero_mask(zero_board: bool, zero_kind: bool, zero_geo: bool, zero_candidate: bool) -> np.ndarray:
        mask = np.ones(TOTAL_DIM, dtype=np.float32)
        if zero_board:
            mask[:BOARD_DIM] = 0.0
        if zero_candidate:
            mask[BOARD_DIM:] = 0.0
        if zero_kind:
            mask[BOARD_DIM + KIND_ONEHOT_OFFSET : BOARD_DIM + KIND_ONEHOT_OFFSET + 5] = 0.0
        if zero_geo:
            mask[BOARD_DIM + CLEARANCE_OFFSET - 1 : BOARD_DIM + CLEARANCE_OFFSET + 1] = 0.0
        return mask

    def run_variant(name: str, key: str, zero_board: bool, zero_kind: bool, zero_geo: bool, zero_candidate: bool, save: bool):
        mask = zero_mask(zero_board, zero_kind, zero_geo, zero_candidate)
        Xtr_v = Xtr * mask
        Xval_v = Xval * mask

        per_seed = []
        for seed in SEEDS:
            model, val_bce, epoch = train_model_early_stopping(Xtr_v, ytr, Xval_v, yval, TOTAL_DIM, max_epochs=MAX_EPOCHS, seed=seed)
            with torch.no_grad():
                logits_val = model(torch.from_numpy(Xval_v)).numpy()
            pred_val = 1.0 / (1.0 + np.exp(-logits_val))
            a, b = fit_platt(logits_val, yval)
            pred_val_calibrated = apply_platt(logits_val, a, b)
            if save:
                torch.save(model.state_dict(), ckpt_dir / f"{key}_seed{seed}.pt")
            per_seed.append({
                "seed": seed,
                "best_epoch": epoch,
                "val_bce": val_bce,
                "val_brier": brier(pred_val, yval),
                "val_brier_calibrated": brier(pred_val_calibrated, yval),
                "val_ece": ece(pred_val, yval),
                "val_ece_calibrated": ece(pred_val_calibrated, yval),
                "val_pooled_spearman": pooled_spearman(pred_val, yval),
                "val_group_aware_spearman": group_aware_spearman(pred_val, yval, sval)[0],
                "platt_a": a,
                "platt_b": b,
                "per_kind": per_kind_metrics(pred_val, yval, kval, sval),
            })
        bces = [s["val_bce"] for s in per_seed]
        spearmans = [s["val_group_aware_spearman"] for s in per_seed]
        summary = {
            "per_seed": per_seed,
            "mask": mask.tolist() if save else None,  # only persisted for variants with saved checkpoints
            "val_bce_mean": float(np.mean(bces)),
            "val_bce_std": float(np.std(bces)),
            "val_group_aware_spearman_mean": float(np.mean(spearmans)),
            "val_group_aware_spearman_std": float(np.std(spearmans)),
        }
        print(f"[phase2d] {name}: val_bce={summary['val_bce_mean']:.4f}+-{summary['val_bce_std']:.4f}, "
              f"group_aware_spearman={summary['val_group_aware_spearman_mean']:.4f}+-{summary['val_group_aware_spearman_std']:.4f}")
        return summary

    results["candidate_conditioned"] = run_variant("candidate-conditioned (main model)", "main", False, False, False, False, save=True)
    results["ablation_state_only"] = run_variant("ablation: state-only", "state_only", False, False, False, True, save=True)
    results["ablation_candidate_only"] = run_variant("ablation: candidate-only", "candidate_only", True, False, False, False, save=False)
    results["ablation_no_kind"] = run_variant("ablation: no candidate-kind encoding", "no_kind", False, True, False, False, save=False)
    results["ablation_no_geo"] = run_variant("ablation: no geometric features", "no_geo", False, False, True, False, save=False)

    # --- Minimum-evidence checklist (val-based; test-set confirmation happens separately in evaluate_test.py) ---
    main_bce = results["candidate_conditioned"]["val_bce_mean"]
    checklist = {
        "beats_constant_mean": main_bce < const_val_bce,
        "beats_candidate_kind_mean": main_bce < kind_mean_val_bce,
        "beats_state_only": main_bce < results["ablation_state_only"]["val_bce_mean"],
        "beats_heuristic_on_group_aware_spearman": results["candidate_conditioned"]["val_group_aware_spearman_mean"] > results["baseline_heuristic"]["val_group_aware_spearman"],
    }
    # Checked per-seed, not just on the mean: all four primary comparisons
    # above must hold for every seed individually, not only in aggregate.
    state_only_by_seed = {s["seed"]: s["val_bce"] for s in results["ablation_state_only"]["per_seed"]}
    heuristic_gas = results["baseline_heuristic"]["val_group_aware_spearman"]
    checklist["consistent_direction_across_seeds"] = all(
        s["val_bce"] < const_val_bce
        and s["val_bce"] < kind_mean_val_bce
        and s["val_bce"] < state_only_by_seed[s["seed"]]
        and s["val_group_aware_spearman"] > heuristic_gas
        for s in results["candidate_conditioned"]["per_seed"]
    )
    per_kind_spearman = {
        k: np.mean([s["per_kind"].get(k, {}).get("group_aware_spearman", np.nan) for s in results["candidate_conditioned"]["per_seed"]])
        for k in KINDS
    }
    pooled_gas = results["candidate_conditioned"]["val_group_aware_spearman_mean"]
    checklist["no_kind_catastrophically_worse"] = all(
        (np.isnan(v) or v >= pooled_gas - 0.3) for v in per_kind_spearman.values()
    )
    results["minimum_evidence_checklist"] = checklist
    results["per_kind_group_aware_spearman_mean"] = {k: (None if np.isnan(v) else float(v)) for k, v in per_kind_spearman.items()}
    print(f"[phase2d] minimum-evidence checklist (val): {checklist}")

    results["test_split_touched"] = False
    results["note"] = (
        "This script never reads test_rows for training/evaluation (loaded above only to print its "
        "count) -- per the frozen protocol, the test split is touched exactly once, by evaluate_test.py, "
        "after this script's results are reviewed and the configuration is confirmed final. Checkpoints "
        "for the main model and the state-only ablation (the two variants the minimum-evidence checklist "
        "needs on test) are saved to checkpoints/ for that script to load directly -- not retrained."
    )

    with open(args.out / "phase2d_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[phase2d] wrote {args.out / 'phase2d_results.json'}")


if __name__ == "__main__":
    main()
