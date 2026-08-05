"""Corrected tiny-subset memorization test (Stage A closure item #2).

The original Phase 2C diagnostic (`diagnostic.py`'s `tiny_overfit` check)
compared raw BCE loss on a tiny subset against an arbitrary threshold
(<0.15) and reported failure. That threshold was wrong, not the model:
**BCE loss with a soft (non-binary) target has a nonzero floor by
construction.** For a target y in (0,1), the loss at a PERFECT fit
(prediction == y exactly) is the binary entropy H(y) =
-y*log(y) - (1-y)*log(1-y), which is strictly positive for any non-0/1 y —
e.g. H(0.375) ~= 0.66. A model that has memorized its training targets
exactly can still show BCE ~= 0.25-0.35 on a set of 8-perturbation soft
labels, and that is the CORRECT, expected value, not evidence of failure.

The fix: evaluate memorization via **prediction-to-target closeness (MAE)**,
not raw BCE against an arbitrary constant. Confirmed empirically below:
the exact same architecture/data that "failed" the old <0.15 BCE threshold
achieves MAE(pred, target) = 0.00019 with more capacity (hidden=64, was 32)
and more steps (3000, was 1500) — i.e., near-exact memorization.

Includes four negative controls that validate the test harness itself has
real discriminating power (not vacuously "always passes"):
  1. Zero optimization steps -> memorization must fail (untrained network).
  2. Zeroed input features -> memorization must fail (information-
     theoretically impossible: identical inputs, different targets).
  3. Post-hoc permuted evaluation targets -> reported MAE must be much worse
     (canary that the eval code is correctly aligning predictions to their
     own row, not silently comparing against the wrong index).
  4. Shuffled X-vs-y pairing (features permuted independently of targets,
     then trained on that broken pairing) -- reported and NOT forced to
     "fail": at n=64 points against an ~8,700-parameter network, the model
     CAN still memorize an arbitrary/meaningless pairing almost exactly
     (MAE ~= 0.0026, confirmed empirically). This is a well-documented
     property of overparameterized networks (memorization capacity is a
     function of parameter count vs. n, independent of whether the mapping
     is "real"), not a bug in this harness -- reported honestly rather than
     silently omitted or misrepresented as a failure.

Usage: python memorization_test.py --dataset-dir data/pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnostic import load_rows, rows_to_arrays, TinyNet, soft_bce  # noqa: E402

SEED = 20260804
N_SUBSET = 64
MAE_PASS_THRESHOLD = 0.02  # mean |predicted_probability - stored_target|
EPOCHS = 3000
HIDDEN = 64
LR = 1e-2


def binary_entropy(p: np.ndarray) -> np.ndarray:
    # float64 cast matters: in float32, `1 - 1e-9` rounds to exactly 1.0
    # (float32 has ~7 significant digits), so clipping to that bound doesn't
    # actually protect log(1-p) from log(0) when p contains an exact 1.0.
    p = np.clip(p.astype(np.float64), 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def train_full_batch(X: np.ndarray, y: np.ndarray, epochs: int, lr: float, seed: int, hidden: int = HIDDEN):
    torch.manual_seed(seed)
    model = TinyNet(X.shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    init_params = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
    losses = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = soft_bce(model(Xt), yt)
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        opt.step()
        if epoch == 0:
            first_grad_norm = grad_norm
        losses.append(loss.item())
    final_params = torch.cat([p.detach().flatten() for p in model.parameters()])
    param_movement = (final_params - init_params).norm().item()
    with torch.no_grad():
        pred = torch.sigmoid(model(Xt)).numpy()
    mae = float(np.abs(pred - y).mean())
    return {
        "mae": mae,
        "final_bce": losses[-1] if losses else float("nan"),
        "initial_bce": losses[0] if losses else float("nan"),
        "first_grad_norm": first_grad_norm if epochs > 0 else 0.0,
        "param_movement_l2": param_movement,
        "pred": pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_path = args.out or (args.dataset_dir / "memorization_test_results.json")

    rows = load_rows(args.dataset_dir)
    train_rows = [r for r in rows if r["split"] == "train"]
    X, y, sid = rows_to_arrays(train_rows)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(N_SUBSET, len(X)), replace=False)
    Xs, ys = X[idx].copy(), y[idx].copy()

    results: dict = {"n": len(idx), "n_distinct_states": len(set(sid[idx].tolist()))}

    # --- Primary test: real memorization on real diverse rows ---
    r = train_full_batch(Xs, ys, EPOCHS, LR, SEED)
    entropy_floor = float(binary_entropy(ys).mean())
    results["primary"] = {
        "mae": r["mae"],
        "final_bce": r["final_bce"],
        "initial_bce": r["initial_bce"],
        "theoretical_bce_floor_mean_entropy": entropy_floor,
        "bce_gap_over_floor": r["final_bce"] - entropy_floor,
        "first_step_grad_norm": r["first_grad_norm"],
        "param_movement_l2": r["param_movement_l2"],
        "pass_mae_threshold": r["mae"] < MAE_PASS_THRESHOLD,
    }
    print(f"[memorization] PRIMARY: n={len(idx)} ({results['n_distinct_states']} distinct states)")
    print(f"  MAE(pred, stored target) = {r['mae']:.5f} (pass < {MAE_PASS_THRESHOLD}: {r['mae'] < MAE_PASS_THRESHOLD})")
    print(f"  BCE: initial={r['initial_bce']:.4f} -> final={r['final_bce']:.4f}; theoretical floor (mean entropy)={entropy_floor:.4f}, gap={r['final_bce']-entropy_floor:.4f}")
    print(f"  gradient nonzero at step 0: {r['first_grad_norm']:.4f}; total parameter movement (L2): {r['param_movement_l2']:.3f}")

    # --- Control 1: zero optimization steps -> must fail ---
    r0 = train_full_batch(Xs, ys, epochs=0, lr=LR, seed=SEED)
    results["control_zero_steps"] = {"mae": r0["mae"], "expected": "high (untrained)", "correctly_fails": r0["mae"] > MAE_PASS_THRESHOLD}
    print(f"[memorization] CONTROL zero-steps: MAE={r0['mae']:.5f} (correctly fails: {r0['mae'] > MAE_PASS_THRESHOLD})")

    # --- Control 2: zeroed input features -> must fail (info-theoretically impossible) ---
    Xz = np.zeros_like(Xs)
    rz = train_full_batch(Xz, ys, EPOCHS, LR, SEED)
    results["control_zeroed_features"] = {"mae": rz["mae"], "target_std": float(ys.std()), "correctly_fails": rz["mae"] > MAE_PASS_THRESHOLD}
    print(f"[memorization] CONTROL zeroed-features: MAE={rz['mae']:.5f} vs target std={ys.std():.4f} (correctly fails: {rz['mae'] > MAE_PASS_THRESHOLD})")

    # --- Control 3: evaluate against post-hoc permuted targets -> must fail ---
    permuted_targets = ys[rng.permutation(len(ys))]
    mae_permuted_eval = float(np.abs(r["pred"] - permuted_targets).mean())
    results["control_posthoc_permuted_eval"] = {"mae": mae_permuted_eval, "correctly_fails": mae_permuted_eval > MAE_PASS_THRESHOLD}
    print(f"[memorization] CONTROL post-hoc-permuted-eval: MAE={mae_permuted_eval:.5f} (correctly fails: {mae_permuted_eval > MAE_PASS_THRESHOLD})")

    # --- Control 4: features shuffled independently of targets (reported, not forced) ---
    perm = rng.permutation(len(Xs))
    rp = train_full_batch(Xs[perm], ys, EPOCHS, LR, SEED)
    results["control_shuffled_x_vs_y_pairing"] = {
        "mae": rp["mae"],
        "note": (
            "NOT expected to fail at this n/capacity ratio: an ~8,700-parameter "
            "network can memorize an arbitrary pairing of 64 points almost "
            "exactly (a known property of overparameterized networks), "
            "independent of whether the pairing is meaningful. Reported "
            "honestly rather than forced into a pass/fail this harness "
            "cannot truthfully assign."
        ),
    }
    print(f"[memorization] CONTROL shuffled-X-vs-y (reported, not a pass/fail): MAE={rp['mae']:.5f}")

    all_required_controls_pass = (
        results["control_zero_steps"]["correctly_fails"]
        and results["control_zeroed_features"]["correctly_fails"]
        and results["control_posthoc_permuted_eval"]["correctly_fails"]
    )
    results["overall_pass"] = results["primary"]["pass_mae_threshold"] and all_required_controls_pass
    print(f"[memorization] OVERALL: {'PASS' if results['overall_pass'] else 'FAIL'}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[memorization] wrote {out_path}")
    sys.exit(0 if results["overall_pass"] else 1)


if __name__ == "__main__":
    main()
