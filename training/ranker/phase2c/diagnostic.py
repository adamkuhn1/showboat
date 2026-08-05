"""Phase 2C diagnostic training checks — NOT the Phase 2D model.

Purpose (per the task brief): validate structural learnability and rule out
leakage in the new dataset/schema/splits, using a tiny throwaway MLP. Does
not replace the existing public ranker artifact (artifacts/run1/*) and is
not evaluated as a final model.

Four checks, all reported (including unfavorable results, not cherry-picked):
  1. Tiny-subset overfit: can the architecture drive training loss near zero
     on a handful of examples? (Rules out a frozen-weights / wrong-loss-sign
     / shape-mismatch bug that would silently prevent ANY learning.)
  2. Shuffled-label control: train on the same inputs with labels randomly
     shuffled. Held-out performance should collapse to roughly the constant-
     mean baseline. If a shuffled-label model still "generalizes", something
     is leaking information through row order/grouping, not genuine signal.
  3. Candidate-conditioned vs. state-only: does adding the candidate block
     (the back TOTAL_DIM-BOARD_DIM features) measurably beat a model with
     only the board block? This is the central claim Showboat's ML redirect
     rests on — a candidate-blind model can't tell candidates at the same
     state apart, so if it does just as well, the candidate features aren't
     adding real signal yet.
  4. Constant-mean baseline comparison: does the real (unshuffled) model beat
     a trivial "always predict the training-set mean" predictor on held-out
     data? (Recorded honestly — Phase 2C's 03-split-quality-audit.md found
     the OLD dataset/model did NOT clear this bar; this check exists so that
     claim gets a real, reproducible answer on the new dataset too.)

Usage:
    python diagnostic.py --dataset-dir data/pilot --out data/pilot/diagnostics

Label: empirical legal-pot rate under perturbation
(raw_counts.legal_pot / n_perturbations) — a soft target in [0,1], trained
with soft-label BCE (a standard proper scoring rule for probabilistic
targets), NOT a hard binary label.

Splits: reuses the family-aware `split` field already assigned by
gen_dataset_v3.ts (train/val/test) — this script does not re-split, since
re-splitting here would bypass the leakage protections in split.ts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # training/ranker/, for schema.py
from schema import BOARD_DIM, TOTAL_DIM  # noqa: E402

DIAGNOSTIC_SEED = 20260804


def load_rows(dataset_dir: Path) -> list[dict]:
    rows = []
    for shard in sorted(dataset_dir.glob("*.ndjson")):
        with open(shard, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([r["features"] for r in rows], dtype=np.float32)
    y = np.array(
        [r["raw_counts"]["legal_pot"] / r["n_perturbations"] for r in rows],
        dtype=np.float32,
    )
    state_ids = np.array([r["state_id"] for r in rows])
    return X, y, state_ids


class TinyNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def soft_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, target)


def train_model_early_stopping(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    input_dim: int,
    max_epochs: int = 500,
    patience: int = 30,
    lr: float = 1e-2,
    seed: int = DIAGNOSTIC_SEED,
) -> tuple[TinyNet, float, int]:
    """Early-stops on validation BCE (best-checkpoint, not best-of-fixed-N):
    the fair way to compare architectures/feature sets on a small, noisy
    dataset without one run's overfitting point contaminating the
    comparison. Returns (best model, best val BCE, epoch it occurred at)."""
    torch.manual_seed(seed)
    model = TinyNet(input_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xv, yv = torch.from_numpy(Xval), torch.from_numpy(yval)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = soft_bce(model(Xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss = soft_bce(model(Xv), yv).item()
        if vloss < best_val:
            best_val = vloss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val, best_epoch


def constant_mean_bce(train_y: np.ndarray, eval_y: np.ndarray) -> float:
    p = float(np.clip(train_y.mean(), 1e-6, 1 - 1e-6))
    eval_y_c = np.clip(eval_y, 1e-6, 1 - 1e-6)
    return float(-(eval_y_c * np.log(p) + (1 - eval_y_c) * np.log(1 - p)).mean())


def pooled_spearman(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman's rho with proper tie-averaged ranks (scipy.stats.rankdata,
    method='average'), NOT the double-argsort ordinal-rank shortcut used
    earlier -- ordinal ranks silently break ties by array order, which
    matters a great deal here: legal_pot_rate only takes 9 or 33 distinct
    values (n_perturbations=8 or 32), so a large fraction of within-state
    pairs are genuinely tied (measured ~30% on the baseline dataset, with
    42.7% of all rows at exactly 0). Found by independent review comparing
    this function's output against a tie-corrected reference implementation
    on real checkpoint predictions -- see PHASE_2D REVIEW section of
    docs/repair/showboat-ml/phase-2d/REVIEW.md."""
    if len(pred) < 3 or np.std(pred) == 0 or np.std(target) == 0:
        return float("nan")
    from scipy.stats import rankdata

    rp = rankdata(pred, method="average")
    rt = rankdata(target, method="average")
    return float(np.corrcoef(rp, rt)[0, 1])


def group_aware_spearman(pred: np.ndarray, target: np.ndarray, state_ids: np.ndarray) -> tuple[float, int]:
    """Mean of per-state Spearman correlations, over states with >=3 candidates
    and non-constant target within the state (a meaningful within-decision
    ranking signal) — the metric the 03-split-quality-audit.md found the old
    pipeline never computed, using pooled Spearman across unrelated states
    instead (which conflates "spread across easy vs hard states" with "did we
    rank candidates within a state correctly", the latter being the actual
    decision task the ranker is used for at inference time)."""
    per_state = []
    for sid in np.unique(state_ids):
        mask = state_ids == sid
        if mask.sum() < 3:
            continue
        t = target[mask]
        if np.std(t) == 0:
            continue
        p = pred[mask]
        rho = pooled_spearman(p, t)
        if not np.isnan(rho):
            per_state.append(rho)
    if not per_state:
        return float("nan"), 0
    return float(np.mean(per_state)), len(per_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    rows = load_rows(args.dataset_dir)
    if not rows:
        print(f"No rows found in {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"[diagnostic] rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    results: dict = {"dataset_dir": str(args.dataset_dir), "n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows)}

    Xtr, ytr, sid_tr = rows_to_arrays(train_rows)
    Xval, yval, sid_val = rows_to_arrays(val_rows) if val_rows else (Xtr[:0], ytr[:0], sid_tr[:0])

    if len(val_rows) == 0:
        print("[diagnostic] no val rows — cannot run early-stopped comparisons; smoke-scale datasets are expected to hit this.", file=sys.stderr)
        results["note"] = "no val rows available at this dataset scale; comparisons skipped"
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "diagnostic_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # --- Check 1: tiny-subset memorization ----------------------------------
    # See memorization_test.py for the authoritative implementation and full
    # explanation. Do NOT duplicate a raw-BCE-vs-threshold check here again —
    # that was the original bug (Stage A closure): BCE against a soft,
    # non-binary target has a nonzero floor (the target's own binary entropy)
    # even at a perfect fit, so "BCE < 0.15" was never an achievable or
    # meaningful bar. Run `python memorization_test.py --dataset-dir <dir>`
    # separately for the rigorous version (MAE-to-target + 3 negative
    # controls); this just points at it so results aren't duplicated/stale.
    print("[diagnostic] tiny-overfit/memorization check moved to memorization_test.py — run it separately.")
    results["tiny_overfit"] = {"note": "see memorization_test.py output (memorization_test_results.json) for the corrected version of this check"}

    baseline_bce = constant_mean_bce(ytr, yval)

    # --- Full-size candidate-conditioned model (early-stopped on val) -------
    full_model, full_val_bce, full_epoch = train_model_early_stopping(Xtr, ytr, Xval, yval, TOTAL_DIM, max_epochs=args.epochs)
    with torch.no_grad():
        full_pred = full_model(torch.from_numpy(Xval)).numpy()
    pooled_rho = pooled_spearman(full_pred, yval)
    grouped_rho, n_states_ranked = group_aware_spearman(full_pred, yval, sid_val)
    results["candidate_conditioned"] = {
        "val_bce": full_val_bce,
        "best_epoch": full_epoch,
        "constant_mean_baseline_val_bce": baseline_bce,
        "beats_baseline": full_val_bce < baseline_bce,
        "pooled_spearman": pooled_rho,
        "group_aware_spearman": grouped_rho,
        "n_states_ranked_for_group_spearman": n_states_ranked,
    }
    print(
        f"[diagnostic] candidate-conditioned (early-stopped @ epoch {full_epoch}): val BCE={full_val_bce:.4f} vs constant-mean baseline={baseline_bce:.4f} "
        f"(beats baseline: {full_val_bce < baseline_bce}); "
        f"pooled_spearman={pooled_rho:.4f} group_aware_spearman={grouped_rho:.4f} (n_states={n_states_ranked})"
    )

    # --- Check 3: state-only (board features only, candidate block zeroed) --
    Xtr_board = Xtr.copy()
    Xtr_board[:, BOARD_DIM:] = 0.0
    Xval_board = Xval.copy()
    Xval_board[:, BOARD_DIM:] = 0.0
    board_model, board_val_bce, board_epoch = train_model_early_stopping(Xtr_board, ytr, Xval_board, yval, TOTAL_DIM, max_epochs=args.epochs)
    results["state_only"] = {"val_bce": board_val_bce, "best_epoch": board_epoch}
    candidate_conditioned_wins = full_val_bce < board_val_bce
    results["candidate_conditioned_beats_state_only"] = candidate_conditioned_wins
    print(
        f"[diagnostic] state-only (early-stopped @ epoch {board_epoch}, candidate features zeroed): val BCE={board_val_bce:.4f} "
        f"— candidate-conditioned beats state-only: {candidate_conditioned_wins}"
    )

    # --- Check 2: shuffled-label control (early-stopped on the REAL val set,
    # giving the "does shuffled training still exploit some leak" hypothesis
    # its best possible chance rather than an arbitrary fixed epoch count) ---
    rng = np.random.default_rng(DIAGNOSTIC_SEED)
    y_shuffled = ytr.copy()
    rng.shuffle(y_shuffled)
    shuffled_model, shuffled_val_bce, shuffled_epoch = train_model_early_stopping(Xtr, y_shuffled, Xval, yval, TOTAL_DIM, max_epochs=args.epochs)
    # "Collapse to baseline" means shuffled training doesn't generalize any
    # better than just predicting the training mean — allow a small tolerance
    # since finite-sample noise means these will never be bit-identical.
    shuffled_collapses = shuffled_val_bce >= baseline_bce - 0.02
    results["shuffled_label_control"] = {
        "val_bce": shuffled_val_bce,
        "best_epoch": shuffled_epoch,
        "constant_mean_baseline_val_bce": baseline_bce,
        "collapses_to_baseline_or_worse": shuffled_collapses,
    }
    print(
        f"[diagnostic] shuffled-label control (early-stopped @ epoch {shuffled_epoch}): val BCE={shuffled_val_bce:.4f} vs baseline={baseline_bce:.4f} "
        f"(collapses to baseline or worse: {shuffled_collapses})"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[diagnostic] wrote {args.out / 'diagnostic_results.json'}")


if __name__ == "__main__":
    main()
