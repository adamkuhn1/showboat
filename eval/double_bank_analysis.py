"""Why is `double-bank` the weakest candidate kind for the Phase 2D ranker?

Diagnostic only. TRAIN and VAL splits exclusively -- the test split is loaded
for its row count and never read, so nothing here can be an optimisation
against test.

Hypotheses checked, in the order the Phase 2D review listed them:
  H1 low representation          -- row/state counts per kind
  H2 label uncertainty           -- n_perturbations and binomial SE per kind
  H3 candidate-feature weakness  -- within-state feature spread, and the best
                                    single-feature within-state Spearman as an
                                    upper bound on what ANY model reading these
                                    features could rank with
  H4 candidate-generator defects -- duplicate/degenerate double-bank candidates
  H5 mixed semantics             -- does `banks`/path-length agree with the kind
  H6 physical sensitivity        -- share of rows at exactly 0, within-state
                                    label spread, and how many states are even
                                    eligible for the group-aware metric
  H7 model capacity              -- a train-only reweighting run with the frozen
                                    architecture, reported for every kind so a
                                    "fix" that trades one kind for another is
                                    visible rather than hidden

Usage:
    python eval/double_bank_analysis.py \
        --dataset-dir <path to phase2c/data/baseline> \
        --checkpoints <path to phase2d/results/checkpoints> \
        --out eval/results
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "training/ranker/phase2c"))
sys.path.insert(0, str(APP_ROOT / "training/ranker"))
sys.path.insert(0, str(APP_ROOT / "training/ranker/phase2d"))

from diagnostic import TinyNet, group_aware_spearman, pooled_spearman  # noqa: E402
from schema import BOARD_DIM, TOTAL_DIM, load_schema  # noqa: E402
from train_baseline import load_rows, SEEDS, MAX_EPOCHS  # noqa: E402

KINDS = ["direct", "bank", "double-bank", "combo", "rail-combo"]
FEATURE_NAMES = load_schema()["candidate_features"]


def to_arrays(rows):
    X = np.array([r["features"] for r in rows], dtype=np.float32)
    y = np.array([r["raw_counts"]["legal_pot"] / r["n_perturbations"] for r in rows], dtype=np.float32)
    kind = np.array([r["candidate_kind"] for r in rows])
    sid = np.array([r["state_id"] for r in rows])
    npert = np.array([r["n_perturbations"] for r in rows], dtype=np.float32)
    return X, y, kind, sid, npert


def eligible_states(y, sid, kind_mask):
    """States the group-aware metric can even score for this kind: >=3 rows of
    the kind and a non-constant target among them."""
    n_eligible, n_states = 0, 0
    spreads = []
    for s in np.unique(sid[kind_mask]):
        m = kind_mask & (sid == s)
        if m.sum() < 1:
            continue
        n_states += 1
        if m.sum() >= 3 and np.std(y[m]) > 0:
            n_eligible += 1
            spreads.append(float(y[m].max() - y[m].min()))
    return n_states, n_eligible, spreads


def best_single_feature_spearman(X, y, sid, kind_mask):
    """Upper bound on within-state rankability from the candidate block alone:
    for each candidate feature, the group-aware Spearman of that raw feature
    (and its negation) against the label, restricted to this kind. If the best
    single feature is near zero, the information simply is not in the encoding
    -- no amount of model capacity recovers it."""
    best = {"feature": None, "spearman": float("-inf")}
    per_feature = {}
    for j, name in enumerate(FEATURE_NAMES):
        col = X[kind_mask, BOARD_DIM + j]
        rho, n = group_aware_spearman(col, y[kind_mask], sid[kind_mask])
        rho = 0.0 if np.isnan(rho) else float(rho)
        per_feature[name] = {"spearman": rho, "abs": abs(rho), "n_states": int(n)}
        if abs(rho) > best["spearman"]:
            best = {"feature": name, "spearman": abs(rho), "signed": rho}
    return best, per_feature


def within_state_feature_degeneracy(X, sid, kind_mask):
    """Fraction of same-state candidate pairs of this kind whose 20-dim
    candidate block is identical (or near-identical). Identical inputs cannot
    be ranked apart by any deterministic model."""
    identical = 0
    total = 0
    near = 0
    for s in np.unique(sid[kind_mask]):
        m = kind_mask & (sid == s)
        idx = np.where(m)[0]
        if len(idx) < 2:
            continue
        C = X[idx, BOARD_DIM:]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                total += 1
                d = np.abs(C[a] - C[b]).max()
                if d == 0:
                    identical += 1
                if d < 1e-3:
                    near += 1
    return {
        "pairs": total,
        "identical_pairs": identical,
        "identical_frac": identical / total if total else float("nan"),
        "near_identical_pairs": near,
        "near_identical_frac": near / total if total else float("nan"),
    }


def per_kind_spearman(pred, y, sid, kind):
    out = {}
    for k in KINDS:
        m = kind == k
        rho, n = group_aware_spearman(pred[m], y[m], sid[m])
        out[k] = {"group_aware_spearman": None if np.isnan(rho) else float(rho), "n_states": int(n), "n_rows": int(m.sum())}
    return out


def train_weighted(Xtr, ytr, wtr, Xval, yval, seed, max_epochs=MAX_EPOCHS, patience=30, lr=1e-2):
    """Same frozen architecture/optimizer as train_baseline.py, with per-row
    loss weights. Early stopping on UNWEIGHTED validation BCE, so the weighting
    cannot flatter its own model-selection metric."""
    torch.manual_seed(seed)
    model = TinyNet(TOTAL_DIM, hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    wt = torch.from_numpy(wtr)
    Xv, yv = torch.from_numpy(Xval), torch.from_numpy(yval)
    best_val, best_state, best_epoch, since = float("inf"), None, -1, 0
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xt), yt, weight=wt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xv), yv).item()
        if vloss < best_val:
            best_val, best_epoch, since = vloss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_val, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--checkpoints", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--weights", type=float, nargs="*", default=[2.0, 4.0, 8.0])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.dataset_dir)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    n_test = sum(1 for r in rows if r["split"] == "test")
    print(f"[db] train={len(train_rows)} val={len(val_rows)} test={n_test} (test never read)")

    Xtr, ytr, ktr, str_, ptr = to_arrays(train_rows)
    Xval, yval, kval, sval, pval = to_arrays(val_rows)

    report = {"test_split_touched": False, "splits": {"train": len(train_rows), "val": len(val_rows)}}

    # ---- H1: representation ------------------------------------------------
    rep = {}
    for k in KINDS:
        mtr, mval = ktr == k, kval == k
        rep[k] = {
            "train_rows": int(mtr.sum()),
            "train_row_share": float(mtr.mean()),
            "train_states_containing": int(len(np.unique(str_[mtr]))),
            "val_rows": int(mval.sum()),
            "mean_per_state_when_present": float(
                np.mean([np.sum((ktr == k) & (str_ == s)) for s in np.unique(str_[mtr])])
            ) if mtr.sum() else float("nan"),
        }
    report["h1_representation"] = rep

    # ---- H2: label uncertainty --------------------------------------------
    unc = {}
    for k in KINDS:
        m = ktr == k
        p = ytr[m]
        n = ptr[m]
        se = np.sqrt(np.clip(p * (1 - p), 0, None) / n)
        unc[k] = {
            "mean_label": float(p.mean()),
            "frac_exactly_zero": float((p == 0).mean()),
            "frac_exactly_one": float((p == 1).mean()),
            "mean_n_perturbations": float(n.mean()),
            "mean_binomial_se": float(se.mean()),
            "se_over_label_spread": float(se.mean() / (p.std() + 1e-9)),
        }
    report["h2_label_uncertainty"] = unc

    # ---- H3/H6: within-state rankability -----------------------------------
    rank = {}
    for k in KINDS:
        m = kval == k
        n_states, n_eligible, spreads = eligible_states(yval, sval, m)
        best, per_feature = best_single_feature_spearman(Xval, yval, sval, m)
        rank[k] = {
            "val_states_containing_kind": n_states,
            "val_states_scorable_by_group_metric": n_eligible,
            "scorable_share": n_eligible / n_states if n_states else float("nan"),
            "mean_within_state_label_spread": float(np.mean(spreads)) if spreads else float("nan"),
            "best_single_candidate_feature": best["feature"],
            "best_single_feature_abs_spearman": float(best["spearman"]),
            "top_features": sorted(
                ({"feature": f, **v} for f, v in per_feature.items()),
                key=lambda d: -d["abs"],
            )[:5],
        }
    report["h3_h6_within_state_rankability"] = rank

    # ---- H4/H5: generator defects / mixed semantics -------------------------
    defects = {}
    for k in KINDS:
        m = kval == k
        deg = within_state_feature_degeneracy(Xval, sval, m)
        banks_col = Xval[m, BOARD_DIM + FEATURE_NAMES.index("banks_norm")]
        path_col = Xval[m, BOARD_DIM + FEATURE_NAMES.index("path_length_norm")]
        clear_col = Xval[m, BOARD_DIM + FEATURE_NAMES.index("obstruction_margin_norm")]
        defects[k] = {
            **deg,
            "banks_norm_unique_values": sorted(set(np.round(banks_col, 6).tolist())),
            "path_length_norm_mean": float(path_col.mean()),
            "path_length_norm_frac_clipped_at_1": float((path_col >= 1.0).mean()),
            "clearance_norm_mean": float(clear_col.mean()),
            "clearance_norm_frac_at_sentinel_1": float((clear_col >= 1.0).mean()),
        }
    report["h4_h5_generator_and_semantics"] = defects

    # ---- shipped checkpoint's own per-kind behaviour ------------------------
    ckpt = args.checkpoints / f"main_seed{SEEDS[0]}.pt"
    model = TinyNet(TOTAL_DIM, hidden=32)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits_val = model(torch.from_numpy(Xval)).numpy()
    pred_val = 1 / (1 + np.exp(-logits_val))
    report["shipped_checkpoint"] = {
        "checkpoint": str(ckpt.name),
        "per_kind": per_kind_spearman(pred_val, yval, sval, kval),
        "pooled_val_group_aware_spearman": float(group_aware_spearman(pred_val, yval, sval)[0]),
    }
    # Within-state prediction spread: if the model outputs near-identical scores
    # for every double-bank candidate in a state, it cannot rank them whatever
    # the labels say.
    spread = {}
    for k in KINDS:
        m = kval == k
        vals = []
        for s in np.unique(sval[m]):
            mm = m & (sval == s)
            if mm.sum() >= 3:
                vals.append(float(pred_val[mm].max() - pred_val[mm].min()))
        spread[k] = {
            "mean_within_state_prediction_spread": float(np.mean(vals)) if vals else float("nan"),
            "n_states": len(vals),
        }
    report["shipped_checkpoint"]["within_state_prediction_spread"] = spread

    # ---- H7: train-only reweighting experiment ------------------------------
    print("[db] reweighting experiment (train/val only)...")
    reweight = {}
    baseline_seeds = SEEDS[:3]
    for w in [1.0] + list(args.weights):
        wtr = np.where(ktr == "double-bank", w, 1.0).astype(np.float32)
        per_seed = []
        for seed in baseline_seeds:
            m, vbce, ep = train_weighted(Xtr, ytr, wtr, Xval, yval, seed)
            with torch.no_grad():
                lg = m(torch.from_numpy(Xval)).numpy()
            pv = 1 / (1 + np.exp(-lg))
            per_seed.append(
                {
                    "seed": seed,
                    "val_bce": float(vbce),
                    "best_epoch": ep,
                    "pooled_group_aware_spearman": float(group_aware_spearman(pv, yval, sval)[0]),
                    "per_kind": per_kind_spearman(pv, yval, sval, kval),
                }
            )
        agg = {
            "weight": w,
            "val_bce_mean": float(np.mean([s["val_bce"] for s in per_seed])),
            "pooled_group_aware_spearman_mean": float(
                np.mean([s["pooled_group_aware_spearman"] for s in per_seed])
            ),
            "per_kind_mean": {
                k: float(
                    np.mean(
                        [
                            s["per_kind"][k]["group_aware_spearman"]
                            for s in per_seed
                            if s["per_kind"][k]["group_aware_spearman"] is not None
                        ]
                    )
                )
                for k in KINDS
            },
            "per_seed": per_seed,
        }
        reweight[f"w={w}"] = agg
        print(
            f"[db]  double-bank weight {w}: val_bce {agg['val_bce_mean']:.4f}, "
            f"pooled {agg['pooled_group_aware_spearman_mean']:.4f}, "
            + ", ".join(f"{k}={agg['per_kind_mean'][k]:.4f}" for k in KINDS)
        )
    report["h7_train_only_reweighting"] = reweight

    out = args.out / "double_bank_analysis.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"[db] wrote {out}")


if __name__ == "__main__":
    main()
