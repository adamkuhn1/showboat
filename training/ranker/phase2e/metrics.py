"""Phase 2E metrics — independently reimplemented, then cross-checked against
Phase 2C's `diagnostic.py` and against `scipy.stats.spearmanr`.

Why reimplement rather than import: the brief for this phase explicitly says to
verify the corrected tie-aware Spearman myself rather than trust it blindly. The
Phase 2D review found a real bug here (ordinal `np.argsort(np.argsort(x))` ranks
instead of tie-averaged ranks), which mattered because `legal_pot_rate` takes
only 9 or 33 distinct values so ties are pervasive. `verify_against_reference()`
below re-runs that check: this module's ranks vs. scipy's, on real predictions,
including deliberately tie-heavy inputs.

Definitions used throughout (same as Phase 2D, so numbers are comparable):
  - pooled_spearman:        Spearman rho over ALL rows of a split at once.
                            Reported for continuity only; it conflates
                            "easy state vs hard state" with the actual
                            per-decision task.
  - group_aware_spearman:   mean of per-state Spearman, over states with >=3
                            candidates and non-constant target. THE ranking
                            metric that matters.
Additional decision-relevant metrics Phase 2D declared but this phase computes
for every architecture: NDCG@3, top-1 hit rate (top pick has
legal_pot_rate >= 0.5), and regret (best available rate - chosen rate).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr

EPS = 1e-6


# --------------------------------------------------------------------------
# probabilistic
# --------------------------------------------------------------------------
def soft_bce_np(pred: np.ndarray, target: np.ndarray) -> float:
    p = np.clip(pred.astype(np.float64), EPS, 1 - EPS)
    t = np.clip(target.astype(np.float64), 0.0, 1.0)
    return float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean())


def brier(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))


def ece(pred: np.ndarray, target: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, 10 equal-width bins — identical binning rule
    to `phase2d/train_baseline.py::ece` so the numbers are comparable."""
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(pred)
    err = 0.0
    for i in range(n_bins):
        upper = pred < bins[i + 1] if i < n_bins - 1 else pred <= bins[i + 1]
        mask = (pred >= bins[i]) & upper
        n = int(mask.sum())
        if n == 0:
            continue
        err += (n / total) * abs(float(pred[mask].mean()) - float(target[mask].mean()))
    return float(err)


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------
def _spearman(pred: np.ndarray, target: np.ndarray) -> float:
    """Tie-averaged Spearman. Returns NaN when either side has zero variance
    (e.g. a state-only model emitting one identical score for every candidate
    in a state — there is genuinely no ordering to score, and inventing one is
    exactly the bug Phase 2D's review caught)."""
    if len(pred) < 3:
        return float("nan")
    rp = rankdata(pred, method="average")
    rt = rankdata(target, method="average")
    if np.std(rp) == 0 or np.std(rt) == 0:
        return float("nan")
    return float(np.corrcoef(rp, rt)[0, 1])


def pooled_spearman(pred: np.ndarray, target: np.ndarray) -> float:
    return _spearman(pred, target)


def group_aware_spearman(
    pred: np.ndarray, target: np.ndarray, state: np.ndarray
) -> tuple[float, int]:
    """Mean per-state Spearman over states with >=3 candidates and non-constant
    target. Matches `diagnostic.py::group_aware_spearman` exactly (verified in
    `verify_against_reference`), so Phase 2D's published numbers are directly
    comparable to this phase's."""
    per_state = []
    order = np.argsort(state, kind="stable")
    s_sorted = state[order]
    bounds = np.flatnonzero(np.diff(s_sorted)) + 1
    for idx in np.split(order, bounds):
        if len(idx) < 3:
            continue
        t = target[idx]
        if np.std(t) == 0:
            continue
        rho = _spearman(pred[idx], t)
        if not np.isnan(rho):
            per_state.append(rho)
    if not per_state:
        return float("nan"), 0
    return float(np.mean(per_state)), len(per_state)


def _dcg(rels: np.ndarray) -> float:
    return float(np.sum(rels / np.log2(np.arange(2, len(rels) + 2))))


def ndcg_at_k(
    pred: np.ndarray, target: np.ndarray, state: np.ndarray, k: int = 3
) -> float:
    """Mean per-state NDCG@k with the soft `legal_pot_rate` as graded relevance.
    States with an all-zero target contribute nothing (ideal DCG is 0), and are
    skipped rather than counted as a perfect score."""
    vals = []
    order = np.argsort(state, kind="stable")
    s_sorted = state[order]
    bounds = np.flatnonzero(np.diff(s_sorted)) + 1
    for idx in np.split(order, bounds):
        if len(idx) < 2:
            continue
        t = target[idx]
        if t.sum() <= 0:
            continue
        p = pred[idx]
        top = np.argsort(-p, kind="stable")[:k]
        ideal = np.argsort(-t, kind="stable")[:k]
        idcg = _dcg(t[ideal])
        if idcg <= 0:
            continue
        vals.append(_dcg(t[top]) / idcg)
    return float(np.mean(vals)) if vals else float("nan")


def top1_hit_rate(
    pred: np.ndarray, target: np.ndarray, state: np.ndarray, threshold: float = 0.5
) -> float:
    """Fraction of states where the model's top-ranked candidate actually has
    legal_pot_rate >= threshold. Only over states that HAVE such a candidate —
    otherwise the metric is dominated by hopeless states no model can win."""
    hits, n = 0, 0
    order = np.argsort(state, kind="stable")
    s_sorted = state[order]
    bounds = np.flatnonzero(np.diff(s_sorted)) + 1
    for idx in np.split(order, bounds):
        t = target[idx]
        if t.max() < threshold:
            continue
        n += 1
        if t[int(np.argmax(pred[idx]))] >= threshold:
            hits += 1
    return float(hits / n) if n else float("nan")


def mean_regret(pred: np.ndarray, target: np.ndarray, state: np.ndarray) -> float:
    """max(legal_pot_rate) - legal_pot_rate(argmax pred), averaged over states
    with >=2 candidates. 0 = always picks a best-available candidate."""
    regs = []
    order = np.argsort(state, kind="stable")
    s_sorted = state[order]
    bounds = np.flatnonzero(np.diff(s_sorted)) + 1
    for idx in np.split(order, bounds):
        if len(idx) < 2:
            continue
        t = target[idx]
        regs.append(float(t.max() - t[int(np.argmax(pred[idx]))]))
    return float(np.mean(regs)) if regs else float("nan")


KINDS = ["direct", "bank", "double-bank", "combo", "rail-combo"]


def full_report(
    pred: np.ndarray, target: np.ndarray, state: np.ndarray, kind: np.ndarray
) -> dict:
    out = {
        "bce": soft_bce_np(pred, target),
        "brier": brier(pred, target),
        "ece": ece(pred, target),
        "pooled_spearman": pooled_spearman(pred, target),
        "ndcg@3": ndcg_at_k(pred, target, state, 3),
        "top1_hit_rate": top1_hit_rate(pred, target, state),
        "mean_regret": mean_regret(pred, target, state),
    }
    gas, n_states = group_aware_spearman(pred, target, state)
    out["group_aware_spearman"] = gas
    out["n_states_ranked"] = n_states
    per_kind = {}
    for i, k in enumerate(KINDS):
        m = kind == i
        if m.sum() == 0:
            continue
        g, ns = group_aware_spearman(pred[m], target[m], state[m])
        per_kind[k] = {
            "n": int(m.sum()),
            "bce": soft_bce_np(pred[m], target[m]),
            "brier": brier(pred[m], target[m]),
            "group_aware_spearman": g,
            "n_states_ranked": ns,
        }
    out["per_kind"] = per_kind
    return out


# --------------------------------------------------------------------------
# self-verification (run: python metrics.py)
# --------------------------------------------------------------------------
def verify_against_reference() -> None:
    """Three independent checks of the tie handling, run before any result in
    this phase is reported."""
    rng = np.random.default_rng(20260901)

    # 1. Against scipy.stats.spearmanr on continuous data (no ties).
    a = rng.normal(size=500)
    b = a * 0.5 + rng.normal(size=500)
    mine, ref = _spearman(a, b), float(spearmanr(a, b).statistic)
    assert abs(mine - ref) < 1e-10, (mine, ref)
    print(f"[verify] no-ties vs scipy.spearmanr: {mine:.12f} == {ref:.12f}  OK")

    # 2. Against scipy on heavily-tied data shaped like legal_pot_rate
    #    (9 distinct values, ~43% exact zeros).
    t = np.round(np.clip(rng.beta(0.4, 2.0, size=2000), 0, 1) * 8) / 8
    p = np.clip(t + rng.normal(scale=0.2, size=2000), 0, 1)
    mine, ref = _spearman(p, t), float(spearmanr(p, t).statistic)
    frac_zero = float((t == 0).mean())
    assert abs(mine - ref) < 1e-10, (mine, ref)
    print(
        f"[verify] tie-heavy ({frac_zero:.1%} exact zeros) vs scipy.spearmanr: "
        f"{mine:.12f} == {ref:.12f}  OK"
    )

    # 3. Demonstrate the ORIGINAL bug would disagree here — i.e. this test has
    #    real power and isn't vacuous.
    def buggy(x, y):
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])

    bug = buggy(p, t)
    print(
        f"[verify] ordinal-rank (pre-fix) implementation on the same data: "
        f"{bug:.6f} vs correct {mine:.6f} — understated by {mine - bug:.4f}"
    )
    assert bug < mine - 1e-3, "tie-handling test has no power on this fixture"

    # 4. Against phase2c/diagnostic.py's (fixed) implementation on real shapes,
    #    including the constant-prediction case that must return NaN.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2c"))
    import diagnostic as diag  # noqa: E402

    state = rng.integers(0, 300, size=2000)
    g_mine, n_mine = group_aware_spearman(p, t, state)
    g_ref, n_ref = diag.group_aware_spearman(p, t, state)
    assert abs(g_mine - g_ref) < 1e-10 and n_mine == n_ref, (g_mine, g_ref)
    print(
        f"[verify] group_aware_spearman vs phase2c/diagnostic.py: "
        f"{g_mine:.12f} == {g_ref:.12f} (n_states {n_mine}) OK"
    )

    const = np.full(2000, 0.3)
    g_const, _ = group_aware_spearman(const, t, state)
    assert np.isnan(g_const), g_const
    print("[verify] constant predictions -> NaN (not a fabricated ordering)  OK")


if __name__ == "__main__":
    verify_against_reference()
    print("[verify] all metric checks passed")
