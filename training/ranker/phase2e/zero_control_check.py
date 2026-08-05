"""Forensics on the full-scale all-zero-input control.

The control itself is in `train.py` (`zero_input_mlp`, `zero_input_deepsets`):
train the frozen architecture on inputs where all 68 dims are zeroed, and check
it cannot beat the constant-mean baseline. Both collapse exactly as required.

This script exists because of a subtlety worth recording rather than papering
over. With an all-zero input every row is literally the same input, so a
deterministic network must emit ONE constant value and the group-aware Spearman
must be undefined (NaN) — that is precisely the property Phase 2D's tie-handling
fix was introduced to respect. The MLP arm reports NaN for every seed, as
expected. The Deep Sets arm reported a small *non*-NaN Spearman on some seeds
(-0.094 +- 0.137 across seeds).

This measures why. The predictions differ at the 1e-8 level between rows
evaluated in different forward-pass chunks (different GEMM tiling for a
16384-row chunk vs an 850-row remainder), which is enough to give one or two
states a defined-but-meaningless ordering. Reported here with the actual peak-
to-peak spread so the number in the results JSON is interpretable and nobody
later reads "-0.094" as a real (or a suspicious) correlation.

Usage:  python zero_control_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data import load
from metrics import group_aware_spearman
from models import DeepSetsRanker, TinyNet

BUILDERS = {
    "zero_input_mlp": lambda: TinyNet(68, 32),
    "zero_input_deepsets": lambda: DeepSetsRanker(64, 64),
    # state-only has the same structural property for a different reason: the
    # candidate block is zeroed, so every candidate WITHIN a state shares an
    # input and the within-state ranking is undefined (Phase 2D found exactly
    # this and reported NaN). Included here so the same float-noise caveat is
    # measured rather than assumed.
    "deepsets_state_only": lambda: DeepSetsRanker(64, 64),
}


def main():
    va = load(Path("results/dataset.npz"))["val"]
    out = {}
    for name, build in BUILDERS.items():
        if name == "deepsets_state_only":
            Xn = va.X.copy()
            Xn[:, 48:] = 0.0
            X = torch.from_numpy(Xn)
        else:
            X = torch.zeros(len(va.y), 68)
        res = json.load(open(f"results/variant_{name}.json"))
        rows = []
        for s in res["per_seed"]:
            m = build()
            m.load_state_dict(
                torch.load(f"results/checkpoints/{name}_seed{s['seed']}.pt", map_location="cpu")
            )
            m.eval()
            with torch.no_grad():
                chunked = np.concatenate(
                    [m(X[i : i + 16384]).numpy() for i in range(0, len(X), 16384)]
                )
                whole = m(X).numpy()  # single forward pass, one GEMM tiling
            gas_chunked, n_c = group_aware_spearman(chunked, va.y, va.state)
            gas_whole, n_w = group_aware_spearman(whole, va.y, va.state)
            rows.append({
                "seed": s["seed"],
                "n_distinct_predictions_chunked": int(len(np.unique(chunked))),
                "peak_to_peak_chunked": float(np.ptp(chunked)),
                "n_distinct_predictions_single_pass": int(len(np.unique(whole))),
                "peak_to_peak_single_pass": float(np.ptp(whole)),
                "group_aware_spearman_chunked": gas_chunked,
                "n_states_ranked_chunked": n_c,
                "group_aware_spearman_single_pass": gas_whole,
                "n_states_ranked_single_pass": n_w,
            })
            print(f"{name} seed {s['seed']}: distinct={rows[-1]['n_distinct_predictions_chunked']} "
                  f"ptp={rows[-1]['peak_to_peak_chunked']:.3e} "
                  f"GAS(chunked)={gas_chunked} over {n_c} states | "
                  f"GAS(single pass)={gas_whole} over {n_w} states")
        out[name] = rows
    Path("results/zero_control_forensics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
