"""Phase 2E dataset loading + caching.

Reads the SAME frozen Phase 2C baseline dataset Phase 2D used
(`training/ranker/phase2c/data/baseline/`), strictly read-only, and caches it
to a single `.npz` inside this phase's own results directory so every
experiment re-reads a 40MB array instead of re-parsing 1,800 NDJSON shards.

Nothing here re-splits, re-encodes, or regenerates anything: the per-row
`split` field assigned by `phase2c/split.ts` is carried through untouched.

Extra columns kept beyond Phase 2D's (X, y, kind, state_id):
  - `pot_id`   : the ball the candidate INTENDS to pot. For combos this differs
                 from `target` (the first ball contacted). It is present in the
                 dataset rows but NOT in the 68-dim encoded feature vector, so
                 a model can only use it if the input contract is extended.
                 Phase 2E measures whether that is worth doing.
  - `target`   : first-contact ball id (already recoverable from feature 48+11,
                 kept explicitly so the derivation can be cross-checked).
  - `family_id`: for reporting only; never used to re-split.

Usage:
    python data.py --dataset-dir <read-only baseline dir> --cache results/dataset.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KINDS = ["direct", "bank", "double-bank", "combo", "rail-combo"]
SPLITS = ["train", "val", "test"]


def build_cache(dataset_dir: Path, cache_path: Path) -> None:
    shards = sorted(dataset_dir.glob("*.ndjson"))
    if not shards:
        raise SystemExit(f"no .ndjson shards under {dataset_dir}")

    X, y, kind, split, state, pot, tgt, fam, npert = [], [], [], [], [], [], [], [], []
    for shard in shards:
        with open(shard) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                X.append(r["features"])
                y.append(r["raw_counts"]["legal_pot"] / r["n_perturbations"])
                kind.append(KINDS.index(r["candidate_kind"]))
                split.append(SPLITS.index(r["split"]))
                state.append(r["state_id"])
                pot.append(r["candidate_pot_id"])
                tgt.append(r["candidate_target"])
                fam.append(r["family_id"])
                npert.append(r["n_perturbations"])

    state_u, state_idx = np.unique(np.array(state), return_inverse=True)
    fam_u, fam_idx = np.unique(np.array(fam), return_inverse=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        kind=np.asarray(kind, dtype=np.int16),
        split=np.asarray(split, dtype=np.int8),
        state=state_idx.astype(np.int32),
        pot=np.asarray(pot, dtype=np.int16),
        target=np.asarray(tgt, dtype=np.int16),
        family=fam_idx.astype(np.int32),
        n_perturbations=np.asarray(npert, dtype=np.int16),
    )
    print(
        f"[data] cached {len(y)} rows, {len(state_u)} states, {len(fam_u)} families -> {cache_path}"
    )


class Split:
    """One split's arrays, as plain numpy."""

    def __init__(self, X, y, kind, state, pot, target, n_perturbations):
        self.X = X
        self.y = y
        self.kind = kind
        self.state = state
        self.pot = pot
        self.target = target
        self.n_perturbations = n_perturbations

    def __len__(self) -> int:
        return len(self.y)


def load(cache_path: Path) -> dict[str, Split]:
    d = np.load(cache_path)
    out = {}
    for i, name in enumerate(SPLITS):
        m = d["split"] == i
        out[name] = Split(
            d["X"][m],
            d["y"][m],
            d["kind"][m],
            d["state"][m],
            d["pot"][m],
            d["target"][m],
            d["n_perturbations"][m],
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    args = ap.parse_args()
    build_cache(args.dataset_dir, args.cache)
