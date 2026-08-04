"""Train the Phase 2A candidate-ranker baseline (flat MLP, Option 1 from
docs/repair/showboat-ml/ARCHITECTURE_DECISION.md).

Usage:
    python train.py --dataset dataset/showboat-ranker-v1.ndjson --out artifacts/run1

Splits by state_id (not by row) to prevent leakage between near-identical
candidates drawn from the same board state — see ARCHITECTURE_DECISION.md's
"Dataset split method". Trains a 2-hidden-layer MLP with BCE loss against the
empirical pot-success-under-perturbation label the dataset generator recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from schema import SCHEMA_VERSION, TOTAL_DIM


class RankerNet(nn.Module):
    """Same architectural shape as the existing (parked) training/showboat_env/network.py
    ShowboatNet — Gemm/Relu only, safe on every ONNX Runtime Web execution provider."""

    def __init__(self, input_dim: int = TOTAL_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # pot_success_logit


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_by_state(rows: list[dict], val_frac: float, test_frac: float, seed: int):
    state_ids = sorted({r["state_id"] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(state_ids)
    n = len(state_ids)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_states = set(state_ids[:n_test])
    val_states = set(state_ids[n_test : n_test + n_val])
    train_states = set(state_ids[n_test + n_val :])
    train = [r for r in rows if r["state_id"] in train_states]
    val = [r for r in rows if r["state_id"] in val_states]
    test = [r for r in rows if r["state_id"] in test_states]
    return train, val, test


def to_tensors(rows: list[dict]):
    x = torch.tensor([r["features"] for r in rows], dtype=torch.float32)
    y = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
    return x, y


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("dataset/showboat-ranker-v1.ndjson"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/run1"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = load_rows(args.dataset)
    for r in rows:
        if r["schema_version"] != SCHEMA_VERSION:
            print(
                f"FATAL: dataset row has schema_version {r['schema_version']!r}, "
                f"training code expects {SCHEMA_VERSION!r}. Regenerate the dataset.",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(r["features"]) != TOTAL_DIM:
            print(
                f"FATAL: dataset row has {len(r['features'])} features, schema declares {TOTAL_DIM}.",
                file=sys.stderr,
            )
            sys.exit(1)

    train_rows, val_rows, test_rows = split_by_state(rows, args.val_frac, args.test_frac, args.seed)
    print(f"rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    if not train_rows or not val_rows or not test_rows:
        print("FATAL: a split is empty — dataset too small for this split ratio.", file=sys.stderr)
        sys.exit(1)

    x_train, y_train = to_tensors(train_rows)
    x_val, y_val = to_tensors(val_rows)
    x_test, y_test = to_tensors(test_rows)

    model = RankerNet(hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = loss_fn(val_logits, y_val).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_logits = model(x_test)
        test_probs = torch.sigmoid(test_logits).numpy()
        test_labels = y_test.numpy()
        test_loss = loss_fn(test_logits, y_test).item()

    metrics = {
        "best_val_loss": best_val_loss,
        "test_bce_loss": test_loss,
        "test_brier_score": brier_score(test_probs, test_labels),
        "test_spearman_vs_label": spearman(test_probs, test_labels),
        "test_n": len(test_rows),
        "mean_test_label": float(test_labels.mean()),
        "mean_test_pred": float(test_probs.mean()),
    }
    print("Test metrics:", json.dumps(metrics, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "hidden": args.hidden,
            "input_dim": TOTAL_DIM,
            "schema_version": SCHEMA_VERSION,
        },
        ckpt_path,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_architecture": f"MLP({TOTAL_DIM}->{args.hidden}->{args.hidden}->1), ReLU, BCEWithLogitsLoss",
        "dataset_path": str(args.dataset),
        "dataset_sha256": file_sha256(args.dataset),
        "dataset_n_rows": len(rows),
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "test_n": len(test_rows),
        "split_method": "by_state_id",
        "training_config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden": args.hidden,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
        },
        "seed": args.seed,
        "git_commit": git_commit(),
        "eval_metrics": metrics,
        "calibration_applied": False,
        "random_init": False,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(args.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote checkpoint to {ckpt_path}")
    print(f"Wrote manifest to {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
