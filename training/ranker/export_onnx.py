"""Export a trained ranker checkpoint to ONNX, with a required manifest.

Unlike the old (parked) training/export_onnx.py, this script REFUSES to
export a randomly-initialized network by default — see
docs/repair/showboat-ml/ARCHITECTURE_DECISION.md's "Real model artifact
integrity" section and 01-current-ml-audit.md §3, which documents the old
script's silent-random-export failure mode as the thing being fixed here.

Usage:
    python export_onnx.py --run artifacts/run1 --out artifacts/run1/showboat-ranker.onnx
    python export_onnx.py --allow-random-init --out /tmp/smoke.onnx   # explicit opt-in only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import onnx
import torch

from schema import TOTAL_DIM
from train import RankerNet


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, help="Directory containing checkpoint.pt + manifest.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Export a randomly-initialized network with no checkpoint. "
        "Sets manifest random_init=true. Never use for a shipped artifact.",
    )
    args = ap.parse_args()

    if not args.run and not args.allow_random_init:
        print(
            "FATAL: no --run checkpoint directory given and --allow-random-init not "
            "passed. Refusing to export an untrained network silently — this is the "
            "exact failure mode documented in docs/repair/showboat-ml/01-current-ml-audit.md §3 "
            "for the old export script, fixed deliberately here.",
            file=sys.stderr,
        )
        sys.exit(1)

    manifest: dict = {}
    if args.run:
        ckpt = torch.load(args.run / "checkpoint.pt", map_location="cpu")
        model = RankerNet(input_dim=ckpt["input_dim"], hidden=ckpt["hidden"])
        model.load_state_dict(ckpt["model"])
        with open(args.run / "manifest.json") as f:
            manifest = json.load(f)
        print(f"Loaded real checkpoint from {args.run / 'checkpoint.pt'}")
    else:
        model = RankerNet(input_dim=TOTAL_DIM)
        manifest = {"random_init": True, "note": "SMOKE TEST ONLY — not a trained model."}
        print("WARNING: exporting a randomly-initialized network (--allow-random-init).", file=sys.stderr)

    model.eval()
    dummy = torch.zeros(1, TOTAL_DIM, dtype=torch.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False (the legacy TorchScript-based exporter): for a model this
    # tiny, the newer dynamo exporter defaults to writing weights into a
    # separate `.onnx.data` external-data file, which onnxruntime-web cannot
    # load from a plain in-memory buffer (only via browser fetch/MountedFiles
    # plumbing this app doesn't have). A single self-contained .onnx file is
    # both simpler to ship and the only form the browser loader supports.
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["candidate_features"],
        output_names=["pot_success_logit"],
        dynamic_axes={"candidate_features": {0: "batch"}, "pot_success_logit": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx_model = onnx.load(str(args.out))
    onnx.checker.check_model(onnx_model)
    print(f"Exported ONNX model to {args.out} (checked, valid graph).")

    manifest["onnx_file_sha256"] = file_sha256(args.out)
    manifest["onnx_export_timestamp"] = datetime.now(timezone.utc).isoformat()
    manifest["onnx_input_names"] = [i.name for i in onnx_model.graph.input]
    manifest["onnx_output_names"] = [o.name for o in onnx_model.graph.output]
    manifest["onnx_opset"] = 17
    manifest_path = args.out.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
