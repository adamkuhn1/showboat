"""Export the trained policy/value net to ONNX for onnxruntime-web.

Produces ``public/model/showboat.onnx`` with the exact I/O contract the browser
reads (``src/ai/onnx.ts``):

    input  "obs"    : float32[1, OBS_DIM]
    output "value"  : float32[1, 1]   (win expectation, tanh)
    output "policy" : float32[1, ACTION_DIM]  (action mean, tanh)

Run after training (or with a smoke-trained / random-init net to prove the
plumbing end to end -- the browser only needs a valid ONNX graph to switch from
"search baseline" to "trained net"):

    python export_onnx.py --checkpoint runs/showboat_8ball_sez/ckpt/best.pt \
                          --out ../public/model/showboat.onnx
"""

from __future__ import annotations

import argparse
import os

from showboat_env.action_space import ACTION_DIM
from showboat_env.network import build_network
from showboat_env.observation import OBS_DIM


def export(checkpoint: str | None, out_path: str) -> None:
    import torch

    net = build_network()
    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        # LightZero checkpoints nest the prediction net; accept either a raw
        # state_dict or a wrapped dict, and load what matches by shape.
        sd = state.get("model", state) if isinstance(state, dict) else state
        missing = net.load_state_dict(sd, strict=False)
        print(f"loaded checkpoint (unmatched keys: {missing})")
    else:
        print("no checkpoint found -> exporting a randomly-initialized net "
              "(plumbing smoke test; browser will read it as a real ONNX model)")

    net.eval()
    dummy = torch.zeros(1, OBS_DIM, dtype=torch.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        net,
        dummy,
        out_path,
        input_names=["obs"],
        output_names=["value", "policy"],
        dynamic_axes={"obs": {0: "batch"}, "value": {0: "batch"}, "policy": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported ONNX -> {out_path}  (obs_dim={OBS_DIM}, action_dim={ACTION_DIM})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="../public/model/showboat.onnx")
    args = ap.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
