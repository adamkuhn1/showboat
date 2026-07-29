"""Policy + value network for the Showboat agent.

A small MLP over the low-dimensional observation, exactly the regime the research
recommends (kilobytes-to-low-MB, sub-millisecond in-browser inference). This is
the module that gets exported to ONNX and shipped to ``public/model/showboat.onnx``
for onnxruntime-web.

It intentionally exposes a plain ``(value, policy)`` forward signature so the ONNX
graph has a stable, documented contract that ``src/ai/onnx.ts`` reads (output 0 =
value, output 1 = policy). During LightZero training the same trunk feeds the
MuZero/EfficientZero prediction heads; this standalone module is the *exported*
inference net (the learned policy prior + value), which is all the browser needs
to drive the client-side TS-MCTS.

torch is imported lazily so the rest of the training package (env, action space,
reward) lints and imports without a torch install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .action_space import ACTION_DIM
from .observation import OBS_DIM

if TYPE_CHECKING:  # pragma: no cover
    import torch
    import torch.nn as nn


def build_network(hidden: int = 128):
    """Construct the policy/value MLP. Requires torch at call time."""
    import torch  # noqa: F401
    import torch.nn as nn

    class ShowboatNet(nn.Module):
        """Shared trunk -> (value head, policy head).

        - value head: scalar win expectation in [-1, 1] (tanh).
        - policy head: mean of a squashed Gaussian over the 4-D continuous
          action, in [-1, 1] (tanh). Continuous-control policy prior for the
          sampled MCTS; the browser search samples around this mean.
        """

        def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACTION_DIM, h: int = hidden):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
            )
            self.value_head = nn.Sequential(nn.Linear(h, 1), nn.Tanh())
            self.policy_head = nn.Sequential(nn.Linear(h, act_dim), nn.Tanh())

        def forward(self, obs):
            z = self.trunk(obs)
            value = self.value_head(z)
            policy = self.policy_head(z)
            return value, policy

    return ShowboatNet()
