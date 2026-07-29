"""Cue action space for the Showboat 8-ball agent.

This is the single most important file for the trick-shot constraint. The action
space has exactly FOUR continuous parameters:

    phi   : aim direction            (radians, wrapped to [-pi, pi])
    v0    : cue-ball launch speed    (m/s, capped at V0_MAX)
    a     : side english / sidespin  (normalized [-1, 1])
    b     : top/bottom english       (normalized [-1, 1], draw<0 / follow>0)

There is NO `theta` (cue elevation) parameter. pooltool's physics is fully
capable of jump and masse shots when the cue is elevated, but with elevation
absent from the action space the agent literally cannot express the parameters
that produce them. This is "impossible at the action level" as the spec demands
-- a capability restriction, not a post-hoc legality filter. The in-browser
TypeScript/Rust engine mirrors the same four-parameter action for train/play
parity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Must match the Rust/TS core (constants.rs / cue.ts) so a policy trained here
# behaves the same at play time.
V0_MAX = 8.5  # m/s
MAX_SIDE_SPIN = 25.0  # rad/s at |a| = 1
MAX_ROLL_SPIN = 40.0  # rad/s at |b| = 1
BALL_RADIUS = 0.028575

# The network emits actions in a normalized [-1, 1]^4 box (standard for
# continuous-control MuZero/EfficientZero); we map that to physical parameters
# here. Keeping the network's action box symmetric and bounded is what lets the
# same policy head serve both discrete-sampled and continuous search.
ACTION_DIM = 4
ACTION_LOW = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class CueAction:
    phi: float
    v0: float
    a: float
    b: float

    def as_pooltool_kwargs(self) -> dict:
        """Map to pooltool's ``Cue.strike`` parameters.

        NOTE: ``theta`` is deliberately omitted (defaults to 0 in pooltool),
        which is exactly the jump/masse prohibition. ``V0`` is pre-capped.
        """
        return dict(
            V0=self.v0,
            phi=math.degrees(self.phi) % 360.0,  # pooltool takes degrees
            a=self.a,  # side english fraction
            b=self.b,  # vertical english fraction
            # theta intentionally not passed -> stays 0 -> no jump/masse.
        )


def decode(norm_action: np.ndarray) -> CueAction:
    """Map a normalized [-1,1]^4 network action to a physical cue action."""
    n = np.clip(np.asarray(norm_action, dtype=np.float32), -1.0, 1.0)
    phi = float(n[0] * math.pi)  # [-pi, pi]
    v0 = float((n[1] * 0.5 + 0.5) * V0_MAX)  # [0, V0_MAX]
    a = float(n[2])
    b = float(n[3])
    return CueAction(phi=phi, v0=v0, a=a, b=b)


def encode(action: CueAction) -> np.ndarray:
    """Inverse of :func:`decode` (useful for tests / imitation warm-starts)."""
    phi_n = action.phi / math.pi
    v0_n = (action.v0 / V0_MAX) * 2.0 - 1.0
    return np.array([phi_n, v0_n, action.a, action.b], dtype=np.float32)
