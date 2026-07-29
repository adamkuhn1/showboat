"""Observation encoding for the Showboat 8-ball agent.

Low-dimensional by design (ball coordinates + flags, NOT pixels) -- this is what
keeps the network small and training tractable on free/consumer compute, per the
research budget analysis. The layout MUST match the TypeScript ``encodeObservation``
in ``src/ai/onnx.ts`` so the exported ONNX net receives identical inputs at play
time.

Layout (per ball, id 0..15, stride 3): [x_norm, y_norm, pocketed_flag]
Positions normalized to [-1, 1] by half the table dimensions. Two extra
scalars are appended describing whose turn / group state it is.
"""

from __future__ import annotations

import numpy as np

N_BALLS = 16
PER_BALL = 3
# 16 balls * 3 + [group_sign, table_open] = 50
OBS_DIM = N_BALLS * PER_BALL + 2

# Match the Rust/TS bar-box table.
TABLE_LENGTH = 1.9812
TABLE_WIDTH = 0.9906
HALF_LEN = TABLE_LENGTH / 2
HALF_WID = TABLE_WIDTH / 2


def encode_observation(
    positions: dict[int, tuple[float, float]],
    pocketed: set[int],
    group_sign: float,
    table_open: bool,
) -> np.ndarray:
    """Encode board state into the fixed-length observation vector.

    ``positions`` maps ball id -> (x, y) in table coordinates (origin centre).
    ``group_sign`` is +1 if the to-move player is on solids, -1 on stripes, 0 if
    the table is open.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    for bid in range(N_BALLS):
        o = bid * PER_BALL
        if bid in pocketed:
            obs[o + 2] = 1.0
            continue
        if bid in positions:
            x, y = positions[bid]
            obs[o] = np.clip(x / HALF_LEN, -1.0, 1.0)
            obs[o + 1] = np.clip(y / HALF_WID, -1.0, 1.0)
    obs[N_BALLS * PER_BALL] = float(group_sign)
    obs[N_BALLS * PER_BALL + 1] = 1.0 if table_open else 0.0
    return obs
