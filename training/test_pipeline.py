"""Unit tests for the pooltool-independent training core.

Runnable on any Python with numpy (no pooltool / no torch needed):
    python -m pytest test_pipeline.py        # if pytest installed
    python test_pipeline.py                   # plain-python fallback runner

Covers the load-bearing invariants: the no-theta action-space guarantee, V0
capping, outcome-only reward (no shot-type term), and the observation layout
that must match the TypeScript ONNX consumer.
"""

from __future__ import annotations

import math

import numpy as np

from showboat_env.action_space import ACTION_DIM, V0_MAX, CueAction, decode, encode
from showboat_env.observation import OBS_DIM, encode_observation
from showboat_env.reward import RewardConfig, ShotSummary, shot_reward


def test_action_has_no_theta():
    a = decode(np.zeros(ACTION_DIM, dtype=np.float32))
    assert not hasattr(a, "theta")
    # The pooltool kwargs must not include a theta key either.
    assert "theta" not in a.as_pooltool_kwargs()


def test_v0_is_capped():
    a = decode(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))  # max speed
    assert a.v0 <= V0_MAX + 1e-6
    a2 = decode(np.array([0.0, 5.0, 0.0, 0.0], dtype=np.float32))  # over-range
    assert a2.v0 <= V0_MAX + 1e-6


def test_decode_encode_roundtrip():
    orig = CueAction(phi=0.7, v0=4.0, a=0.3, b=-0.2)
    back = decode(encode(orig))
    assert math.isclose(back.phi, orig.phi, abs_tol=1e-5)
    assert math.isclose(back.v0, orig.v0, abs_tol=1e-4)
    assert math.isclose(back.a, orig.a, abs_tol=1e-6)
    assert math.isclose(back.b, orig.b, abs_tol=1e-6)


def test_reward_is_outcome_only():
    cfg = RewardConfig()
    assert shot_reward(ShotSummary(True, False, 3, 0, False, False, True), cfg) == cfg.win
    assert shot_reward(ShotSummary(False, True, 0, 0, False, False, False), cfg) == cfg.loss
    # A foul-with-scratch is negative; nothing about *how* a ball dropped matters.
    assert shot_reward(ShotSummary(False, False, 0, 0, True, True, False), cfg) < 0
    # Potting more of your own balls yields strictly more reward (monotone).
    r1 = shot_reward(ShotSummary(False, False, 1, 0, False, False, True), cfg)
    r2 = shot_reward(ShotSummary(False, False, 2, 0, False, False, True), cfg)
    assert r2 > r1


def test_observation_layout_matches_ts():
    obs = encode_observation({0: (0.0, 0.0), 3: (0.5, 0.25)}, {7}, -1.0, False)
    assert obs.shape == (OBS_DIM,)
    assert obs[7 * 3 + 2] == 1.0  # pocketed flag for ball 7
    assert obs[OBS_DIM - 1] == 0.0  # table not open
    assert obs[OBS_DIM - 2] == -1.0  # group sign = stripes


if __name__ == "__main__":  # plain-python runner (no pytest dependency)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[ok] {fn.__name__}")
    print(f"all {len(fns)} pipeline tests passed")
