"""Adapter registering :class:`EightBallEnv` as a LightZero/DI-engine env.

LightZero consumes environments through DI-engine's ``BaseEnv`` interface and a
registry name (referenced by ``create_config.env.type`` in the config). This
thin adapter wraps our plain Gym-style :class:`EightBallEnv` in that interface.
Kept separate from the env itself so the core env has no hard DI-engine
dependency and stays unit-testable on a bare Python.

DI-engine imports are guarded so this module is importable for linting without
the full stack; the registry decorator only binds when DI-engine is present.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .action_space import ACTION_DIM
from .eight_ball_env import EightBallEnv, EnvConfig
from .observation import OBS_DIM

try:  # pragma: no cover - only available in a full LightZero install
    from ding.envs import BaseEnv, BaseEnvTimestep
    from ding.utils import ENV_REGISTRY

    _DING_AVAILABLE = True
except Exception:  # pragma: no cover
    _DING_AVAILABLE = False
    BaseEnv = object  # type: ignore
    BaseEnvTimestep = tuple  # type: ignore

    class _Reg:
        def register(self, *_a, **_k):
            def deco(cls):
                return cls

            return deco

    ENV_REGISTRY = _Reg()  # type: ignore


@ENV_REGISTRY.register("showboat-8ball")
class ShowboatLightZeroEnv(BaseEnv):  # pragma: no cover - needs DI-engine
    """DI-engine BaseEnv wrapper around EightBallEnv."""

    def __init__(self, cfg: Any = None):
        self._cfg = cfg
        self._env = EightBallEnv(EnvConfig())
        self._init_flag = False

    def reset(self):
        obs = self._env.reset()
        self._init_flag = True
        return {"observation": obs.astype(np.float32),
                "action_mask": None,
                "to_play": self._env._to_move}

    def step(self, action):
        obs, rew, done, info = self._env.step(np.asarray(action, dtype=np.float32))
        obs_dict = {"observation": obs.astype(np.float32),
                    "action_mask": None,
                    "to_play": self._env._to_move}
        return BaseEnvTimestep(obs_dict, np.array([rew], dtype=np.float32), done, info)

    def close(self):
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._env = EightBallEnv(EnvConfig(seed=seed))

    @property
    def observation_space(self):
        return OBS_DIM

    @property
    def action_space(self):
        return ACTION_DIM

    def __repr__(self) -> str:
        return "ShowboatLightZeroEnv(8-ball, continuous 4-D action, no theta)"
