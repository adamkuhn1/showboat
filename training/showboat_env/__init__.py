"""Showboat 8-ball self-play training environment (pooltool + LightZero)."""

from .action_space import ACTION_DIM, CueAction, decode, encode
from .eight_ball_env import EightBallEnv, EnvConfig, pooltool_available
from .observation import OBS_DIM, encode_observation
from .reward import RewardConfig, ShotSummary, shot_reward

__all__ = [
    "ACTION_DIM",
    "OBS_DIM",
    "CueAction",
    "EightBallEnv",
    "EnvConfig",
    "RewardConfig",
    "ShotSummary",
    "decode",
    "encode",
    "encode_observation",
    "pooltool_available",
    "shot_reward",
]
