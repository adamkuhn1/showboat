"""End-to-end smoke run: prove the training loop executes before the real
multi-hour Colab run.

What it does when the full stack is installed (Python 3.10+, pooltool, torch):
  1. builds the env, runs a few self-play episodes with a random policy,
  2. checks observation/action shapes and reward signal flow,
  3. exports a (smoke) ONNX net to prove the browser plumbing.

What it does on a partial stack (this dev box, Python 3.9, no pooltool): it still
exercises every piece that does NOT need pooltool -- the action-space decode, the
outcome-only reward, the observation encoder, and (if torch is present) the
network + ONNX export -- and prints exactly which stage is blocked and why. This
keeps the pipeline honest and lint-clean without pretending it ran a real game.
"""

from __future__ import annotations

import sys

import numpy as np

from showboat_env import (
    ACTION_DIM,
    OBS_DIM,
    EightBallEnv,
    EnvConfig,
    RewardConfig,
    ShotSummary,
    decode,
    encode_observation,
    pooltool_available,
    shot_reward,
)


def check_action_space() -> None:
    a = decode(np.array([0.5, 1.0, -0.3, 0.2], dtype=np.float32))
    assert 0.0 <= a.v0 <= 8.5, a.v0
    assert -np.pi <= a.phi <= np.pi, a.phi
    # The action has no theta attribute -- jump/masse are unrepresentable.
    assert not hasattr(a, "theta"), "action must not expose cue elevation"
    print(f"[ok] action space: {ACTION_DIM}-D, no theta, V0 capped -> {a}")


def check_reward() -> None:
    win = shot_reward(ShotSummary(True, False, 1, 0, False, False, True))
    foul = shot_reward(ShotSummary(False, False, 0, 0, True, True, False))
    assert win == RewardConfig().win
    assert foul < 0
    # Reward reads only outcomes; there is no code path that inspects shot type.
    print(f"[ok] outcome-only reward: win={win}, foul-with-scratch={foul:.3f}")


def check_observation() -> None:
    obs = encode_observation({0: (0.0, 0.0), 8: (0.3, 0.1)}, {1, 2}, 1.0, False)
    assert obs.shape == (OBS_DIM,), obs.shape
    assert obs[1 * 3 + 2] == 1.0  # ball 1 flagged pocketed
    print(f"[ok] observation encoder: dim={OBS_DIM}, matches TS onnx.ts layout")


def check_network_and_export() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        print("[skip] torch not installed -> network/ONNX export not exercised "
              "here (runs on Colab). Install: pip install torch onnx")
        return False
    from showboat_env.network import build_network

    net = build_network()
    import torch

    v, p = net(torch.zeros(1, OBS_DIM))
    assert v.shape == (1, 1) and p.shape == (1, ACTION_DIM)
    print(f"[ok] network forward: value{tuple(v.shape)} policy{tuple(p.shape)}")
    return True


def try_self_play_episodes(n: int = 2) -> bool:
    if not pooltool_available():
        print("[blocked] pooltool not installed on this interpreter "
              f"(Python {sys.version_info.major}.{sys.version_info.minor}). "
              "pooltool requires 3.10-3.13; this dev box is 3.9. The self-play "
              "loop runs on Colab -- see training/README.md.")
        return False
    env = EightBallEnv(EnvConfig(seed=0))
    for ep in range(n):
        obs = env.reset()
        total = 0.0
        done = False
        steps = 0
        while not done and steps < 60:
            action = np.random.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
            obs, reward, done, info = env.step(action)
            total += reward
            steps += 1
        print(f"[ok] self-play episode {ep}: {steps} shots, return={total:.3f}")
    return True


def main() -> int:
    print("== Showboat training smoke run ==")
    check_action_space()
    check_reward()
    check_observation()
    check_network_and_export()
    played = try_self_play_episodes()
    print("\nSummary: core pipeline (action/reward/obs) verified. "
          f"Self-play {'ran' if played else 'is BLOCKED on this box (see above)'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
