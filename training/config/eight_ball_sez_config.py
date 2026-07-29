"""LightZero Sampled EfficientZero config for Showboat 8-ball self-play.

Mirrors the structure of the ``ekiefl/LightZero`` ``dev-pooltool`` sum-to-three
config, retargeted to our full 8-ball env with a 4-D continuous action (NO
theta) and the low-dimensional coordinate observation. This is the de-risking
starting point; EfficientZero V2 is the SOTA upgrade target once the pipeline is
proven end to end (see README).

The dict shape follows LightZero's ``EasyDict`` main/create split so it can be
handed to ``lzero.entry.train_muzero`` with minimal glue. Values are chosen for
free/consumer compute (small MLP, modest simulation count).
"""

from __future__ import annotations

# These imports are only present in a full LightZero install; guarded so the
# file is importable for linting on a bare Python.
try:  # pragma: no cover
    from easydict import EasyDict
except Exception:  # pragma: no cover
    def EasyDict(d):  # type: ignore
        return d


from showboat_env.action_space import ACTION_DIM
from showboat_env.observation import OBS_DIM

# --- knobs ---
collector_env_num = 8
evaluator_env_num = 3
n_episode = 8
num_simulations = 50
K = 20  # num_of_sampled_actions for Sampled EfficientZero
update_per_collect = 100
batch_size = 256
max_env_step = int(2e6)
reanalyze_ratio = 0.0

main_config = EasyDict(
    dict(
        exp_name="showboat_8ball_sez",
        env=dict(
            env_id="showboat-8ball",
            continuous=True,
            obs_shape=OBS_DIM,
            action_space_size=ACTION_DIM,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False),
        ),
        policy=dict(
            model=dict(
                observation_shape=OBS_DIM,
                action_space_size=ACTION_DIM,
                continuous_action_space=True,
                num_of_sampled_actions=K,
                model_type="mlp",
                latent_state_dim=128,
                lstm_hidden_size=128,
                self_supervised_learning_loss=True,
                # No theta: the action head is 4-D. Jump/masse cannot be sampled
                # because the parameter does not exist in the action space.
            ),
            cuda=True,
            policy_entropy_weight=5e-3,
            game_segment_length=50,
            random_collect_episode_num=0,
            use_augmentation=False,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type="Adam",
            learning_rate=3e-4,
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            eval_freq=int(2e3),
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
        ),
    )
)

create_config = EasyDict(
    dict(
        env=dict(
            type="showboat-8ball",
            import_names=["showboat_env.lightzero_adapter"],
        ),
        env_manager=dict(type="subprocess"),
        policy=dict(
            type="sampled_efficientzero",
            import_names=["lzero.policy.sampled_efficientzero"],
        ),
    )
)


if __name__ == "__main__":  # pragma: no cover
    # Entry used on Colab: `python config/eight_ball_sez_config.py`.
    from lzero.entry import train_muzero

    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
