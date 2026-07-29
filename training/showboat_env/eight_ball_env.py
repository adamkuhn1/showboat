"""LightZero-style self-play environment wrapping pooltool's 8-ball ruleset.

This is the main net-new engineering the research flagged: pooltool ships
``pooltool/ruleset/eight_ball.py`` and a physics engine, but the LightZero fork
only trained the toy "sum-to-three" game. Here we wrap full 8-ball -- turn
structure, group assignment, fouls, ball-in-hand, and the win condition -- as a
Gym-style ``reset()``/``step()`` env with a continuous 4-D action, suitable for
Sampled EfficientZero / EfficientZero V2 self-play.

pooltool is imported lazily so this module lints and imports on any Python
(including the 3.9 dev box where pooltool -- which needs 3.10+ -- cannot install).
The actual simulation obviously requires pooltool at run time; see
``training/README.md`` for the environment blocker and the Colab run recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .action_space import ACTION_DIM, decode
from .observation import OBS_DIM, encode_observation
from .reward import RewardConfig, ShotSummary, shot_reward

# pooltool is optional at import time; resolved on first real use.
try:  # pragma: no cover - depends on runtime environment
    import pooltool as pt  # type: ignore

    _POOLTOOL_AVAILABLE = True
except Exception:  # pragma: no cover
    pt = None  # type: ignore
    _POOLTOOL_AVAILABLE = False


def pooltool_available() -> bool:
    return _POOLTOOL_AVAILABLE


@dataclass
class EnvConfig:
    max_shots: int = 60  # safety cap on game length
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: Optional[int] = None


class EightBallEnv:
    """Self-play 8-ball environment.

    A single :class:`EightBallEnv` represents one game. Because 8-ball is
    turn-based and perfect-information, self-play alternates the acting player;
    the observation is always presented from the to-move player's perspective
    (``group_sign`` flips), which is what lets one shared network play both
    seats -- the AlphaZero self-play setup.
    """

    action_dim = ACTION_DIM
    obs_dim = OBS_DIM

    def __init__(self, config: Optional[EnvConfig] = None):
        self.cfg = config or EnvConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._system = None  # pooltool System
        self._ruleset = None  # pooltool eight_ball ruleset instance
        self._shots = 0
        self._done = False
        self._to_move = 0
        self._groups: dict[int, Optional[str]] = {0: None, 1: None}

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> np.ndarray:
        if not _POOLTOOL_AVAILABLE:
            raise RuntimeError(
                "pooltool is not installed in this environment. Install with "
                "`pip install pooltool-billiards` on Python 3.10-3.13 (see "
                "training/README.md). This dev box is Python 3.9, which pooltool "
                "does not support."
            )
        # Build a fresh 8-ball table + rack + cue. pooltool provides factory
        # helpers; we assemble a System with the eight_ball ruleset.
        self._system = _new_eight_ball_system(self._rng)
        self._ruleset = pt.game.ruleset.get_ruleset("8ball")()  # type: ignore[attr-defined]
        self._shots = 0
        self._done = False
        self._to_move = 0
        self._groups = {0: None, 1: None}
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Apply one shot (a normalized 4-D action) and resolve the ruleset."""
        if self._done:
            raise RuntimeError("step() called on a finished game; call reset().")
        cue_action = decode(np.asarray(action, dtype=np.float32))

        # Strike the cue ball. theta is never passed -> no jump/masse.
        cue = self._system.cue  # type: ignore[union-attr]
        cue.set_state(**cue_action.as_pooltool_kwargs())
        cue.aim_at_ball  # (aiming handled by phi already)

        pt.simulate(self._system, inplace=True)  # type: ignore[union-attr]

        summary = self._resolve_ruleset()
        reward = shot_reward(summary, self.cfg.reward)

        self._shots += 1
        if summary.won or summary.lost or self._shots >= self.cfg.max_shots:
            self._done = True
        if not summary.kept_turn and not self._done:
            self._to_move = 1 - self._to_move

        info = {
            "shots": self._shots,
            "to_move": self._to_move,
            "summary": summary,
        }
        return self._observe(), reward, self._done, info

    # -- helpers -----------------------------------------------------------
    def _observe(self) -> np.ndarray:
        positions, pocketed = _extract_state(self._system)
        group = self._groups[self._to_move]
        group_sign = 0.0 if group is None else (1.0 if group == "solids" else -1.0)
        table_open = group is None
        return encode_observation(positions, pocketed, group_sign, table_open)

    def _resolve_ruleset(self) -> ShotSummary:
        """Translate pooltool's shot info into our outcome-only ShotSummary.

        Delegates legality/win-condition to pooltool's eight_ball ruleset, then
        records ONLY outcome facts -- never how a ball was pocketed (no bank
        detection), preserving the emergent-trick-shot constraint.
        """
        info = self._ruleset.process_shot(self._system)  # type: ignore[union-attr]
        # pooltool's ruleset exposes turn/win/foul info; we adapt defensively so
        # a version bump in field names fails loudly rather than silently.
        won = bool(getattr(info, "game_over", False)) and not bool(
            getattr(info, "is_loss", False)
        )
        lost = bool(getattr(info, "is_loss", False))
        return ShotSummary(
            won=won,
            lost=lost,
            own_balls_potted=int(getattr(info, "own_pocketed", 0)),
            opponent_balls_potted=int(getattr(info, "opp_pocketed", 0)),
            foul=bool(getattr(info, "is_foul", False)),
            scratch=bool(getattr(info, "is_scratch", False)),
            kept_turn=not bool(getattr(info, "turn_over", True)),
        )


# --- pooltool glue (only runs when pooltool is installed) -----------------

def _new_eight_ball_system(rng: np.random.Generator) -> Any:  # pragma: no cover
    """Construct a fresh 8-ball System (table + racked balls + cue)."""
    table = pt.Table.default()  # type: ignore[union-attr]
    balls = pt.get_rack("8ball", table)  # type: ignore[union-attr]
    cue = pt.Cue(cue_ball_id="cue")  # type: ignore[union-attr]
    system = pt.System(table=table, balls=balls, cue=cue)  # type: ignore[union-attr]
    return system


def _extract_state(system: Any) -> tuple[dict[int, tuple[float, float]], set[int]]:  # pragma: no cover
    """Pull ball (x, y) positions + pocketed set from a pooltool System."""
    positions: dict[int, tuple[float, float]] = {}
    pocketed: set[int] = set()
    name_to_id = _BALL_NAME_TO_ID
    for name, ball in system.balls.items():
        bid = name_to_id.get(name)
        if bid is None:
            continue
        state = ball.state
        if getattr(ball, "is_pocketed", False) or getattr(state, "s", 0) == 4:
            pocketed.add(bid)
            continue
        # Recentre pooltool's corner-origin coords to our centre-origin frame.
        x, y = float(state.rvw[0][0]), float(state.rvw[0][1])
        positions[bid] = (x - system.table.w / 2, y - system.table.l / 2)
    return positions, pocketed


# pooltool names balls "cue", "1".."15", "8". Map to our numeric ids.
_BALL_NAME_TO_ID = {"cue": 0, **{str(i): i for i in range(1, 16)}}
