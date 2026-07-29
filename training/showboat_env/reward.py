"""Reward shaping for the Showboat 8-ball agent.

HARD RULE (PLAN.md S5): banks and multi-wall combos must be EMERGENT from reward,
never scripted. So the reward here is a function of OUTCOMES ONLY -- it never
inspects *how* a ball was pocketed, never pays a bonus for "banking", never
detects a trick-shot geometry. The agent discovers banks because they win, not
because we paid it to bank.

The default weights below reward: winning the game (terminal), pocketing your own
legal ball, keeping your turn, and clearing toward the 8. Fouls and losing are
penalized. Nothing rewards a shot *type*.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    win: float = 1.0
    loss: float = -1.0
    pot_own_ball: float = 0.12  # per legal ball of your own group pocketed
    foul: float = -0.15
    scratch: float = -0.20
    opponent_ball_potted: float = -0.05  # helping the opponent
    keep_turn: float = 0.02  # small bonus for continuing (encourages runs)
    # No "bank" / "combo" / "trick" term exists here BY DESIGN.


@dataclass
class ShotSummary:
    """Outcome facts about one resolved shot (produced by the ruleset)."""

    won: bool
    lost: bool
    own_balls_potted: int
    opponent_balls_potted: int
    foul: bool
    scratch: bool
    kept_turn: bool


def shot_reward(summary: ShotSummary, cfg: RewardConfig = RewardConfig()) -> float:
    """Outcome-only reward. Reads no information about shot geometry/type."""
    if summary.won:
        return cfg.win
    if summary.lost:
        return cfg.loss

    r = 0.0
    r += cfg.pot_own_ball * summary.own_balls_potted
    r += cfg.opponent_ball_potted * summary.opponent_balls_potted
    if summary.foul:
        r += cfg.foul
    if summary.scratch:
        r += cfg.scratch
    if summary.kept_turn:
        r += cfg.keep_turn
    return r
