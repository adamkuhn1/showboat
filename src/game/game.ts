import { type Table, makeTable } from "../physics/table";
import { type Ball, cloneBall, Motion } from "../physics/ball";
import { simulateShot, type SimOptions, type SimResult } from "../physics/engine";
import { applyCue, type CueAction } from "../physics/cue";
import { type GameState } from "./state";
import { rackEightBall, CUE_ID } from "./rack";
import { applyShotRules, type ShotOutcome } from "./rules";

// The game controller. Owns the table geometry and drives one shot at a time:
// apply the cue action to the cue ball, run the event-based simulation, then
// resolve the ruleset. Pure with respect to its inputs — takes a state, returns
// the next state plus the sim result and rule outcome (used by both the UI and
// the ML environment / search).

export interface ShotReport {
  next: GameState;
  sim: SimResult;
  outcome: ShotOutcome;
}

export const makeGame = (): { state: GameState; table: Table } => {
  const balls = rackEightBall();
  return {
    table: makeTable(),
    state: {
      balls,
      turn: 0,
      groups: { 0: null, 1: null },
      ballInHand: false,
      winner: null,
      broken: false,
      shotCount: 0,
    },
  };
};

export const cloneState = (s: GameState): GameState => ({
  balls: s.balls.map(cloneBall),
  turn: s.turn,
  groups: { ...s.groups },
  ballInHand: s.ballInHand,
  winner: s.winner,
  broken: s.broken,
  shotCount: s.shotCount,
});

// Place the cue ball (ball-in-hand). Returns a new state with the cue moved.
export const placeCueBall = (
  s: GameState,
  x: number,
  y: number,
): GameState => {
  const next = cloneState(s);
  const cue = next.balls.find((b) => b.id === CUE_ID);
  if (cue) {
    cue.pos = { x, y };
    cue.pocketed = false;
    cue.vel = { x: 0, y: 0 };
    cue.motion = Motion.Stationary;
  }
  next.ballInHand = false;
  return next;
};

// Execute one shot. Does not mutate the input state.
export const takeShot = (
  s: GameState,
  table: Table,
  action: CueAction,
  simOpts: SimOptions = {},
): ShotReport => {
  const pre = cloneState(s);
  const next = cloneState(s);

  const cue = next.balls.find((b) => b.id === CUE_ID);
  if (!cue) throw new Error("no cue ball in state");
  applyCue(cue, action);

  const sim = simulateShot(next.balls, table, simOpts);

  // Re-spot the cue ball if it was scratched: it comes back into play as
  // ball-in-hand for the opponent, so we lift it off the table until placed.
  const outcome = applyShotRules(next, pre, sim);
  next.broken = true;
  next.shotCount = pre.shotCount + 1;
  if (outcome.ballInHandForNext) next.ballInHand = "anywhere";
  // Restore a scratched cue ball as a placeable (not pocketed) ball for the
  // incoming player.
  if (outcome.pocketedThisShot.includes(CUE_ID) && !outcome.gameOver) {
    const c = next.balls.find((b) => b.id === CUE_ID);
    if (c) {
      c.pocketed = false;
      c.vel = { x: 0, y: 0 };
      c.motion = Motion.Stationary;
      // Park it off-centre; UI/env will place it via placeCueBall.
      c.pos = { x: 0, y: 0 };
    }
  }

  return { next, sim, outcome };
};

// Convenience for tests / search: pull the live (non-pocketed) balls.
export const liveBalls = (s: GameState): Ball[] =>
  s.balls.filter((b) => !b.pocketed);
