import { type Table, makeTable, HEAD_STRING_X } from "../physics/table";
import { BALL_RADIUS } from "../physics/constants";
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
// The caller is responsible for keeping the spot legal (see cueSpotLimits).
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

  const outcome = applyShotRules(next, pre, sim);
  next.broken = true;
  next.shotCount = pre.shotCount + 1;
  // A scratched cue ball stays off the table (pocketed) until the incoming
  // player puts it down behind the head string with placeCueBall. Nothing is
  // drawn in the meantime, so it never appears anywhere it wasn't put.
  if (outcome.ballInHandForNext && !outcome.gameOver) {
    next.ballInHand = "behind-head";
  }

  return { next, sim, outcome };
};

// Where the cue ball may go for the current ball-in-hand: behind the head
// string after a scratch, anywhere otherwise. One ball radius in from the
// rails, and at or behind the line (a ball on the line counts as behind it).
export const cueSpotLimits = (
  s: GameState,
  table: Table,
): { minX: number; maxX: number; minY: number; maxY: number } => {
  const hx = table.length / 2 - BALL_RADIUS;
  const hy = table.width / 2 - BALL_RADIUS;
  return {
    minX: -hx,
    maxX: s.ballInHand === "behind-head" ? HEAD_STRING_X : hx,
    minY: -hy,
    maxY: hy,
  };
};

// Convenience for tests / search: pull the live (non-pocketed) balls.
export const liveBalls = (s: GameState): Ball[] =>
  s.balls.filter((b) => !b.pocketed);
