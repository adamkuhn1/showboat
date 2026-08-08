import { type Table, makeTable, HEAD_SPOT } from "../physics/table";
import { type Ball, cloneBall, Motion } from "../physics/ball";
import { simulateShot, type SimResult } from "../physics/engine";
import { applyCue, type CueAction } from "../physics/cue";
import { BALL_DIAMETER, BALL_RADIUS, ENGINE_MIN_GAP } from "../physics/constants";
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

// The nearest spot to (x, y) where the cue ball can legally sit: inside the
// cushions, and not inside another ball.
//
// A ball-in-hand placement is a raw canvas point, so nothing stops it landing
// on top of an object ball. Left alone, that overlap is resolved by the
// simulation engine on the first frame of the shot: both balls are shoved apart
// before the cue is struck — measured at 27 mm, half a ball width, for a
// placement 5 mm off an object ball's centre. Two balls that leap apart on
// their own are the clearest possible way to look like broken physics, and the
// cue ball is always one of them.
//
// Resolved by moving only the cue: object balls are where the last shot left
// them and nothing about picking a spot may disturb them.
export const clearCuePlacement = (
  balls: Ball[],
  x: number,
  y: number,
  table: Table,
): { x: number; y: number } => {
  const hx = table.length / 2 - BALL_RADIUS;
  const hy = table.width / 2 - BALL_RADIUS;
  const clamp = (p: { x: number; y: number }) => ({
    x: Math.max(-hx, Math.min(hx, p.x)),
    y: Math.max(-hy, Math.min(hy, p.y)),
  });
  const others = balls.filter((b) => b.id !== CUE_ID && !b.pocketed);
  // Sit at the clearance the engine keeps between resting balls, so the first
  // thing the simulation does is not to separate the board.
  const min = BALL_DIAMETER + ENGINE_MIN_GAP;
  const deepestOverlap = (p: { x: number; y: number }) => {
    let worst: { dx: number; dy: number; dist: number; overlap: number } | null = null;
    for (const b of others) {
      const dx = p.x - b.pos.x;
      const dy = p.y - b.pos.y;
      const dist = Math.hypot(dx, dy);
      const overlap = min - dist;
      if (overlap > 0 && (worst === null || overlap > worst.overlap)) {
        worst = { dx, dy, dist, overlap };
      }
    }
    return worst;
  };

  // Push out of the deepest overlap, repeatedly: clearing one ball can push the
  // cue into the next in a cluster.
  let p = clamp({ x, y });
  for (let pass = 0; pass < 16; pass++) {
    const worst = deepestOverlap(p);
    if (worst === null) return p;
    const [nx, ny] =
      worst.dist > 1e-9 ? [worst.dx / worst.dist, worst.dy / worst.dist] : [1, 0];
    p = clamp({ x: p.x + nx * worst.overlap, y: p.y + ny * worst.overlap });
  }

  // Still stuck — the push kept running the cue into a rail or into the next
  // ball of a cluster. Fall back to the clear point on a table-wide lattice
  // nearest the one that was asked for, which cannot fail while the table has
  // room for one more ball.
  const step = BALL_DIAMETER / 2;
  let best: { x: number; y: number } | null = null;
  let bestDist = Infinity;
  for (let gx = -hx; gx <= hx; gx += step) {
    for (let gy = -hy; gy <= hy; gy += step) {
      const c = { x: gx, y: gy };
      if (deepestOverlap(c) !== null) continue;
      const d = Math.hypot(c.x - x, c.y - y);
      if (d < bestDist) {
        bestDist = d;
        best = c;
      }
    }
  }
  return best ?? p;
};

// Place the cue ball (ball-in-hand). Returns a new state with the cue moved to
// the nearest legal spot to the one requested.
export const placeCueBall = (
  s: GameState,
  x: number,
  y: number,
  table: Table,
): GameState => {
  const next = cloneState(s);
  const cue = next.balls.find((b) => b.id === CUE_ID);
  if (cue) {
    cue.pos = clearCuePlacement(next.balls, x, y, table);
    cue.pocketed = false;
    cue.vel = { x: 0, y: 0 };
    cue.motion = Motion.Stationary;
  }
  next.ballInHand = false;
  return next;
};

// A shot simulator: applies the cue action to the cue ball and evolves the
// world to rest, returning the trace. The default is the pure-TS reference
// engine; the App injects the Rust→WASM implementation for play. Making this an
// injected function keeps the game/rules layer engine-agnostic and lets tests
// run against the deterministic TS oracle without the wasm toolchain.
export type Simulator = (balls: Ball[], action: CueAction) => SimResult;

const defaultSimulator: Simulator = (balls, action) => {
  const cue = balls.find((b) => b.id === CUE_ID);
  if (cue) applyCue(cue, action);
  return simulateShot(balls, makeTable());
};

// Execute one shot. Does not mutate the input state.
export const takeShot = (
  s: GameState,
  table: Table,
  action: CueAction,
  simulate: Simulator = defaultSimulator,
): ShotReport => {
  const pre = cloneState(s);
  const next = cloneState(s);

  const cue = next.balls.find((b) => b.id === CUE_ID);
  if (!cue) throw new Error("no cue ball in state");

  const sim = simulate(next.balls, action);

  const outcome = applyShotRules(next, pre, sim);
  next.broken = true;
  next.shotCount = pre.shotCount + 1;
  if (outcome.ballInHandForNext) next.ballInHand = "anywhere";
  // Restore a scratched cue ball as a placeable (not pocketed) ball for the
  // incoming player. It waits on the head spot — the conventional respot, and
  // a spot the incoming player will usually be happy with — cleared of any ball
  // resting there, because this position is drawn on the felt from the moment
  // the shot commits until the placement is made.
  if (outcome.pocketedThisShot.includes(CUE_ID) && !outcome.gameOver) {
    const c = next.balls.find((b) => b.id === CUE_ID);
    if (c) {
      c.pocketed = false;
      c.vel = { x: 0, y: 0 };
      c.motion = Motion.Stationary;
      c.pos = clearCuePlacement(next.balls, HEAD_SPOT.x, HEAD_SPOT.y, table);
    }
  }

  return { next, sim, outcome };
};

// Convenience for tests / search: pull the live (non-pocketed) balls.
export const liveBalls = (s: GameState): Ball[] =>
  s.balls.filter((b) => !b.pocketed);
