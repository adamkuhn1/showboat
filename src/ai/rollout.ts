import { type Table } from "../physics/table";
import { type CueAction } from "../physics/cue";
import { type GameState } from "../game/state";
import { takeShot, type ShotReport } from "../game/game";
import { CUE_ID, EIGHT_ID } from "../game/rack";

// Shared success definition + jittered rollouts. Used to LABEL candidates in
// the training pipeline and to measure robustness of the AI's verified shots
// at play time — one definition, so the model's training target and the game's
// selection criterion are the same quantity.

// Execution noise for robustness rollouts: a small angular error and a small
// power error. A shot that only works at one exact angle scores lower than a
// forgiving one — this is the property the ranker is trained to predict.
export const NOISE_PHI_SIGMA = 0.008; // radians (~0.46 deg)
export const NOISE_POWER_SIGMA = 0.03;

export interface RolloutResult {
  successes: number;
  n: number;
}

// A shot "succeeds" when it is legal (no foul of any kind, which covers wrong
// first contact, scratch and the no-rail rule) and the intended target ball
// went down — without illegally sinking the 8.
export const shotSucceeded = (report: ShotReport, targetBall: number): boolean => {
  const { sim, outcome } = report;
  if (outcome.foul) return false;
  if (!sim.pocketed.includes(targetBall)) return false;
  if (sim.pocketed.includes(CUE_ID)) return false;
  if (targetBall !== EIGHT_ID && sim.pocketed.includes(EIGHT_ID)) return false;
  return true;
};

// Deterministic-enough gaussian via Box-Muller on the provided RNG (defaults
// to Math.random; the training pipeline passes a seeded RNG so datasets are
// reproducible).
export const gaussian = (rng: () => number): number => {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

export const jitterAction = (a: CueAction, rng: () => number): CueAction => ({
  phi: a.phi + gaussian(rng) * NOISE_PHI_SIGMA,
  power: Math.min(1, Math.max(0.05, a.power + gaussian(rng) * NOISE_POWER_SIGMA)),
  sideSpin: a.sideSpin,
  topSpin: a.topSpin,
});

// Run n jittered executions of the action and count successes. Never records
// frames (these are measurement rollouts, not the published shot).
export const rolloutSuccess = (
  g: GameState,
  table: Table,
  action: CueAction,
  targetBall: number,
  n: number,
  rng: () => number = Math.random,
): RolloutResult => {
  let successes = 0;
  for (let i = 0; i < n; i++) {
    const report = takeShot(g, table, jitterAction(action, rng));
    if (shotSucceeded(report, targetBall)) successes++;
  }
  return { successes, n };
};

// Small seeded RNG (mulberry32) so training data generation is reproducible.
export const makeRng = (seed: number): (() => number) => {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};
