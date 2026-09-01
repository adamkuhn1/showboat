import { type Ball, Motion } from "./ball";
import { fromAngle, scale, add } from "./vec";
import { BALL_RADIUS } from "./constants";

// CueAction is planar: no cue-elevation field, so jump and massé shots are
// unrepresentable rather than merely discouraged (a jump needs downward
// elevation into the ball; massé needs steep elevation plus extreme side
// english). V0 is capped (MAX_V0) so pure speed can't launch the ball
// airborne either.
//
//   - phi: aim direction in radians (table frame).
//   - power: normalized [0,1]; maps to V0 in [0, MAX_V0].
//   - sideSpin (a): horizontal english in [-1, 1] (left/right of centre).
//   - topSpin (b): vertical english in [-1, 1] (draw/follow). Bounded so it
//     contributes roll, never lift.
export interface CueAction {
  phi: number;
  power: number; // [0,1]
  sideSpin: number; // [-1,1]  (english / "a")
  topSpin: number; // [-1,1]  (draw<0 / follow>0 / "b")
}

// Maximum cue-ball launch speed (m/s). A hard break is ~8 m/s; we cap a little
// above typical to allow a strong break while keeping the ball on the bed.
export const MAX_V0 = 8.5;

// Maximum spin the strike can impart, in rad/s, per unit of normalized english.
// Bounded so english influences path/throw realistically without the pathologies
// that elevation would enable.
export const MAX_SIDE_SPIN = 25; // rad/s of sidespin (wz) at full english
export const MAX_ROLL_SPIN = 40; // rad/s of roll (follow/draw) at full english

// Apply a cue action to the cue ball, setting its initial velocity and spin.
// Enforces the action-space bounds defensively even though the type already
// forbids elevation.
export const applyCue = (cueBall: Ball, action: CueAction): void => {
  const power = clamp(action.power, 0, 1);
  const a = clamp(action.sideSpin, -1, 1);
  const b = clamp(action.topSpin, -1, 1);

  const v0 = power * MAX_V0;
  const dir = fromAngle(action.phi);

  cueBall.vel = scale(dir, v0);

  // Sidespin becomes vertical-axis spin (wz), which curves throw and cushion
  // rebound.
  cueBall.wz = a * MAX_SIDE_SPIN;

  // Follow/draw sets the initial roll relative to velocity. Follow (b>0) means
  // the ball is over-rolling in its travel direction; draw (b<0) means
  // back-spin (surface moving backward at contact) → the classic draw shot.
  // We seed the roll vector along the aim direction, scaled by b.
  const rollMag = (v0 / BALL_RADIUS) + b * MAX_ROLL_SPIN;
  cueBall.roll = scale(dir, rollMag);

  cueBall.motion = v0 > 0 ? Motion.Sliding : Motion.Stationary;
  void add; // silences unused-import lint; add is unused in this file
};

const clamp = (v: number, lo: number, hi: number): number =>
  Math.max(lo, Math.min(hi, v));
