import { type Ball, Motion } from "./ball";
import { fromAngle, scale } from "./vec";
import { BALL_RADIUS } from "./constants";

// CueAction is planar: no cue-elevation field, so jump and massé shots are
// unrepresentable rather than merely discouraged (a jump needs downward
// elevation into the ball; massé needs steep elevation plus extreme side
// english). V0 is capped (MAX_V0) so pure speed can't launch the ball
// airborne either.
//
//   - phi: aim direction in radians (table frame).
//   - power: normalized [0,1]; maps to V0 in [0, MAX_V0].
//   - sideSpin (a): horizontal english in [-1, 1] (left < 0 < right).
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

// How far off centre the tip can strike, as a fraction of the ball radius.
// Past about half a radius a real tip miscues, so full english on the
// sliders means a half-radius offset.
export const MAX_TIP_OFFSET = 0.5;

// Spin from an off-centre strike. A cue tip hitting the ball at an offset d
// from centre, at speed v, gives angular velocity w = 5 v d / (2 R^2): the
// impulse J = m v acts with lever arm d on a solid sphere (I = 2/5 m R^2).
// So spin scales with speed. A centre hit has no spin at all: the ball slides
// off the tip, and friction turns that into rolling a little way down the table.
const spinFromOffset = (v0: number, offset: number): number =>
  (5 * v0 * offset * MAX_TIP_OFFSET) / (2 * BALL_RADIUS);

// Apply a cue action to the cue ball, setting its initial velocity and spin.
// The bounds are clamped here too, even though the type already rules out
// elevation.
export const applyCue = (cueBall: Ball, action: CueAction): void => {
  const power = clamp(action.power, 0, 1);
  const a = clamp(action.sideSpin, -1, 1);
  const b = clamp(action.topSpin, -1, 1);

  const v0 = power * MAX_V0;
  const dir = fromAngle(action.phi);

  cueBall.vel = scale(dir, v0);

  // Right english (a > 0) is clockwise seen from above, which is negative wz
  // in this right-handed, z-up frame.
  cueBall.wz = -spinFromOffset(v0, a);

  // Follow (b > 0) is topspin along the line of travel, draw (b < 0) is
  // backspin. `roll` is stored so that roll * R is the surface speed the
  // spin alone would give, so a centre hit starts at zero and a natural
  // roll is roll = v / R (reached at b = 0.8 with the half-radius limit).
  cueBall.roll = scale(dir, spinFromOffset(v0, b));

  cueBall.motion = v0 > 0 ? Motion.Sliding : Motion.Stationary;
};
const clamp = (v: number, lo: number, hi: number): number =>
  Math.max(lo, Math.min(hi, v));
