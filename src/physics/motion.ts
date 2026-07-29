import { type Vec2, add, scale, mag, normalize, sub } from "./vec";
import {
  type Ball,
  Motion,
  relativeSurfaceVelocity,
  linearDeceleration,
  spinDeceleration,
} from "./ball";
import { BALL_RADIUS, STOP_SPEED, STOP_SPIN } from "./constants";

// Advance a single ball by dt along its current analytic trajectory. This is
// closed-form integration, not a fixed timestep of the whole world: the caller
// (the evolution engine) has already guaranteed no collision happens within dt,
// so we can move the ball exactly along the friction-decelerated path.
export const advanceBall = (b: Ball, dt: number): void => {
  if (b.pocketed || b.motion === Motion.Stationary) {
    decaySpin(b, dt);
    return;
  }

  if (b.motion === Motion.Spinning) {
    // No translation; just bleed sidespin.
    decaySpin(b, dt);
    return;
  }

  const decel = linearDeceleration(b.motion);
  const speed = mag(b.vel);

  if (b.motion === Motion.Sliding) {
    // Sliding: friction acts opposite the *slip* (relative surface velocity),
    // decelerating the slip and simultaneously spinning the ball up toward pure
    // rolling. Position integrates the centre velocity.
    const slipDir = normalize(relativeSurfaceVelocity(b));
    const a = scale(slipDir, -decel);
    b.pos = add(add(b.pos, scale(b.vel, dt)), scale(a, 0.5 * dt * dt));
    b.vel = add(b.vel, scale(a, dt));
    // The friction torque increases the roll toward matching the (new) velocity.
    // In the slide phase the roll grows so that slip shrinks; we model roll as
    // trending to vel/R.
    const targetRoll = scale(b.vel, 1 / BALL_RADIUS);
    // Roll approaches target at the same rate slip decays.
    b.roll = approach(b.roll, targetRoll, decel * dt);
  } else {
    // Rolling: friction acts opposite the velocity, decelerating linearly.
    if (speed > 1e-12) {
      const dir = normalize(b.vel);
      const a = scale(dir, -decel);
      b.pos = add(add(b.pos, scale(b.vel, dt)), scale(a, 0.5 * dt * dt));
      b.vel = add(b.vel, scale(a, dt));
      // Roll stays locked to velocity in pure rolling.
      b.roll = scale(b.vel, 1 / BALL_RADIUS);
    }
  }

  decaySpin(b, dt);
  clampStopped(b);
};

const decaySpin = (b: Ball, dt: number): void => {
  const sd = spinDeceleration();
  if (b.wz > 0) b.wz = Math.max(0, b.wz - sd * dt);
  else if (b.wz < 0) b.wz = Math.min(0, b.wz + sd * dt);
};

// Move `from` toward `to` by at most `maxStep` magnitude of change.
const approach = (from: Vec2, to: Vec2, maxStep: number): Vec2 => {
  const d = sub(to, from);
  const dm = mag(d);
  if (dm <= maxStep || dm < 1e-12) return { ...to };
  return add(from, scale(normalize(d), maxStep));
};

const clampStopped = (b: Ball): void => {
  if (mag(b.vel) < STOP_SPEED && mag(relativeSurfaceVelocity(b)) < STOP_SPEED) {
    b.vel = { x: 0, y: 0 };
    b.roll = { x: 0, y: 0 };
    if (Math.abs(b.wz) < STOP_SPIN) b.wz = 0;
  }
};

// --- Analytic time-to-event solvers -----------------------------------------

// Time until a sliding ball transitions to rolling (slip reaches zero), or a
// rolling/spinning ball stops. Returns Infinity if it never happens in this
// phase. This is what lets the engine schedule "phase change" events precisely.
export const timeToPhaseChange = (b: Ball): number => {
  if (b.pocketed) return Infinity;

  if (b.motion === Motion.Sliding) {
    const decel = linearDeceleration(Motion.Sliding);
    const slip = mag(relativeSurfaceVelocity(b));
    return decel > 0 ? slip / decel : Infinity;
  }
  if (b.motion === Motion.Rolling) {
    const decel = linearDeceleration(Motion.Rolling);
    const speed = mag(b.vel);
    return decel > 0 ? speed / decel : Infinity;
  }
  if (b.motion === Motion.Spinning) {
    const sd = spinDeceleration();
    return sd > 0 ? Math.abs(b.wz) / sd : Infinity;
  }
  return Infinity;
};
