import { add, scale, mag, normalize } from "./vec";
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
    // Sliding: friction F = mu*m*g acts against the slip (the contact
    // point's velocity over the cloth). That direction stays fixed for the
    // whole sliding phase, so the centre decelerates uniformly at mu*g while
    // the same force's torque spins the surface up at (5/2)*mu*g
    // (I = 2/5 m R^2). Slip therefore closes at (7/2)*mu*g, and the phase
    // ends exactly when it reaches zero.
    const slip = relativeSurfaceVelocity(b);
    const slipDir = normalize(slip);
    const tEnd = slidingTime(mag(slip), decel);
    const h = Math.min(dt, tEnd);
    const a = scale(slipDir, -decel);
    b.pos = add(add(b.pos, scale(b.vel, h)), scale(a, 0.5 * h * h));
    b.vel = add(b.vel, scale(a, h));
    if (h >= tEnd) {
      // Slip is gone: lock the spin to the velocity (natural roll) and spend
      // what is left of dt rolling.
      b.roll = scale(b.vel, 1 / BALL_RADIUS);
      b.motion = Motion.Rolling;
      decaySpin(b, h);
      clampStopped(b);
      if (dt > h) advanceBall(b, dt - h);
      return;
    }
    b.roll = add(b.roll, scale(slipDir, (2.5 * decel / BALL_RADIUS) * h));
  } else {
    // Rolling: friction acts opposite the velocity, decelerating linearly.
    if (speed > 1e-12) {
      // Never integrate past the stop: beyond it the quadratic would carry
      // the ball backwards.
      const h = Math.min(dt, speed / decel);
      const dir = normalize(b.vel);
      const a = scale(dir, -decel);
      b.pos = add(add(b.pos, scale(b.vel, h)), scale(a, 0.5 * h * h));
      b.vel = add(b.vel, scale(a, h));
      // Roll stays locked to velocity in pure rolling.
      b.roll = scale(b.vel, 1 / BALL_RADIUS);
    }
  }

  decaySpin(b, dt);
  clampStopped(b);
};

// Time for slip to close under sliding friction: it shrinks at (7/2)*mu*g
// (linear deceleration mu*g plus spin-up (5/2)*mu*g), which is where the
// textbook "a stun shot rolls on at 5/7 of its speed" comes from.
export const slidingTime = (slip: number, decel: number): number =>
  decel > 0 ? slip / (3.5 * decel) : Infinity;

const decaySpin = (b: Ball, dt: number): void => {
  const sd = spinDeceleration();
  if (b.wz > 0) b.wz = Math.max(0, b.wz - sd * dt);
  else if (b.wz < 0) b.wz = Math.min(0, b.wz + sd * dt);
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
    return slidingTime(mag(relativeSurfaceVelocity(b)), decel);
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
