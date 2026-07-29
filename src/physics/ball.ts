import { type Vec2, vec, mag, scale, sub } from "./vec";
import {
  BALL_RADIUS,
  G,
  MU_SLIDING,
  MU_ROLLING,
  MU_SPINNING,
  STOP_SPEED,
  STOP_SPIN,
} from "./constants";

// Motion phases. This is the core of the event-based model: within a single
// phase a ball follows a closed-form trajectory (uniformly accelerated for
// sliding, decelerating-linear for rolling), so we can solve analytically for
// when the next event happens rather than stepping time forward blindly.
export enum Motion {
  Stationary = "stationary",
  Sliding = "sliding", // surface velocity != contact-point velocity (has "english"/slip)
  Rolling = "rolling", // pure rolling; surface velocity matches translation
  Spinning = "spinning", // stationary translation but still spinning about vertical
}

export interface Ball {
  id: number;
  // 0 = cue ball; 1..7 solids; 8 = eight; 9..15 stripes. Kept as plain data.
  pos: Vec2; // centre position on the bed, metres
  vel: Vec2; // linear velocity of the centre, m/s
  // Angular velocity vector. We track z (spin about vertical, i.e. "english"
  // sidespin / masse component — capped to zero at the action layer) and the
  // in-plane roll components implicitly via the rolling model. For a 2D
  // top-down engine the load-bearing angular quantity is wz (sidespin) plus the
  // roll that opposes sliding, which we derive from the slip velocity.
  wz: number; // spin about the vertical axis, rad/s (sidespin)
  // Perpendicular angular velocity that resists sliding is captured by the
  // relative surface velocity; we store the ball's "spin" contribution as a
  // separate roll vector so throw and follow/draw are represented.
  roll: Vec2; // effective rolling angular velocity mapped into the bed plane
  motion: Motion;
  pocketed: boolean;
}

export const makeBall = (id: number, x: number, y: number): Ball => ({
  id,
  pos: vec(x, y),
  vel: vec(0, 0),
  wz: 0,
  roll: vec(0, 0),
  motion: Motion.Stationary,
  pocketed: false,
});

export const cloneBall = (b: Ball): Ball => ({
  id: b.id,
  pos: { ...b.pos },
  vel: { ...b.vel },
  wz: b.wz,
  roll: { ...b.roll },
  motion: b.motion,
  pocketed: b.pocketed,
});

// The velocity of the ball's contact point with the cloth (the "relative
// surface velocity"). Sliding friction acts opposite to this vector. In the
// classic model this is v + R * (w x k) where k is the surface normal; with our
// roll vector convention this reduces to vel + R*(perp of roll contribution).
export const relativeSurfaceVelocity = (b: Ball): Vec2 => {
  // roll encodes the peripheral velocity the ball *would* have if rolling; the
  // slip is the difference between actual translation and that rolling speed.
  return sub(b.vel, scale(b.roll, BALL_RADIUS));
};

// Classify a ball's motion from its current kinematic state. This is what the
// evolution loop calls after every event to decide which analytic trajectory to
// integrate next.
export const classifyMotion = (b: Ball): Motion => {
  if (b.pocketed) return Motion.Stationary;
  const speed = mag(b.vel);
  const slip = mag(relativeSurfaceVelocity(b));
  const spin = Math.abs(b.wz);

  if (speed < STOP_SPEED && slip < STOP_SPEED) {
    return spin > STOP_SPIN ? Motion.Spinning : Motion.Stationary;
  }
  // If the contact point is essentially not slipping, the ball is rolling.
  if (slip < STOP_SPEED) return Motion.Rolling;
  return Motion.Sliding;
};

// Linear deceleration magnitude for the current phase (m/s^2). Sliding uses the
// full sliding-friction deceleration; rolling uses the much smaller rolling
// resistance. These feed the analytic time-of-next-event solver.
export const linearDeceleration = (motion: Motion): number => {
  switch (motion) {
    case Motion.Sliding:
      return MU_SLIDING * G;
    case Motion.Rolling:
      return MU_ROLLING * G;
    default:
      return 0;
  }
};

// Spin (sidespin) decays under spinning friction — a torque about the vertical
// axis. Rate is (5/2) * mu_sp * g / R (from I = 2/5 m R^2). rad/s^2.
export const spinDeceleration = (): number =>
  (5 / 2) * (MU_SPINNING * G) / BALL_RADIUS;
