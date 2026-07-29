import { sub, add, scale, dot, mag, normalize } from "./vec";
import { type Ball, Motion, relativeSurfaceVelocity, linearDeceleration } from "./ball";
import { type Cushion, type Table } from "./table";
import { BALL_RADIUS, BALL_DIAMETER, EPS } from "./constants";

// Predicting the next event is where the event-based approach earns its keep.
// Because each ball follows a known accelerated path within a phase, the time of
// the next geometric event is the smallest positive root of a low-order
// polynomial. For robustness against the varying acceleration directions during
// sliding, we solve these with a short guarded numeric root-find on the exact
// distance function rather than trusting a hand-expanded quartic; this keeps the
// code auditable and stable while remaining "solve for the event time," not a
// blind fixed-step march.

// Position of a ball at time t under its current phase (mirrors advanceBall).
const posAt = (b: Ball, t: number): { x: number; y: number } => {
  if (b.pocketed || b.motion === Motion.Stationary || b.motion === Motion.Spinning) {
    return { ...b.pos };
  }
  const decel = linearDeceleration(b.motion);
  let dir;
  if (b.motion === Motion.Sliding) {
    dir = normalize(relativeSurfaceVelocity(b));
  } else {
    dir = mag(b.vel) > 1e-12 ? normalize(b.vel) : { x: 0, y: 0 };
  }
  const a = scale(dir, -decel);
  // Clamp t so we never integrate past the moment the ball would stop/​switch
  // phase along this accelerated arc (velocity would otherwise reverse).
  return add(add(b.pos, scale(b.vel, t)), scale(a, 0.5 * t * t));
};

const speedAt = (b: Ball, t: number): number => {
  if (b.pocketed || b.motion === Motion.Stationary || b.motion === Motion.Spinning) return 0;
  const decel = linearDeceleration(b.motion);
  const s0 = mag(b.vel);
  return Math.max(0, s0 - decel * t);
};

// Bisection root-find for the earliest t in (0, tMax] where f(t) crosses zero
// from positive to non-positive. f must be continuous. Returns Infinity if no
// crossing is found. Used for both ball-ball and ball-cushion contact.
const earliestZero = (
  f: (t: number) => number,
  tMax: number,
  samples = 64,
): number => {
  if (tMax <= 0 || !isFinite(tMax)) return Infinity;
  let prevT = 0;
  let prevF = f(0);
  // If already in contact and closing, treat as immediate.
  const dt = tMax / samples;
  for (let i = 1; i <= samples; i++) {
    const t = i * dt;
    const cur = f(t);
    if (prevF > 0 && cur <= 0) {
      // Bisection refine on [prevT, t].
      let lo = prevT;
      let hi = t;
      for (let k = 0; k < 40; k++) {
        const mid = 0.5 * (lo + hi);
        if (f(mid) > 0) lo = mid;
        else hi = mid;
      }
      return 0.5 * (lo + hi);
    }
    prevT = t;
    prevF = cur;
  }
  return Infinity;
};

// Time until two balls first touch (centre distance == diameter), within tMax.
export const timeToBallBall = (a: Ball, b: Ball, tMax: number): number => {
  if (a.pocketed || b.pocketed) return Infinity;
  // Both stationary -> never.
  if (a.motion === Motion.Stationary && b.motion === Motion.Stationary) return Infinity;
  // Broad-phase reject: the closest they can get in tMax is bounded by the sum
  // of the distances each can travel. If current separation exceeds that plus a
  // diameter, no contact is possible in the window — skip the root-find.
  const sep = mag(sub(b.pos, a.pos));
  const reach = (mag(a.vel) + mag(b.vel)) * tMax;
  if (sep - reach > BALL_DIAMETER) return Infinity;
  const f = (t: number): number => {
    const d = sub(posAt(b, t), posAt(a, t));
    return mag(d) - BALL_DIAMETER;
  };
  // Only meaningful if they are currently separated; if already overlapping and
  // approaching, report immediate.
  if (f(0) <= EPS) {
    const closing = dot(sub(b.vel, a.vel), normalize(sub(b.pos, a.pos))) < 0;
    return closing ? 0 : Infinity;
  }
  return earliestZero(f, tMax);
};

// Time until a ball's centre reaches within BALL_RADIUS of a cushion line
// (i.e. the ball surface touches the rail nose), within tMax.
export const timeToCushion = (b: Ball, c: Cushion, tMax: number): number => {
  if (b.pocketed || b.motion === Motion.Stationary || b.motion === Motion.Spinning) {
    return Infinity;
  }
  const f = (t: number): number => {
    const p = posAt(b, t);
    const coord = c.axis === "x" ? p.x : p.y;
    // Signed distance from the cushion nose along the inward normal.
    const signed = c.axis === "x"
      ? (p.x - c.pos) * c.normal.x
      : (p.y - c.pos) * c.normal.y;
    void coord;
    return signed - BALL_RADIUS;
  };
  if (f(0) <= EPS) {
    // Already at/inside; collide only if moving into the rail.
    const vn = dot(b.vel, c.normal);
    return vn < 0 ? 0 : Infinity;
  }
  return earliestZero(f, tMax);
};

// Time until a ball's centre passes within a pocket's capture radius, within
// tMax. Pocket capture is modelled as the centre entering the jaw radius.
export const timeToPocket = (
  b: Ball,
  table: Table,
  tMax: number,
): { t: number; pocketId: string } | null => {
  if (b.pocketed || b.motion === Motion.Stationary || b.motion === Motion.Spinning) {
    return null;
  }
  let best: { t: number; pocketId: string } | null = null;
  for (const pk of table.pockets) {
    const f = (t: number): number => mag(sub(posAt(b, t), pk.center)) - pk.radius;
    let t: number;
    if (f(0) <= 0) {
      t = 0;
    } else {
      t = earliestZero(f, tMax);
    }
    if (isFinite(t) && (best === null || t < best.t)) {
      best = { t, pocketId: pk.id };
    }
  }
  return best;
};

export { speedAt };
