import { add, sub, scale, dot, mag, normalize, perp } from "./vec";
import { type Ball } from "./ball";
import { type Cushion } from "./table";
import {
  BALL_RADIUS,
  E_BALL_BALL,
  E_BALL_CUSHION,
  MU_BALL_BALL,
  MU_BALL_CUSHION,
  CUSHION_HEIGHT_FRACTION,
} from "./constants";

// --- Ball-ball collision -----------------------------------------------------
//
// Impulse-based resolution along the line of centres, with a tangential
// friction impulse that models "throw" (the collision-induced friction that
// nudges the object ball off the pure geometric line and transfers a little
// spin). Both balls have equal mass, so the normal exchange is symmetric.
//
// Mutates a and b in place.
export const resolveBallBall = (a: Ball, b: Ball): void => {
  const n = normalize(sub(b.pos, a.pos)); // unit line of centres, a -> b
  const t = perp(n); // tangent

  // Positional correction: if the balls are interpenetrating (numerical
  // overshoot from the discrete event step), separate them symmetrically along
  // the line of centres so they touch exactly. Prevents balls resting inside
  // one another and stops the "already overlapping" re-collision loop.
  const dist = mag(sub(b.pos, a.pos));
  const overlap = 2 * BALL_RADIUS - dist;
  if (overlap > 0) {
    const push = scale(n, overlap / 2 + 1e-7);
    a.pos = sub(a.pos, push);
    b.pos = add(b.pos, push);
  }

  // Decompose each velocity into normal / tangential components.
  const van = dot(a.vel, n);
  const vat = dot(a.vel, t);
  const vbn = dot(b.vel, n);
  const vbt = dot(b.vel, t);

  const approach = van - vbn; // closing speed along the normal
  if (approach <= 0) return; // already separating; nothing to do

  // Equal-mass elastic-with-restitution exchange along the normal.
  const e = E_BALL_BALL;
  const newVan = van - (1 + e) * 0.5 * approach;
  const newVbn = vbn + (1 + e) * 0.5 * approach;

  // Throw: a small friction impulse acting along the tangent, proportional to
  // the normal impulse and bounded by the relative tangential surface speed.
  // The relative tangential surface velocity includes each ball's sidespin
  // contribution at the contact point (w x r ~ wz * R along the tangent).
  const surfATan = vat + a.wz * BALL_RADIUS;
  const surfBTan = vbt - b.wz * BALL_RADIUS;
  const relTan = surfATan - surfBTan;

  const normalImpulse = 0.5 * (1 + e) * approach; // per-ball magnitude
  const maxFric = MU_BALL_BALL * normalImpulse;
  // Friction opposes the relative surface slip, up to the Coulomb limit, and
  // never more than it takes to stop the slip. Each unit of tangential
  // impulse (per unit mass) changes each ball's surface speed by 1 through
  // translation and 5/2 through spin (I = 2/5 m R^2), so the two surfaces
  // close at 2 * (1 + 5/2) = 7 per unit of impulse.
  const fric = Math.sign(relTan) * Math.min(maxFric, Math.abs(relTan) / 7);

  const newVat = vat - fric;
  const newVbt = vbt + fric;

  a.vel = add(scale(n, newVan), scale(t, newVat));
  b.vel = add(scale(n, newVbn), scale(t, newVbt));

  // The same impulse twists both balls the same way. Its contact point is
  // at +R n on a (force -fric t) and at -R n on b (force +fric t); in this
  // z-up frame (perp is a +90 degree turn) both torques are -R * fric.
  const spinTransfer = (2.5 * fric) / BALL_RADIUS;
  a.wz -= spinTransfer;
  b.wz -= spinTransfer;
};

// --- Ball-cushion collision (Han 2005 lineage) -------------------------------
//
// The Han model treats the cushion nose as contacting the ball above its
// equator (at CUSHION_HEIGHT_FRACTION of the radius). That offset couples the
// ball's translation and spin so that (a) the rebound angle differs from the
// naive mirror, and (b) sidespin ("english") changes the outgoing direction and
// speed. We implement the 2D projection of that coupling:
//   - normal component reverses and loses energy by restitution;
//   - a friction impulse along the rail acts on the contact-point surface
//     velocity, which includes the sidespin term, producing spin-dependent
//     rebound and post-collision spin change.
//
// Mutates b in place.
export const resolveBallCushion = (b: Ball, cushion: Cushion): void => {
  const n = cushion.normal; // inward normal (unit)
  const t = perp(n); // rail tangent

  const vn = dot(b.vel, n); // normal velocity (negative = into rail)
  const vt = dot(b.vel, t); // tangential velocity along rail
  if (vn >= 0) return; // moving away from rail already

  // The contact point sits above the equator; the effective restitution and
  // the sidespin coupling both scale with the contact-height geometry.
  const heightCoupling = CUSHION_HEIGHT_FRACTION;

  // Normal rebound with restitution.
  const newVn = -E_BALL_CUSHION * vn;

  // Surface velocity along the rail at the contact point. The rail is behind
  // the ball (contact at -R n), where spin wz moves the surface at
  // -wz * R * t, reduced by how high on the ball the nose sits. So right
  // english (wz < 0) on a ball running along the rail in +t is "running"
  // english: it lowers the slip and widens the rebound angle.
  const surfTan = vt - b.wz * BALL_RADIUS * heightCoupling;

  // Friction along the rail opposes that slip, capped by Coulomb and by the
  // impulse that would just stop it (translation plus the spin response,
  // 1 + (5/2) h^2 per unit impulse). This is what makes english grab the rail.
  const normalImpulse = (1 + E_BALL_CUSHION) * Math.abs(vn);
  const maxFric = MU_BALL_CUSHION * normalImpulse;
  const stopSlip = Math.abs(surfTan) / (1 + 2.5 * heightCoupling * heightCoupling);
  const fric = Math.sign(surfTan) * Math.min(maxFric, stopSlip);

  const newVt = vt - fric;

  b.vel = add(scale(n, newVn), scale(t, newVt));

  // The reaction torque of that friction (force -fric t at -R n) is
  // +R h * fric about z, so the rail takes off english the ball brings in.
  b.wz += (2.5 * fric * heightCoupling) / BALL_RADIUS;

  // The cushion also absorbs the roll component along its normal: a ball
  // arriving with follow (rolling into the rail) cannot keep that forward
  // roll through the rebound — the raised nose grips the ball's surface and
  // partially reverses it (the familiar "check" off a cushion). Without this,
  // residual roll re-accelerates the ball into the rail after every rebound
  // and the ball pins itself against the cushion indefinitely.
  const rollN = dot(b.roll, n);
  b.roll = add(b.roll, scale(n, -(1 + E_BALL_CUSHION) * rollN));

  // Nudge the ball just clear of the rail to avoid immediate re-collision from
  // floating-point contact.
  b.pos = add(b.pos, scale(n, 1e-6));
};

// Speed after a cushion hit, used for event-trace annotation.
export const speedOf = (b: Ball): number => mag(b.vel);
