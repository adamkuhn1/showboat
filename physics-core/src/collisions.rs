//! Ball-ball collision (with throw) and Han-2005-lineage ball-cushion rebound
//! (spin/english coupled to outgoing angle and speed). Ported from the TS ref.

use crate::ball::Ball;
use crate::constants::*;
use crate::table::Cushion;
use crate::vec::Vec2;

/// Impulse-based ball-ball resolution along the line of centres plus a tangential
/// friction impulse modelling throw. Equal masses -> symmetric normal exchange.
pub fn resolve_ball_ball(a: &mut Ball, b: &mut Ball) {
    let n = b.pos.sub(a.pos).normalize();
    let t = n.perp();

    // Positional de-overlap: push apart by overlap/2 plus 1 mm clearance so
    // the next event scan sees gap > EPS and won't fire a t=0 event for this
    // pair again immediately.  1e-7 (0.2 nm) was too tight — float jitter
    // would re-trigger the t=0 guard on the very next iteration.
    let dist = b.pos.sub(a.pos).mag();
    let overlap = 2.0 * BALL_RADIUS - dist;
    if overlap > 0.0 {
        let push = n.scale(overlap / 2.0 + 0.001);
        a.pos = a.pos.sub(push);
        b.pos = b.pos.add(push);
    }

    let van = a.vel.dot(n);
    let vat = a.vel.dot(t);
    let vbn = b.vel.dot(n);
    let vbt = b.vel.dot(t);

    let approach = van - vbn;
    if approach <= 0.0 {
        return;
    }

    let e = E_BALL_BALL;
    let new_van = van - (1.0 + e) * 0.5 * approach;
    let new_vbn = vbn + (1.0 + e) * 0.5 * approach;

    // Throw: tangential friction impulse bounded by relative tangential surface
    // speed (which includes each ball's sidespin contribution).
    let surf_a_tan = vat + a.wz * BALL_RADIUS;
    let surf_b_tan = vbt - b.wz * BALL_RADIUS;
    let rel_tan = surf_a_tan - surf_b_tan;

    let normal_impulse = 0.5 * (1.0 + e) * approach;
    let max_fric = MU_BALL_BALL * normal_impulse;
    let desired = 0.5 * rel_tan.abs();
    let fric = rel_tan.signum() * max_fric.min(desired);

    let new_vat = vat - fric;
    let new_vbt = vbt + fric;

    a.vel = n.scale(new_van).add(t.scale(new_vat));
    b.vel = n.scale(new_vbn).add(t.scale(new_vbt));

    let spin_transfer = (fric / BALL_RADIUS) * 0.5;
    a.wz -= spin_transfer;
    b.wz += spin_transfer;
}

/// Han-2005-lineage cushion rebound: normal reverses with restitution; a rail
/// friction impulse acts on the raised-contact-point surface velocity (which
/// includes sidespin), producing spin-dependent rebound and post-hit spin.
pub fn resolve_ball_cushion(b: &mut Ball, c: &Cushion) {
    let n = c.normal;
    let t = n.perp();

    let vn = b.vel.dot(n);
    let vt = b.vel.dot(t);
    if vn >= 0.0 {
        return;
    }

    let height_coupling = CUSHION_HEIGHT_FRACTION;
    let new_vn = -E_BALL_CUSHION * vn;

    let surf_tan = vt + b.wz * BALL_RADIUS * height_coupling;
    let normal_impulse = (1.0 + E_BALL_CUSHION) * vn.abs();
    let max_fric = MU_BALL_CUSHION * normal_impulse;
    let fric = surf_tan.signum() * max_fric.min(surf_tan.abs());

    let new_vt = vt - fric;
    b.vel = n.scale(new_vn).add(t.scale(new_vt));
    b.wz -= (fric / BALL_RADIUS) * height_coupling;

    // Nudge clear of the rail to avoid immediate re-collision.
    b.pos = b.pos.add(n.scale(1e-6));
    let _ = Vec2::ZERO;
}
