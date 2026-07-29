//! Closed-form per-phase trajectory integration and analytic time-to-phase-change
//! solvers. Ported from the TypeScript reference.

use crate::ball::*;
use crate::constants::*;
use crate::vec::Vec2;

/// Advance a single ball by `dt` along its current analytic trajectory. The
/// caller guarantees no collision happens within `dt`, so this moves the ball
/// exactly along the friction-decelerated path.
pub fn advance_ball(b: &mut Ball, dt: f64) {
    if b.pocketed || b.motion == Motion::Stationary {
        decay_spin(b, dt);
        return;
    }
    if b.motion == Motion::Spinning {
        decay_spin(b, dt);
        return;
    }

    let decel = linear_deceleration(b.motion);
    let speed = b.vel.mag();

    if b.motion == Motion::Sliding {
        let slip_dir = b.relative_surface_velocity().normalize();
        let a = slip_dir.scale(-decel);
        b.pos = b.pos.add(b.vel.scale(dt)).add(a.scale(0.5 * dt * dt));
        b.vel = b.vel.add(a.scale(dt));
        let target_roll = b.vel.scale(1.0 / BALL_RADIUS);
        b.roll = approach(b.roll, target_roll, decel * dt);
    } else {
        // Rolling
        if speed > 1e-12 {
            let dir = b.vel.normalize();
            let a = dir.scale(-decel);
            b.pos = b.pos.add(b.vel.scale(dt)).add(a.scale(0.5 * dt * dt));
            b.vel = b.vel.add(a.scale(dt));
            b.roll = b.vel.scale(1.0 / BALL_RADIUS);
        }
    }

    decay_spin(b, dt);
    clamp_stopped(b);
}

fn decay_spin(b: &mut Ball, dt: f64) {
    let sd = spin_deceleration();
    if b.wz > 0.0 {
        b.wz = (b.wz - sd * dt).max(0.0);
    } else if b.wz < 0.0 {
        b.wz = (b.wz + sd * dt).min(0.0);
    }
}

fn approach(from: Vec2, to: Vec2, max_step: f64) -> Vec2 {
    let d = to.sub(from);
    let dm = d.mag();
    if dm <= max_step || dm < 1e-12 {
        to
    } else {
        from.add(d.normalize().scale(max_step))
    }
}

fn clamp_stopped(b: &mut Ball) {
    if b.vel.mag() < STOP_SPEED && b.relative_surface_velocity().mag() < STOP_SPEED {
        b.vel = Vec2::ZERO;
        b.roll = Vec2::ZERO;
        if b.wz.abs() < STOP_SPIN {
            b.wz = 0.0;
        }
    }
}

/// Time until a ball's motion phase changes (slide->roll, roll/spin->stop), or
/// infinity if it never does within this phase.
pub fn time_to_phase_change(b: &Ball) -> f64 {
    if b.pocketed {
        return f64::INFINITY;
    }
    match b.motion {
        Motion::Sliding => {
            let decel = linear_deceleration(Motion::Sliding);
            let slip = b.relative_surface_velocity().mag();
            if decel > 0.0 {
                slip / decel
            } else {
                f64::INFINITY
            }
        }
        Motion::Rolling => {
            let decel = linear_deceleration(Motion::Rolling);
            let speed = b.vel.mag();
            if decel > 0.0 {
                speed / decel
            } else {
                f64::INFINITY
            }
        }
        Motion::Spinning => {
            let sd = spin_deceleration();
            if sd > 0.0 {
                b.wz.abs() / sd
            } else {
                f64::INFINITY
            }
        }
        Motion::Stationary => f64::INFINITY,
    }
}
