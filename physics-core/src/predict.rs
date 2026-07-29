//! Analytic time-to-next-event solvers. Each ball follows a known accelerated
//! path within a phase, so the next geometric event is a root of a low-order
//! distance function; we find it with a guarded scan + bisection on the exact
//! distance (stable and auditable, still "solve for the event time" rather than
//! a blind fixed-step march). Ported from the TS reference.

use crate::ball::*;
use crate::constants::*;
use crate::table::Table;
use crate::vec::Vec2;

fn pos_at(b: &Ball, t: f64) -> Vec2 {
    if b.pocketed || b.motion == Motion::Stationary || b.motion == Motion::Spinning {
        return b.pos;
    }
    let decel = linear_deceleration(b.motion);
    let dir = if b.motion == Motion::Sliding {
        b.relative_surface_velocity().normalize()
    } else if b.vel.mag() > 1e-12 {
        b.vel.normalize()
    } else {
        Vec2::ZERO
    };
    let a = dir.scale(-decel);
    b.pos.add(b.vel.scale(t)).add(a.scale(0.5 * t * t))
}

/// Earliest t in (0, t_max] where f crosses from positive to <= 0, via a coarse
/// scan then bisection refine. Returns infinity if no crossing found.
fn earliest_zero<F: Fn(f64) -> f64>(f: F, t_max: f64, samples: usize) -> f64 {
    if t_max <= 0.0 || !t_max.is_finite() {
        return f64::INFINITY;
    }
    let dt = t_max / samples as f64;
    let mut prev_t = 0.0;
    let mut prev_f = f(0.0);
    for i in 1..=samples {
        let t = i as f64 * dt;
        let cur = f(t);
        if prev_f > 0.0 && cur <= 0.0 {
            let mut lo = prev_t;
            let mut hi = t;
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if f(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return 0.5 * (lo + hi);
        }
        prev_t = t;
        prev_f = cur;
    }
    f64::INFINITY
}

/// Time until two balls first touch (centre distance == diameter) within t_max.
pub fn time_to_ball_ball(a: &Ball, b: &Ball, t_max: f64) -> f64 {
    if a.pocketed || b.pocketed {
        return f64::INFINITY;
    }
    if a.motion == Motion::Stationary && b.motion == Motion::Stationary {
        return f64::INFINITY;
    }
    // Broad-phase reject.
    let sep = b.pos.sub(a.pos).mag();
    let reach = (a.vel.mag() + b.vel.mag()) * t_max;
    if sep - reach > BALL_DIAMETER {
        return f64::INFINITY;
    }
    let f = |t: f64| pos_at(b, t).sub(pos_at(a, t)).mag() - BALL_DIAMETER;
    if f(0.0) <= EPS {
        let closing = b.vel.sub(a.vel).dot(b.pos.sub(a.pos).normalize()) < 0.0;
        return if closing { 0.0 } else { f64::INFINITY };
    }
    earliest_zero(f, t_max, 64)
}

/// Time until a ball's surface touches a cushion within t_max.
pub fn time_to_cushion(b: &Ball, c: &crate::table::Cushion, t_max: f64) -> f64 {
    if b.pocketed || b.motion == Motion::Stationary || b.motion == Motion::Spinning {
        return f64::INFINITY;
    }
    let f = |t: f64| {
        let p = pos_at(b, t);
        let signed = if c.axis_x {
            (p.x - c.pos) * c.normal.x
        } else {
            (p.y - c.pos) * c.normal.y
        };
        signed - BALL_RADIUS
    };
    if f(0.0) <= EPS {
        let vn = b.vel.dot(c.normal);
        return if vn < 0.0 { 0.0 } else { f64::INFINITY };
    }
    earliest_zero(f, t_max, 64)
}

/// Earliest pocket capture (centre entering jaw radius) within t_max.
pub fn time_to_pocket(b: &Ball, table: &Table, t_max: f64) -> Option<(f64, u8)> {
    if b.pocketed || b.motion == Motion::Stationary || b.motion == Motion::Spinning {
        return None;
    }
    let mut best: Option<(f64, u8)> = None;
    for pk in table.pockets.iter() {
        let f = |t: f64| pos_at(b, t).sub(pk.center).mag() - pk.radius;
        let t = if f(0.0) <= 0.0 { 0.0 } else { earliest_zero(f, t_max, 64) };
        if t.is_finite() && best.map_or(true, |(bt, _)| t < bt) {
            best = Some((t, pk.id));
        }
    }
    best
}
