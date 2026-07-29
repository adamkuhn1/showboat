//! Native unit tests, ported from the TypeScript reference spec. Run with
//! `cargo test` (no wasm toolchain needed).

use crate::ball::{Ball, Motion};
use crate::collisions::resolve_ball_ball;
use crate::constants::*;
use crate::cue::{apply_cue, CueAction};
use crate::engine::{simulate_shot, EventKind};
use crate::rollout::{rollout_value, Rng};
use crate::table::Table;
use crate::vec::Vec2;

fn moving(mut b: Ball, vx: f64, vy: f64) -> Ball {
    b.vel = Vec2::new(vx, vy);
    b.motion = Motion::Sliding;
    b
}

#[test]
fn stun_shot_transfers_momentum() {
    let mut cue = moving(Ball::new(0, 0.0, 0.0), 2.0, 0.0);
    let mut obj = Ball::new(1, BALL_DIAMETER, 0.0);
    resolve_ball_ball(&mut cue, &mut obj);
    assert!(obj.vel.x > 1.8, "object x vel {}", obj.vel.x);
    assert!(obj.vel.y.abs() < 0.05);
    assert!(cue.vel.x < 0.2, "cue x vel {}", cue.vel.x);
}

#[test]
fn head_on_conserves_linear_momentum() {
    let mut cue = moving(Ball::new(0, 0.0, 0.0), 3.0, 0.0);
    let mut obj = Ball::new(1, BALL_DIAMETER, 0.0);
    let before = cue.vel.x + obj.vel.x;
    resolve_ball_ball(&mut cue, &mut obj);
    let after = cue.vel.x + obj.vel.x;
    assert!((after - before).abs() < 1e-6);
}

#[test]
fn half_ball_cut_separates_near_ninety_degrees() {
    let mut cue = moving(Ball::new(0, 0.0, 0.0), 2.0, 0.0);
    let off = BALL_DIAMETER / std::f64::consts::SQRT_2;
    let mut obj = Ball::new(1, off, off);
    resolve_ball_ball(&mut cue, &mut obj);
    let cosang = cue.vel.dot(obj.vel) / (cue.vel.mag() * obj.vel.mag() + 1e-12);
    assert!(cosang.abs() < 0.26, "cos angle {}", cosang);
}

#[test]
fn straight_shot_pockets_in_corner() {
    let mut cue = Ball::new(0, 0.0, 0.0);
    let table = Table::bar_box();
    let corner = table.pockets.iter().find(|p| p.id == 3).unwrap(); // top-right
    let phi = (corner.center.y - 0.0).atan2(corner.center.x - 0.0);
    apply_cue(&mut cue, &CueAction { phi, power: 0.6, side_spin: 0.0, top_spin: 0.0 });
    let mut balls = vec![cue];
    let res = simulate_shot(&mut balls, &table);
    assert!(res.pocketed.contains(&0), "pocketed: {:?}", res.pocketed);
}

#[test]
fn cushion_rebound_records_event_and_returns_ball() {
    let mut cue = Ball::new(0, 0.0, 0.0);
    apply_cue(&mut cue, &CueAction { phi: 0.0, power: 0.3, side_spin: 0.0, top_spin: 0.0 });
    let mut balls = vec![cue];
    let table = Table::bar_box();
    let res = simulate_shot(&mut balls, &table);
    let hit = res
        .events
        .iter()
        .any(|e| e.kind == EventKind::BallCushion && e.cushion == "right");
    assert!(hit, "expected a right-rail cushion event");
    assert!(balls[0].vel.mag() < 0.01, "ball should be at rest");
}

#[test]
fn energy_never_increases_and_all_balls_settle() {
    let mut cue = Ball::new(0, 0.0, 0.0);
    apply_cue(&mut cue, &CueAction { phi: 0.7, power: 1.0, side_spin: 0.3, top_spin: 0.2 });
    let start_ke = 0.5 * cue.vel.mag().powi(2);
    let mut balls = vec![cue];
    let table = Table::bar_box();
    let res = simulate_shot(&mut balls, &table);
    let end_ke = 0.5 * balls[0].vel.mag().powi(2);
    for b in &balls {
        assert!(b.pocketed || b.vel.mag() < 0.01);
    }
    assert!(end_ke <= start_ke + 1e-9);
    let _ = res;
}

#[test]
fn no_resting_overlap_after_multiball() {
    let mut cue = Ball::new(0, -0.3, 0.0);
    apply_cue(&mut cue, &CueAction { phi: 0.0, power: 0.8, side_spin: 0.0, top_spin: 0.0 });
    let b1 = Ball::new(1, 0.2, 0.0);
    let b2 = Ball::new(2, 0.2 + BALL_DIAMETER * 1.02, 0.0);
    let mut balls = vec![cue, b1, b2];
    let table = Table::bar_box();
    simulate_shot(&mut balls, &table);
    for i in 0..balls.len() {
        for j in (i + 1)..balls.len() {
            if balls[i].pocketed || balls[j].pocketed {
                continue;
            }
            let d = balls[i].pos.sub(balls[j].pos).mag();
            assert!(d > BALL_DIAMETER - 1e-3, "overlap d={}", d);
        }
    }
}

#[test]
fn cue_action_has_no_elevation_and_caps_speed() {
    let mut cue = Ball::new(0, 0.0, 0.0);
    // power > 1 must be clamped; there is no elevation field to set at all.
    apply_cue(&mut cue, &CueAction { phi: 1.2, power: 5.0, side_spin: 1.0, top_spin: 1.0 });
    assert!(cue.vel.mag() <= MAX_V0 + 1e-6);
    assert!(cue.wz.is_finite());
}

#[test]
fn rng_is_deterministic() {
    let mut a = Rng::new(42);
    let mut b = Rng::new(42);
    for _ in 0..1000 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn rollout_value_is_finite_and_nonnegative() {
    // A cue ball plus one target near a corner; the baseline rollout must run
    // the hot loop without panicking and return a sane value.
    let cue = Ball::new(0, 0.0, 0.0);
    let target = Ball::new(1, 0.5, 0.2);
    let balls = vec![cue, target];
    let table = Table::bar_box();
    let action = CueAction { phi: 0.3, power: 0.6, side_spin: 0.0, top_spin: 0.0 };
    let v = rollout_value(&balls, &table, &action, &[1], 3, 8, 7);
    assert!(v.is_finite() && v >= 0.0, "rollout value {}", v);
}
