//! Event-based evolution engine: advances balls along analytic trajectories,
//! resolving the earliest event each step, and emits a real event trace (the raw
//! material for the reasoning overlay's "cue → rail → 3-ball → corner" captions).

use crate::ball::*;
use crate::collisions::*;
use crate::motion::*;
use crate::predict::*;
use crate::table::{cushion_side_str, Table};
use crate::vec::Vec2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    BallBall,
    BallCushion,
    Pocket,
    Stop,
}

#[derive(Clone, Debug)]
pub struct ShotEvent {
    pub time: f64,
    pub kind: EventKind,
    pub ball_a: i32,
    pub ball_b: i32, // -1 if not applicable
    pub cushion: &'static str, // "" if not applicable
    pub pocket: i32, // -1 if not applicable
}

#[derive(Clone, Debug)]
pub struct Waypoint {
    pub time: f64,
    pub balls: Vec<Ball>,
}

#[derive(Clone, Debug)]
pub struct SimResult {
    pub events: Vec<ShotEvent>,
    pub pocketed: Vec<u8>,
    pub first_contact: i32, // -1 if none
    pub duration: f64,
    /// Full ball-state snapshots at every resolved step, captured only when
    /// requested (see `simulate_shot`'s `capture_waypoints` flag). Lets the
    /// caller replay the EXACT trajectory that produced `pocketed`/the final
    /// resting state, instead of re-simulating separately for animation and
    /// risking the two runs diverging (see render/animate.ts's removal note).
    pub waypoints: Vec<Waypoint>,
}

const MAX_SIM_TIME: f64 = 8.0; // 8 s simulated ≫ any realistic shot; prevents runaway sims
const LOOKAHEAD: f64 = 0.05;
const MAX_ITERATIONS: u32 = 500; // hard guard; a realistic shot needs ≤~200 iterations at LOOKAHEAD=0.05

/// Multi-pass positional separation: push overlapping pairs apart until all
/// gaps ≥ 2 mm.  Mirrors the JS `separateOverlaps` used on the bridge side.
/// Called at the start of `simulate_shot` so the event-detection scan never
/// sees a pair at contact distance before a collision has been resolved.
fn separate_overlaps(balls: &mut [Ball]) {
    let min_dist = crate::constants::BALL_DIAMETER + 0.002; // 2 mm clearance
    for _ in 0..20 {
        let mut any = false;
        for i in 0..balls.len() {
            for j in (i + 1)..balls.len() {
                if balls[i].pocketed || balls[j].pocketed {
                    continue;
                }
                let d = balls[j].pos.sub(balls[i].pos);
                let dist = d.mag();
                if dist < min_dist {
                    any = true;
                    let overlap = min_dist - dist;
                    let n = if dist > 1e-9 {
                        d.scale(1.0 / dist)
                    } else {
                        Vec2::new(1.0, 0.0)
                    };
                    let push = n.scale(overlap / 2.0);
                    balls[i].pos = balls[i].pos.sub(push);
                    balls[j].pos = balls[j].pos.add(push);
                }
            }
        }
        if !any {
            break;
        }
    }
}

fn reclassify(balls: &mut [Ball]) {
    for b in balls.iter_mut() {
        if !b.pocketed {
            b.motion = b.classify();
        }
    }
}

fn any_moving(balls: &[Ball]) -> bool {
    balls
        .iter()
        .any(|b| !b.pocketed && b.motion != Motion::Stationary)
}

enum Action {
    None,
    BallBall(usize, usize),
    Cushion(usize, usize), // ball index, cushion index
    Pocket(usize, u8),
}

/// Run one shot to completion, mutating `balls` to the resting state.
///
/// `capture_waypoints` records a full ball-state snapshot after every resolved
/// step (event or LOOKAHEAD-bounded advance) — the same granularity
/// render/animate.ts used to build separately in TS. Only the single real
/// shot the player/AI actually takes needs this (see wasm.rs); the MCTS
/// rollout hot loop calls this thousands of times per turn and must not pay
/// for snapshot allocation it never uses.
pub fn simulate_shot(balls: &mut Vec<Ball>, table: &Table, capture_waypoints: bool) -> SimResult {
    let mut events: Vec<ShotEvent> = Vec::new();
    let mut pocketed: Vec<u8> = Vec::new();
    let mut first_contact: i32 = -1;
    let mut t = 0.0;
    let mut iters: u32 = 0;
    let mut zero_step_count: u32 = 0;
    let mut waypoints: Vec<Waypoint> = Vec::new();

    // Ensure no balls start at contact distance: the JS bridge applies 2 mm
    // clearance too, but round-trips through Float64Array can shave a few nm.
    separate_overlaps(balls);
    reclassify(balls);
    if capture_waypoints {
        waypoints.push(Waypoint { time: 0.0, balls: balls.clone() });
    }

    while any_moving(balls) && t < MAX_SIM_TIME {
        iters += 1;
        if iters >= MAX_ITERATIONS {
            break;
        }
        let window = LOOKAHEAD;
        let mut best_t = window;
        let mut best_action = Action::None;
        let mut best_is_event = false;
        let mut best_kind = EventKind::Stop;

        // Phase changes (do not create trace events, just bound the step).
        for b in balls.iter() {
            let tp = time_to_phase_change(b);
            if tp < best_t {
                best_t = tp;
                best_action = Action::None;
                best_is_event = false;
            }
        }

        // Ball-ball.
        for i in 0..balls.len() {
            for j in (i + 1)..balls.len() {
                let tc = time_to_ball_ball(&balls[i], &balls[j], best_t);
                if tc.is_finite() && tc < best_t {
                    best_t = tc;
                    best_action = Action::BallBall(i, j);
                    best_is_event = true;
                    best_kind = EventKind::BallBall;
                }
            }
        }

        // Ball-cushion.
        for i in 0..balls.len() {
            for (ci, c) in table.cushions.iter().enumerate() {
                let tc = time_to_cushion(&balls[i], c, best_t);
                if tc.is_finite() && tc < best_t {
                    best_t = tc;
                    best_action = Action::Cushion(i, ci);
                    best_is_event = true;
                    best_kind = EventKind::BallCushion;
                }
            }
        }

        // Pockets.
        for i in 0..balls.len() {
            if let Some((tp, pid)) = time_to_pocket(&balls[i], table, best_t) {
                if tp.is_finite() && tp < best_t {
                    best_t = tp;
                    best_action = Action::Pocket(i, pid);
                    best_is_event = true;
                    best_kind = EventKind::Pocket;
                }
            }
        }

        let step = best_t.max(0.0);
        for b in balls.iter_mut() {
            advance_ball(b, step);
        }
        t += step;

        if best_is_event {
            match best_action {
                Action::BallBall(i, j) => {
                    let (a_id, b_id) = (balls[i].id as i32, balls[j].id as i32);
                    // Split borrow for the two mutable balls.
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    let (left, right) = balls.split_at_mut(hi);
                    resolve_ball_ball(&mut left[lo], &mut right[0]);
                    // After resolving a ball-ball collision, nearby balls in a
                    // cluster may now be overlapping (the resolved ball was pushed
                    // into a neighbour). Run a full separation pass so the next
                    // event scan sees no t=0 pairs and the cascade is cut at zero cost.
                    separate_overlaps(balls);
                    if first_contact == -1 && (a_id == 0 || b_id == 0) {
                        first_contact = if a_id == 0 { b_id } else { a_id };
                    }
                    events.push(ShotEvent {
                        time: t,
                        kind: EventKind::BallBall,
                        ball_a: a_id,
                        ball_b: b_id,
                        cushion: "",
                        pocket: -1,
                    });
                }
                Action::Cushion(i, ci) => {
                    let side = cushion_side_str(table.cushions[ci].side);
                    resolve_ball_cushion(&mut balls[i], &table.cushions[ci]);
                    events.push(ShotEvent {
                        time: t,
                        kind: EventKind::BallCushion,
                        ball_a: balls[i].id as i32,
                        ball_b: -1,
                        cushion: side,
                        pocket: -1,
                    });
                }
                Action::Pocket(i, pid) => {
                    balls[i].pocketed = true;
                    balls[i].vel = Vec2::ZERO;
                    balls[i].roll = Vec2::ZERO;
                    balls[i].wz = 0.0;
                    balls[i].motion = Motion::Stationary;
                    pocketed.push(balls[i].id);
                    events.push(ShotEvent {
                        time: t,
                        kind: EventKind::Pocket,
                        ball_a: balls[i].id as i32,
                        ball_b: -1,
                        cushion: "",
                        pocket: pid as i32,
                    });
                    let _ = best_kind;
                }
                Action::None => {}
            }
        }

        reclassify(balls);

        // Capture a waypoint after EVERY iteration — event or no-event window
        // advance — mirroring animate.ts's buildAnimTrack exactly, so replaying
        // these with the same advanceBall/interpolation logic on the JS side
        // never spans a phase transition it wasn't captured across.
        if capture_waypoints {
            waypoints.push(Waypoint { time: t, balls: balls.clone() });
        }

        if step <= 0.0 {
            if !best_is_event {
                break; // no progress and no event: fully stationary
            }
            zero_step_count += 1;
            if zero_step_count > 200 {
                break; // t=0 event cascade: too many consecutive zero-advance steps
            }
        } else {
            zero_step_count = 0;
        }
    }

    // Final settle.
    for b in balls.iter_mut() {
        if !b.pocketed && b.vel.mag() < crate::constants::STOP_SPEED {
            b.vel = Vec2::ZERO;
            b.roll = Vec2::ZERO;
        }
    }

    // Final de-overlap pass: an event step can leave two resting balls touching
    // with a sub-millimetre interpenetration that never triggers another
    // resolve. Separate any remaining overlaps symmetrically so the resting
    // state is physically valid (no ball inside another). A few relaxation
    // iterations converge for the small overlaps that occur here.
    for _ in 0..4 {
        for i in 0..balls.len() {
            for j in (i + 1)..balls.len() {
                if balls[i].pocketed || balls[j].pocketed {
                    continue;
                }
                let d = balls[j].pos.sub(balls[i].pos);
                let dist = d.mag();
                let overlap = crate::constants::BALL_DIAMETER - dist;
                if overlap > 1e-9 {
                    let n = if dist > 1e-12 { d.normalize() } else { Vec2::new(1.0, 0.0) };
                    let push = n.scale(overlap / 2.0 + 1e-7);
                    balls[i].pos = balls[i].pos.sub(push);
                    balls[j].pos = balls[j].pos.add(push);
                }
            }
        }
    }

    events.push(ShotEvent {
        time: t,
        kind: EventKind::Stop,
        ball_a: -1,
        ball_b: -1,
        cushion: "",
        pocket: -1,
    });

    // Final settle + de-overlap ran after the last in-loop waypoint capture;
    // replace that waypoint (same `t`) with the truly-final state so the last
    // frame of a replayed animation matches `balls` — the value this function
    // actually returns — exactly.
    if capture_waypoints {
        if let Some(last) = waypoints.last_mut() {
            if last.time == t {
                last.balls = balls.clone();
            } else {
                waypoints.push(Waypoint { time: t, balls: balls.clone() });
            }
        } else {
            waypoints.push(Waypoint { time: t, balls: balls.clone() });
        }
    }

    SimResult {
        events,
        pocketed,
        first_contact,
        duration: t,
        waypoints,
    }
}
