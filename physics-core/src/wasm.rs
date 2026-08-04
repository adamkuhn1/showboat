//! Thin wasm-bindgen boundary between the Rust physics/rollout core and the
//! TypeScript layer (canvas render, rules, ONNX, overlay).
//!
//! Ball state crosses the boundary as a flat `Float64Array` with a fixed stride
//! so we avoid per-ball object allocation on the JS side — important because the
//! render loop reads state every frame and search writes it thousands of times.
//!
//! Layout per ball (stride = 8):
//!   [id, pos.x, pos.y, vel.x, vel.y, wz, roll.x, roll.y]
//! `pocketed` is derived on the JS side from a sentinel (pocketed balls report
//! NaN position) and also returned explicitly by shot results.

use wasm_bindgen::prelude::*;

use crate::ball::{Ball, Motion};
use crate::cue::CueAction;
use crate::engine::{simulate_shot, EventKind};
use crate::rollout::rollout_value;
use crate::table::Table;
use crate::vec::Vec2;

pub const STRIDE: usize = 8;

fn balls_from_flat(flat: &[f64]) -> Vec<Ball> {
    let mut balls = Vec::with_capacity(flat.len() / STRIDE);
    let mut i = 0;
    while i + STRIDE <= flat.len() {
        let id = flat[i] as u8;
        let px = flat[i + 1];
        let pocketed = px.is_nan();
        let mut b = Ball {
            id,
            pos: Vec2::new(if pocketed { 0.0 } else { px }, flat[i + 2]),
            vel: Vec2::new(flat[i + 3], flat[i + 4]),
            wz: flat[i + 5],
            roll: Vec2::new(flat[i + 6], flat[i + 7]),
            motion: Motion::Stationary,
            pocketed,
        };
        b.motion = b.classify();
        balls.push(b);
        i += STRIDE;
    }
    balls
}

fn balls_to_flat(balls: &[Ball]) -> Vec<f64> {
    let mut out = Vec::with_capacity(balls.len() * STRIDE);
    for b in balls {
        out.push(b.id as f64);
        out.push(if b.pocketed { f64::NAN } else { b.pos.x });
        out.push(b.pos.y);
        out.push(b.vel.x);
        out.push(b.vel.y);
        out.push(b.wz);
        out.push(b.roll.x);
        out.push(b.roll.y);
    }
    out
}

/// Result of a single shot returned to JS. Events are packed into a flat i32
/// array (kind, ball_a, ball_b, pocket) plus a parallel f64 time array so the
/// overlay can reconstruct the trace without a struct-of-arrays dance in JS.
#[wasm_bindgen]
pub struct ShotResult {
    balls: Vec<f64>,
    pocketed: Vec<u8>,
    event_kinds: Vec<i32>,
    event_balls: Vec<i32>, // pairs: [a0,b0,a1,b1,...]
    event_pockets: Vec<i32>,
    event_cushions: Vec<String>,
    event_times: Vec<f64>,
    first_contact: i32,
    duration: f64,
    // Full trajectory for animation playback: `waypoint_times[i]` pairs with
    // the flat (STRIDE=8) ball block at `waypoint_balls[i * ball_count *
    // STRIDE .. (i+1) * ball_count * STRIDE]`. This is the same simulation
    // run that produced `balls`/`pocketed` above — replaying it is what
    // guarantees the animation ends exactly where the authoritative result
    // says it does, instead of a second, independently-run TS simulation
    // that can diverge over a long collision cascade (see App.tsx's removed
    // glideToAuthoritative correction).
    waypoint_times: Vec<f64>,
    waypoint_balls: Vec<f64>,
    ball_count: usize,
}

fn kind_code(k: EventKind) -> i32 {
    match k {
        EventKind::BallBall => 0,
        EventKind::BallCushion => 1,
        EventKind::Pocket => 2,
        EventKind::Stop => 3,
    }
}

#[wasm_bindgen]
impl ShotResult {
    #[wasm_bindgen(getter)]
    pub fn balls(&self) -> Vec<f64> {
        self.balls.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn pocketed(&self) -> Vec<u8> {
        self.pocketed.clone()
    }
    #[wasm_bindgen(getter, js_name = eventKinds)]
    pub fn event_kinds(&self) -> Vec<i32> {
        self.event_kinds.clone()
    }
    #[wasm_bindgen(getter, js_name = eventBalls)]
    pub fn event_balls(&self) -> Vec<i32> {
        self.event_balls.clone()
    }
    #[wasm_bindgen(getter, js_name = eventPockets)]
    pub fn event_pockets(&self) -> Vec<i32> {
        self.event_pockets.clone()
    }
    #[wasm_bindgen(getter, js_name = eventCushions)]
    pub fn event_cushions(&self) -> Vec<JsValue> {
        self.event_cushions
            .iter()
            .map(|s| JsValue::from_str(s))
            .collect()
    }
    #[wasm_bindgen(getter, js_name = eventTimes)]
    pub fn event_times(&self) -> Vec<f64> {
        self.event_times.clone()
    }
    #[wasm_bindgen(getter, js_name = firstContact)]
    pub fn first_contact(&self) -> i32 {
        self.first_contact
    }
    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> f64 {
        self.duration
    }
    #[wasm_bindgen(getter, js_name = waypointTimes)]
    pub fn waypoint_times(&self) -> Vec<f64> {
        self.waypoint_times.clone()
    }
    #[wasm_bindgen(getter, js_name = waypointBalls)]
    pub fn waypoint_balls(&self) -> Vec<f64> {
        self.waypoint_balls.clone()
    }
    #[wasm_bindgen(getter, js_name = ballCount)]
    pub fn ball_count(&self) -> usize {
        self.ball_count
    }
}

/// Simulate one shot to its resting state. `flat` is the ball state; the cue
/// action is applied to ball id 0. Returns the final state + event trace.
#[wasm_bindgen(js_name = simulateShot)]
pub fn simulate_shot_wasm(
    flat: &[f64],
    phi: f64,
    power: f64,
    side_spin: f64,
    top_spin: f64,
) -> ShotResult {
    let mut balls = balls_from_flat(flat);
    let table = Table::bar_box();
    if let Some(cue) = balls.iter_mut().find(|b| b.id == 0) {
        crate::cue::apply_cue(
            cue,
            &CueAction { phi, power, side_spin, top_spin },
        );
    }
    let ball_count = balls.len();
    let res = simulate_shot(&mut balls, &table, true);

    let mut event_kinds = Vec::new();
    let mut event_balls = Vec::new();
    let mut event_pockets = Vec::new();
    let mut event_cushions = Vec::new();
    let mut event_times = Vec::new();
    for e in &res.events {
        event_kinds.push(kind_code(e.kind));
        event_balls.push(e.ball_a);
        event_balls.push(e.ball_b);
        event_pockets.push(e.pocket);
        event_cushions.push(e.cushion.to_string());
        event_times.push(e.time);
    }

    let mut waypoint_times = Vec::with_capacity(res.waypoints.len());
    let mut waypoint_balls = Vec::with_capacity(res.waypoints.len() * ball_count * STRIDE);
    for wp in &res.waypoints {
        waypoint_times.push(wp.time);
        waypoint_balls.extend(balls_to_flat(&wp.balls));
    }

    ShotResult {
        balls: balls_to_flat(&balls),
        pocketed: res.pocketed,
        event_kinds,
        event_balls,
        event_pockets,
        event_cushions,
        event_times,
        first_contact: res.first_contact,
        duration: res.duration,
        waypoint_times,
        waypoint_balls,
        ball_count,
    }
}

/// Evaluate a candidate shot with the native rollout hot loop. Returns the mean
/// target-balls-pocketed value across `n_rollouts` playouts of `depth` shots.
/// This is the pure-search baseline value the MCTS uses before the ONNX net
/// supplies a learned value.
#[wasm_bindgen(js_name = rolloutValue)]
#[allow(clippy::too_many_arguments)]
pub fn rollout_value_wasm(
    flat: &[f64],
    phi: f64,
    power: f64,
    side_spin: f64,
    top_spin: f64,
    targets: &[u8],
    depth: u32,
    n_rollouts: u32,
    seed: f64,
) -> f64 {
    let balls = balls_from_flat(flat);
    let table = Table::bar_box();
    let action = CueAction { phi, power, side_spin, top_spin };
    rollout_value(
        &balls, &table, &action, targets, depth, n_rollouts, seed as u64,
    )
}

/// Table geometry constants exposed so the TS renderer/overlay never hardcodes
/// numbers that could drift from the physics core.
#[wasm_bindgen(js_name = tableDims)]
pub fn table_dims() -> Vec<f64> {
    let t = Table::bar_box();
    let mut v = vec![t.length, t.width];
    for p in t.pockets.iter() {
        v.push(p.center.x);
        v.push(p.center.y);
        v.push(p.radius);
    }
    v
}
