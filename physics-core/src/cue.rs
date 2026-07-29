//! Cue action space. There is NO elevation (`theta`) parameter anywhere — the
//! struct simply has no field for it — so jump and massé are UNREPRESENTABLE,
//! not merely discouraged. V0 is capped. This is the "impossible at the action
//! level" guarantee from the spec, enforced by the type.

use crate::ball::{Ball, Motion};
use crate::constants::*;
use crate::vec::Vec2;

#[derive(Clone, Copy, Debug)]
pub struct CueAction {
    pub phi: f64,       // aim direction, radians
    pub power: f64,     // [0,1] -> V0 in [0, MAX_V0]
    pub side_spin: f64, // [-1,1] english ("a")
    pub top_spin: f64,  // [-1,1] draw(<0)/follow(>0) ("b")
}

/// Apply a cue action to the cue ball. Defensively clamps even though the type
/// already forbids elevation.
pub fn apply_cue(cue: &mut Ball, action: &CueAction) {
    let power = action.power.clamp(0.0, 1.0);
    let a = action.side_spin.clamp(-1.0, 1.0);
    let b = action.top_spin.clamp(-1.0, 1.0);

    let v0 = power * MAX_V0;
    let dir = Vec2::from_angle(action.phi, 1.0);

    cue.vel = dir.scale(v0);
    // Sidespin -> vertical-axis spin (english). Never elevation, so no masse.
    cue.wz = a * MAX_SIDE_SPIN;
    // Follow/draw seeds the roll relative to velocity.
    let roll_mag = (v0 / BALL_RADIUS) + b * MAX_ROLL_SPIN;
    cue.roll = dir.scale(roll_mag);
    cue.motion = if v0 > 0.0 {
        Motion::Sliding
    } else {
        Motion::Stationary
    };
}
