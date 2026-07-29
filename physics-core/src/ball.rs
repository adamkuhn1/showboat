//! Ball state and motion-phase classification. Within a single motion phase a
//! ball follows a closed-form trajectory, which is what lets the engine solve
//! analytically for the next event instead of stepping blindly.

use crate::constants::*;
use crate::vec::Vec2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    Stationary,
    Sliding,
    Rolling,
    Spinning,
}

#[derive(Clone, Copy, Debug)]
pub struct Ball {
    pub id: u8,
    pub pos: Vec2,
    pub vel: Vec2,
    /// Spin about the vertical axis (sidespin / english), rad/s.
    pub wz: f64,
    /// Effective rolling angular velocity mapped into the bed plane; encodes
    /// follow/draw and the roll that opposes sliding.
    pub roll: Vec2,
    pub motion: Motion,
    pub pocketed: bool,
}

impl Ball {
    pub fn new(id: u8, x: f64, y: f64) -> Self {
        Ball {
            id,
            pos: Vec2::new(x, y),
            vel: Vec2::ZERO,
            wz: 0.0,
            roll: Vec2::ZERO,
            motion: Motion::Stationary,
            pocketed: false,
        }
    }

    /// Velocity of the ball's contact point with the cloth. Sliding friction
    /// acts opposite this vector.
    #[inline]
    pub fn relative_surface_velocity(&self) -> Vec2 {
        self.vel.sub(self.roll.scale(BALL_RADIUS))
    }

    /// Classify current motion phase from kinematic state.
    pub fn classify(&self) -> Motion {
        if self.pocketed {
            return Motion::Stationary;
        }
        let speed = self.vel.mag();
        let slip = self.relative_surface_velocity().mag();
        let spin = self.wz.abs();

        if speed < STOP_SPEED && slip < STOP_SPEED {
            return if spin > STOP_SPIN {
                Motion::Spinning
            } else {
                Motion::Stationary
            };
        }
        if slip < STOP_SPEED {
            return Motion::Rolling;
        }
        Motion::Sliding
    }
}

/// Linear deceleration magnitude (m/s^2) for a motion phase.
#[inline]
pub fn linear_deceleration(motion: Motion) -> f64 {
    match motion {
        Motion::Sliding => MU_SLIDING * G,
        Motion::Rolling => MU_ROLLING * G,
        _ => 0.0,
    }
}

/// Sidespin decay under spinning friction: (5/2) * mu_sp * g / R, rad/s^2.
#[inline]
pub fn spin_deceleration() -> f64 {
    (5.0 / 2.0) * (MU_SPINNING * G) / BALL_RADIUS
}
