//! Physical constants and table geometry (SI units), ported from the TypeScript
//! reference so the Rust play/rollout physics and the Python training sim
//! (pooltool, Han-2005 lineage) stay in parity. A 7-foot bar-box table with
//! regulation 2-1/4" balls.

pub const G: f64 = 9.8;

// --- Ball geometry / mass ---
pub const BALL_RADIUS: f64 = 0.028575;
pub const BALL_DIAMETER: f64 = BALL_RADIUS * 2.0;

// --- Friction / restitution coefficients ---
pub const MU_SLIDING: f64 = 0.2;
pub const MU_ROLLING: f64 = 0.01;
pub const MU_SPINNING: f64 = 0.044;

pub const E_BALL_BALL: f64 = 0.95;
pub const E_BALL_CUSHION: f64 = 0.85;
pub const MU_BALL_BALL: f64 = 0.06;
pub const MU_BALL_CUSHION: f64 = 0.2;

/// Cushion nose contacts the ball above its equator; this coupling fraction is
/// what produces the Han-2005 spin/rebound interaction.
pub const CUSHION_HEIGHT_FRACTION: f64 = 0.635;

// --- Table geometry (7ft bar box) ---
pub const TABLE_LENGTH: f64 = 1.9812;
pub const TABLE_WIDTH: f64 = 0.9906;

pub const CORNER_POCKET_RADIUS: f64 = 0.0605;
pub const SIDE_POCKET_RADIUS: f64 = 0.055;

// --- Stopping thresholds ---
pub const STOP_SPEED: f64 = 0.005;
pub const STOP_SPIN: f64 = 0.05;

// --- Cue action bounds (jump/masse unrepresentable: no elevation term at all) ---
pub const MAX_V0: f64 = 8.5;
pub const MAX_SIDE_SPIN: f64 = 25.0;
pub const MAX_ROLL_SPIN: f64 = 40.0;

pub const EPS: f64 = 1e-9;
