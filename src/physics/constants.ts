// Physical constants and table geometry.
//
// Values follow the pooltool / Han-2005 lineage (SI units: metres, seconds,
// kilograms) so that this in-house TypeScript engine stays close to the Python
// training simulator. A 7-foot "bar box" table is modelled: 7ft x 3.5ft playing
// surface, standard 2-1/4" (0.05715 m) balls. Coefficients are the standard
// billiards values used across the literature (Alciatore, Han 2005, pooltool
// defaults) rather than tuned magic numbers.

export const G = 9.8; // gravitational acceleration, m/s^2

// --- Ball geometry / mass ---
export const BALL_RADIUS = 0.028575; // m (2-1/4 inch diameter ball)
export const BALL_DIAMETER = BALL_RADIUS * 2;
export const BALL_MASS = 0.17; // kg, regulation pool ball (~6 oz)

// Moment of inertia of a solid sphere: I = (2/5) m r^2.
export const BALL_INERTIA = (2 / 5) * BALL_MASS * BALL_RADIUS * BALL_RADIUS;

// --- Friction / restitution coefficients ---
// Coefficient of sliding friction between ball and cloth. Governs how fast a
// sliding ball's surface velocity decays toward pure rolling.
export const MU_SLIDING = 0.2;
// Coefficient of rolling resistance. Much smaller than sliding friction;
// slowly bleeds speed from a purely rolling ball until it stops. Set above
// the textbook value (~0.01), which under this engine's deceleration model
// (a = MU_ROLLING * G applied directly to translational speed) would settle
// a medium-power shot only after several real seconds; this value settles
// an ordinary shot in a few seconds, matching a real table.
export const MU_ROLLING = 0.05;
// Spin (about vertical axis) decays through a separate spinning friction.
export const MU_SPINNING = 0.044;

// Ball-ball restitution (nearly elastic; pool balls are hard phenolic).
export const E_BALL_BALL = 0.95;
// Ball-cushion restitution. Cushions absorb more energy than a ball does.
export const E_BALL_CUSHION = 0.85;
// Coefficient of friction during a ball-ball impact — responsible for "throw"
// (transfer of tangential/spin effects between balls).
export const MU_BALL_BALL = 0.06;
// Coefficient of friction during a ball-cushion impact.
export const MU_BALL_CUSHION = 0.2;

// Cushion nose height above the table bed, expressed as a fraction of the ball
// radius. Regulation cushions contact the ball at ~63.5% of its height; this is
// the parameter that produces the Han-2005 vertical rebound coupling.
export const CUSHION_HEIGHT_FRACTION = 0.635;

// --- Table geometry (7ft bar box) ---
// Playing surface dimensions (cushion nose to cushion nose), metres.
export const TABLE_LENGTH = 1.9812; // 78 in
export const TABLE_WIDTH = 0.9906; // 39 in (length/2, regulation 2:1)

// Pocket radii. Corner pockets are wider than side pockets on a bar box.
export const CORNER_POCKET_RADIUS = 0.0605;
export const SIDE_POCKET_RADIUS = 0.055;

// A ball is considered stopped when its speed and spin fall below these floors.
export const STOP_SPEED = 0.005; // m/s
export const STOP_SPIN = 0.05; // rad/s

// Numerical guard for event-time comparisons.
export const EPS = 1e-9;
