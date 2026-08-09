// The wall-clock to simulation-time mapping, and the argument for keeping it
// boring.
//
// Playback consumes simulation seconds at a rate that does not change for the
// duration of a shot. Elapsed screen time `w` shows simulation time `speed * w`,
// end to end, so the mapping is affine and `pacing.test.ts` measures the
// residual against a straight line to prove it.
//
// WHY A SINGLE RATE IS THE ONLY CORRECT CHOICE HERE
//
// A shot is a MULTI-BODY scene. Any rate that varies with simulation time is a
// GLOBAL time warp: it is applied to every ball at once, but it is derived from
// events that belong to particular balls. Slow the clock down for a cushion
// contact and every other ball on the table decelerates with it, including a
// ball rolling alone across open felt that touched nothing. Nothing in the
// physical world can do that, so the eye reads it as the simulation being
// broken rather than as an editorial choice about pacing.
//
// Measured on 82 real shots at the 0.5x default, an earlier per-contact rate
// curve produced a 25x swing in playback rate WITHIN a single shot, ran the
// coast to rest at up to 5.68x real time, and left 32.7% of moving-ball frames
// showing a ball whose apparent speed changed while it had no contact of its
// own within 0.35 s either side. `qa/time-mapping.ts` is that measurement and
// reports the same numbers for the current code.
//
// A ball is therefore only ever seen slowing down because friction slowed it
// down, and only ever seen speeding up because it was hit.
//
// WHAT THIS IS NOT
//
// It is not physics. Nothing here can reach a physics constant, a candidate
// action, a simulation or the committed game state; those are all fixed before
// a frame is painted. This scales time on the way from wall-clock to simulation
// time and does nothing else.

/**
 * Simulation seconds to advance per wall-clock second, at a chosen speed.
 *
 * Constant for the whole shot, by the argument above. Takes a plain number so
 * this module keeps its distance from the React speed control.
 *
 * The animation loops integrate this incrementally (`simTime += dt * rate`)
 * rather than deriving `simTime` from a start timestamp. Both are equivalent
 * while the speed is fixed; integrating is what lets a speed change taken
 * mid-shot bend the rest of the playback instead of teleporting the balls to
 * where the new speed says they should already be.
 */
export const simulationRate = (speed: number): number => speed;

/**
 * Screen seconds a shot of `durationSec` simulation seconds will occupy.
 *
 * Trivial at a single rate, and kept as a named function because it is the
 * quantity the default speed is chosen against and reporting it from one place
 * keeps `qa/time-mapping.ts` and the tests from restating the mapping.
 */
export const screenSeconds = (durationSec: number, speed: number): number =>
  speed > 0 ? durationSec / speed : 0;
