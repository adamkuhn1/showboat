import { type Ball, Motion, classifyMotion, cloneBall } from "../physics/ball";
import { type SimResult } from "../physics/engine";
import { type Table } from "../physics/table";
import { advanceBall, timeToPhaseChange } from "../physics/motion";
import { timeToBallBall, timeToCushion, timeToPocket } from "../physics/predict";
import { resolveBallBall, resolveBallCushion } from "../physics/collisions";

// Maximum simulated time per shot — same cap as the batch engine.
const MAX_SIM_TIME = 30;
// How far ahead to search for the next event per iteration (seconds). Kept
// deliberately small so no phase-transition slips between detection windows.
const LOOKAHEAD = 0.05;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/** Complete snapshot of all ball states at a specific simulation instant. */
export interface AnimWaypoint {
  simTime: number;
  balls: Ball[]; // deep-copied from the simulation state at this moment
}

/**
 * Precomputed animation track for a shot. Built synchronously once via
 * buildAnimTrack(), then replayed frame-by-frame with interpolateBalls() —
 * no real-time physics stepping, no guard loops, no tunnelling.
 */
export interface AnimTrack {
  waypoints: AnimWaypoint[]; // ordered by simTime; first waypoint is always t=0
  duration: number; // total simulation seconds until all balls settle
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

const cloneBalls = (balls: Ball[]): Ball[] => balls.map(cloneBall);

const reclassify = (balls: Ball[]): void => {
  for (const b of balls) {
    if (!b.pocketed) b.motion = classifyMotion(b);
  }
};

const anyMoving = (balls: Ball[]): boolean =>
  balls.some((b) => !b.pocketed && b.motion !== Motion.Stationary);

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Run the TS physics engine to completion on a COPY of `balls`, capturing a
 * waypoint snapshot after every resolved event and at every LOOKAHEAD
 * boundary. The logic mirrors simulateShot() in engine.ts with snapshot
 * capture woven in.
 *
 * This runs in < 1ms for a typical shot (the batch pass is fast; it's the
 * real-time incremental stepping in the old stepWorld that was slow and
 * unstable). The resulting waypoints are replayed smoothly by
 * interpolateBalls() with no real-time physics at all.
 *
 * Why this fixes the bugs in the old stepWorld approach:
 *  - The old stepWorld reused `bestT` as the search window for later event
 *    categories. After a t=0 collision set bestT=0, subsequent
 *    timeToBallBall(a, b, 0) searches could never satisfy tc<0, so
 *    simultaneous rack-ball cascade collisions were silently dropped. The
 *    batch engine here uses a FRESH `window` per iteration and correctly
 *    processes events one at a time in time order.
 *  - No frame-timing guard limit: the batch loop runs until the physics
 *    settles, not until a wall-clock budget expires. Events are never lost.
 */
export const buildAnimTrack = (balls: Ball[], table: Table): AnimTrack => {
  const working = cloneBalls(balls);
  reclassify(working);

  // First waypoint is the initial state (after reclassification).
  const waypoints: AnimWaypoint[] = [
    { simTime: 0, balls: cloneBalls(working) },
  ];
  let t = 0;

  while (anyMoving(working) && t < MAX_SIM_TIME) {
    // Each category searches the FULL window — not a progressively-narrowed
    // bestT like the old stepWorld did. The old code's bug: once a t=0 event
    // set bestT=0, subsequent timeToBallBall(a,b,0) calls used a zero window
    // and found nothing, silently dropping simultaneous rack cascade collisions.
    const window = LOOKAHEAD;

    // Two separate variables so TypeScript's narrowing works without closures
    // obstructing control-flow analysis.
    let bestTime: number = window; // starts at full window; narrows on each find
    let bestApply: (() => void) | null = null;

    const consider = (tc: number, apply: () => void): void => {
      if (tc < 0 || !isFinite(tc)) return;
      if (tc < bestTime) { bestTime = tc; bestApply = apply; }
    };

    // Phase-change events (slide→roll, roll→stop, spin→stop).
    for (const b of working) {
      const tp = timeToPhaseChange(b);
      if (tp < window) {
        consider(tp, () => {
          /* reclassify() below handles the transition; no explicit apply needed */
        });
      }
    }

    // Ball-ball collisions — always searched against the FULL `window`, not
    // against the already-narrowed `bestTime`. This means a phase-change at
    // tp<window does NOT blind the search for collisions between tp and window;
    // consider() will pick whichever is earliest.
    for (let i = 0; i < working.length; i++) {
      for (let j = i + 1; j < working.length; j++) {
        const a = working[i];
        const b = working[j];
        const tc = timeToBallBall(a, b, window);
        if (isFinite(tc)) consider(tc, () => resolveBallBall(a, b));
      }
    }

    // Ball-cushion collisions.
    for (const b of working) {
      for (const c of table.cushions) {
        const tc = timeToCushion(b, c, window);
        if (isFinite(tc)) consider(tc, () => resolveBallCushion(b, c));
      }
    }

    // Pocket captures.
    for (const b of working) {
      const p = timeToPocket(b, table, window);
      if (p && isFinite(p.t)) {
        consider(p.t, () => {
          b.pocketed = true;
          b.vel = { x: 0, y: 0 };
          b.roll = { x: 0, y: 0 };
          b.wz = 0;
          b.motion = Motion.Stationary;
        });
      }
    }

    // Advance to the earliest event (or the full window if none found).
    // bestTime equals `window` when no event was found (bestApply stays null).
    const step = Math.max(bestTime, 0);
    for (const b of working) advanceBall(b, step);
    t += step;

    // Use a typed const so TypeScript's narrowing accepts the call despite the
    // closure mutation on bestApply (same pattern as engine.ts uses for `best`).
    if (bestApply !== null) { const fn: () => void = bestApply; fn(); }
    reclassify(working);

    // Capture a waypoint after EVERY iteration — event or no-event window
    // advance. This guarantees no inter-waypoint interval spans a phase
    // transition, so advanceBall() is accurate across the full interval in
    // interpolateBalls().
    waypoints.push({ simTime: t, balls: cloneBalls(working) });

    // Guard: if the step was zero and nothing happened, something is stuck.
    if (step <= 0 && bestApply === null) break;
  }

  return { waypoints, duration: t };
};

/**
 * Given a precomputed AnimTrack and an elapsed simulation time (wall-clock
 * elapsed × speed factor), return ball states by finding the nearest prior
 * waypoint and advancing analytically from it using advanceBall.
 *
 * O(log n) binary search over waypoints + O(balls) for the advance.
 * Produces smooth, physically accurate motion with no real-time stepping.
 */
/**
 * When the last thing worth watching happened, in simulation seconds.
 *
 * Everything after it is balls coasting to rest — measured at 1.99 s of a 5.39 s
 * median shot, so it is not a rounding detail. Playback uses this to decide
 * where the presentation speed stops applying (`ui/playbackSpeed.ts`).
 *
 * Zero when the shot recorded no contact at all, which makes the whole thing
 * settle: a shot that touched nothing has no part worth slowing down for.
 */
export const lastContactSec = (sim: SimResult): number => {
  let t = 0;
  for (const e of sim.events) {
    if (e.kind === "ball-ball" || e.kind === "ball-cushion" || e.kind === "pocket") {
      if (e.time > t) t = e.time;
    }
  }
  return t;
};

export const interpolateBalls = (track: AnimTrack, simTime: number): Ball[] => {
  const clampedT = Math.min(simTime, track.duration);

  // Binary search for the last waypoint at or before clampedT.
  let lo = 0;
  let hi = track.waypoints.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (track.waypoints[mid].simTime <= clampedT) lo = mid;
    else hi = mid - 1;
  }
  const wp = track.waypoints[lo];
  const dt = clampedT - wp.simTime;

  if (dt <= 1e-9) return cloneBalls(wp.balls);

  // Advance from the waypoint. Each LOOKAHEAD-sized interval contains no
  // phase transitions (those generate their own waypoints), so advanceBall
  // is accurate across the full dt.
  const snapshot = cloneBalls(wp.balls);
  for (const b of snapshot) advanceBall(b, dt);
  return snapshot;
};
