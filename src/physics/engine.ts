import { type Ball, Motion, classifyMotion, cloneBall } from "./ball";
import { type Table } from "./table";
import { advanceBall, timeToPhaseChange } from "./motion";
import { timeToBallBall, timeToCushion, timeToPocket } from "./predict";
import { resolveBallBall, resolveBallCushion } from "./collisions";
import { STOP_SPEED } from "./constants";

// An entry in the shot's event trace. This is the raw material the reasoning
// overlay turns into captions like "cue -> rail -> 3-ball -> corner". It is a
// byproduct of the real simulation, not decoration.
export type ShotEventKind =
  | "ball-ball"
  | "ball-cushion"
  | "pocket"
  | "phase"
  | "stop";

export interface ShotEvent {
  time: number; // seconds from shot start
  kind: ShotEventKind;
  balls: number[]; // ball ids involved
  cushion?: string; // cushion side, for ball-cushion
  pocket?: string; // pocket id, for pocket
}

// A sampled keyframe of every ball's position at a simulation time. Frames are
// recorded by the ONE authoritative simulation as it runs (at every internal
// event/window boundary, so never more than LOOKAHEAD seconds apart). Playback
// interpolates between these frames — it never re-simulates the shot, so what
// is animated is by construction the exact simulation that was resolved.
// (Worst-case linear-interpolation error between 50 ms keyframes under sliding
// friction is a*dt^2/8 ≈ 0.6 mm — invisible at canvas scale.)
export interface Frame {
  t: number; // seconds from shot start
  balls: { id: number; x: number; y: number; pocketed: boolean }[];
}

export interface SimResult {
  balls: Ball[]; // final resting state
  events: ShotEvent[]; // ordered event trace
  pocketed: number[]; // ball ids pocketed during the shot, in order
  firstContact: number | null; // id of first object ball the cue ball hit
  duration: number; // simulated seconds
  // Present only when the caller asked for playback frames (the interactive
  // game does; search/training rollouts skip them to stay allocation-light).
  frames?: Frame[];
}

export interface SimOptions {
  recordFrames?: boolean;
}

// Maximum simulated time for a single shot; a real shot settles in a few
// seconds, this guards against numeric non-termination.
const MAX_SIM_TIME = 30;
// Cap on how far ahead we scan for the next event in one search window. Events
// are found by scanning [0, LOOKAHEAD]; if none, we integrate that far and
// rescan. Keeps the polynomial sampling dense enough to catch fast contacts.
const LOOKAHEAD = 0.05;
// Hard cap on event-loop iterations, independent of simulated time. If a
// resolved collision ever left two balls still within collision distance,
// the same event would be found again at essentially t=0, `step` would stay
// ~0, and `t` would never advance far enough to hit MAX_SIM_TIME — an
// unbounded loop that a purely time-based guard can't catch. A normal shot
// resolves in well under 1,000 iterations, so this is slack for a real
// break-like cascade, not a budget anything legitimate should approach.
const MAX_ITERATIONS = 20_000;

interface Candidate {
  t: number;
  apply: () => void;
  event: ShotEvent;
}

// Recompute every ball's motion phase. Called after each resolved event.
const reclassify = (balls: Ball[]): void => {
  for (const b of balls) {
    if (!b.pocketed) b.motion = classifyMotion(b);
  }
};

const anyMoving = (balls: Ball[]): boolean =>
  balls.some((b) => !b.pocketed && b.motion !== Motion.Stationary);

// Run one shot to completion. `balls` is mutated to the resting state; the
// function also returns a structured result including the event trace.
export const simulateShot = (
  balls: Ball[],
  table: Table,
  opts: SimOptions = {},
): SimResult => {
  const events: ShotEvent[] = [];
  const pocketed: number[] = [];
  let firstContact: number | null = null;
  let t = 0;

  const frames: Frame[] | undefined = opts.recordFrames ? [] : undefined;
  const snapshot = (): void => {
    if (!frames) return;
    frames.push({
      t,
      balls: balls.map((b) => ({
        id: b.id,
        x: b.pos.x,
        y: b.pos.y,
        pocketed: b.pocketed,
      })),
    });
  };

  reclassify(balls);
  snapshot();

  let iterations = 0;
  while (anyMoving(balls) && t < MAX_SIM_TIME) {
    iterations++;
    if (iterations > MAX_ITERATIONS) {
      // Force a settle rather than spin forever; see MAX_ITERATIONS above.
      for (const b of balls) {
        if (!b.pocketed) {
          b.vel = { x: 0, y: 0 };
          b.roll = { x: 0, y: 0 };
          b.wz = 0;
          b.motion = Motion.Stationary;
        }
      }
      break;
    }
    // Find the earliest event across all active balls within the lookahead.
    let best: Candidate | null = null;
    const window = LOOKAHEAD;

    const consider = (c: Candidate): void => {
      if (c.t < 0) return;
      if (best === null || c.t < best.t) best = c;
    };

    // Phase-change events (slide->roll, roll->stop, spin->stop).
    for (const b of balls) {
      const tp = timeToPhaseChange(b);
      if (tp < window) {
        consider({
          t: tp,
          event: { time: t + tp, kind: "phase", balls: [b.id] },
          apply: () => {
            // After advancing, reclassify handles the transition; nothing else.
          },
        });
      }
    }

    // Ball-ball collisions.
    for (let i = 0; i < balls.length; i++) {
      for (let j = i + 1; j < balls.length; j++) {
        const a = balls[i];
        const b = balls[j];
        const tc = timeToBallBall(a, b, window);
        if (isFinite(tc)) {
          consider({
            t: tc,
            event: { time: t + tc, kind: "ball-ball", balls: [a.id, b.id] },
            apply: () => resolveBallBall(a, b),
          });
        }
      }
    }

    // Ball-cushion collisions.
    for (const b of balls) {
      for (const c of table.cushions) {
        const tc = timeToCushion(b, c, window);
        if (isFinite(tc)) {
          consider({
            t: tc,
            event: {
              time: t + tc,
              kind: "ball-cushion",
              balls: [b.id],
              cushion: c.side,
            },
            apply: () => resolveBallCushion(b, c),
          });
        }
      }
    }

    // Pocket capture.
    for (const b of balls) {
      const p = timeToPocket(b, table, window);
      if (p && isFinite(p.t)) {
        consider({
          t: p.t,
          event: {
            time: t + p.t,
            kind: "pocket",
            balls: [b.id],
            pocket: p.pocketId,
          },
          apply: () => {
            b.pocketed = true;
            b.vel = { x: 0, y: 0 };
            b.roll = { x: 0, y: 0 };
            b.wz = 0;
            b.motion = Motion.Stationary;
          },
        });
      }
    }

    // Advance to the earliest event (or the full window if none).
    const step: number = best !== null ? (best as Candidate).t : window;
    for (const b of balls) advanceBall(b, Math.max(step, 0));
    t += Math.max(step, 0);
    snapshot();

    if (best !== null) {
      const c: Candidate = best;
      c.apply();
      // Record first cue-ball contact for foul detection.
      if (
        c.event.kind === "ball-ball" &&
        c.event.balls.includes(0) &&
        firstContact === null
      ) {
        firstContact = c.event.balls.find((id) => id !== 0) ?? null;
      }
      if (c.event.kind === "pocket") {
        pocketed.push(c.event.balls[0]);
      }
      // Only record non-trivial phase events into the trace to keep it legible.
      if (c.event.kind !== "phase") events.push(c.event);
    }

    reclassify(balls);

    // Guard against a stuck state: if the earliest step is ~0 repeatedly the
    // reclassify+resolve above should have separated things; nudge time.
    if (step <= 0 && best === null) break;
  }

  // Final settle: zero out any residual sub-threshold drift so the resting
  // state is exact. The event loop terminates when no ball is classified as
  // moving, which can leave a ball with a velocity just under the stop floor.
  for (const b of balls) {
    if (b.pocketed) continue;
    if (Math.hypot(b.vel.x, b.vel.y) < STOP_SPEED) {
      b.vel = { x: 0, y: 0 };
      b.roll = { x: 0, y: 0 };
    }
  }

  events.push({ time: t, kind: "stop", balls: [] });

  return {
    balls,
    events,
    pocketed,
    firstContact,
    duration: t,
    frames,
  };
};

// Convenience: simulate on a copy, leaving the input untouched (used by search /
// candidate evaluation where we must not disturb the real world state).
export const simulateShotCopy = (
  balls: Ball[],
  table: Table,
  opts: SimOptions = {},
): SimResult => {
  const copy = balls.map(cloneBall);
  return simulateShot(copy, table, opts);
};

export const isSettled = (balls: Ball[]): boolean => {
  return balls.every(
    (b) => b.pocketed || Math.hypot(b.vel.x, b.vel.y) < STOP_SPEED,
  );
};
