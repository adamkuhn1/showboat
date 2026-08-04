import { type Ball, Motion, classifyMotion, cloneBall } from "./ball";
import { type Table } from "./table";
import { advanceBall, timeToPhaseChange } from "./motion";
import { timeToBallBall, timeToCushion, timeToPocket } from "./predict";
import { resolveBallBall, resolveBallCushion } from "./collisions";
import { STOP_SPEED, BALL_RADIUS } from "./constants";

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

export interface SimWaypoint {
  time: number;
  balls: Ball[];
}

export interface SimResult {
  balls: Ball[]; // final resting state
  events: ShotEvent[]; // ordered event trace
  pocketed: number[]; // ball ids pocketed during the shot, in order
  firstContact: number | null; // id of first object ball the cue ball hit
  duration: number; // simulated seconds
  /**
   * Full ball-state snapshots across the shot, only populated by the WASM
   * simulator (simulateShotWasm) for the single real shot the player/AI
   * takes — never by this TS reference engine, and never for UCB search rollouts.
   * When present, this is what the UI replays for the animation, so the
   * on-screen motion and the authoritative outcome are the same simulation
   * run rather than two independently-computed ones that can diverge over a
   * long collision cascade (see render/animate.ts).
   */
  waypoints?: SimWaypoint[];
}

// Maximum simulated time for a single shot; a real shot settles in a few
// seconds, this guards against numeric non-termination.
const MAX_SIM_TIME = 30;
// Cap on how far ahead we scan for the next event in one search window. Events
// are found by scanning [0, LOOKAHEAD]; if none, we integrate that far and
// rescan. Keeps the polynomial sampling dense enough to catch fast contacts.
const LOOKAHEAD = 0.05;

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
export const simulateShot = (balls: Ball[], table: Table): SimResult => {
  const events: ShotEvent[] = [];
  const pocketed: number[] = [];
  let firstContact: number | null = null;
  let t = 0;

  reclassify(balls);

  while (anyMoving(balls) && t < MAX_SIM_TIME) {
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

  // Final de-overlap pass (mirrors the Rust core for train/play parity): an
  // event step can leave two resting balls interpenetrating by a fraction of a
  // millimetre without triggering another resolve. A few relaxation iterations
  // separate them so the resting state is physically valid.
  for (let iter = 0; iter < 4; iter++) {
    for (let i = 0; i < balls.length; i++) {
      for (let j = i + 1; j < balls.length; j++) {
        const a = balls[i];
        const b = balls[j];
        if (a.pocketed || b.pocketed) continue;
        const d = { x: b.pos.x - a.pos.x, y: b.pos.y - a.pos.y };
        const dist = Math.hypot(d.x, d.y);
        const overlap = 2 * BALL_RADIUS - dist;
        if (overlap > 1e-9) {
          const nx = dist > 1e-12 ? d.x / dist : 1;
          const ny = dist > 1e-12 ? d.y / dist : 0;
          const push = overlap / 2 + 1e-7;
          a.pos = { x: a.pos.x - nx * push, y: a.pos.y - ny * push };
          b.pos = { x: b.pos.x + nx * push, y: b.pos.y + ny * push };
        }
      }
    }
  }

  events.push({ time: t, kind: "stop", balls: [] });

  return {
    balls,
    events,
    pocketed,
    firstContact,
    duration: t,
  };
};

// Convenience: simulate on a copy, leaving the input untouched (used by search /
// candidate evaluation where we must not disturb the real world state).
export const simulateShotCopy = (balls: Ball[], table: Table): SimResult => {
  const copy = balls.map(cloneBall);
  return simulateShot(copy, table);
};

export const isSettled = (balls: Ball[]): boolean => {
  return balls.every(
    (b) => b.pocketed || Math.hypot(b.vel.x, b.vel.y) < STOP_SPEED,
  );
};
