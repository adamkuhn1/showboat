import { type Frame, type ShotEvent, type SimResult } from "../physics/engine";
import { CUE_ID } from "../game/rack";

// Playback of a resolved shot. This module consumes the frame samples recorded
// by the ONE authoritative simulation (SimResult.frames) and interpolates
// between them — it never re-simulates, so the animation is the exact shot that
// was resolved by the rules.
//
// Pacing rules (each one answers a previously-shipped bug, do not relax them):
//   - There is a single global timeline for the whole table. Every ball is
//     sampled from the same warped clock. No per-ball rate exists anywhere, so
//     an untouched ball can never appear to speed up because a different ball
//     touched something (the old "5.68x phantom acceleration" bug).
//   - The time warp is PRECOMPUTED from the shot's event log before playback
//     starts, and is a fixed, smooth, monotone mapping for the entire shot.
//     It slows gently around the moments that matter (first cue contact, rail
//     hits, pockets) and returns to the base rate in between — it never reacts
//     to live animation state.
//   - stop() resolves the returned promise itself; it does not depend on the
//     requestAnimationFrame loop reaching another tick (the old unsettled-
//     promise bug when a rack was reset mid-animation).

export interface PlaybackHandle {
  // Resolves when the animation finishes OR is stopped. Never rejects.
  done: Promise<void>;
  // Stop immediately. Draws the final frame and resolves `done` synchronously
  // with respect to this call (no dependence on a future animation tick).
  stop: () => void;
}

// Times (in sim seconds) worth slowing down for, measured from the actual
// event log of the simulation being played.
const highlightTimes = (events: ShotEvent[]): number[] => {
  const times: number[] = [];
  let sawCueContact = false;
  for (const e of events) {
    if (e.kind === "ball-ball" && !sawCueContact && e.balls.includes(CUE_ID)) {
      sawCueContact = true;
      times.push(e.time);
    } else if (e.kind === "ball-cushion" || e.kind === "pocket") {
      times.push(e.time);
    }
  }
  return times;
};

// Instantaneous playback rate at sim-time t. Base rate everywhere, easing down
// to SLOW_FACTOR of base inside a +-WINDOW around each highlight, with a
// cosine ramp so the rate is continuous (no visible jerk). Pure function of
// the (fixed) highlight list -> the warp cannot drift or feed back.
const WINDOW = 0.35; // seconds of sim time around a highlight
const SLOW_FACTOR = 0.55;

const rateAt = (t: number, highlights: number[], base: number): number => {
  let dip = 0;
  for (const h of highlights) {
    const d = Math.abs(t - h);
    if (d < WINDOW) {
      // 1 at the event, 0 at the window edge, cosine-smoothed.
      const w = 0.5 * (1 + Math.cos((d / WINDOW) * Math.PI));
      dip = Math.max(dip, w);
    }
  }
  const factor = 1 - (1 - SLOW_FACTOR) * dip;
  return base * factor;
};

// Precomputed monotone mapping wall-clock seconds -> sim seconds, built by
// integrating the rate function once, up front, on a fixed grid.
interface TimeWarp {
  wall: number[]; // strictly increasing
  sim: number[]; // strictly increasing, same length
  totalWall: number;
}

const buildTimeWarp = (
  duration: number,
  events: ShotEvent[],
  baseRate: number,
): TimeWarp => {
  const highlights = highlightTimes(events);
  const STEP = 1 / 120; // sim-time integration step
  const wall: number[] = [0];
  const sim: number[] = [0];
  let w = 0;
  for (let t = 0; t < duration; t += STEP) {
    const dt = Math.min(STEP, duration - t);
    const r = rateAt(t + dt / 2, highlights, baseRate);
    w += dt / r; // r sim-seconds per wall-second
    wall.push(w);
    sim.push(t + dt);
  }
  return { wall, sim, totalWall: w };
};

// Invert the warp: wall seconds since start -> sim seconds, by binary search +
// linear interpolation over the precomputed grid.
const simTimeAt = (warp: TimeWarp, wallT: number): number => {
  const { wall, sim } = warp;
  if (wallT <= 0) return 0;
  if (wallT >= warp.totalWall) return sim[sim.length - 1];
  let lo = 0;
  let hi = wall.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (wall[mid] <= wallT) lo = mid;
    else hi = mid;
  }
  const span = wall[hi] - wall[lo];
  const f = span > 1e-12 ? (wallT - wall[lo]) / span : 0;
  return sim[lo] + f * (sim[hi] - sim[lo]);
};

// Interpolated ball positions at sim-time t from the recorded frames.
export interface FramePose {
  id: number;
  x: number;
  y: number;
  pocketed: boolean;
}

export const poseAt = (frames: Frame[], t: number): FramePose[] => {
  if (frames.length === 0) return [];
  if (t <= frames[0].t) return frames[0].balls;
  const last = frames[frames.length - 1];
  if (t >= last.t) return last.balls;
  // Binary search for the bracketing pair.
  let lo = 0;
  let hi = frames.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t <= t) lo = mid;
    else hi = mid;
  }
  const a = frames[lo];
  const b = frames[hi];
  const span = b.t - a.t;
  const f = span > 1e-12 ? (t - a.t) / span : 0;
  return a.balls.map((ba, i) => {
    const bb = b.balls[i];
    // A ball that gets pocketed inside this bracket disappears at the later
    // frame; until then it moves toward the jaw.
    return {
      id: ba.id,
      x: ba.x + (bb.x - ba.x) * f,
      y: ba.y + (bb.y - ba.y) * f,
      pocketed: ba.pocketed,
    };
  });
};

// Play a resolved shot. `draw` is called once per animation frame with the
// interpolated poses; the caller owns all canvas work. `baseRate` of 1 is
// real time; the UI's "slow" option passes a smaller value. The rate for a
// given shot never changes after this call.
export const playShot = (
  sim: SimResult,
  baseRate: number,
  draw: (poses: FramePose[], simT: number) => void,
): PlaybackHandle => {
  const frames = sim.frames ?? [];
  let raf = 0;
  let finished = false;
  let resolveDone: () => void = () => undefined;
  const done = new Promise<void>((res) => {
    resolveDone = res;
  });

  const finish = (): void => {
    if (finished) return;
    finished = true;
    if (raf) cancelAnimationFrame(raf);
    // Always land exactly on the final simulated frame.
    if (frames.length > 0) {
      draw(frames[frames.length - 1].balls, sim.duration);
    }
    resolveDone();
  };

  if (frames.length < 2) {
    // Nothing to animate (or frames were not recorded): resolve immediately
    // after drawing the resting state.
    finish();
    return { done, stop: finish };
  }

  const warp = buildTimeWarp(sim.duration, sim.events, baseRate);
  // Wall time is ACCUMULATED tick to tick with each step clamped to 100 ms,
  // rather than read as (now - start). Two reasons: rAF timestamps come from
  // the compositor clock, whose epoch/step can diverge from performance.now()
  // (observed under headless Chromium), and a hidden tab stops producing
  // frames entirely. With accumulation, any timestamp gap simply pauses the
  // animation; it can never fast-forward or skip the shot.
  let wallT = 0;
  let lastNow: number | null = null;

  const tick = (now: number): void => {
    if (finished) return;
    if (lastNow !== null) wallT += Math.min((now - lastNow) / 1000, 0.1);
    lastNow = now;
    if (wallT >= warp.totalWall) {
      finish();
      return;
    }
    const simT = simTimeAt(warp, wallT);
    draw(poseAt(frames, simT), simT);
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);

  return { done, stop: finish };
};
