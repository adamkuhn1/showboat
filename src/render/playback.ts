import { type Frame, type SimResult } from "../physics/engine";

// Playback of a resolved shot. This module consumes the frame samples recorded
// by the ONE authoritative simulation (SimResult.frames) and interpolates
// between them — it never re-simulates, so the animation is the exact shot that
// was resolved by the rules.
//
// Pacing invariants:
//   - One global timeline for the whole table: sim-time advances at a fixed
//     multiple of wall-clock time (baseRate), the same rate for every ball,
//     with no per-ball or per-event rate changes. "Slow" is `baseRate < 1`,
//     not a different curve.
//   - stop() resolves the returned promise itself; it never depends on the
//     requestAnimationFrame loop reaching another tick.

export interface PlaybackHandle {
  // Resolves when the animation finishes OR is stopped. Never rejects.
  done: Promise<void>;
  // Stop immediately. Draws the final frame and resolves `done` synchronously
  // with respect to this call (no dependence on a future animation tick).
  stop: () => void;
}

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
    const simT = wallT * baseRate;
    if (simT >= sim.duration) {
      finish();
      return;
    }
    draw(poseAt(frames, simT), simT);
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);

  return { done, stop: finish };
};
