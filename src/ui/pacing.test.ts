// One rate per shot, checked on real shots and per ball.
//
// The property is not "the animation is slower". It is that nothing about where
// the balls are changes how fast time runs. Asserting on the rate function
// directly cannot catch that failure, because every frame of a warped playback
// is still a correct sample of a correct simulation — what goes wrong is WHICH
// simulation instant each frame samples.
//
// So the strongest assertion here is positional. Walk a shot frame by frame
// the way the animation loop does, then check every ball against where the
// straight line through the sequence's own endpoints says it should be. Under
// one rate the two agree to floating-point residue. Under a warp the balls run
// ahead of or behind that line — and a viewer, seeing a ball surge because a
// DIFFERENT ball hit something, blames the physics.
//
// The last of these tests warps the playback on purpose and asserts the check
// rejects it, because a guard nobody has watched fail is not yet a guard.
//
// The shots are real: the reference engine on real boards, and the contact
// times come from the resulting event log.

import { describe, it, expect } from "vitest";
import { makeTable } from "../physics/table";
import { cloneBall, makeBall, type Ball } from "../physics/ball";
import { CUE_ID } from "../game/rack";
import { applyCue } from "../physics/cue";
import { simulateShot } from "../physics/engine";
import { buildAnimTrack, contactTimes, interpolateBalls, type AnimTrack } from "../render/animate";
import { PLAYBACK_SPEEDS } from "./playbackSpeed";
import { simulationRate, screenSeconds } from "./pacing";

const table = makeTable();
const FRAME_SEC = 1 / 60;

/** Real shots, chosen to include a bank and a multi-ball cascade. */
const shots = () => {
  const boards: Ball[][] = [
    [makeBall(CUE_ID, -0.7, 0), makeBall(1, 0.2, 0.15), makeBall(2, 0.4, -0.2)],
    [
      makeBall(CUE_ID, -0.6, -0.1),
      makeBall(1, 0.3, 0.02),
      makeBall(2, 0.38, 0.1),
      makeBall(4, 0.1, -0.25),
      makeBall(5, -0.2, 0.3),
    ],
    [makeBall(CUE_ID, -0.8, 0.2), makeBall(3, 0.5, -0.3)],
  ];
  const actions = [
    { phi: 0.05, power: 0.8, sideSpin: 0, topSpin: 0 },
    { phi: -0.2, power: 0.65, sideSpin: 0, topSpin: 0.2 },
    { phi: 0.35, power: 0.95, sideSpin: 0.1, topSpin: 0 },
  ];
  const out: { contacts: number[]; duration: number; track: AnimTrack }[] = [];
  for (const balls of boards) {
    for (const action of actions) {
      const work = balls.map(cloneBall);
      const cue = work.find((b) => b.id === CUE_ID)!;
      applyCue(cue, action);
      const sim = simulateShot(work.map(cloneBall), table);
      const contacts = contactTimes(sim);
      if (contacts.length < 2 || sim.duration <= 0.5) continue;
      out.push({ contacts, duration: sim.duration, track: buildAnimTrack(work, table) });
    }
  }
  return out;
};

const REAL_SHOTS = shots();

/** Every simulation instant the loop would sample, at 60 Hz. */
const walk = (speed: number, duration: number): number[] => {
  const rate = simulationRate(speed);
  const times = [0];
  let t = 0;
  let guard = 0;
  while (t < duration && guard++ < 20_000) {
    t = Math.min(t + FRAME_SEC * rate, duration);
    times.push(t);
  }
  return times;
};

describe("the pacing is built from real shots", () => {
  it("the fixture actually produced shots with contacts in them", () => {
    expect(REAL_SHOTS.length).toBeGreaterThan(2);
    for (const s of REAL_SHOTS) {
      expect(s.contacts.length).toBeGreaterThanOrEqual(2);
      expect(s.track.waypoints.length).toBeGreaterThan(4);
    }
  });
});

describe("one rate per shot", () => {
  it("the rate depends on the speed and on nothing else", () => {
    // The structural half of the guarantee: the rate is not a function of
    // simulation time, so no caller can make playback depend on where the balls
    // are. All a caller can pass is the speed.
    expect(simulationRate.length).toBe(1);
    for (const { duration } of REAL_SHOTS) {
      for (const speed of PLAYBACK_SPEEDS) {
        expect(simulationRate(speed)).toBe(speed);
        expect(screenSeconds(duration, speed)).toBeCloseTo(duration / speed, 9);
      }
    }
  });

  it("playback is never faster than real time at any offered speed", () => {
    for (const speed of PLAYBACK_SPEEDS) {
      expect(simulationRate(speed)).toBeLessThanOrEqual(1);
    }
  });

  it("simulation time advances by the same amount on every frame", () => {
    for (const { duration } of REAL_SHOTS) {
      for (const speed of PLAYBACK_SPEEDS) {
        const times = walk(speed, duration);
        // The last step is short because it is clamped to the end of the shot.
        const steps = times.slice(1, -1).map((t, i) => t - times[i]);
        for (const s of steps) expect(s).toBeCloseTo(FRAME_SEC * speed, 12);
      }
    }
  });

  it("screen time is the shot divided by the rate, and scales with the speed", () => {
    for (const { duration } of REAL_SHOTS) {
      expect(screenSeconds(duration, 1)).toBeCloseTo(duration, 9);
      expect(screenSeconds(duration, 0.5) / screenSeconds(duration, 1)).toBeCloseTo(2, 9);
    }
  });
});

/**
 * How far a ball set has travelled, as a fraction of the whole shot.
 *
 * Positions only. A frame is compared against the track at the fraction of
 * simulation time that a single-rate playback would have reached by then, so
 * the assertion is about where the balls actually are, not about the helper
 * that produced the frame.
 */
const worstOffsetFrom = (track: AnimTrack, frame: Ball[], simTime: number): number => {
  const ref = interpolateBalls(track, simTime);
  let worst = 0;
  for (const b of frame) {
    const r = ref.find((x) => x.id === b.id);
    if (!r || b.pocketed || r.pocketed) continue;
    worst = Math.max(worst, Math.hypot(b.pos.x - r.pos.x, b.pos.y - r.pos.y));
  }
  return worst;
};

/**
 * The largest distance, in metres, by which any ball sits away from where a
 * constant rate would have put it. Zero for single-rate playback; a warp
 * shows up as balls running ahead of or behind the straight line.
 */
const worstDriftFromConstantRate = (track: AnimTrack, times: number[]): number => {
  // The straight line through the sequence's own first and last samples. Using
  // the endpoints rather than the per-frame step keeps this independent of the
  // formula being tested, and sidesteps the final frame, which is clamped to
  // the end of the shot and so is not on the line by construction.
  const end = times.length - 2;
  if (end < 2) return 0;
  const span = times[end] - times[0];
  let worst = 0;
  for (let i = 1; i < end; i++) {
    const onTheLine = times[0] + span * (i / end);
    worst = Math.max(worst, worstOffsetFrom(track, interpolateBalls(track, times[i]), onTheLine));
  }
  return worst;
};

describe("no ball changes speed because of something another ball did", () => {
  // The regression this file exists for, measured the way a viewer sees it.
  //
  // Asserting on the frame times alone would prove nothing: those are produced
  // by the helper under test. So the check is positional. Under one rate, a
  // ball a third of the way through the wall-clock is a third of the way
  // through its journey, and every ball agrees at once.
  it("every ball sits where a single rate would put it, on every frame", () => {
    for (const { duration, track } of REAL_SHOTS) {
      for (const speed of PLAYBACK_SPEEDS) {
        const drift = worstDriftFromConstantRate(track, walk(speed, duration));
        // Micrometres: floating-point residue, not a warp. The test below
        // shows what a real one costs.
        expect(drift, `speed ${speed}: worst drift ${drift} m`).toBeLessThan(1e-6);
      }
    }
  });

  it("a warped playback fails that check", () => {
    // The guard above is only worth having if it rejects the defect it names.
    // This is the shape the removed code produced: simulation time advancing
    // as a curve against wall-clock instead of a straight line.
    const { duration, track } = REAL_SHOTS[0];
    const frames = walk(0.5, duration).length - 1;
    const warped = Array.from({ length: frames + 1 }, (_, i) => duration * (i / frames) ** 2);
    const drift = worstDriftFromConstantRate(track, warped);
    expect(drift).toBeGreaterThan(0.05);
  });
});
