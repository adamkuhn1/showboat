// One rate per shot, checked on real shots and per ball.
//
// The property is not "the animation is slower". It is that nothing about where
// the balls are changes how fast time runs. Asserting on the rate function
// directly cannot catch that failure, because every frame of a warped playback
// is still a correct sample of a correct simulation — what goes wrong is WHICH
// simulation instant each frame samples.
//
// So the strongest assertion here is measured off the balls themselves: walk a
// shot frame by frame the way the animation loop does, and for each ball divide
// the distance it moved on screen by the distance it moved in the simulation.
// That ratio IS the playback rate, read from that one ball's motion. If it ever
// changes — and in particular if it changes because a DIFFERENT ball hit
// something — the mapping is warping time and the physics will be blamed.
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

describe("no ball changes speed because of something another ball did", () => {
  // The regression this file exists for, measured the way a viewer sees it.
  it("every moving ball reports the same playback rate on every frame", () => {
    for (const { duration, track } of REAL_SHOTS) {
      for (const speed of PLAYBACK_SPEEDS) {
        const times = walk(speed, duration);
        const implied: number[] = [];
        let prev = interpolateBalls(track, times[0]);
        for (let i = 1; i < times.length - 1; i++) {
          const now = interpolateBalls(track, times[i]);
          const dSim = times[i] - times[i - 1];
          for (const b of now) {
            const p = prev.find((x) => x.id === b.id);
            if (!p || b.pocketed || p.pocketed) continue;
            if (Math.hypot(b.pos.x - p.pos.x, b.pos.y - p.pos.y) < 1e-4) continue;
            // Simulation seconds consumed per wall-clock second, read off this
            // one ball's motion. dWall is one frame by construction.
            implied.push(dSim / FRAME_SEC);
          }
          prev = now;
        }
        expect(implied.length).toBeGreaterThan(20);
        const min = Math.min(...implied);
        const max = Math.max(...implied);
        expect(max / min, `speed ${speed}: implied rate ranged ${min}..${max}`).toBeCloseTo(1, 9);
        expect(min).toBeCloseTo(speed, 9);
      }
    }
  });

  it("the mapping from wall-clock to simulation time is exactly affine", () => {
    // A single-rate playback satisfies simTime = a*wall + b with zero residual.
    // The residual is the amount of time-warping in the shot, in simulation
    // seconds, and it is the one number that summarises the whole defect.
    for (const { duration } of REAL_SHOTS) {
      for (const speed of PLAYBACK_SPEEDS) {
        const times = walk(speed, duration);
        // Drop the clamped final frame; it is deliberately short.
        const frames = times.slice(0, -1).map((simTime, i) => ({ wall: i * FRAME_SEC, simTime }));
        const n = frames.length;
        let sw = 0, ss = 0, sww = 0, sws = 0;
        for (const f of frames) {
          sw += f.wall;
          ss += f.simTime;
          sww += f.wall * f.wall;
          sws += f.wall * f.simTime;
        }
        const a = (n * sws - sw * ss) / (n * sww - sw * sw);
        const b = (ss - a * sw) / n;
        let residual = 0;
        for (const f of frames) residual = Math.max(residual, Math.abs(f.simTime - (a * f.wall + b)));
        expect(residual, `speed ${speed}`).toBeLessThan(1e-9);
        expect(a).toBeCloseTo(speed, 9);
      }
    }
  });
});
