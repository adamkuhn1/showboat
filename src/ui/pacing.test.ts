// Where the slowness goes — checked on real shots.
//
// The property that matters is not "the animation is slower". It is that the
// screen time is spent on the parts of the shot that carry information. So the
// assertions below are about DISTRIBUTION: what share of the screen time lands
// near a real contact, versus what share the old flat multiplier gave it.
//
// The shots are real. `buildAnimTrack` runs the reference engine on real boards
// and the contact times come from the resulting event log, so the curves under
// test are the curves a visitor actually sees.

import { describe, it, expect } from "vitest";
import { makeTable } from "../physics/table";
import { cloneBall, makeBall } from "../physics/ball";
import { CUE_ID } from "../game/rack";
import { applyCue } from "../physics/cue";
import { simulateShot } from "../physics/engine";
import { contactTimes } from "../render/animate";
import { PLAYBACK_SPEEDS } from "./playbackSpeed";
import {
  buildPacing,
  screenSeconds,
  CONTACT_RATE,
  CONTACT_WINDOW_SEC,
  COAST_MAX_ABSOLUTE,
} from "./pacing";

const table = makeTable();

/** A few real shots, chosen to include a bank and a multi-ball cascade. */
const shots = () => {
  const boards = [
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
  const out: { contacts: number[]; duration: number }[] = [];
  for (const balls of boards) {
    for (const action of actions) {
      const work = balls.map(cloneBall);
      const cue = work.find((b) => b.id === CUE_ID)!;
      applyCue(cue, action);
      const sim = simulateShot(work, table);
      const contacts = contactTimes(sim);
      if (contacts.length >= 2 && sim.duration > 0.5) out.push({ contacts, duration: sim.duration });
    }
  }
  return out;
};

const REAL_SHOTS = shots();

describe("the pacing is built from real shots", () => {
  it("the fixture actually produced shots with contacts in them", () => {
    expect(REAL_SHOTS.length).toBeGreaterThan(2);
    for (const s of REAL_SHOTS) expect(s.contacts.length).toBeGreaterThanOrEqual(2);
  });
});

describe("slowness lands on the contacts", () => {
  it("the rate at a contact is slower than the rate on open felt", () => {
    for (const { contacts, duration } of REAL_SHOTS) {
      const p = buildPacing({ contactTimes: contacts, durationSec: duration, speed: 0.6 });
      // Compare a contact against a point provably outside every window, and
      // before the coast to rest (which has its own rule).
      const c = contacts[0];
      const open = contacts
        .slice(0, -1)
        .map((a, i) => (a + contacts[i + 1]) / 2)
        .find(
          (m) => contacts.every((x) => Math.abs(m - x) > CONTACT_WINDOW_SEC) && m < p.lastContactSec,
        );
      if (open === undefined) continue; // this shot has no open stretch; nothing to compare
      expect(p.rateAt(c)).toBeLessThan(p.rateAt(open));
    }
  });

  it("the rate exactly at a contact is the contact rate", () => {
    const p = buildPacing({ contactTimes: [1, 3], durationSec: 5, speed: 0.6 });
    expect(p.rateAt(1)).toBeCloseTo(0.6 * CONTACT_RATE, 6);
  });

  it("contacts get a larger share of the screen time than of the shot", () => {
    // The whole point, stated as a measurement: the fraction of SCREEN seconds
    // spent within a contact window exceeds the fraction of SIMULATION seconds
    // those windows occupy.
    for (const { contacts, duration } of REAL_SHOTS) {
      const p = buildPacing({ contactTimes: contacts, durationSec: duration, speed: 0.6 });
      const steps = 4000;
      const dt = duration / steps;
      let simNear = 0;
      let screenNear = 0;
      let screenTotal = 0;
      for (let i = 0; i < steps; i++) {
        const t = i * dt + dt / 2;
        const rate = p.rateAt(t);
        const screen = dt / rate;
        screenTotal += screen;
        const near = contacts.some((c) => Math.abs(t - c) < CONTACT_WINDOW_SEC);
        if (near) {
          simNear += dt;
          screenNear += screen;
        }
      }
      const simShare = simNear / duration;
      const screenShare = screenNear / screenTotal;
      expect(screenShare, `sim ${simShare} vs screen ${screenShare}`).toBeGreaterThan(simShare);
    }
  });

  it("no coasting stretch is played faster than real time", () => {
    for (const { contacts, duration } of REAL_SHOTS) {
      for (const speed of PLAYBACK_SPEEDS) {
        const p = buildPacing({ contactTimes: contacts, durationSec: duration, speed });
        // Everything before the coast to rest. The coast has its own cap and is
        // deliberately allowed to run fast — it is balls rolling to a stop.
        for (let t = 0; t < p.lastContactSec; t += 0.02) {
          expect(p.rateAt(t), `speed ${speed} at ${t}`).toBeLessThanOrEqual(COAST_MAX_ABSOLUTE + 1e-9);
          expect(p.rateAt(t)).toBeGreaterThan(0);
        }
      }
    }
  });
});

describe("the shot does not get longer to achieve it", () => {
  it("total screen time stays close to the flat multiplier it replaces", () => {
    // The old model: flat `speed` up to the last contact, capped settle after.
    // The new one redistributes rather than inflates, so the totals should be
    // comparable — the contacts got longer by making the open felt shorter.
    for (const { contacts, duration } of REAL_SHOTS) {
      const speed = 0.6;
      const p = buildPacing({ contactTimes: contacts, durationSec: duration, speed });
      const now = screenSeconds(p, duration);
      const flat = p.lastContactSec / speed + Math.min(1, p.tailSec);
      // Within a factor of 1.5 either way. This is a sanity bound on the
      // redistribution, not a target: the point is that it is a redistribution.
      expect(now, `now ${now} vs flat ${flat}`).toBeLessThan(flat * 1.5);
      expect(now).toBeGreaterThan(flat / 1.5);
    }
  });

  it("the coast to rest is still capped and still never slower than asked", () => {
    for (const { contacts, duration } of REAL_SHOTS) {
      for (const speed of PLAYBACK_SPEEDS) {
        const p = buildPacing({ contactTimes: contacts, durationSec: duration, speed });
        const settle = p.rateAt(p.lastContactSec + 1e-6);
        expect(settle).toBeGreaterThanOrEqual(speed - 1e-9);
        if (p.tailSec > 0) expect(p.tailSec / settle).toBeLessThanOrEqual(1 + 1e-6);
      }
    }
  });
});

describe("degenerate shots behave", () => {
  it("a shot with no contacts has one constant rate and no dips", () => {
    const p = buildPacing({ contactTimes: [], durationSec: 2, speed: 0.6 });
    const rates = [0, 0.5, 1, 1.5, 1.99].map((t) => p.rateAt(t));
    expect(new Set(rates.map((r) => r.toFixed(9))).size).toBe(1);
    expect(p.lastContactSec).toBe(0);
  });

  it("the rate is finite and positive everywhere, at every offered speed", () => {
    for (const { contacts, duration } of REAL_SHOTS) {
      for (const speed of PLAYBACK_SPEEDS) {
        const p = buildPacing({ contactTimes: contacts, durationSec: duration, speed });
        for (let t = 0; t <= duration; t += duration / 200) {
          const r = p.rateAt(t);
          expect(Number.isFinite(r), `speed ${speed} at ${t}`).toBe(true);
          expect(r).toBeGreaterThan(0);
        }
      }
    }
  });

  it("the rate is continuous — a step would read as a physics stutter", () => {
    for (const { contacts, duration } of REAL_SHOTS) {
      const p = buildPacing({ contactTimes: contacts, durationSec: duration, speed: 0.6 });
      const dt = 0.002;
      let worst = 0;
      // Up to the last contact; the settle is deliberately a different regime.
      for (let t = 0; t < p.lastContactSec - dt; t += dt) {
        worst = Math.max(worst, Math.abs(p.rateAt(t + dt) - p.rateAt(t)));
      }
      // The cosine ramp spans CONTACT_WINDOW_SEC, so per 2 ms step the change
      // is small. A switch instead of a blend would show up here immediately.
      expect(worst).toBeLessThan(0.05);
    }
  });
});
