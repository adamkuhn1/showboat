// Contact marks must come from the shot that actually happened.
//
// Run against a real WASM simulation, not a hand-built fixture, so a mark that
// does not correspond to a real emitted event is a failure rather than an
// agreement between two pieces of test data.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { initPhysics, simulateShotWasm } from "../physics/wasm-bridge";
import { makeBall } from "../physics/ball";
import { CUE_ID } from "../game/rack";
import type { SimResult } from "../physics/engine";
import { contactMarks } from "./annotate";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../..");

let sim: SimResult;

describe("contact marks are derived from the executed simulation", () => {
  beforeAll(async () => {
    await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
    // A cue ball driven into a ball near the top rail: guaranteed to produce a
    // ball-ball contact and at least one cushion contact.
    const balls = [makeBall(CUE_ID, -0.7, 0.0), makeBall(3, 0.2, 0.05), makeBall(5, 0.5, -0.3)];
    sim = simulateShotWasm(balls, { phi: 0.06, power: 0.85, sideSpin: 0, topSpin: 0 });
  }, 60_000);

  it("produces a mark for every marked event, in time order, up to the cap", () => {
    const marks = contactMarks(sim, 12);
    const marked = sim.events.filter((e) =>
      ["ball-ball", "ball-cushion", "pocket"].includes(e.kind),
    );
    expect(marked.length).toBeGreaterThan(0);
    expect(marks.length).toBe(Math.min(12, marked.length));
    for (let i = 1; i < marks.length; i++) {
      expect(marks[i].timeSec).toBeGreaterThanOrEqual(marks[i - 1].timeSec);
    }
    // The cap drops the latest events, so the marks are always a true prefix.
    for (let i = 0; i < marks.length; i++) {
      expect(marks[i].timeSec).toBeCloseTo(marked[i].time, 6);
      expect(marks[i].ballId).toBe(marked[i].balls[0]);
    }
  });

  it("combination order is 1-based and consecutive", () => {
    const marks = contactMarks(sim);
    expect(marks.map((m) => m.order)).toEqual(marks.map((_, i) => i + 1));
  });

  it("each mark sits where that ball really was at that instant", () => {
    const marks = contactMarks(sim);
    for (const m of marks) {
      // Find the waypoint nearest the mark's time and confirm the mark is on
      // that ball's position there — i.e. the position was looked up, not
      // interpolated from the candidate's planned geometry.
      const wp = sim.waypoints!.reduce((best, w) =>
        Math.abs(w.time - m.timeSec) < Math.abs(best.time - m.timeSec) ? w : best,
      );
      const ball = wp.balls.find((b) => b.id === m.ballId)!;
      expect(m.at.x).toBeCloseTo(ball.pos.x, 9);
      expect(m.at.y).toBeCloseTo(ball.pos.y, 9);
    }
  });

  it("returns nothing when the simulation captured no waypoints", () => {
    expect(contactMarks({ ...sim, waypoints: undefined })).toEqual([]);
  });
});
