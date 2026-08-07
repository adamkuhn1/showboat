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
import { makeTable } from "../physics/table";
import { BALL_RADIUS } from "../physics/constants";
import { CUE_ID } from "../game/rack";
import type { SimResult } from "../physics/engine";
import { extractExecutedMotion } from "../ai/trace/executed";
import { contactMarks, contactMarksFromExecuted } from "./annotate";

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
    expect(marks.some((m) => m.kind !== "pocket")).toBe(true);
    for (const m of marks) {
      if (m.kind === "pocket") continue; // see the pocket case below
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

  // ---------------------------------------------------------------------
  // The pocket case, and a correction to what this file used to assert.
  //
  // The test above ran over EVERY mark, pockets included, and it passed. It
  // passed because it compared the mark against the same waypoint the mark was
  // built from, so it could only ever fail if the lookup were inconsistent with
  // itself — and both sides were wrong together.
  //
  // A waypoint is emitted after the capture is resolved, at which point the
  // ball is off the table and `wasm-bridge.ts`'s `ballsFromFlat` records it at
  // the x=0 sentinel. So every pocket mark was being drawn on the centre line
  // of the cloth. The two tests below replace that expectation: one pins where
  // a pocket mark belongs, the other measures how far the old rule was out, so
  // "this was worth changing" is a number rather than an assertion.
  // ---------------------------------------------------------------------

  it("a pocket mark sits on the pocket it names", () => {
    // The simulator captures a ball when its centre reaches the pocket radius,
    // so the capture-instant position lies on that circle. A ball radius of
    // slack covers the gap between the Rust core's event solve and the TS
    // `advanceBall` that reproduces the position for it.
    const table = makeTable();
    const marks = contactMarks(sim).filter((m) => m.kind === "pocket");
    expect(marks.length).toBeGreaterThan(0);
    for (const m of marks) {
      const pocket = table.pockets.find((p) => p.id === m.detail)!;
      const off = Math.hypot(m.at.x - pocket.center.x, m.at.y - pocket.center.y);
      expect(off, `pocket mark for ball ${m.ballId} is ${off.toFixed(3)} m from ${m.detail}`)
        .toBeLessThan(pocket.radius + BALL_RADIUS);
    }
  });

  it("the rule it replaces put at least one pocket mark half a table away", () => {
    const table = makeTable();
    const wrong = sim.events
      .filter((e) => e.kind === "pocket")
      .map((e) => {
        // Exactly what the old lookup did: nearest waypoint, whatever it says.
        const wp = sim.waypoints!.reduce((best, w) =>
          Math.abs(w.time - e.time) < Math.abs(best.time - e.time) ? w : best,
        );
        const ball = wp.balls.find((b) => b.id === e.balls[0])!;
        const pocket = table.pockets.find((p) => p.id === e.pocket)!;
        return Math.hypot(ball.pos.x - pocket.center.x, ball.pos.y - pocket.center.y);
      });
    expect(wrong.length).toBeGreaterThan(0);
    // The worst of them on this fixture is the top-left pocket: the sentinel
    // puts the ball on the centre line, ~0.95 m from where it went in. Half the
    // table is 0.99 m, so this is not a rounding error being dressed up.
    expect(Math.max(...wrong)).toBeGreaterThan(0.9);
  });

  it("the trace-derived marks the app draws are the same marks", () => {
    // `useAiTurn` reads marks off the published trace, not off the raw
    // `SimResult`, so that each one lands on a vertex of the route drawn
    // beside it. That only helps if the two agree — this is where they are
    // held together.
    const executed = extractExecutedMotion(sim)!;
    expect(executed).not.toBeNull();
    const fromTrace = contactMarksFromExecuted(executed);
    const fromSim = contactMarks(sim);
    expect(fromTrace.length).toBe(fromSim.length);
    for (let i = 0; i < fromTrace.length; i++) {
      expect(fromTrace[i].kind).toBe(fromSim[i].kind);
      expect(fromTrace[i].ballId).toBe(fromSim[i].ballId);
      expect(fromTrace[i].order).toBe(fromSim[i].order);
      expect(fromTrace[i].detail).toBe(fromSim[i].detail);
      expect(fromTrace[i].timeSec).toBeCloseTo(fromSim[i].timeSec, 12);
      expect(fromTrace[i].at.x).toBeCloseTo(fromSim[i].at.x, 12);
      expect(fromTrace[i].at.y).toBeCloseTo(fromSim[i].at.y, 12);
    }
  });

  it("returns nothing when the simulation captured no waypoints", () => {
    expect(contactMarks({ ...sim, waypoints: undefined })).toEqual([]);
  });
});
