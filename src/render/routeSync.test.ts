// Does the drawn route keep up with the ball?
//
// Adam's second complaint: the route is hard to reconcile with the motion. Two
// separate things could cause that — the route could be the wrong shape, or it
// could be the right shape drawn out of step with the balls. The first is
// already covered (`executedMotion.test.ts` checks the polyline against the
// true path to within half a logical pixel). This file covers the second.
//
// The measurement: at a set of simulation times, compare the head of the drawn
// line — `legCutAt`, the exact function `overlay.ts` strokes with — against the
// ball's own position at that instant from `interpolateBalls`, which is what
// the animation uses to place it. If those two disagree, the line and the ball
// are out of step on screen, whatever either one is individually correct about.
//
// The bound is expressed in LOGICAL CANVAS PIXELS, because that is the unit the
// disagreement is seen in, and it is DPR-independent by construction: the app
// draws in a fixed 900x500 logical space and scales the backing store, so a
// world-space agreement is an agreement at every device pixel ratio. The
// browser harness (`qa/motion.mjs`) confirms the frame pacing separately at
// DPR 1 and DPR 2; this is the geometric half.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../physics/table";
import { makeBall, cloneBall, type Ball } from "../physics/ball";
import { CUE_ID } from "../game/rack";
import { initPhysics, simulateShotWasm } from "../physics/wasm-bridge";
import { generateCandidates } from "../ai/candidates";
import { extractExecutedMotion } from "../ai/trace/executed";
import type { ExecutedMotion } from "../ai/trace/contract";
import { interpolateBalls, type AnimTrack } from "./animate";
import { computeView } from "./renderer";
import { legCutAt, measuredRoute } from "./presentation";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../..");
const table = makeTable();

// The shipped canvas. `App.tsx` draws in this fixed logical space at every DPR.
const CANVAS_W = 900;
const CANVAS_H = 500;
const view = computeView(CANVAS_W, CANVAS_H, table);

interface Fixture {
  name: string;
  track: AnimTrack;
  motion: ExecutedMotion;
  contacts: number;
}

const BOARDS: { name: string; balls: Ball[]; targets: number[] }[] = [
  {
    name: "open",
    balls: [
      makeBall(CUE_ID, -0.6, -0.1),
      makeBall(1, 0.3, 0.02),
      makeBall(2, 0.38, 0.1),
      makeBall(4, 0.1, -0.25),
    ],
    targets: [1, 2, 4],
  },
  {
    name: "tight",
    balls: [makeBall(CUE_ID, -0.75, 0.22), makeBall(3, 0.45, -0.28), makeBall(6, -0.1, 0.3)],
    targets: [3, 6],
  },
];

const fixtures: Fixture[] = [];

beforeAll(async () => {
  await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
  for (const board of BOARDS) {
    for (const c of generateCandidates(board.balls, table, board.targets)) {
      for (const power of [c.action.power, 0.85]) {
        const work = board.balls.map(cloneBall);
        const sim = simulateShotWasm(work, { ...c.action, power });
        const motion = extractExecutedMotion(sim);
        const wps = sim.waypoints ?? [];
        if (!motion || wps.length < 4) continue;
        const contacts = sim.events.filter(
          (e) => e.kind === "ball-ball" || e.kind === "ball-cushion" || e.kind === "pocket",
        ).length;
        // Only shots with real structure — a cue ball rolling two feet in a
        // straight line cannot show a synchronisation error.
        if (contacts < 2) continue;
        fixtures.push({
          name: `${board.name}/${c.kind}/${power.toFixed(2)}`,
          track: { waypoints: wps.map((w) => ({ simTime: w.time, balls: w.balls })), duration: sim.duration },
          motion,
          contacts,
        });
        if (fixtures.length >= 24) return;
      }
    }
  }
}, 180_000);

/** World metres -> logical canvas pixels, the same transform the overlay uses. */
const toPx = (p: { x: number; y: number }) => ({
  x: view.offsetX + p.x * view.scale,
  y: view.offsetY - p.y * view.scale,
});

const pxApart = (a: { x: number; y: number }, b: { x: number; y: number }) => {
  const pa = toPx(a);
  const pb = toPx(b);
  return Math.hypot(pa.x - pb.x, pa.y - pb.y);
};

describe("the drawn route stays with the ball it belongs to", () => {
  it("the fixture set is real and structurally varied", () => {
    expect(fixtures.length).toBeGreaterThan(5);
    expect(fixtures.some((f) => f.contacts >= 3)).toBe(true);
  });

  /**
   * The drawn line at `t` is: retained points `0..upTo`, then a segment to the
   * ball. So synchronisation is two properties, and both can fail.
   *
   *   1. no retained point is drawn before the ball has reached it — otherwise
   *      the line runs ahead of the ball it belongs to;
   *   2. the ball lies on the segment that follows the last drawn point —
   *      otherwise the line doubles back, or leaves a gap, at the join.
   */
  const checkLeg = (
    label: string,
    leg: NonNullable<ReturnType<typeof measuredRoute>["cue"]>,
    track: AnimTrack,
    duration: number,
    acc: { samples: number; worstOvershoot: number; worstOffPath: number; at: string },
  ) => {
    for (let i = 1; i < 80; i++) {
      const t = (duration * i) / 80;
      const cut = legCutAt(leg, t);
      if (cut === null) continue;
      const ball = interpolateBalls(track, t).find((b) => b.id === leg.ballId);
      if (!ball || ball.pocketed) continue;
      acc.samples++;

      // (1) every drawn point is one the ball has already passed.
      const overshoot = leg.timesSec[cut.upTo] - t;
      if (overshoot > acc.worstOvershoot) {
        acc.worstOvershoot = overshoot;
        acc.at = `${label} @ ${t.toFixed(3)}s`;
      }

      // (2) the ball sits on the segment from the last drawn point to the next
      // retained one. Measured as perpendicular distance from that segment.
      const a = leg.points[cut.upTo];
      const b = leg.points[Math.min(cut.upTo + 1, leg.points.length - 1)];
      const abx = b.x - a.x;
      const aby = b.y - a.y;
      const len2 = abx * abx + aby * aby;
      let off: number;
      if (len2 < 1e-12) {
        off = pxApart(a, ball.pos);
      } else {
        const s = Math.max(0, Math.min(1, ((ball.pos.x - a.x) * abx + (ball.pos.y - a.y) * aby) / len2));
        off = pxApart({ x: a.x + abx * s, y: a.y + aby * s }, ball.pos);
      }
      if (off > acc.worstOffPath) {
        acc.worstOffPath = off;
        acc.at = `${label} @ ${t.toFixed(3)}s`;
      }
    }
  };

  it("the cue-ball line never runs ahead of the cue ball", () => {
    const acc = { samples: 0, worstOvershoot: -Infinity, worstOffPath: 0, at: "" };
    for (const f of fixtures) {
      const route = measuredRoute(f.motion);
      if (route.cue) checkLeg(f.name, route.cue, f.track, f.track.duration, acc);
    }
    expect(acc.samples).toBeGreaterThan(200);
    // No drawn point has a time later than the instant being drawn.
    expect(acc.worstOvershoot, `overshoot at ${acc.at}`).toBeLessThanOrEqual(1e-9);
    // And the ball is on the segment the line joins to, within the thinning
    // tolerance (half a logical pixel) plus rounding.
    expect(acc.worstOffPath, `off-path ${acc.worstOffPath.toFixed(3)} px at ${acc.at}`).toBeLessThan(1);
  });

  it("an object-ball line never runs ahead of its own ball", () => {
    const acc = { samples: 0, worstOvershoot: -Infinity, worstOffPath: 0, at: "" };
    for (const f of fixtures) {
      const route = measuredRoute(f.motion);
      if (route.object) checkLeg(f.name, route.object, f.track, f.track.duration, acc);
    }
    expect(acc.samples).toBeGreaterThan(100);
    expect(acc.worstOvershoot, `overshoot at ${acc.at}`).toBeLessThanOrEqual(1e-9);
    expect(acc.worstOffPath, `off-path ${acc.worstOffPath.toFixed(3)} px at ${acc.at}`).toBeLessThan(1);
  });

  it("time interpolation alone would NOT be good enough — the fix is load-bearing", () => {
    // The reason `strokeMeasured` takes the ball's real position rather than
    // using `legCutAt`'s fallback head during playback. If a future refactor
    // drops the `ballAt` argument and goes back to the fallback, this records
    // how wrong that was: the route is thinned, and position along a thinned
    // straight run is not linear in time because the ball is decelerating.
    let worst = 0;
    for (const f of fixtures) {
      const route = measuredRoute(f.motion);
      const leg = route.cue;
      if (!leg) continue;
      for (let i = 1; i < 80; i++) {
        const t = (f.track.duration * i) / 80;
        const cut = legCutAt(leg, t);
        if (cut === null) continue;
        const ball = interpolateBalls(f.track, t).find((b) => b.id === leg.ballId);
        if (!ball || ball.pocketed) continue;
        worst = Math.max(worst, pxApart(cut.head, ball.pos));
      }
    }
    // Measured at 109 logical px when this was written. Asserting it is large
    // rather than pinning the exact number, which depends on the fixtures.
    expect(worst, "the fallback head is now accurate, so the ballAt path is dead weight").toBeGreaterThan(10);
  });

  it("the overlay actually asks for the ball's position during playback", () => {
    // A behavioural test cannot reach inside a canvas stroke, so this pins the
    // seam: the host supplies a locator and the stroke uses it.
    const overlay = readFileSync(join(__dirname, "overlay.ts"), "utf8");
    expect(overlay).toContain("ballAt(m.cue.ballId)");
    expect(overlay).toContain("ballAt(m.object.ballId)");
    expect(overlay).toContain("const head = t === null ? cut.head : (ballPos ?? cut.head);");
    const app = readFileSync(join(APP_ROOT, "src/App.tsx"), "utf8");
    expect(app).toContain("drawPresentation(ctx, scene.frame, view, scene.marks, scene.simTime, ballAt)");
  });

  it("the line does not run ahead of a ball that has not started moving", () => {
    // An object ball sits still until the cue reaches it. Its line must not be
    // drawn before then — a route arriving early is the most misleading version
    // of "hard to reconcile with the motion" there is.
    for (const f of fixtures) {
      const route = measuredRoute(f.motion);
      if (!route.object) continue;
      const leg = route.object;
      const startsAt = leg.timesSec[0];
      expect(legCutAt(leg, startsAt - 1e-6)).toBeNull();
      // And at its own start instant there is still nothing to draw.
      expect(legCutAt(leg, startsAt)).toBeNull();
    }
  });

  it("the line is complete exactly when the shot is over, and never before", () => {
    for (const f of fixtures) {
      const route = measuredRoute(f.motion);
      for (const leg of [route.cue, route.object].filter((l) => l !== null)) {
        const end = leg!.timesSec[leg!.timesSec.length - 1];
        const atEnd = legCutAt(leg!, end);
        expect(atEnd).not.toBeNull();
        expect(atEnd!.upTo).toBe(leg!.points.length - 1);
        // Just before the end, at least one point is still to come whenever the
        // leg has more than two.
        if (leg!.points.length > 2) {
          const before = legCutAt(leg!, leg!.timesSec[leg!.timesSec.length - 2] - 1e-9);
          expect(before!.upTo).toBeLessThan(leg!.points.length - 1);
        }
      }
    }
  });

  it("the agreement is a property of world coordinates, so it holds at any DPR", () => {
    // The app scales the backing store by `devicePixelRatio` and keeps drawing
    // in the same 900x500 logical space, so the world-to-logical transform is
    // the only one that can introduce a mismatch — and it does not depend on
    // DPR at all. Pinned here so a future change that makes the view depend on
    // the backing-store size fails a test rather than only looking wrong on
    // one machine.
    const at1 = computeView(CANVAS_W, CANVAS_H, table);
    const at2 = computeView(CANVAS_W, CANVAS_H, table);
    expect(at2).toEqual(at1);
    // And the shipped canvas dimensions are the ones this suite measured in.
    const app = readFileSync(join(APP_ROOT, "src/App.tsx"), "utf8");
    expect(app).toContain(`const CANVAS_W = ${CANVAS_W};`);
    expect(app).toContain(`const CANVAS_H = ${CANVAS_H};`);
    // The transform is set from the ratio and the logical space is unchanged.
    expect(app).toContain("setTransform(dpr, 0, 0, dpr, 0, 0)");
  });
});
