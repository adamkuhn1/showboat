// What the engine accepts as a starting board, and how long it takes.
//
// The bridge used to separate every ball pair to 2 mm of clearance before each
// WASM call, because an input at contact distance once made the Rust engine
// emit thousands of t≈0 collision events and block for 10–30 s. That belongs to
// the engine and the engine does it: `simulate_shot` separates before its first
// event scan, again after every ball-ball resolve, and once more at the end
// (physics-core/src/engine.rs).
//
// Doing it on this side as well is not free. It moves the caller's balls — the
// racked triangle, the board the search is about to explore — before a
// simulation that is going to do the same thing to its own copy, so positions
// drift for reasons that have nothing to do with the shot.
//
// This file is the evidence for leaving it out: the pathological inputs still
// simulate promptly, the resting state still has no ball inside another, and a
// fresh rack now reaches the first frame exactly as it was drawn.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeBall, cloneBall, type Ball } from "./ball";
import { BALL_DIAMETER } from "./constants";
import { initPhysics, simulateShotWasm, resolvePenetration } from "./wasm-bridge";
import { CUE_ID, rackEightBall } from "../game/rack";

const APP_ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");

/**
 * Generous by two orders of magnitude. Every board below simulates in ~100 ms;
 * the failure this guards against was measured in tens of seconds, so a
 * threshold this loose cannot flake on a busy machine and still cannot miss it.
 */
const BUDGET_MS = 3000;

/** A rack at an arbitrary row spacing, for the pathological inputs. */
const rackAtSpacing = (gap: number): Ball[] => {
  const rows = [[1], [9, 2], [3, 8, 10], [11, 4, 5, 12], [6, 13, 14, 7, 15]];
  const out: Ball[] = [makeBall(CUE_ID, -0.4953, 0)];
  const rowDx = (gap * Math.sqrt(3)) / 2;
  rows.forEach((row, r) => {
    const y0 = -((row.length - 1) / 2) * gap;
    row.forEach((id, i) => out.push(makeBall(id, 0.4953 + r * rowDx, y0 + i * gap)));
  });
  return out;
};

const closestPair = (balls: Ball[]): number => {
  let min = Infinity;
  for (let i = 0; i < balls.length; i++) {
    for (let j = i + 1; j < balls.length; j++) {
      if (balls[i].pocketed || balls[j].pocketed) continue;
      min = Math.min(min, Math.hypot(balls[j].pos.x - balls[i].pos.x, balls[j].pos.y - balls[i].pos.y));
    }
  }
  return min;
};

describe("the engine takes its input as it finds it", () => {
  beforeAll(async () => {
    await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
  }, 60_000);

  const boards: [string, () => Ball[]][] = [
    ["a fresh rack", rackEightBall],
    ["a rack at exact contact distance", () => rackAtSpacing(BALL_DIAMETER)],
    ["a rack penetrating by 1 mm", () => rackAtSpacing(BALL_DIAMETER - 0.001)],
    [
      "fifteen coincident balls",
      () => [makeBall(CUE_ID, -0.6, 0), ...Array.from({ length: 15 }, (_, i) => makeBall(i + 1, 0.3, 0))],
    ],
  ];

  for (const [name, build] of boards) {
    it(`breaks ${name} promptly and leaves no ball inside another`, () => {
      const balls = build();
      const started = performance.now();
      const sim = simulateShotWasm(balls, { phi: 0, power: 1, sideSpin: 0, topSpin: 0 });
      const elapsed = performance.now() - started;

      expect(elapsed, `${name} took ${elapsed.toFixed(0)} ms`).toBeLessThan(BUDGET_MS);
      // A real shot, not an instant bail-out.
      expect(sim.duration).toBeGreaterThan(0.5);
      expect(sim.events.length).toBeGreaterThan(5);
      // The committed resting board. Touching is legal; overlapping is not.
      expect(closestPair(balls)).toBeGreaterThanOrEqual(BALL_DIAMETER - 1e-9);
    }, 60_000);
  }

  it("commits exactly the board the last frame of the animation showed", () => {
    // The animation ends on the last waypoint; the game then commits `balls`.
    // If the bridge adjusts `balls` after the engine returns, those two are
    // different boards and every ball whose position changed appears to move
    // once more, by itself, after the shot has visibly finished. The engine
    // deliberately rewrites its final waypoint to match (engine.rs), so the
    // only way to break this is to touch the state on the way out.
    const boards: [string, Ball[], number][] = [
      ["break", rackEightBall(), 1],
      // A frozen pair: the case where a correction to touching distance would
      // be most tempting and most visible.
      [
        "frozen pair",
        [makeBall(CUE_ID, -0.6, 0), makeBall(1, 0.2, 0), makeBall(2, 0.2 + BALL_DIAMETER, 0)],
        0.6,
      ],
    ];
    for (const [name, balls, power] of boards) {
      const sim = simulateShotWasm(balls, { phi: 0, power, sideSpin: 0, topSpin: 0 });
      const last = (sim.waypoints ?? [])[(sim.waypoints ?? []).length - 1];
      expect(last, name).toBeDefined();
      for (const shown of last.balls) {
        const committed = balls.find((b) => b.id === shown.id)!;
        expect(committed.pocketed, `${name}: ball ${shown.id}`).toBe(shown.pocketed);
        if (shown.pocketed) continue;
        expect(
          Math.hypot(committed.pos.x - shown.pos.x, committed.pos.y - shown.pos.y),
          `${name}: ball ${shown.id} moves after the animation ends`,
        ).toBe(0);
      }
    }
  }, 60_000);

  it("plays the rack that was drawn — the engine finds nothing to correct", () => {
    // Racking tighter than the engine's own clearance used to have it walk the
    // triangle apart by up to 4.7 mm before the cue ball moved, which is a
    // crooked rack nobody built and 1.9 logical px of unexplained motion in the
    // first frame of the one shot every visitor sees.
    const rack = rackEightBall();
    const sim = simulateShotWasm(rack.map(cloneBall), { phi: 0, power: 0, sideSpin: 0, topSpin: 0 });
    const first = (sim.waypoints ?? [])[0];
    expect(first).toBeDefined();
    for (const b of first.balls) {
      const drawn = rack.find((x) => x.id === b.id)!;
      expect(
        Math.hypot(b.pos.x - drawn.pos.x, b.pos.y - drawn.pos.y),
        `ball ${b.id} moved before the shot`,
      ).toBeLessThan(1e-9);
    }
  }, 60_000);
});

describe("resolvePenetration corrects penetration and nothing else", () => {
  const pairAt = (dist: number): Ball[] => [makeBall(1, 0, 0), makeBall(2, dist, 0)];
  const gap = (balls: Ball[]) => Math.hypot(balls[1].pos.x - balls[0].pos.x, balls[1].pos.y - balls[0].pos.y);

  it("leaves balls that are merely close exactly where they are", () => {
    for (const d of [BALL_DIAMETER, BALL_DIAMETER + 1e-6, BALL_DIAMETER + 0.002, 0.3]) {
      const balls = pairAt(d);
      expect(resolvePenetration(balls)).toBe(0);
      expect(gap(balls)).toBe(d);
    }
  });

  it("pushes a genuinely overlapping pair to touching, not past it", () => {
    for (const d of [BALL_DIAMETER - 1e-4, BALL_DIAMETER - 0.01, BALL_DIAMETER / 2]) {
      const balls = pairAt(d);
      expect(resolvePenetration(balls)).toBeGreaterThan(0);
      expect(gap(balls)).toBeCloseTo(BALL_DIAMETER, 12);
      // Symmetric: neither ball is privileged.
      expect(balls[0].pos.x).toBeCloseTo(-(BALL_DIAMETER - d) / 2, 12);
    }
  });

  it("separates coincident balls rather than dividing by zero", () => {
    const balls = pairAt(0);
    resolvePenetration(balls);
    expect(gap(balls)).toBeCloseTo(BALL_DIAMETER, 12);
    expect(Number.isFinite(balls[0].pos.x)).toBe(true);
  });

  it("resolves the worst cluster that can exist: a whole rack, 1 mm into itself", () => {
    // Clearing one pair pushes a ball into the next, so a full rack needs many
    // relaxation passes. This is the case the pass budget is sized against.
    const balls = rackAtSpacing(BALL_DIAMETER - 0.001).filter((b) => b.id !== CUE_ID);
    const passes = resolvePenetration(balls);
    expect(passes).toBeGreaterThan(0);
    expect(passes, `converged in ${passes} passes`).toBeLessThan(48);
    expect(closestPair(balls)).toBeGreaterThanOrEqual(BALL_DIAMETER - 1e-9);
  });

  it("ignores pocketed balls, whose positions are not on the table", () => {
    const balls = pairAt(0.001);
    balls[1].pocketed = true;
    expect(resolvePenetration(balls)).toBe(0);
    expect(balls[0].pos.x).toBe(0);
  });
});
