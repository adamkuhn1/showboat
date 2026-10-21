import { describe, it, expect } from "vitest";
import { makeGame, takeShot, cloneState, cueSpotLimits, placeCueBall } from "./game";
import { HEAD_STRING_X } from "../physics/table";
import { bestCuePlacement } from "../ai/agent";
import { makeRanker } from "../ai/ranker";
import { rackEightBall, CUE_ID } from "./rack";
import { applyShotRules } from "./rules";
import type { SimResult } from "../physics/engine";
import type { GameState } from "./state";

// Build a minimal SimResult for unit-testing the pure ruleset without running
// the physics engine — lets us assert rule branches deterministically.
const simOf = (partial: Partial<SimResult>): SimResult => ({
  balls: [],
  events: partial.events ?? [{ time: 0, kind: "stop", balls: [] }],
  pocketed: partial.pocketed ?? [],
  firstContact: partial.firstContact ?? null,
  duration: 1,
});

const openBrokenState = (): GameState => {
  const g = makeGame().state;
  g.broken = true;
  g.groups = { 0: null, 1: null };
  return g;
};

describe("8-ball ruleset", () => {
  it("rack has 16 balls including cue and eight", () => {
    const balls = rackEightBall();
    expect(balls).toHaveLength(16);
    expect(balls.filter((b) => b.id === 8)).toHaveLength(1);
    expect(balls.filter((b) => b.id === CUE_ID)).toHaveLength(1);
  });

  it("no object-ball contact is a foul that passes the turn", () => {
    const pre = openBrokenState();
    const g = cloneState(pre);
    const out = applyShotRules(g, pre, simOf({ firstContact: null }));
    expect(out.foul).toBe(true);
    expect(out.turnPasses).toBe(true);
    expect(g.turn).toBe(1);
  });

  it("a wrong-ball foul passes the turn but gives no ball in hand", () => {
    const pre = openBrokenState();
    pre.groups = { 0: "solids", 1: "stripes" };
    const g = cloneState(pre);
    const out = applyShotRules(
      g,
      pre,
      simOf({
        firstContact: 9,
        events: [
          { time: 0.1, kind: "ball-ball", balls: [CUE_ID, 9] },
          { time: 0.2, kind: "ball-cushion", balls: [9], cushion: "top" },
          { time: 1, kind: "stop", balls: [] },
        ],
      }),
    );
    expect(out.foul).toBe(true);
    expect(out.turnPasses).toBe(true);
    expect(out.ballInHandForNext).toBe(false);
  });

  it("a rail the cue ball hit before contact does not satisfy the rail rule", () => {
    const pre = openBrokenState();
    const g = cloneState(pre);
    const out = applyShotRules(
      g,
      pre,
      simOf({
        firstContact: 1,
        events: [
          { time: 0.1, kind: "ball-cushion", balls: [CUE_ID], cushion: "top" },
          { time: 0.3, kind: "ball-ball", balls: [CUE_ID, 1] },
          { time: 1, kind: "stop", balls: [] },
        ],
      }),
    );
    expect(out.foul).toBe(true);
    expect(out.foulReason).toMatch(/rail/);
    expect(out.ballInHandForNext).toBe(false);
  });

  it("scratch is a foul giving opponent ball-in-hand", () => {
    const pre = openBrokenState();
    const g = cloneState(pre);
    const out = applyShotRules(
      g,
      pre,
      simOf({
        firstContact: 1,
        pocketed: [1, CUE_ID],
        events: [
          { time: 0.1, kind: "ball-cushion", balls: [1], cushion: "top" },
          { time: 1, kind: "stop", balls: [] },
        ],
      }),
    );
    expect(out.foul).toBe(true);
    expect(out.ballInHandForNext).toBe(true);
  });

  it("legally pocketing a solid on an open table claims solids and keeps turn", () => {
    const pre = openBrokenState();
    const g = cloneState(pre);
    const out = applyShotRules(
      g,
      pre,
      simOf({
        firstContact: 1,
        pocketed: [1],
        events: [
          { time: 0.1, kind: "ball-cushion", balls: [2], cushion: "top" },
          { time: 1, kind: "stop", balls: [] },
        ],
      }),
    );
    expect(out.foul).toBe(false);
    expect(out.assignedGroups).toBe(true);
    expect(g.groups[0]).toBe("solids");
    expect(g.groups[1]).toBe("stripes");
    expect(out.turnPasses).toBe(false); // continue turn
  });

  it("hitting opponent's ball first is a foul once groups are assigned", () => {
    const pre = openBrokenState();
    pre.groups = { 0: "solids", 1: "stripes" };
    const g = cloneState(pre);
    const out = applyShotRules(
      g,
      pre,
      simOf({
        firstContact: 9, // a stripe
        pocketed: [],
        events: [
          { time: 0.1, kind: "ball-cushion", balls: [9], cushion: "top" },
          { time: 1, kind: "stop", balls: [] },
        ],
      }),
    );
    expect(out.foul).toBe(true);
    expect(out.foulReason).toMatch(/opponent/);
  });

  it("pocketing the 8 early loses the game", () => {
    const pre = openBrokenState();
    pre.groups = { 0: "solids", 1: "stripes" };
    // Player 0 still has solids on the table (default rack), so potting 8 loses.
    const g = cloneState(pre);
    const out = applyShotRules(
      g,
      pre,
      simOf({ firstContact: 8, pocketed: [8] }),
    );
    expect(out.gameOver).toBe(true);
    expect(out.winner).toBe(1);
  });

  it("pocketing the 8 after clearing your group wins", () => {
    const pre = openBrokenState();
    pre.groups = { 0: "solids", 1: "stripes" };
    // Remove all solids from the table so player 0's group is cleared.
    for (const b of pre.balls) {
      if ([1, 2, 3, 4, 5, 6, 7].includes(b.id)) b.pocketed = true;
    }
    const g = cloneState(pre);
    const out = applyShotRules(
      g,
      pre,
      simOf({
        firstContact: 8,
        pocketed: [8],
        events: [
          { time: 0.1, kind: "ball-cushion", balls: [8], cushion: "top" },
          { time: 1, kind: "stop", balls: [] },
        ],
      }),
    );
    expect(out.gameOver).toBe(true);
    expect(out.winner).toBe(0);
  });
});

describe("ball in hand after a scratch", () => {
  it("keeps the cue ball off the table until placed, then only in the kitchen", () => {
    const { state, table } = makeGame();
    state.broken = true;
    // Clear the rack away from the side pocket and fire the cue ball into it.
    for (const b of state.balls) if (b.id !== CUE_ID) b.pocketed = true;
    const one = state.balls.find((b) => b.id === 1)!;
    one.pocketed = false;
    one.pos = { x: 0.7, y: -0.3 };
    const cue = state.balls.find((b) => b.id === CUE_ID)!;
    cue.pos = { x: 0, y: 0 };
    const report = takeShot(state, table, {
      phi: Math.PI / 2,
      power: 0.3,
      sideSpin: 0,
      topSpin: 0,
    });
    expect(report.outcome.pocketedThisShot).toContain(CUE_ID);
    expect(report.next.ballInHand).toBe("behind-head");
    // Not parked anywhere visible: it stays pocketed until someone puts it down.
    expect(report.next.balls.find((b) => b.id === CUE_ID)!.pocketed).toBe(true);
    const lim = cueSpotLimits(report.next, table);
    expect(lim.maxX).toBeCloseTo(HEAD_STRING_X, 9);
    const placed = placeCueBall(report.next, -0.7, 0.1);
    expect(placed.ballInHand).toBe(false);
    expect(placed.balls.find((b) => b.id === CUE_ID)!.pocketed).toBe(false);
  });

  it("the AI only ever places the cue ball behind the head string", () => {
    const { state, table } = makeGame();
    state.broken = true;
    state.ballInHand = "behind-head";
    state.balls.find((b) => b.id === CUE_ID)!.pocketed = true;
    const spot = bestCuePlacement(state, table, makeRanker("classical"));
    expect(spot).not.toBeNull();
    expect(spot!.x).toBeLessThanOrEqual(HEAD_STRING_X + 1e-9);
  });
});

describe("full-shot integration through physics", () => {
  it("break shot resolves without error and marks the game broken", () => {
    const { state, table } = makeGame();
    // A firm break straight into the rack.
    const report = takeShot(state, table, {
      phi: 0,
      power: 0.9,
      sideSpin: 0,
      topSpin: 0,
    });
    expect(report.next.broken).toBe(true);
    expect(report.sim.firstContact).not.toBeNull();
    // Every ball is at rest or pocketed after the shot resolves.
    for (const b of report.next.balls) {
      expect(b.pocketed || Math.hypot(b.vel.x, b.vel.y) < 0.01).toBe(true);
    }
  });
});
