// Ball-in-hand puts the cue ball somewhere legal, or the shot starts wrong.
//
// A placement is a raw canvas point: a tap 5 mm from an object ball's centre is
// a perfectly ordinary thing for a visitor to do, and it puts the cue ball
// inside that object ball. Nothing downstream treats that as an error — the
// simulation engine simply separates the pair on the first frame of the shot,
// so both balls leap 27 mm apart before the cue is struck. Two balls moving on
// their own is the single most damaging thing this app can show, because it is
// indistinguishable from the physics being broken, and the cue ball is always
// one of the two.
//
// So the placement is resolved where it is made, by moving the cue ball only.

import { describe, it, expect } from "vitest";
import { makeTable } from "../physics/table";
import { makeBall, type Ball } from "../physics/ball";
import { BALL_DIAMETER, BALL_RADIUS, ENGINE_MIN_GAP } from "../physics/constants";
import { CUE_ID } from "./rack";
import { clearCuePlacement, makeGame, placeCueBall } from "./game";
import type { GameState } from "./state";

const table = makeTable();

const stateWith = (balls: Ball[]): GameState => ({
  ...makeGame().state,
  balls,
  broken: true,
  ballInHand: "anywhere",
});

const clearOf = (p: { x: number; y: number }, balls: Ball[]) => {
  let min = Infinity;
  for (const b of balls) {
    if (b.id === CUE_ID || b.pocketed) continue;
    min = Math.min(min, Math.hypot(p.x - b.pos.x, p.y - b.pos.y));
  }
  return min;
};

describe("ball-in-hand placement", () => {
  const board = (): Ball[] => [
    makeBall(CUE_ID, -0.6, 0),
    makeBall(1, 0.2, 0),
    makeBall(8, 0.5, 0.1),
  ];

  it("leaves a placement with room exactly where it was asked for", () => {
    const balls = board();
    for (const p of [
      { x: -0.6, y: 0 },
      { x: 0, y: 0.3 },
      { x: 0.8, y: -0.2 },
    ]) {
      expect(clearCuePlacement(balls, p.x, p.y, table)).toEqual(p);
    }
  });

  it("clears a placement made on top of an object ball", () => {
    const balls = board();
    // 5 mm from ball 1's centre: deep inside it.
    const p = clearCuePlacement(balls, 0.205, 0, table);
    expect(clearOf(p, balls)).toBeGreaterThanOrEqual(BALL_DIAMETER + ENGINE_MIN_GAP - 1e-9);
  });

  it("moves the cue ball and nothing else", () => {
    const before = board();
    const next = placeCueBall(stateWith(before.map((b) => ({ ...b }))), 0.205, 0, table);
    for (const b of next.balls) {
      if (b.id === CUE_ID) continue;
      const was = before.find((x) => x.id === b.id)!;
      expect(b.pos).toEqual(was.pos);
    }
    expect(next.ballInHand).toBe(false);
  });

  it("keeps the cue ball inside the cushions however far outside the tap was", () => {
    const balls = board();
    const hx = table.length / 2 - BALL_RADIUS;
    const hy = table.width / 2 - BALL_RADIUS;
    for (const p of [
      { x: 99, y: 99 },
      { x: -99, y: 0 },
      { x: 0, y: -99 },
    ]) {
      const out = clearCuePlacement(balls, p.x, p.y, table);
      expect(Math.abs(out.x)).toBeLessThanOrEqual(hx + 1e-9);
      expect(Math.abs(out.y)).toBeLessThanOrEqual(hy + 1e-9);
    }
  });

  it("finds a legal spot even when the tap lands in the middle of a cluster", () => {
    // A tight wall of balls: pushing out of one lands the cue in the next.
    const balls: Ball[] = [makeBall(CUE_ID, -0.8, 0)];
    for (let i = 0; i < 7; i++) {
      balls.push(makeBall(i + 1, 0, (i - 3) * (BALL_DIAMETER + 0.0001)));
    }
    const p = clearCuePlacement(balls, 0, 0, table);
    expect(clearOf(p, balls)).toBeGreaterThanOrEqual(BALL_DIAMETER + ENGINE_MIN_GAP - 1e-9);
  });

  it("clears a placement wedged against a cushion", () => {
    // Against the end rail with a ball sitting right there, so pushing straight
    // out of the overlap runs the cue into the cushion.
    const railX = table.length / 2 - BALL_RADIUS;
    const balls: Ball[] = [makeBall(CUE_ID, -0.8, 0), makeBall(1, railX, 0)];
    const p = clearCuePlacement(balls, railX, 0.002, table);
    expect(Math.abs(p.x)).toBeLessThanOrEqual(railX + 1e-9);
    expect(clearOf(p, balls)).toBeGreaterThanOrEqual(BALL_DIAMETER + ENGINE_MIN_GAP - 1e-9);
  });
});
