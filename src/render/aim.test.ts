import { describe, expect, it } from "vitest";
import { makeBall } from "../physics/ball";
import { makeTable } from "../physics/table";
import { BALL_RADIUS, BALL_DIAMETER } from "../physics/constants";
import { aimContact } from "./aim";

// Expected values are worked out by hand from circle/line geometry, not by
// re-running aimContact.

const table = makeTable();
const R = BALL_RADIUS;
const D = BALL_DIAMETER;

describe("aimContact", () => {
  it("full-ball hit: ghost sits one diameter short of the object ball", () => {
    const cue = makeBall(0, -0.5, 0);
    const obj = makeBall(1, 0.3, 0);
    const c = aimContact(cue, 0, [cue, obj], table);
    expect(c.kind).toBe("ball");
    if (c.kind !== "ball") return;
    expect(c.target.id).toBe(1);
    expect(c.ghost.x).toBeCloseTo(0.3 - D, 9);
    expect(c.ghost.y).toBeCloseTo(0, 9);
    expect(c.distance).toBeCloseTo(0.8 - D, 9);
    expect(c.objectDir.x).toBeCloseTo(1, 9);
    expect(c.objectDir.y).toBeCloseTo(0, 9);
  });

  it("cut shot: object ball is sent along the line of centres", () => {
    // Object ball offset by one radius: a half-ball hit. The ghost centre is
    // where the line y=0 meets the circle of radius D around (0.3, R):
    // x = 0.3 - sqrt(D^2 - R^2) = 0.3 - R*sqrt(3).
    const cue = makeBall(0, -0.5, 0);
    const obj = makeBall(1, 0.3, R);
    const c = aimContact(cue, 0, [cue, obj], table);
    expect(c.kind).toBe("ball");
    if (c.kind !== "ball") return;
    expect(c.ghost.x).toBeCloseTo(0.3 - R * Math.sqrt(3), 9);
    expect(c.ghost.y).toBeCloseTo(0, 9);
    // Half-ball hit: the object ball leaves at 30 degrees to the aim line.
    expect(c.objectDir.x).toBeCloseTo(Math.cos(Math.PI / 6), 9);
    expect(c.objectDir.y).toBeCloseTo(Math.sin(Math.PI / 6), 9);
  });

  it("a ball just over one diameter off the line is missed", () => {
    const cue = makeBall(0, -0.5, 0);
    const obj = makeBall(1, 0.3, D + 1e-4);
    expect(aimContact(cue, 0, [cue, obj], table).kind).toBe("cushion");
  });

  it("returns the nearer of two balls on the line", () => {
    const cue = makeBall(0, -0.5, 0);
    const far = makeBall(2, 0.6, 0);
    const near = makeBall(3, 0.1, 0.01);
    const c = aimContact(cue, 0, [cue, far, near], table);
    expect(c.kind === "ball" && c.target.id).toBe(3);
  });

  it("ignores pocketed balls and balls behind the cue ball", () => {
    const cue = makeBall(0, 0, 0);
    const pocketed = { ...makeBall(1, 0.3, 0), pocketed: true };
    const behind = makeBall(2, -0.3, 0);
    expect(aimContact(cue, 0, [cue, pocketed, behind], table).kind).toBe("cushion");
  });

  it("with nothing in the way, stops one radius short of the first cushion", () => {
    const cue = makeBall(0, 0, 0);
    const right = aimContact(cue, 0, [cue], table);
    expect(right.kind).toBe("cushion");
    expect(right.ghost.x).toBeCloseTo(table.length / 2 - R, 9);
    const up = aimContact(cue, Math.PI / 2, [cue], table);
    expect(up.ghost.y).toBeCloseTo(table.width / 2 - R, 9);
    expect(up.ghost.x).toBeCloseTo(0, 9);
  });
});
