// Which way the cue ball goes when you drag, and what a drag is allowed to do.
//
// The aim angle is one `atan2` over an inverted y axis, which is exactly the
// kind of expression that can be wrong by a sign for a long time without
// looking wrong in code review. It shipped inverted: the cue ball travelled
// AWAY from the pointer while the UI said "drag to aim", so aiming at the ball
// you wanted to hit sent the cue 180 degrees the other way. Pinned here in the
// only terms that matter — a canvas point, and where the ball ends up.
//
// The second half is about authority rather than geometry. A pointer drag may
// aim and may do nothing else; it may not set power and it may not take the
// shot. Checked against the source, because the rule is about which handlers
// exist rather than about a value one of them returns.

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../physics/table";
import { aimTowards, computeView } from "./renderer";

const SRC = join(dirname(fileURLToPath(import.meta.url)), "..");
const table = makeTable();
const view = computeView(900, 500, table);
const toPx = (x: number, y: number): [number, number] => [
  view.offsetX + x * view.scale,
  view.offsetY - y * view.scale,
];

describe("dragging aims the cue ball at the pointer", () => {
  const cue = { x: -0.5, y: 0.1 };

  it("points at a target the pointer is resting on, in every direction", () => {
    // Eight targets around the cue ball. The aim angle must be the world-space
    // bearing from the cue ball to that target, which is the definition of
    // "aim at what you are pointing at".
    for (let k = 0; k < 8; k++) {
      const bearing = (k * Math.PI) / 4;
      const target = { x: cue.x + 0.3 * Math.cos(bearing), y: cue.y + 0.3 * Math.sin(bearing) };
      const angle = aimTowards(cue, ...toPx(target.x, target.y), view);
      // Compared as a direction, so ±π does not read as a failure.
      expect(Math.cos(angle - bearing), `bearing ${bearing}`).toBeCloseTo(1, 9);
    }
  });

  it("aims UP the table when the pointer is ABOVE the cue ball on screen", () => {
    // The sign that was wrong. Canvas y grows downward, table y grows upward.
    const [px, py] = toPx(cue.x, cue.y);
    expect(Math.sin(aimTowards(cue, px, py - 100, view))).toBeGreaterThan(0.99);
    expect(Math.sin(aimTowards(cue, px, py + 100, view))).toBeLessThan(-0.99);
    expect(Math.cos(aimTowards(cue, px + 100, py, view))).toBeGreaterThan(0.99);
    expect(Math.cos(aimTowards(cue, px - 100, py, view))).toBeLessThan(-0.99);
  });

  it("is unchanged by how far away the pointer is — distance is not power", () => {
    const [px, py] = toPx(cue.x, cue.y);
    const near = aimTowards(cue, px + 5, py - 5, view);
    const far = aimTowards(cue, px + 400, py - 400, view);
    expect(near).toBeCloseTo(far, 9);
  });
});

describe("a pointer drag aims and does nothing else", () => {
  const app = readFileSync(join(SRC, "App.tsx"), "utf8");
  const code = app
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .split("\n")
    .filter((l) => !/^\s*(\/\/|\*)/.test(l))
    .join("\n");

  it("the drag handlers set the aim and no other shot parameter", () => {
    const handlers = code.slice(code.indexOf("const aimAt ="), code.indexOf("const shoot ="));
    expect(handlers).toContain("setAim(aimTowards(cue.pos, px, py, view))");
    expect(handlers, "a drag sets the power").not.toContain("setPower(");
    expect(handlers, "a drag sets spin").not.toMatch(/setSide\(|setTop\(/);
  });

  it("releasing a drag cannot take the shot", () => {
    const endDrag = code.slice(code.indexOf("const endDrag ="), code.indexOf("const shoot ="));
    expect(endDrag, "release fires a shot").not.toContain("shootRef.current()");
    expect(endDrag).toContain("releasePointerCapture");
  });

  it("the shot is taken only from the button, the keyboard, or the opponent", () => {
    // Three call sites and no more: the Shoot button, the space key, and the
    // ref the two of them share.
    const calls = code.match(/shootRef\.current\(\)/g) ?? [];
    expect(calls.length).toBe(1);
    expect(code).toContain("onClick={shoot}");
  });
});
