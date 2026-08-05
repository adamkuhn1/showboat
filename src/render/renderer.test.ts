// Renderer invariants that are cheap to check without rasterising.
//
// These exist because a real rendering defect (accumulating cue-stick pixels
// outside the table box, because `render` repainted only the table+rail
// rectangle and `drawAim` draws the stick unclipped) survived every headless
// suite and was only found by looking at a screenshot from a real browser.
// A recording stub is not a substitute for looking, but it does pin the one
// property that would have caught it.

import { describe, it, expect } from "vitest";
import { makeTable } from "../physics/table";
import { makeBall } from "../physics/ball";
import { CUE_ID } from "../game/rack";
import type { GameState } from "../game/state";
import { computeView, render } from "./renderer";

const CANVAS_W = 900;
const CANVAS_H = 500;

/** Records the calls `render` makes, so ordering can be asserted. */
function recordingContext() {
  const calls: { fn: string; args: unknown[] }[] = [];
  const noop = (fn: string) => (...args: unknown[]) => {
    calls.push({ fn, args });
    return undefined;
  };
  const gradient = { addColorStop: noop("addColorStop") };
  const ctx = {
    canvas: { width: CANVAS_W, height: CANVAS_H },
    clearRect: noop("clearRect"),
    fillRect: noop("fillRect"),
    strokeRect: noop("strokeRect"),
    beginPath: noop("beginPath"),
    closePath: noop("closePath"),
    moveTo: noop("moveTo"),
    lineTo: noop("lineTo"),
    arc: noop("arc"),
    ellipse: noop("ellipse"),
    rect: noop("rect"),
    clip: noop("clip"),
    fill: noop("fill"),
    stroke: noop("stroke"),
    save: noop("save"),
    restore: noop("restore"),
    translate: noop("translate"),
    rotate: noop("rotate"),
    scale: noop("scale"),
    fillText: noop("fillText"),
    measureText: () => ({ width: 10 }),
    setLineDash: noop("setLineDash"),
    createLinearGradient: () => gradient,
    createRadialGradient: () => gradient,
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 0,
    lineCap: "butt",
    lineJoin: "miter",
    font: "",
    textAlign: "left",
    textBaseline: "alphabetic",
    globalAlpha: 1,
    shadowBlur: 0,
    shadowColor: "",
  };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, calls };
}

const state: GameState = {
  balls: [makeBall(CUE_ID, -0.4, 0.05), makeBall(1, 0.25, 0.15), makeBall(9, 0.5, -0.2)],
  turn: 0,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 1,
};

describe("render", () => {
  it("clears the entire canvas before drawing anything", () => {
    const { ctx, calls } = recordingContext();
    const table = makeTable();
    render(ctx, state, table, computeView(CANVAS_W, CANVAS_H, table));

    const first = calls.find((c) => c.fn === "clearRect" || c.fn === "fillRect");
    expect(first, "render drew nothing").toBeDefined();
    expect(first!.fn, "the first paint op must be a full clear, not a partial repaint").toBe(
      "clearRect",
    );
    expect(first!.args).toEqual([0, 0, CANVAS_W, CANVAS_H]);
  });

  it("clears using the canvas's intrinsic size, not a hard-coded one", () => {
    const { ctx, calls } = recordingContext();
    (ctx.canvas as { width: number }).width = 1234;
    (ctx.canvas as { height: number }).height = 567;
    const table = makeTable();
    render(ctx, state, table, computeView(1234, 567, table));
    const clear = calls.find((c) => c.fn === "clearRect")!;
    expect(clear.args).toEqual([0, 0, 1234, 567]);
  });
});
