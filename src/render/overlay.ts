// The reasoning overlay: routes on the felt.
//
// Everything drawn here comes from a `PresentationFrame`, which is computed by
// `presentation.ts` from the real `DecisionTraceV1`. This file makes no
// judgement of its own: it decides where a line goes, never why a line died.
// If the trace did not carry a rejection reason, none is printed — the route
// simply dims. That is a cut, not a fabrication.
//
// Role is the primary visual channel, because kind is not: on a typical table
// most candidates are banks, so colouring by kind left every line looking the
// same. Kind survives as a dash signature, which does not compete with the
// brightness that carries "this one is still alive".

import { type Vec2 } from "../physics/vec";
import { type ViewTransform } from "./renderer";
import { BALL_RADIUS } from "../physics/constants";
import {
  REJECTION_TEXT,
  type MeasuredLeg,
  type PresentationFrame,
  type RouteRender,
  type RouteRole,
} from "./presentation";
import { type ContactMark } from "./annotate";

const toPx = (p: { x: number; y: number }, v: ViewTransform): [number, number] => [
  v.offsetX + p.x * v.scale,
  v.offsetY - p.y * v.scale,
];

/** Dash signature per candidate kind. Solid = no cushion in the object route. */
const DASH: Record<string, number[]> = {
  direct: [],
  bank: [9, 6],
  "double-bank": [16, 6],
  combo: [3, 4],
  "rail-combo": [12, 4, 3, 4],
  "safety-kick": [5, 5],
};

const ROLE_STROKE: Record<RouteRole, string> = {
  // Chalk on felt, not UI colour.
  candidate: "215, 226, 232",
  verified: "168, 224, 190",
  rejected: "196, 122, 108",
  selected: "126, 233, 174",
};

/** Total length of a polyline in pixels. */
function pxLength(pts: [number, number][]): number {
  let n = 0;
  for (let i = 1; i < pts.length; i++) n += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  return n;
}

/** Stroke the first `fraction` of a polyline, by arc length. */
function strokePartial(
  ctx: CanvasRenderingContext2D,
  pts: [number, number][],
  fraction: number,
): void {
  if (pts.length < 2) return;
  const f = Math.max(0, Math.min(1, fraction));
  if (f === 0) return;
  const target = pxLength(pts) * f;

  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  let travelled = 0;
  for (let i = 1; i < pts.length; i++) {
    const [x0, y0] = pts[i - 1];
    const [x1, y1] = pts[i];
    const seg = Math.hypot(x1 - x0, y1 - y0);
    if (travelled + seg >= target) {
      const t = seg === 0 ? 1 : (target - travelled) / seg;
      ctx.lineTo(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
      break;
    }
    ctx.lineTo(x1, y1);
    travelled += seg;
  }
  ctx.stroke();
}

/**
 * Stroke a measured leg up to simulation time `t`.
 *
 * The cut lands where the ball actually was at `t`, because every point carries
 * its own simulation time — the final partial segment is interpolated between
 * two recorded positions rather than eyeballed by arc length. `t === null`
 * draws the whole leg, which is the pre-stroke case: the route is known in full
 * before the cue moves, and pretending otherwise would be theatre.
 */
function strokeMeasured(
  ctx: CanvasRenderingContext2D,
  leg: MeasuredLeg,
  v: ViewTransform,
  t: number | null,
): void {
  const pts = leg.points;
  if (pts.length < 2) return;
  const times = leg.timesSec;
  if (t !== null && t <= times[0]) return;

  ctx.beginPath();
  const [x0, y0] = toPx(pts[0], v);
  ctx.moveTo(x0, y0);
  for (let i = 1; i < pts.length; i++) {
    if (t !== null && times[i] > t) {
      const span = times[i] - times[i - 1];
      const f = span <= 0 ? 1 : (t - times[i - 1]) / span;
      const [ax, ay] = toPx(pts[i - 1], v);
      const [bx, by] = toPx(pts[i], v);
      ctx.lineTo(ax + (bx - ax) * f, ay + (by - ay) * f);
      ctx.stroke();
      return;
    }
    const [x, y] = toPx(pts[i], v);
    ctx.lineTo(x, y);
  }
  ctx.stroke();
}

/**
 * The measured route of the shot being played: the cue ball, the ball that
 * drops, and any ball in between. Solid, because it is not a plan — dashes on
 * this canvas mean "intended".
 */
function drawMeasured(
  ctx: CanvasRenderingContext2D,
  r: RouteRender,
  v: ViewTransform,
  simTime: number | null,
): void {
  const m = r.measured;
  if (!m) return;
  const rgb = ROLE_STROKE[r.role];
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.setLineDash([]);

  if (m.cue) {
    ctx.strokeStyle = `rgba(${rgb}, ${r.alpha * 0.55})`;
    ctx.lineWidth = 1 + r.weight * 1.4;
    strokeMeasured(ctx, m.cue, v, simTime);
  }
  for (const other of m.others) {
    ctx.strokeStyle = `rgba(${rgb}, ${r.alpha * 0.6})`;
    ctx.lineWidth = 1 + r.weight * 1.8;
    strokeMeasured(ctx, other, v, simTime);
  }
  if (m.object) {
    ctx.strokeStyle = `rgba(${rgb}, ${r.alpha})`;
    ctx.lineWidth = 1 + r.weight * 2.6;
    strokeMeasured(ctx, m.object, v, simTime);
  }
}

function drawRoute(
  ctx: CanvasRenderingContext2D,
  r: RouteRender,
  v: ViewTransform,
  simTime: number | null,
): void {
  if (r.alpha <= 0.01) return;
  if (r.source === "simulated") {
    drawMeasured(ctx, r, v, simTime);
    return;
  }
  const rgb = ROLE_STROKE[r.role];
  const objectPts = r.objectLeg.map((p) => toPx(p, v));

  // The cue leg first — this is the fix for the chosen line "floating" with no
  // connection to the white ball. `candidate.path` starts at the OBJECT ball;
  // the cue's own travel is [cue ball, ghost-ball contact point] and is what
  // makes the picture tell the shot.
  // Drawn only for routes that are still in play. Forty cue legs radiating
  // from one point is a starburst, not a picture — during VERIFYING the dead
  // routes keep their object leg (so you can see what was considered) and lose
  // the leg that says "the cue would go here".
  const showCueLeg =
    r.role === "selected" || r.role === "verified" || r.resolving || r.role === "candidate";
  if (r.cueLeg && showCueLeg) {
    const cuePts = r.cueLeg.map((p) => toPx(p, v));
    ctx.strokeStyle = `rgba(${rgb}, ${r.alpha * 0.7})`;
    ctx.lineWidth = 1 + r.weight * 1.6;
    ctx.setLineDash([2, 5]);
    strokePartial(ctx, cuePts, r.reveal);
    ctx.setLineDash([]);
  }

  if (objectPts.length >= 2) {
    ctx.strokeStyle = `rgba(${rgb}, ${r.alpha})`;
    ctx.lineWidth = 1 + r.weight * 2.6;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.setLineDash(DASH[r.kind] ?? []);
    strokePartial(ctx, objectPts, r.reveal);
    ctx.setLineDash([]);
  }

  // Ghost-ball contact circle: where the cue must arrive for this route.
  if (r.role === "selected" && r.cueLeg) {
    const [gx, gy] = toPx(r.cueLeg[1], v);
    ctx.strokeStyle = `rgba(${rgb}, ${r.alpha * 0.85})`;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(gx, gy, BALL_RADIUS * v.scale, 0, Math.PI * 2);
    ctx.stroke();
  }
}

/**
 * The elimination reason, on the felt beside the route it belongs to — never
 * as a list in the panel, which would grow the panel and shrink the table.
 * Drawn for one route at a time: the one resolving this instant.
 */
function drawReason(ctx: CanvasRenderingContext2D, r: RouteRender, v: ViewTransform): void {
  if (r.reason === null) return;
  const pts = r.objectLeg;
  if (pts.length === 0) return;
  const anchor: Vec2 = pts[Math.floor(pts.length / 2)];
  const [ax, ay] = toPx(anchor, v);
  ctx.font = "500 11px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const text = REJECTION_TEXT[r.reason];
  const w = ctx.measureText(text).width;
  // Keep the caption on the cloth: anchored at a route midpoint it otherwise
  // runs over the rail and reads as a truncated word. `INSET` is the rail
  // width drawn by `drawTable`, plus a little air.
  const INSET = 30;
  const x = Math.min(Math.max(ax + 6, INSET), ctx.canvas.width - w - INSET);
  const y = Math.min(Math.max(ay, INSET), ctx.canvas.height - INSET);
  ctx.fillStyle = `rgba(8, 12, 10, ${0.62 * r.alpha + 0.2})`;
  ctx.fillRect(x, y - 8, w + 8, 16);
  ctx.fillStyle = `rgba(${ROLE_STROKE.rejected}, ${Math.max(0.55, r.alpha)})`;
  ctx.fillText(text, x + 4, y + 1);
}

/** How long a contact stays emphasised after it happens, in simulation seconds. */
const MARK_FLASH_SEC = 0.25;

/**
 * Cushion contacts and combination order from the executed simulation.
 *
 * Two readings, and they are different claims:
 *   - `simTime === null` — before the cue moves. Every mark is drawn: this is
 *     the whole verified route, contacts included, and it is known in full.
 *   - a number — during playback. A contact appears when it OCCURS and not
 *     before. It used to be the other way round: every mark was bright from the
 *     first frame and dimmed once the ball passed it, so the table showed the
 *     end of the shot while the balls were still on their way to it.
 */
export function drawContactMarks(
  ctx: CanvasRenderingContext2D,
  marks: ContactMark[],
  v: ViewTransform,
  simTime: number | null,
  alpha = 1,
): void {
  for (const m of marks) {
    let a: number;
    if (simTime === null) {
      a = 0.9 * alpha;
    } else {
      if (simTime < m.timeSec) continue;
      a = (simTime - m.timeSec < MARK_FLASH_SEC ? 0.95 : 0.55) * alpha;
    }
    if (a <= 0.02) continue;
    const [x, y] = toPx(m.at, v);
    const r = m.kind === "cushion" ? 5 : 4;

    ctx.strokeStyle = `rgba(126, 233, 174, ${a})`;
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    if (m.kind === "cushion") {
      // A tick, not a ring: a cushion contact is a bounce off a line.
      ctx.moveTo(x - r, y - r);
      ctx.lineTo(x + r, y + r);
      ctx.moveTo(x + r, y - r);
      ctx.lineTo(x - r, y + r);
    } else {
      ctx.arc(x, y, r, 0, Math.PI * 2);
    }
    ctx.stroke();

    // Combination order, so a multi-ball route reads in sequence. Set BESIDE
    // the ring, not inside it: a numbered circle the size of a ball reads as a
    // ball, which is the one thing on this canvas it must not be confused with.
    if (m.kind === "ball") {
      ctx.font = "600 8px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillStyle = `rgba(126, 233, 174, ${a * 0.85})`;
      ctx.fillText(String(m.order), x + r + 2, y - r);
    }
  }
}

/** Draw one presentation frame. This is the whole public surface. */
export function drawPresentation(
  ctx: CanvasRenderingContext2D,
  frame: PresentationFrame,
  v: ViewTransform,
  marks: ContactMark[] = [],
  simTime: number | null = null,
): void {
  ctx.save();
  // Weakest first, so the live routes sit on top of the dead ones.
  const ordered = [...frame.routes].sort((a, b) => a.alpha - b.alpha);
  for (const r of ordered) drawRoute(ctx, r, v, simTime);

  // One caption at a time. During VERIFYING it belongs to the candidate the
  // physics just finished with, so each elimination reads as it happens;
  // during SELECTED it belongs to the loudest survivor that lost, so the
  // dimming is a reason and not a blanket fade.
  const captioned =
    frame.routes.find((r) => r.justResolved && r.reason !== null) ??
    (frame.state === "SELECTED"
      ? frame.routes
          .filter((r) => r.role === "rejected" && r.reason !== null && r.alpha > 0.2)
          .sort((a, b) => b.alpha - a.alpha)[0]
      : undefined);
  if (captioned) drawReason(ctx, captioned, v);

  if (frame.showContacts && marks.length > 0) drawContactMarks(ctx, marks, v, simTime);
  ctx.restore();
}
