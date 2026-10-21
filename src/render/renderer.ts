import { type Ball } from "../physics/ball";
import { type Table, FOOT_SPOT, HEAD_SPOT, HEAD_STRING_X } from "../physics/table";
import { BALL_RADIUS } from "../physics/constants";
import { type GameState } from "../game/state";
import { CUE_ID, EIGHT_ID, SOLIDS } from "../game/rack";
import { aimContact } from "./aim";

// Canvas renderer for the 2D bar-pool table. Draws in metres, mapping the table
// coordinate system (origin at centre) to canvas pixels with a fixed scale plus
// margin for the rails. Pure drawing — takes state, no game logic here.

export interface ViewTransform {
  scale: number; // pixels per metre
  offsetX: number; // canvas px of table origin
  offsetY: number;
}

// Standard pool ball colors by id (solids 1-7, eight 8, stripes 9-15 share the
// solid's hue with a white band drawn over them).
const BALL_COLORS: Record<number, string> = {
  1: "#f4c724", // yellow
  2: "#1f4fd8", // blue
  3: "#e23c2e", // red
  4: "#6b2fb3", // purple
  5: "#e8792b", // orange
  6: "#1f8a4c", // green
  7: "#8c2f2a", // maroon
  8: "#111111", // black
};

export const computeView = (
  canvasW: number,
  canvasH: number,
  table: Table,
): ViewTransform => {
  const margin = 44;
  const usableW = canvasW - margin * 2;
  const usableH = canvasH - margin * 2;
  const scale = Math.min(usableW / table.length, usableH / table.width);
  return { scale, offsetX: canvasW / 2, offsetY: canvasH / 2 };
};

const toPx = (x: number, y: number, v: ViewTransform): [number, number] => [
  v.offsetX + x * v.scale,
  v.offsetY - y * v.scale, // flip y so +y is up on screen
];

// Wood, cloth and fittings. Picked off a well-kept bar box: walnut rails,
// tournament-green cloth a shade lighter under the lamp, pearl diamond sights.
const WOOD_LIGHT = "#5a3a22";
const WOOD_DARK = "#2e1c10";
const CLOTH = "#17704a";
const CLOTH_EDGE = "#0f5537";
const CUSHION = "#125a3b";
const SIGHT = "#e9e2cf";
const CUE_WHITE = "#f6f2e6";

export const drawTable = (
  ctx: CanvasRenderingContext2D,
  table: Table,
  v: ViewTransform,
  opts: { kitchen?: boolean } = {},
): void => {
  const hx = table.length / 2;
  const hy = table.width / 2;
  const railPx = 32;
  const cushionPx = 5;

  const [x0, y0] = toPx(-hx, hy, v);
  const w = table.length * v.scale;
  const h = table.width * v.scale;

  // Rail frame: walnut, lit from the same lamp as the cloth, with a thin
  // bevel where the wood meets the rubber.
  const rx = x0 - railPx;
  const ry = y0 - railPx;
  const rw = w + railPx * 2;
  const rh = h + railPx * 2;
  const wood = ctx.createRadialGradient(
    x0 + w / 2, y0 + h / 2, h * 0.3,
    x0 + w / 2, y0 + h / 2, rw * 0.6,
  );
  wood.addColorStop(0, WOOD_LIGHT);
  wood.addColorStop(1, WOOD_DARK);
  ctx.fillStyle = wood;
  roundRect(ctx, rx, ry, rw, rh, 12);
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.55)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.strokeStyle = "rgba(255,226,180,0.12)";
  ctx.strokeRect(x0 - cushionPx - 2.5, y0 - cushionPx - 2.5, w + cushionPx * 2 + 5, h + cushionPx * 2 + 5);

  // Diamond sights: three between each pair of pockets on the long rails,
  // three on each short rail, at quarter points, set into the middle of
  // the wood.
  ctx.fillStyle = SIGHT;
  const mid = railPx / 2 + cushionPx / 2;
  for (let i = 1; i < 8; i++) {
    if (i === 4) continue; // side pocket
    const sx = x0 + (w * i) / 8;
    sight(ctx, sx, y0 - mid);
    sight(ctx, sx, y0 + h + mid);
  }
  for (let i = 1; i < 4; i++) {
    const sy = y0 + (h * i) / 4;
    sight(ctx, x0 - mid, sy);
    sight(ctx, x0 + w + mid, sy);
  }

  // Cloth, lit from a lamp over the middle of the table.
  const lamp = ctx.createRadialGradient(
    x0 + w / 2, y0 + h / 2, h * 0.1,
    x0 + w / 2, y0 + h / 2, w * 0.62,
  );
  lamp.addColorStop(0, CLOTH);
  lamp.addColorStop(1, CLOTH_EDGE);
  ctx.fillStyle = lamp;
  ctx.fillRect(x0 - cushionPx, y0 - cushionPx, w + cushionPx * 2, h + cushionPx * 2);

  // Cushion noses: a darker band just outside the playing line, where the
  // ball actually rebounds.
  ctx.fillStyle = CUSHION;
  ctx.fillRect(x0 - cushionPx, y0 - cushionPx, w + cushionPx * 2, cushionPx);
  ctx.fillRect(x0 - cushionPx, y0 + h, w + cushionPx * 2, cushionPx);
  ctx.fillRect(x0 - cushionPx, y0, cushionPx, h);
  ctx.fillRect(x0 + w, y0, cushionPx, h);

  // Ball in hand after a scratch: lift the kitchen so the legal area is
  // obvious without a word of explanation.
  const [hsx] = toPx(HEAD_STRING_X, 0, v);
  if (opts.kitchen) {
    ctx.fillStyle = "rgba(255,255,255,0.07)";
    ctx.fillRect(x0, y0, hsx - x0, h);
  }

  // Head string, head spot and foot spot, the way they're chalked on a
  // real cloth: faint.
  ctx.strokeStyle = opts.kitchen ? "rgba(255,255,255,0.35)" : "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(hsx + 0.5, y0);
  ctx.lineTo(hsx + 0.5, y0 + h);
  ctx.stroke();
  ctx.fillStyle = "rgba(255,255,255,0.28)";
  for (const spot of [HEAD_SPOT, FOOT_SPOT]) {
    const [sx, sy] = toPx(spot.x, spot.y, v);
    ctx.beginPath();
    ctx.arc(sx, sy, 2, 0, Math.PI * 2);
    ctx.fill();
  }

  // Pockets: a leather-dark ring around the drop.
  for (const p of table.pockets) {
    const [px, py] = toPx(p.center.x, p.center.y, v);
    const r = p.radius * v.scale;
    ctx.fillStyle = "#1b120b";
    ctx.beginPath();
    ctx.arc(px, py, r + 3, 0, Math.PI * 2);
    ctx.fill();
    const drop = ctx.createRadialGradient(px, py, 0, px, py, r);
    drop.addColorStop(0, "#000");
    drop.addColorStop(1, "#0d0d0d");
    ctx.fillStyle = drop;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fill();
  }
};

const sight = (ctx: CanvasRenderingContext2D, x: number, y: number): void => {
  const s = 3.2;
  ctx.beginPath();
  ctx.moveTo(x, y - s);
  ctx.lineTo(x + s * 0.7, y);
  ctx.lineTo(x, y + s);
  ctx.lineTo(x - s * 0.7, y);
  ctx.closePath();
  ctx.fill();
};

const roundRect = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void => {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
};

// Balls get a soft contact shadow (the lamp is overhead, slightly toward
// the viewer) and a spherical shade, drawn over the flat colour and band.
const drawShadows = (
  ctx: CanvasRenderingContext2D,
  balls: { pos: { x: number; y: number }; pocketed: boolean }[],
  v: ViewTransform,
): void => {
  const r = BALL_RADIUS * v.scale;
  ctx.fillStyle = "rgba(0,0,0,0.28)";
  for (const b of balls) {
    if (b.pocketed) continue;
    const [px, py] = toPx(b.pos.x, b.pos.y, v);
    ctx.beginPath();
    ctx.ellipse(px + r * 0.18, py + r * 0.3, r * 1.02, r * 0.92, 0, 0, Math.PI * 2);
    ctx.fill();
  }
};

export const drawBall = (
  ctx: CanvasRenderingContext2D,
  ball: Ball,
  v: ViewTransform,
): void => {
  if (ball.pocketed) return;
  const [px, py] = toPx(ball.pos.x, ball.pos.y, v);
  const r = BALL_RADIUS * v.scale;

  ctx.fillStyle =
    ball.id === CUE_ID
      ? CUE_WHITE
      : BALL_COLORS[ball.id] ?? BALL_COLORS[((ball.id - 1) % 7) + 1];
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fill();

  ctx.save();
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.clip();

  // Stripe band for high balls.
  if (ball.id > EIGHT_ID) {
    ctx.fillStyle = CUE_WHITE;
    ctx.fillRect(px - r, py - r * 0.45, r * 2, r * 0.9);
  }

  // Number circle.
  if (ball.id !== CUE_ID) {
    ctx.fillStyle = CUE_WHITE;
    ctx.beginPath();
    ctx.arc(px, py, r * 0.46, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#16130f";
    ctx.font = `700 ${Math.round(r * (ball.id > 9 ? 0.62 : 0.72))}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(ball.id), px, py + 0.5);
  }

  // Spherical shade: bright where the lamp hits, darker toward the rim.
  const shade = ctx.createRadialGradient(
    px - r * 0.35, py - r * 0.4, r * 0.1,
    px, py, r * 1.05,
  );
  shade.addColorStop(0, "rgba(255,255,255,0.45)");
  shade.addColorStop(0.35, "rgba(255,255,255,0)");
  shade.addColorStop(1, "rgba(0,0,0,0.38)");
  ctx.fillStyle = shade;
  ctx.fillRect(px - r, py - r, r * 2, r * 2);
  ctx.restore();
};

// Aim overlay for the human shooter: a dashed line from the cue ball to the
// first thing it would touch on a straight path. When that is a ball, a ghost
// ball marks where the cue ball's centre will be at contact, and a short line
// off the object ball shows the direction it will be sent (no throw or spin).
export const drawAim = (
  ctx: CanvasRenderingContext2D,
  cue: Ball,
  phi: number,
  balls: readonly Ball[],
  v: ViewTransform,
  table: Table,
): void => {
  if (cue.pocketed) return;
  const contact = aimContact(cue, phi, balls, table);
  const [cx, cy] = toPx(cue.pos.x, cue.pos.y, v);
  const [gx, gy] = toPx(contact.ghost.x, contact.ghost.y, v);
  const r = BALL_RADIUS * v.scale;

  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.8)";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(gx, gy);
  ctx.stroke();
  ctx.setLineDash([]);

  if (contact.kind === "ball") {
    // Ghost ball.
    ctx.fillStyle = "rgba(255,255,255,0.14)";
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(gx, gy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Object-ball direction, clipped to the cloth so it never runs onto the rail.
    const [x0, y0] = toPx(-table.length / 2, table.width / 2, v);
    ctx.beginPath();
    ctx.rect(x0, y0, table.length * v.scale, table.width * v.scale);
    ctx.clip();
    // Starts at the object ball's far edge so it doesn't cover the number.
    const [ox, oy] = toPx(contact.target.pos.x, contact.target.pos.y, v);
    const ux = contact.objectDir.x;
    const uy = -contact.objectDir.y; // screen y flip
    const len = OBJECT_LINE_LENGTH * v.scale;
    ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(ox + ux * r, oy + uy * r);
    ctx.lineTo(ox + ux * (r + len), oy + uy * (r + len));
    ctx.stroke();
  }
  ctx.restore();
};

// Kept short on purpose: it shows the direction, not a promise of where the
// object ball ends up.
const OBJECT_LINE_LENGTH = 0.3; // m

export const render = (
  ctx: CanvasRenderingContext2D,
  state: GameState,
  table: Table,
  v: ViewTransform,
): void => {
  drawTable(ctx, table, v, { kitchen: state.ballInHand === "behind-head" });
  drawShadows(ctx, state.balls, v);
  for (const b of state.balls) drawBall(ctx, b, v);
};

// Draw interpolated playback poses (from the recorded simulation frames).
// Reuses drawBall by constructing a minimal Ball-shaped pose — the renderer
// only reads id/pos/pocketed.
export interface Pose {
  id: number;
  x: number;
  y: number;
  pocketed: boolean;
}

export const renderPoses = (
  ctx: CanvasRenderingContext2D,
  poses: Pose[],
  table: Table,
  v: ViewTransform,
): void => {
  drawTable(ctx, table, v);
  drawShadows(ctx, poses.map((p) => ({ pos: { x: p.x, y: p.y }, pocketed: p.pocketed })), v);
  for (const p of poses) {
    drawBall(
      ctx,
      {
        id: p.id,
        pos: { x: p.x, y: p.y },
        pocketed: p.pocketed,
      } as Ball,
      v,
    );
  }
};

// A polyline route in table coordinates (metres). Used for both the dashed
// hypothetical geometry shown while the AI is still considering a candidate
// and the solid verified route extracted from the chosen simulation's frames.
export const drawRoute = (
  ctx: CanvasRenderingContext2D,
  points: { x: number; y: number }[],
  v: ViewTransform,
  style: "hypothetical" | "verified",
  color: string,
): void => {
  if (points.length < 2) return;
  ctx.save();
  if (style === "hypothetical") {
    ctx.setLineDash([5, 7]);
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.65;
  } else {
    ctx.setLineDash([]);
    ctx.lineWidth = 2.5;
    ctx.globalAlpha = 0.9;
  }
  ctx.strokeStyle = color;
  ctx.beginPath();
  const [x0, y0] = toPx(points[0].x, points[0].y, v);
  ctx.moveTo(x0, y0);
  for (let i = 1; i < points.length; i++) {
    const [px, py] = toPx(points[i].x, points[i].y, v);
    ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.restore();
};

export { SOLIDS };
