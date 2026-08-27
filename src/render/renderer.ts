import { type Ball } from "../physics/ball";
import { type Table } from "../physics/table";
import { BALL_RADIUS } from "../physics/constants";
import { type GameState } from "../game/state";
import { CUE_ID, EIGHT_ID, SOLIDS } from "../game/rack";

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
  const margin = 40;
  const usableW = canvasW - margin * 2;
  const usableH = canvasH - margin * 2;
  const scale = Math.min(usableW / table.length, usableH / table.width);
  return { scale, offsetX: canvasW / 2, offsetY: canvasH / 2 };
};

const toPx = (x: number, y: number, v: ViewTransform): [number, number] => [
  v.offsetX + x * v.scale,
  v.offsetY - y * v.scale, // flip y so +y is up on screen
];

export const drawTable = (
  ctx: CanvasRenderingContext2D,
  table: Table,
  v: ViewTransform,
): void => {
  const hx = table.length / 2;
  const hy = table.width / 2;
  const railPx = 22;

  // Rail frame.
  const [x0, y0] = toPx(-hx, hy, v);
  const w = table.length * v.scale;
  const h = table.width * v.scale;
  ctx.fillStyle = "#3a2416";
  ctx.fillRect(x0 - railPx, y0 - railPx, w + railPx * 2, h + railPx * 2);

  // Cloth.
  ctx.fillStyle = "#12603a";
  ctx.fillRect(x0, y0, w, h);

  // Head string (behind-line for the break).
  const [hsx1, hsy1] = toPx(-table.length / 4, hy, v);
  const [, hsy2] = toPx(-table.length / 4, -hy, v);
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(hsx1, hsy1);
  ctx.lineTo(hsx1, hsy2);
  ctx.stroke();

  // Pockets.
  ctx.fillStyle = "#0a0a0a";
  for (const p of table.pockets) {
    const [px, py] = toPx(p.center.x, p.center.y, v);
    ctx.beginPath();
    ctx.arc(px, py, p.radius * v.scale, 0, Math.PI * 2);
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

  if (ball.id === CUE_ID) {
    ctx.fillStyle = "#f7f5ee";
  } else {
    ctx.fillStyle = BALL_COLORS[ball.id] ?? BALL_COLORS[((ball.id - 1) % 7) + 1];
  }

  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fill();

  // Stripe band for high balls.
  if (ball.id > EIGHT_ID) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.clip();
    ctx.fillStyle = "#f7f5ee";
    ctx.fillRect(px - r, py - r * 0.42, r * 2, r * 0.84);
    ctx.restore();
  }

  // Number pip for non-cue balls.
  if (ball.id !== CUE_ID) {
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(px, py, r * 0.42, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#111";
    ctx.font = `${Math.round(r * 0.7)}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(ball.id), px, py + 0.5);
  }

  // Subtle highlight.
  ctx.fillStyle = "rgba(255,255,255,0.18)";
  ctx.beginPath();
  ctx.arc(px - r * 0.3, py - r * 0.3, r * 0.35, 0, Math.PI * 2);
  ctx.fill();
};

// Draw the aim line + power indicator for the human shooter.
export const drawAim = (
  ctx: CanvasRenderingContext2D,
  cue: Ball,
  phi: number,
  power: number,
  v: ViewTransform,
): void => {
  if (cue.pocketed) return;
  const [cx, cy] = toPx(cue.pos.x, cue.pos.y, v);
  const len = (0.3 + power * 0.7) * table_diag(v);
  const dx = Math.cos(phi);
  const dy = -Math.sin(phi); // screen y flip
  ctx.strokeStyle = "rgba(255,255,255,0.8)";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + dx * len, cy + dy * len);
  ctx.stroke();
  ctx.setLineDash([]);
};

const table_diag = (v: ViewTransform): number => Math.max(140, v.scale * 0.6);

export const render = (
  ctx: CanvasRenderingContext2D,
  state: GameState,
  table: Table,
  v: ViewTransform,
): void => {
  drawTable(ctx, table, v);
  // Draw eight last-but-cue-first-ish; order by id so overlaps look natural.
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
