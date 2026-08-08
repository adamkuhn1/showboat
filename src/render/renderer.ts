import { type Ball } from "../physics/ball";
import type { Table } from "../physics/table";
import { BALL_RADIUS } from "../physics/constants";
import { type GameState } from "../game/state";
import { CUE_ID, EIGHT_ID, SOLIDS } from "../game/rack";

export interface ViewTransform {
  scale: number;
  offsetX: number;
  offsetY: number;
}

const BALL_COLORS: Record<number, string> = {
  1: "#f4c724",
  2: "#1f4fd8",
  3: "#e23c2e",
  4: "#6b2fb3",
  5: "#e8792b",
  6: "#1f8a4c",
  7: "#8c2f2a",
  8: "#111111",
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
  v.offsetY - y * v.scale,
];

/**
 * The aim angle that sends the cue ball TOWARDS a canvas point.
 *
 * Lives beside the transform it inverts, and is a function rather than four
 * lines in a pointer handler so the one thing about it that matters — which way
 * the ball goes — is pinned by a test. The canvas y axis points down and the
 * table's points up, which is the sign that has to be right.
 */
export const aimTowards = (
  cue: { x: number; y: number },
  canvasX: number,
  canvasY: number,
  v: ViewTransform,
): number => {
  const [cx, cy] = toPx(cue.x, cue.y, v);
  return Math.atan2(-(canvasY - cy), canvasX - cx);
};

export const drawTable = (
  ctx: CanvasRenderingContext2D,
  table: Table,
  v: ViewTransform,
): void => {
  const hx = table.length / 2;
  const hy = table.width / 2;
  const railPx = 22;

  const [x0, y0] = toPx(-hx, hy, v);
  const w = table.length * v.scale;
  const h = table.width * v.scale;

  // Rail frame — base dark wood
  ctx.fillStyle = "#2e1c0e";
  ctx.fillRect(x0 - railPx, y0 - railPx, w + railPx * 2, h + railPx * 2);

  // Wood grain sheen on top rail edge
  const woodGrad = ctx.createLinearGradient(x0, y0 - railPx, x0, y0);
  woodGrad.addColorStop(0, "rgba(255,200,130,0.10)");
  woodGrad.addColorStop(0.5, "rgba(255,200,130,0.04)");
  woodGrad.addColorStop(1, "rgba(0,0,0,0.06)");
  ctx.fillStyle = woodGrad;
  ctx.fillRect(x0 - railPx, y0 - railPx, w + railPx * 2, h + railPx * 2);

  // Inner rail edge highlight — physical ledge illusion
  ctx.strokeStyle = "rgba(160,110,55,0.32)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(x0 + 0.75, y0 + 0.75, w - 1.5, h - 1.5);

  // Cloth base
  ctx.fillStyle = "#11603a";
  ctx.fillRect(x0, y0, w, h);

  // Felt texture — very subtle diagonal hatch
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, w, h);
  ctx.clip();
  ctx.strokeStyle = "rgba(0,0,0,0.05)";
  ctx.lineWidth = 1;
  const step = 9;
  for (let i = -h; i < w + h; i += step) {
    ctx.beginPath();
    ctx.moveTo(x0 + i, y0);
    ctx.lineTo(x0 + i - h, y0 + h);
    ctx.stroke();
  }
  ctx.restore();

  // Cloth vignette — table edges slightly darker
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, w, h);
  ctx.clip();
  const vig = ctx.createRadialGradient(
    x0 + w / 2, y0 + h / 2, Math.min(w, h) * 0.15,
    x0 + w / 2, y0 + h / 2, Math.max(w, h) * 0.65,
  );
  vig.addColorStop(0, "rgba(0,0,0,0)");
  vig.addColorStop(1, "rgba(0,0,0,0.20)");
  ctx.fillStyle = vig;
  ctx.fillRect(x0, y0, w, h);
  ctx.restore();

  // Head string
  const [hsx1, hsy1] = toPx(-table.length / 4, hy, v);
  const [, hsy2] = toPx(-table.length / 4, -hy, v);
  ctx.strokeStyle = "rgba(255,255,255,0.09)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 6]);
  ctx.beginPath();
  ctx.moveTo(hsx1, hsy1);
  ctx.lineTo(hsx1, hsy2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Pockets — with depth shadow
  for (const p of table.pockets) {
    const [px, py] = toPx(p.center.x, p.center.y, v);
    const pr = p.radius * v.scale;

    // Shadow halo
    const pShadow = ctx.createRadialGradient(px, py, pr * 0.2, px, py, pr * 1.6);
    pShadow.addColorStop(0, "rgba(0,0,0,0.85)");
    pShadow.addColorStop(0.6, "rgba(0,0,0,0.42)");
    pShadow.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = pShadow;
    ctx.beginPath();
    ctx.arc(px, py, pr * 1.6, 0, Math.PI * 2);
    ctx.fill();

    // Pocket void
    ctx.fillStyle = "#040404";
    ctx.beginPath();
    ctx.arc(px, py, pr, 0, Math.PI * 2);
    ctx.fill();

    // Leather rim
    ctx.strokeStyle = "rgba(55,32,14,0.65)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, pr + 1, 0, Math.PI * 2);
    ctx.stroke();
    // No pocket-id label here on purpose — a real table doesn't print BL/TR/
    // etc. into its pockets. The reasoning panel still names the target
    // pocket in its own (secondary, non-table) text.
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

  // Drop shadow (offset down-right)
  const shadowGrad = ctx.createRadialGradient(px + 3, py + 5, 0, px + 3, py + 5, r * 1.45);
  shadowGrad.addColorStop(0, "rgba(0,0,0,0.36)");
  shadowGrad.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = shadowGrad;
  ctx.beginPath();
  ctx.ellipse(px + 3, py + 5, r * 1.45, r * 0.78, 0, 0, Math.PI * 2);
  ctx.fill();

  const baseColor =
    ball.id === CUE_ID
      ? "#f0ece0"
      : (BALL_COLORS[ball.id] ?? BALL_COLORS[((ball.id - 1) % 7) + 1]);

  ctx.fillStyle = baseColor;
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fill();

  // Stripe band for high balls
  if (ball.id > EIGHT_ID) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.clip();
    ctx.fillStyle = "#f0ece0";
    ctx.fillRect(px - r, py - r * 0.44, r * 2, r * 0.88);
    ctx.restore();
  }

  // Number pip
  if (ball.id !== CUE_ID) {
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(px, py, r * 0.42, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#111";
    ctx.font = `bold ${Math.round(r * 0.68)}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(ball.id), px, py + 0.5);
  }

  // Spherical shading — dark edge gives roundness
  const edgeShade = ctx.createRadialGradient(px, py, r * 0.6, px, py, r);
  edgeShade.addColorStop(0, "rgba(0,0,0,0)");
  edgeShade.addColorStop(1, "rgba(0,0,0,0.30)");
  ctx.fillStyle = edgeShade;
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fill();

  // Primary highlight
  const hl = ctx.createRadialGradient(
    px - r * 0.28, py - r * 0.32, 0,
    px - r * 0.28, py - r * 0.32, r * 0.56,
  );
  hl.addColorStop(0, "rgba(255,255,255,0.44)");
  hl.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = hl;
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fill();

  // Pinpoint specular
  ctx.fillStyle = "rgba(255,255,255,0.72)";
  ctx.beginPath();
  ctx.arc(px - r * 0.27, py - r * 0.31, r * 0.11, 0, Math.PI * 2);
  ctx.fill();
};

// Draw the cue stick + aim ghost line for the human shooter.
// Pass balls to enable ghost-ball and deflection preview (real geometry, not decoration).
export const drawAim = (
  ctx: CanvasRenderingContext2D,
  cue: Ball,
  phi: number,
  power: number,
  v: ViewTransform,
  table: Table,
  balls: Ball[] = [],
): void => {
  if (cue.pocketed) return;
  const [cx, cy] = toPx(cue.pos.x, cue.pos.y, v);
  const r = BALL_RADIUS * v.scale;

  const dx = Math.cos(phi);
  const dy = -Math.sin(phi); // screen y flip
  const perX = dy;
  const perY = -dx;

  // --- Ghost ball detection (physics coords) ---
  // Sweep a ball of radius BALL_RADIUS along the aim ray; find the first collision.
  const aimDxP = Math.cos(phi);
  const aimDyP = Math.sin(phi);
  let hitBall: Ball | null = null;
  let ghostSx = 0, ghostSy = 0; // ghost center in screen coords
  let bestT = Infinity;

  for (const b of balls) {
    if (b.id === CUE_ID || b.pocketed) continue;
    const fx = b.pos.x - cue.pos.x;
    const fy = b.pos.y - cue.pos.y;
    const proj = fx * aimDxP + fy * aimDyP;
    if (proj <= 0) continue;
    const perp = Math.abs(fx * aimDyP - fy * aimDxP);
    const sumR = 2 * BALL_RADIUS;
    if (perp >= sumR) continue;
    const t = proj - Math.sqrt(sumR * sumR - perp * perp);
    if (t >= 0 && t < bestT) {
      bestT = t;
      hitBall = b;
      const [gx, gy] = toPx(cue.pos.x + aimDxP * t, cue.pos.y + aimDyP * t, v);
      ghostSx = gx;
      ghostSy = gy;
    }
  }

  // --- Cue stick (unclipped — can extend into rail area) ---
  drawStick(ctx, cx, cy, dx, dy, perX, perY, r + 10 + power * 30);

  // --- Aim ghost line (clipped to table+rail, cut at ghost ball) ---
  let lineLen = (0.25 + power * 0.75) * tableDiag(v);
  if (hitBall) {
    const ghostDist = Math.hypot(ghostSx - cx, ghostSy - cy);
    lineLen = Math.min(lineLen, Math.max(r + 2, ghostDist - r));
  }

  const railPx = 22;
  const [rx, ry] = toPx(-table.length / 2, table.width / 2, v);
  const rw = table.length * v.scale;
  const rh = table.width * v.scale;

  ctx.save();
  ctx.beginPath();
  ctx.rect(rx - railPx, ry - railPx, rw + railPx * 2, rh + railPx * 2);
  ctx.clip();
  ctx.strokeStyle = "rgba(255,255,255,0.42)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 8]);
  ctx.beginPath();
  ctx.moveTo(cx + dx * (r + 2), cy + dy * (r + 2));
  ctx.lineTo(cx + dx * lineLen, cy + dy * lineLen);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  // --- Ghost ball circle + object ball deflection ---
  if (hitBall) {
    const [bx, by] = toPx(hitBall.pos.x, hitBall.pos.y, v);

    // Ghost ball (faint outline at contact point)
    ctx.strokeStyle = "rgba(255,255,255,0.30)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(ghostSx, ghostSy, r, 0, Math.PI * 2);
    ctx.stroke();

    // Object ball deflection: direction from ghost contact toward ball center
    const distPx = Math.hypot(bx - ghostSx, by - ghostSy);
    if (distPx > 1) {
      const ddx = (bx - ghostSx) / distPx;
      const ddy = (by - ghostSy) / distPx;
      ctx.strokeStyle = "rgba(255,255,255,0.20)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 7]);
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(bx + ddx * 150, by + ddy * 150);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
};

const tableDiag = (v: ViewTransform): number => Math.max(140, v.scale * 0.6);

/**
 * The cue stick itself. `tipGap` is the distance in pixels from the cue ball's
 * centre back to the tip along the aim direction, so callers control the
 * backswing without duplicating the stick's geometry.
 */
function drawStick(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  dx: number,
  dy: number,
  perX: number,
  perY: number,
  tipGap: number,
): void {
  const tipX = cx - dx * tipGap;
  const tipY = cy - dy * tipGap;
  const stickLen = 190;
  const buttX = tipX - dx * stickLen;
  const buttY = tipY - dy * stickLen;

  const stickGrad = ctx.createLinearGradient(tipX, tipY, buttX, buttY);
  stickGrad.addColorStop(0, "#8a6030");
  stickGrad.addColorStop(0.06, "#c09050");
  stickGrad.addColorStop(0.45, "#d4aa70");
  stickGrad.addColorStop(0.88, "#b08040");
  stickGrad.addColorStop(1, "#3e2008");

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(tipX - perX * 2.5, tipY - perY * 2.5);
  ctx.lineTo(tipX + perX * 2.5, tipY + perY * 2.5);
  ctx.lineTo(buttX + perX * 7.5, buttY + perY * 7.5);
  ctx.lineTo(buttX - perX * 7.5, buttY - perY * 7.5);
  ctx.closePath();
  ctx.fillStyle = stickGrad;
  ctx.fill();

  // Wrap band near butt
  const wrapT = 0.8;
  const wrapX = tipX - dx * stickLen * wrapT;
  const wrapY = tipY - dy * stickLen * wrapT;
  ctx.strokeStyle = "rgba(40,20,5,0.55)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(wrapX - perX * 6.8, wrapY - perY * 6.8);
  ctx.lineTo(wrapX + perX * 6.8, wrapY + perY * 6.8);
  ctx.stroke();

  // Chalk tip
  ctx.fillStyle = "#5890b0";
  ctx.beginPath();
  ctx.arc(tipX, tipY, 2.8, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/**
 * The opponent's stroke. The AI's shot used to begin with the cue ball already
 * moving, which reads as teleportation and severs the chosen line from the
 * ball. This is a backswing and a strike over `progress` 0..1, with the
 * direction and the length of the backswing taken from the real `CueAction`
 * the search chose — nothing here is invented, and the balls do not move until
 * it finishes.
 */
export const drawCueStroke = (
  ctx: CanvasRenderingContext2D,
  cue: Ball,
  phi: number,
  power: number,
  progress: number,
  v: ViewTransform,
): void => {
  if (cue.pocketed) return;
  const [cx, cy] = toPx(cue.pos.x, cue.pos.y, v);
  const r = BALL_RADIUS * v.scale;
  const dx = Math.cos(phi);
  const dy = -Math.sin(phi);

  // Backswing scales with the real power: a soft safety is a short stroke.
  const rest = 10;
  const back = 18 + power * 46;
  const t = Math.max(0, Math.min(1, progress));
  // Draw back over the first 45%, then accelerate through the ball.
  const gap =
    t < 0.45
      ? rest + (back - rest) * (t / 0.45)
      : back * Math.pow(1 - (t - 0.45) / 0.55, 2);

  drawStick(ctx, cx, cy, dx, dy, dy, -dx, r + gap);
};

export const render = (
  ctx: CanvasRenderingContext2D,
  state: GameState,
  table: Table,
  v: ViewTransform,
): void => {
  // Clear the WHOLE canvas first, not just the table+rail box `drawTable`
  // repaints. `drawAim` draws the cue stick deliberately unclipped ("can extend
  // into rail area"), and at low power / shallow angles it lands outside that
  // box — where nothing ever painted over it again, so stick pixels
  // accumulated as permanent tan smears along the canvas edges for the life of
  // the page. Found in live Chrome QA (visible in
  // docs/repair/release-candidate/showboat/); invisible to the headless suites,
  // which never rasterise.
  //
  // Cleared in DEVICE pixels, with the transform temporarily reset. The context
  // carries a devicePixelRatio scale (see App.tsx), so `ctx.canvas.width` —
  // which is in device pixels — must not be passed through it: on a 2x display
  // that would clear a region four times the surface. Resetting first makes the
  // clear exactly the drawing surface, whatever the ratio.
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.restore();
  drawTable(ctx, table, v);
  for (const b of state.balls) drawBall(ctx, b, v);
};

export { SOLIDS };
