import { type Vec2 } from "../physics/vec";
import { type Table } from "../physics/table";
import { type ViewTransform } from "./renderer";
import { type CandidateStat, type SearchResult } from "../ai/mcts";

// Reasoning overlay renderer. Every number and path drawn here comes from the
// REAL search output (candidate geometry, MCTS visit counts, rollout value/
// win-prob, and the rails-before-pot measured from the actual simulated event
// trace). Nothing here is decorative or invented — if the search didn't produce
// it, it isn't drawn. This is the constraint from PLAN.md §5 / CLAUDE.md #2.

const toPx = (p: Vec2, v: ViewTransform): [number, number] => [
  v.offsetX + p.x * v.scale,
  v.offsetY - p.y * v.scale,
];

// Draw the candidate ghost paths, weighting opacity/width by visit share so the
// most-searched lines read as the strongest — a faithful picture of where the
// search actually spent its budget.
export const drawCandidatePaths = (
  ctx: CanvasRenderingContext2D,
  result: SearchResult,
  v: ViewTransform,
  topN = 6,
): void => {
  const shown = result.stats.slice(0, topN);
  const maxVisits = Math.max(1, ...shown.map((s) => s.visits));

  // Draw weakest first so the best line sits on top.
  for (let i = shown.length - 1; i >= 0; i--) {
    const s = shown[i];
    const isBest = result.best === s;
    const weight = s.visits / maxVisits;
    const path = s.candidate.path;
    if (path.length < 2) continue;

    ctx.lineWidth = isBest ? 3 : 1 + weight * 1.5;
    ctx.strokeStyle = colorFor(s, isBest, weight);
    ctx.setLineDash(s.candidate.kind === "direct" ? [] : [7, 5]);
    ctx.beginPath();
    const [x0, y0] = toPx(path[0], v);
    ctx.moveTo(x0, y0);
    for (let k = 1; k < path.length; k++) {
      const [x, y] = toPx(path[k], v);
      ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Label the pocket end of the best few with win-prob. Trick shots also get
    // a style badge derived from the real styleScore (rails + combo bonus).
    if (isBest || weight > 0.4) {
      const end = path[path.length - 1];
      const [ex, ey] = toPx(end, v);
      ctx.fillStyle = isBest ? "#eafff5" : "rgba(230,230,230,0.7)";
      ctx.font = `${isBest ? 12 : 10}px ui-monospace, monospace`;
      ctx.textAlign = "center";
      ctx.fillText(`${Math.round(s.winProb * 100)}%`, ex, ey - 8);
      if (s.styleScore >= 2) {
        ctx.fillStyle = "rgba(255,100,100,0.9)";
        ctx.font = "9px ui-monospace, monospace";
        ctx.fillText(`★${s.styleScore}`, ex, ey - 20);
      }
    }
  }
};

const colorFor = (s: CandidateStat, isBest: boolean, weight: number): string => {
  if (isBest) return "rgba(92, 214, 160, 0.95)";
  // Trick routes are tinted by kind so they read distinctly from direct pots.
  // Colors label the real candidate kind — not scripted shots.
  const alpha = 0.25 + weight * 0.5;
  if (s.candidate.kind === "double-bank") return `rgba(255, 100, 100, ${alpha})`;
  if (s.candidate.kind === "bank") return `rgba(255, 179, 111, ${alpha})`;
  if (s.candidate.kind === "combo") return `rgba(191, 143, 255, ${alpha})`;
  return `rgba(140, 200, 255, ${alpha})`;
};

// Mark the aim (ghost-ball) contact point of the chosen shot.
export const drawChosenAim = (
  ctx: CanvasRenderingContext2D,
  result: SearchResult,
  v: ViewTransform,
): void => {
  if (!result.best) return;
  const [x, y] = toPx(result.best.candidate.aimPoint, v);
  ctx.strokeStyle = "rgba(92,214,160,0.9)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(x, y, 6, 0, Math.PI * 2);
  ctx.stroke();
};

// The table isn't needed for drawing but is accepted to keep the overlay's
// signature aligned with the renderer family and allow future rail annotations.
export const drawOverlay = (
  ctx: CanvasRenderingContext2D,
  result: SearchResult,
  _table: Table,
  v: ViewTransform,
): void => {
  drawCandidatePaths(ctx, result, v);
  drawChosenAim(ctx, result, v);
};
