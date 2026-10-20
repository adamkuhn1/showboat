import { type Ball } from "../physics/ball";
import { type Table } from "../physics/table";
import { BALL_RADIUS, BALL_DIAMETER } from "../physics/constants";
import { type Vec2 } from "../physics/vec";

// Straight-line aim geometry for the human shooter's overlay: where the cue
// ball's centre will be at first contact (the "ghost ball"), and which way the
// struck object ball heads. Pure geometry with no spin, throw or friction, the
// same approximation a player makes when aiming with a ghost ball.

export type AimContact =
  | {
      kind: "ball";
      ghost: Vec2; // cue-ball centre at the moment of contact
      distance: number; // metres travelled by the cue-ball centre to get there
      target: Ball;
      objectDir: Vec2; // unit vector, ghost centre -> object centre
    }
  | {
      kind: "cushion";
      ghost: Vec2; // cue-ball centre where it meets the first cushion
      distance: number;
    };

export const aimContact = (
  cue: Ball,
  phi: number,
  balls: readonly Ball[],
  table: Table,
): AimContact => {
  const d = { x: Math.cos(phi), y: Math.sin(phi) };
  const c = cue.pos;

  let best: { t: number; ball: Ball } | null = null;
  for (const b of balls) {
    if (b.pocketed || b.id === cue.id) continue;
    const wx = b.pos.x - c.x;
    const wy = b.pos.y - c.y;
    const along = wx * d.x + wy * d.y;
    if (along <= 0) continue; // behind the cue ball
    const perp2 = wx * wx + wy * wy - along * along;
    const D2 = BALL_DIAMETER * BALL_DIAMETER;
    if (perp2 >= D2) continue; // the line passes clear of this ball
    // Clamp: a ball already frozen to the cue ball is contacted immediately.
    const t = Math.max(0, along - Math.sqrt(D2 - perp2));
    if (!best || t < best.t) best = { t, ball: b };
  }

  if (best) {
    const ghost = { x: c.x + d.x * best.t, y: c.y + d.y * best.t };
    const ox = best.ball.pos.x - ghost.x;
    const oy = best.ball.pos.y - ghost.y;
    const n = Math.hypot(ox, oy) || 1;
    return {
      kind: "ball",
      ghost,
      distance: best.t,
      target: best.ball,
      objectDir: { x: ox / n, y: oy / n },
    };
  }

  // No ball in the way: the cue-ball centre stops one radius short of the
  // cushion nose line.
  const hx = table.length / 2 - BALL_RADIUS;
  const hy = table.width / 2 - BALL_RADIUS;
  const tx = d.x > 0 ? (hx - c.x) / d.x : d.x < 0 ? (-hx - c.x) / d.x : Infinity;
  const ty = d.y > 0 ? (hy - c.y) / d.y : d.y < 0 ? (-hy - c.y) / d.y : Infinity;
  const t = Math.max(0, Math.min(tx, ty));
  return {
    kind: "cushion",
    ghost: { x: c.x + d.x * t, y: c.y + d.y * t },
    distance: t,
  };
};
