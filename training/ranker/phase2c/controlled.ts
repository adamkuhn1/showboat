// Controlled (non-trajectory) state generation — the Phase 2A/2B approach,
// extracted unchanged as one generation mode among several (see selfplay.ts
// for the other). Kept because it's a legitimate way to guarantee coverage
// of specific geometric setups a self-play game may rarely reach (obstructed
// paths, engineered bank/combo opportunities, sparse late-game-like layouts)
// — not because it's a substitute for real trajectory states. See
// DATASET_DESIGN.md's "State generation" section.

import { makeBall, type Ball } from "../../../src/physics/ball";
import { BALL_RADIUS } from "../../../src/physics/constants";
import { CUE_ID, EIGHT_ID, SOLIDS, STRIPES } from "../../../src/game/rack";
import { type Table } from "../../../src/physics/table";

/** A random legal-ish open-table board: random subset of balls, non-overlapping. */
export function randomControlledState(table: Table, rng: () => number): Ball[] {
  const hx = table.length / 2 - BALL_RADIUS - 0.01;
  const hy = table.width / 2 - BALL_RADIUS - 0.01;
  const allIds = [...SOLIDS, EIGHT_ID, ...STRIPES];
  const keepFraction = 0.6 + rng() * 0.4;
  const kept = allIds.filter(() => rng() < keepFraction);
  const nonEight = kept.filter((id) => id !== EIGHT_ID);
  if (nonEight.length < 2) {
    return randomControlledState(table, rng); // resample rather than emit a degenerate state
  }
  const ids = [CUE_ID, ...kept];
  const balls: Ball[] = [];
  const MIN_DIST = 2 * BALL_RADIUS + 0.01;
  for (const id of ids) {
    let placed = false;
    for (let attempt = 0; attempt < 200 && !placed; attempt++) {
      const x = (rng() * 2 - 1) * hx;
      const y = (rng() * 2 - 1) * hy;
      if (balls.every((b) => Math.hypot(b.pos.x - x, b.pos.y - y) >= MIN_DIST)) {
        balls.push(makeBall(id, x, y));
        placed = true;
      }
    }
    if (!placed) return randomControlledState(table, rng); // resample on a packed failure
  }
  return balls;
}
