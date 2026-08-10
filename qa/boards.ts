// Board families for the measured-shot gate.
//
// `eval/harness.ts`'s `makeFixtures` is the right fixture set for the ML
// evaluation it was written for, and the wrong one for an acceptance run: it
// keeps only states with at least six generated candidates, which quietly
// removes every sparse and every snookered board — exactly the positions where
// the trick-only ladder has to fall through to a safety. A gate whose fixtures
// cannot reach the fallback has not tested the fallback.
//
// So the families below are stated rather than sampled: an open table, a
// crowded cluster, a late game with two or three balls left, a ball-in-hand
// turn, and a deliberately snookered cue. Every one is placed from a seeded
// PRNG, so a family is reproducible from its seed alone, and every one is a
// legal resting board — no interpenetration, nothing off the cloth.
//
// BALL IN HAND is not a layout, it is a turn state, and the thing worth testing
// about it is the code the opponent actually runs. `ui/useAiTurn.ts` places the
// cue at `(-length/4, 0)` through `placeCueBall` and re-enters planning on the
// next render; this module performs the same call on the same state, so an
// in-hand fixture is the board the opponent really plans from rather than one
// invented here.

import { makeBall, type Ball } from "../src/physics/ball";
import { BALL_RADIUS } from "../src/physics/constants";
import { CUE_ID, EIGHT_ID, SOLIDS, STRIPES } from "../src/game/rack";
import { type Table } from "../src/physics/table";
import { type GameState } from "../src/game/state";
import { placeCueBall } from "../src/game/game";
import { legalTargets } from "../src/ai/turn";
import { mulberry32 } from "../eval/harness";

export type BoardType = "open" | "crowded" | "late" | "ball-in-hand" | "snookered";

export const BOARD_TYPES: readonly BoardType[] = [
  "open",
  "crowded",
  "late",
  "ball-in-hand",
  "snookered",
];

export interface BoardFixture {
  id: string;
  boardType: BoardType;
  /** Ready to plan from: ball-in-hand has already been resolved the way the app resolves it. */
  state: GameState;
  /** True when `placeCueBall` ran, i.e. this turn began with ball in hand. */
  ballInHandApplied: boolean;
  targets: number[];
  /** Shape of the board, so the report can describe the mix in numbers. */
  metrics: {
    objectBalls: number;
    /** Closest pair of object balls, in millimetres. */
    minPairMm: number;
    /** Mean nearest-neighbour distance over the object balls, in millimetres. */
    meanNearestMm: number;
    /** Distance from the cue ball to the nearest legal target, in millimetres. */
    cueToTargetMm: number;
  };
}

const D = 2 * BALL_RADIUS;
const mm = (m: number) => Math.round(m * 10000) / 10;

const state = (
  balls: Ball[],
  groups: GameState["groups"],
  ballInHand: GameState["ballInHand"],
): GameState => ({
  balls,
  turn: 0,
  groups,
  ballInHand,
  winner: null,
  broken: true,
  shotCount: 4,
});

/** Place `id` at a random point at least `minGap` from every ball already down. */
const scatter = (
  balls: Ball[],
  id: number,
  rng: () => number,
  table: Table,
  minGap: number,
  bounds = 1,
): boolean => {
  const hx = (table.length / 2 - BALL_RADIUS - 0.02) * bounds;
  const hy = (table.width / 2 - BALL_RADIUS - 0.02) * bounds;
  for (let i = 0; i < 400; i++) {
    const x = (rng() * 2 - 1) * hx;
    const y = (rng() * 2 - 1) * hy;
    if (balls.every((b) => Math.hypot(b.pos.x - x, b.pos.y - y) >= minGap)) {
      balls.push(makeBall(id, x, y));
      return true;
    }
  }
  return false;
};

const shuffled = <T>(xs: readonly T[], rng: () => number): T[] => {
  const out = [...xs];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
};

/** Five or six object balls, none of them near another. */
function openBoard(table: Table, rng: () => number): Ball[] | null {
  const n = 4 + Math.floor(rng() * 3);
  const ids = shuffled([...SOLIDS, ...STRIPES], rng).slice(0, n);
  const balls: Ball[] = [];
  if (!scatter(balls, CUE_ID, rng, table, 0, 0.9)) return null;
  for (const id of [...ids, EIGHT_ID]) {
    if (!scatter(balls, id, rng, table, 5 * D, 0.9)) return null;
  }
  return balls;
}

/** A packed blob of five to seven balls, plus two loose ones and the cue away from it. */
function crowdedBoard(table: Table, rng: () => number): Ball[] | null {
  const ids = shuffled([...SOLIDS, ...STRIPES], rng);
  const clusterN = 5 + Math.floor(rng() * 3);
  const cx = (rng() * 2 - 1) * (table.length / 2 - 6 * D);
  const cy = (rng() * 2 - 1) * (table.width / 2 - 6 * D);
  const balls: Ball[] = [];
  const cluster = [...ids.slice(0, clusterN - 1), EIGHT_ID];
  for (const id of cluster) {
    let placed = false;
    for (let i = 0; i < 500 && !placed; i++) {
      // A blob two and a half diameters across: contacts are near-frozen, which
      // is where collision chatter and ambiguous drivers live.
      const r = rng() * 2.5 * D;
      const th = rng() * Math.PI * 2;
      const x = cx + Math.cos(th) * r;
      const y = cy + Math.sin(th) * r;
      if (Math.abs(x) > table.length / 2 - BALL_RADIUS - 0.02) continue;
      if (Math.abs(y) > table.width / 2 - BALL_RADIUS - 0.02) continue;
      if (!balls.every((b) => Math.hypot(b.pos.x - x, b.pos.y - y) >= D + 0.0015)) continue;
      balls.push(makeBall(id, x, y));
      placed = true;
    }
    if (!placed) return null;
  }
  for (const id of ids.slice(clusterN - 1, clusterN + 1)) {
    if (!scatter(balls, id, rng, table, 2 * D, 0.9)) return null;
  }
  if (!scatter(balls, CUE_ID, rng, table, 4 * D, 0.9)) return null;
  return balls;
}

/** Two or three object balls and the eight. */
function lateBoard(table: Table, rng: () => number): Ball[] | null {
  const n = 2 + Math.floor(rng() * 2);
  const ids = shuffled(SOLIDS, rng).slice(0, n);
  const balls: Ball[] = [];
  if (!scatter(balls, CUE_ID, rng, table, 0, 0.9)) return null;
  for (const id of [...ids, EIGHT_ID]) {
    if (!scatter(balls, id, rng, table, 3 * D, 0.9)) return null;
  }
  return balls;
}

/**
 * The cue tucked into a corner behind a wall of three balls, with the legal
 * targets on the far side of it. Built to make the trick rungs fail: the
 * generator's straight and bank routes are all screened, so what the gate sees
 * here is the safety ladder doing its job or not doing it.
 */
function snookeredBoard(table: Table, rng: () => number): Ball[] | null {
  const sx = rng() < 0.5 ? -1 : 1;
  const sy = rng() < 0.5 ? -1 : 1;
  const hx = table.length / 2 - BALL_RADIUS - 0.02;
  const hy = table.width / 2 - BALL_RADIUS - 0.02;
  const balls: Ball[] = [];
  // Cue hard into the corner.
  balls.push(makeBall(CUE_ID, sx * (hx - 0.5 * D), sy * (hy - 0.5 * D)));
  // Three blockers on the diagonal arc in front of it, a ball's width apart, so
  // nothing leaves the corner on a straight line.
  const wall = 2.1 * D;
  const blockers = shuffled([...SOLIDS, ...STRIPES], rng);
  const angles = [0.15, 0.5, 0.85].map((f) => (Math.PI / 2) * f);
  angles.forEach((a, i) => {
    balls.push(
      makeBall(
        blockers[i],
        sx * (hx - 0.5 * D - Math.cos(a) * wall),
        sy * (hy - 0.5 * D - Math.sin(a) * wall),
      ),
    );
  });
  // Targets, far away, past the wall.
  const targets = blockers.slice(3, 3 + 2 + Math.floor(rng() * 2));
  for (const id of [...targets, EIGHT_ID]) {
    let placed = false;
    for (let i = 0; i < 400 && !placed; i++) {
      const x = -sx * (rng() * hx * 0.7 + 0.1);
      const y = -sy * (rng() * hy * 0.7 + 0.05);
      if (!balls.every((b) => Math.hypot(b.pos.x - x, b.pos.y - y) >= 3 * D)) continue;
      balls.push(makeBall(id, x, y));
      placed = true;
    }
    if (!placed) return null;
  }
  return balls;
}

const metricsOf = (balls: Ball[], targets: number[]): BoardFixture["metrics"] => {
  const objs = balls.filter((b) => b.id !== CUE_ID && !b.pocketed);
  const cue = balls.find((b) => b.id === CUE_ID)!;
  let minPair = Infinity;
  const nearest: number[] = [];
  for (const a of objs) {
    let best = Infinity;
    for (const b of objs) {
      if (a === b) continue;
      const d = Math.hypot(a.pos.x - b.pos.x, a.pos.y - b.pos.y);
      if (d < best) best = d;
      if (d < minPair) minPair = d;
    }
    if (isFinite(best)) nearest.push(best);
  }
  const live = objs.filter((b) => targets.includes(b.id));
  const cueTo = live.length
    ? Math.min(...live.map((b) => Math.hypot(b.pos.x - cue.pos.x, b.pos.y - cue.pos.y)))
    : NaN;
  return {
    objectBalls: objs.length,
    minPairMm: mm(minPair),
    meanNearestMm: mm(nearest.reduce((a, b) => a + b, 0) / Math.max(1, nearest.length)),
    cueToTargetMm: mm(cueTo),
  };
};

/**
 * `count` fixtures, dealt round-robin across the five families so the mix does
 * not depend on how many of each the generator happened to succeed at.
 *
 * `groups` is set for the late family — a real late game is a player on their
 * own two remaining balls, or on the eight — and left open elsewhere, which is
 * what `legalTargets` reads.
 */
export function makeBoardMix(table: Table, count: number, seed: number): BoardFixture[] {
  const rng = mulberry32(seed);
  const out: BoardFixture[] = [];
  let i = 0;
  let guard = 0;
  while (out.length < count && guard++ < count * 60) {
    const boardType = BOARD_TYPES[i++ % BOARD_TYPES.length];
    let balls: Ball[] | null = null;
    let groups: GameState["groups"] = { 0: null, 1: null };
    let inHand: GameState["ballInHand"] = false;

    switch (boardType) {
      case "open":
        balls = openBoard(table, rng);
        break;
      case "crowded":
        balls = crowdedBoard(table, rng);
        break;
      case "late":
        balls = lateBoard(table, rng);
        groups = { 0: "solids", 1: "stripes" };
        break;
      case "ball-in-hand":
        balls = rng() < 0.5 ? openBoard(table, rng) : crowdedBoard(table, rng);
        inHand = "anywhere";
        break;
      case "snookered":
        balls = snookeredBoard(table, rng);
        break;
    }
    if (!balls) continue;

    let s = state(balls, groups, inHand);
    const ballInHandApplied = s.ballInHand !== false;
    // Exactly what `ui/useAiTurn.ts` does on an opponent ball-in-hand turn.
    if (ballInHandApplied) s = placeCueBall(s, -table.length / 4, 0, table);

    const targets = legalTargets(s, 0);
    if (targets.length === 0) continue;
    out.push({
      id: `${boardType}-${out.length}`,
      boardType,
      state: s,
      ballInHandApplied,
      targets,
      metrics: metricsOf(s.balls, targets),
    });
  }
  return out;
}
