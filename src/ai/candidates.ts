import { type Vec2, add, dot, mag, normalize, scale, sub } from "../physics/vec";
import { BALL_DIAMETER, BALL_RADIUS } from "../physics/constants";
import { type Cushion, type Pocket, type Table } from "../physics/table";
import { type CueAction } from "../physics/cue";
import { type GameState } from "../game/state";
import { CUE_ID, EIGHT_ID, SOLIDS, STRIPES } from "../game/rack";

// Candidate shot generation. Produces concrete, parameterised cue actions from
// pure geometry: ghost-ball aiming for direct shots, mirror-image reflection
// for banks (object ball off one or two cushions) and kicks (cue ball off a
// cushion first), and two-stage ghost-ball chaining for combos.
//
// Two properties matter here:
//   - The `kind` on a candidate is only a *generation* label. The published
//     shot kind shown to the player is always re-measured from the event log
//     of the actual simulation (see classify.ts) — never from this intent.
//   - Jump and massé shots are not generated: CueAction has no elevation
//     axis (see physics/cue.ts), so the action space can't express them.

export type CandidateKind = "direct" | "bank" | "kick" | "combo";

export interface Candidate {
  id: number;
  kind: CandidateKind;
  targetBall: number; // the ball this shot intends to pot
  comboBall?: number; // for combos: the ball the cue strikes first
  pocketId: string;
  railsPlanned: number; // cushions in the *planned* geometry (0 for direct)
  action: CueAction;
  // Planned geometry for the dashed "hypothetical" preview. These are the
  // idealised mirror-line polylines, NOT a simulation result — the UI must
  // draw them dashed and visually distinct from the verified route.
  cuePath: Vec2[]; // cue position -> (cushion points) -> ghost point
  objectPath: Vec2[]; // object ball -> (cushion points) -> pocket
}

// --- geometry helpers --------------------------------------------------------

// The line a ball's CENTRE reflects across is the cushion nose pushed one ball
// radius into the table (the centre never reaches the nose itself).
const centreLine = (c: Cushion): number =>
  c.axis === "x" ? c.pos + c.normal.x * BALL_RADIUS : c.pos + c.normal.y * BALL_RADIUS;

const reflectAcross = (p: Vec2, c: Cushion): Vec2 => {
  const line = centreLine(c);
  return c.axis === "x" ? { x: 2 * line - p.x, y: p.y } : { x: p.x, y: 2 * line - p.y };
};

// Intersection of segment a->b with a cushion's centre line, if it crosses
// inside the playing area. Returns null if the segment doesn't cross.
const crossingPoint = (a: Vec2, b: Vec2, c: Cushion, table: Table): Vec2 | null => {
  const line = centreLine(c);
  const pa = c.axis === "x" ? a.x : a.y;
  const pb = c.axis === "x" ? b.x : b.y;
  const denom = pb - pa;
  if (Math.abs(denom) < 1e-12) return null;
  const t = (line - pa) / denom;
  if (t <= 1e-6 || t >= 1 - 1e-6) return null;
  const pt = add(a, scale(sub(b, a), t));
  // The crossing must lie on the physical rail, inside the table bounds.
  const hx = table.length / 2 - BALL_RADIUS;
  const hy = table.width / 2 - BALL_RADIUS;
  if (c.axis === "x") {
    if (Math.abs(pt.y) > hy) return null;
  } else if (Math.abs(pt.x) > hx) {
    return null;
  }
  return pt;
};

// Ghost-ball point: where the cue (or striking ball) centre must be at impact
// to send `target` along `dir`.
const ghostPoint = (target: Vec2, dir: Vec2): Vec2 =>
  sub(target, scale(dir, BALL_DIAMETER));

// A shot is only geometrically possible if the striker approaches the ghost
// point from "behind" the intended direction (cut angle < ~80 degrees).
const cutFeasible = (striker: Vec2, ghost: Vec2, dir: Vec2): boolean => {
  const approach = normalize(sub(ghost, striker));
  return dot(approach, dir) > 0.17; // cos(80deg)
};

// Power heuristic from total planned travel distance and planned cushions.
// Each cushion hit costs energy (restitution 0.85 + rail friction), so banks
// and kicks are struck harder. This is only the candidate's opening guess —
// verification simulates the real outcome.
const powerFor = (pathLen: number, rails: number, combo: boolean): number => {
  const p = 0.3 + 0.17 * pathLen + 0.09 * rails + (combo ? 0.1 : 0);
  return Math.min(1, Math.max(0.2, p));
};

const pathLength = (pts: Vec2[]): number => {
  let L = 0;
  for (let i = 1; i < pts.length; i++) L += mag(sub(pts[i], pts[i - 1]));
  return L;
};

// --- target selection --------------------------------------------------------

// Which balls may the shooter legally aim to pot? Own group while any remain,
// then the 8. On an open table, any object ball except the 8.
export const legalTargets = (g: GameState): number[] => {
  const group = g.groups[g.turn];
  const onTable = (id: number): boolean =>
    g.balls.some((b) => b.id === id && !b.pocketed);
  if (group === null) {
    return [...SOLIDS, ...STRIPES].filter(onTable);
  }
  const own = (group === "solids" ? SOLIDS : STRIPES).filter(onTable);
  return own.length > 0 ? own : [EIGHT_ID];
};

// --- generation --------------------------------------------------------------

export const generateCandidates = (g: GameState, table: Table): Candidate[] => {
  const cue = g.balls.find((b) => b.id === CUE_ID);
  if (!cue || cue.pocketed) return [];
  const targets = legalTargets(g);
  const out: Candidate[] = [];
  let nextId = 0;

  const push = (
    kind: CandidateKind,
    targetBall: number,
    pocket: Pocket,
    railsPlanned: number,
    cuePath: Vec2[],
    objectPath: Vec2[],
    comboBall?: number,
  ): void => {
    const phi = Math.atan2(
      cuePath[1].y - cuePath[0].y,
      cuePath[1].x - cuePath[0].x,
    );
    const len = pathLength(cuePath) + pathLength(objectPath);
    out.push({
      id: nextId++,
      kind,
      targetBall,
      comboBall,
      pocketId: pocket.id,
      railsPlanned,
      action: {
        phi,
        power: powerFor(len, railsPlanned, comboBall !== undefined),
        sideSpin: 0,
        topSpin: 0,
      },
      cuePath,
      objectPath,
    });
  };

  for (const tid of targets) {
    const target = g.balls.find((b) => b.id === tid);
    if (!target || target.pocketed) continue;
    const T = target.pos;

    for (const pocket of table.pockets) {
      const P = pocket.center;

      // Direct: cue -> ghost, object -> pocket.
      {
        const dir = normalize(sub(P, T));
        const ghost = ghostPoint(T, dir);
        if (cutFeasible(cue.pos, ghost, dir)) {
          push("direct", tid, pocket, 0, [cue.pos, ghost], [T, P]);
        }
      }

      // Bank (1 rail): object ball reflects off one cushion into the pocket.
      for (const c of table.cushions) {
        const P1 = reflectAcross(P, c);
        const dir = normalize(sub(P1, T));
        const ghost = ghostPoint(T, dir);
        const x1 = crossingPoint(T, P1, c, table);
        if (!x1) continue;
        if (!cutFeasible(cue.pos, ghost, dir)) continue;
        push("bank", tid, pocket, 1, [cue.pos, ghost], [T, x1, P]);
      }

      // Bank (2 rails, multi-wall): reflect the pocket across cushion b, then
      // cushion a; the object ball hits a first, then b, then the pocket.
      for (const ca of table.cushions) {
        for (const cb of table.cushions) {
          if (ca.side === cb.side) continue;
          const P2 = reflectAcross(reflectAcross(P, cb), ca);
          const dir = normalize(sub(P2, T));
          const ghost = ghostPoint(T, dir);
          const xa = crossingPoint(T, P2, ca, table);
          if (!xa) continue;
          const xb = crossingPoint(xa, reflectAcross(P, cb), cb, table);
          if (!xb) continue;
          if (!cutFeasible(cue.pos, ghost, dir)) continue;
          push("bank", tid, pocket, 2, [cue.pos, ghost], [T, xa, xb, P]);
        }
      }

      // Kick (1 rail): the cue ball banks off a cushion before striking the
      // object ball, which then runs straight to the pocket.
      {
        const dir = normalize(sub(P, T));
        const ghost = ghostPoint(T, dir);
        for (const c of table.cushions) {
          const G1 = reflectAcross(ghost, c);
          const xc = crossingPoint(cue.pos, G1, c, table);
          if (!xc) continue;
          // The post-rail approach into the ghost must still make the cut.
          const approach = normalize(sub(ghost, xc));
          if (dot(approach, dir) <= 0.17) continue;
          push("kick", tid, pocket, 1, [cue.pos, xc, ghost], [T, P]);
        }
      }
    }

    // Combos: cue strikes an own-group ball A, which drives target T into a
    // pocket. First contact (A) must be legal, so A comes from the same legal
    // target set.
    for (const aid of targets) {
      if (aid === tid) continue;
      const aBall = g.balls.find((b) => b.id === aid);
      if (!aBall || aBall.pocketed) continue;
      const A = aBall.pos;
      for (const pocket of table.pockets) {
        const P = pocket.center;
        const dirTP = normalize(sub(P, T));
        const ghostT = ghostPoint(T, dirTP);
        const dirAT = normalize(sub(ghostT, A));
        const ghostA = ghostPoint(A, dirAT);
        // Both stages need a feasible cut.
        if (dot(normalize(sub(ghostT, A)), dirTP) <= 0.25) continue;
        if (!cutFeasible(cue.pos, ghostA, dirAT)) continue;
        push(
          "combo",
          tid,
          pocket,
          0,
          [cue.pos, ghostA],
          [A, ghostT, T, P],
          aid,
        );
      }
    }
  }

  return out;
};

// A soft "safety" action used when no candidate survives verification: roll
// gently into the nearest legal ball, aiming slightly through it so both balls
// are likely to reach a rail (avoiding the no-rail foul).
export const safetyAction = (g: GameState, _table: Table): CueAction | null => {
  const cue = g.balls.find((b) => b.id === CUE_ID);
  if (!cue || cue.pocketed) return null;
  const targets = legalTargets(g);
  let best: { d: number; phi: number } | null = null;
  for (const tid of targets) {
    const t = g.balls.find((b) => b.id === tid);
    if (!t || t.pocketed) continue;
    const d = mag(sub(t.pos, cue.pos));
    const phi = Math.atan2(t.pos.y - cue.pos.y, t.pos.x - cue.pos.x);
    if (!best || d < best.d) best = { d, phi };
  }
  if (!best) return null;
  return {
    phi: best.phi,
    power: Math.min(0.55, 0.3 + 0.15 * best.d),
    sideSpin: 0,
    topSpin: 0,
  };
};
