import { type Vec2 } from "../physics/vec";
import { type Ball } from "../physics/ball";
import { type Table } from "../physics/table";
import { BALL_RADIUS } from "../physics/constants";
import { type CueAction } from "../physics/cue";
import { CUE_ID } from "../game/rack";

// Enumerate aiming paths geometrically: direct pots, single- and double-cushion
// banks (via pocket mirroring), simple combos through an intermediate ball, and
// rail-plus-combination routes (combo where the intermediate ball's leg to the
// pocket banks off one rail). Emits aim angle + suggested power — the UCB
// search value function picks what's good.

export type CandidateKind = "direct" | "bank" | "double-bank" | "combo" | "rail-combo";

export interface Candidate {
  kind: CandidateKind;
  target: number; // ball the CUE must legally strike first (first-contact legality)
  // Ball that should actually end up in the pocket. Equal to `target` for
  // direct/bank/double-bank (the struck ball travels straight to the
  // pocket). Distinct from `target` for combo/rail-combo, where the cue
  // strikes `target` (satisfying legal first contact) which then drives a
  // *different* ball (`potId`) into the pocket — checking `pocketed.includes`
  // against the wrong one of these two ids is a real, confirmed bug: for a
  // combo shot simulated at realistic (non-degenerate) range, `target`
  // generally does NOT itself reach the pocket, only `potId` does.
  potId: number;
  pocket: string; // pocket id aimed at
  aimPoint: Vec2; // the "ghost ball" point the cue is aimed through
  action: CueAction; // phi + a suggested power (spin left to search)
  // For overlay: the polyline the intended object ball follows (world coords).
  path: Vec2[];
  banks: number; // number of cushions in the object-ball route (0 = direct)
}

const sub = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y });
const add = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x + b.x, y: a.y + b.y });
const scale = (a: Vec2, s: number): Vec2 => ({ x: a.x * s, y: a.y * s });
const mag = (a: Vec2): number => Math.hypot(a.x, a.y);
const norm = (a: Vec2): Vec2 => {
  const m = mag(a);
  return m < 1e-9 ? { x: 0, y: 0 } : { x: a.x / m, y: a.y / m };
};

// The ghost-ball contact point: to send the object ball toward `aimTarget`, the
// cue ball must strike it at the point one ball-diameter back along the line
// from aimTarget through the object ball.
const ghostBall = (obj: Vec2, aimTarget: Vec2): Vec2 => {
  const dir = norm(sub(aimTarget, obj));
  return sub(obj, scale(dir, 2 * BALL_RADIUS));
};

// Power heuristic from distance (longer paths need more speed). Kept modest and
// only a *suggestion* — search/spin refine it, nothing here is a tuned trick.
const powerFor = (dist: number): number =>
  Math.max(0.35, Math.min(0.95, 0.35 + dist * 0.35));

const aimAction = (cue: Vec2, ghost: Vec2, power: number): CueAction => {
  const d = sub(ghost, cue);
  return { phi: Math.atan2(d.y, d.x), power, sideSpin: 0, topSpin: 0 };
};

// Mirror a point across a cushion line (for bank-shot geometry).
const mirrorAcross = (p: Vec2, table: Table, side: string): Vec2 => {
  const hx = table.length / 2;
  const hy = table.width / 2;
  switch (side) {
    case "left":
      return { x: -2 * hx - p.x, y: p.y };
    case "right":
      return { x: 2 * hx - p.x, y: p.y };
    case "bottom":
      return { x: p.x, y: -2 * hy - p.y };
    case "top":
      return { x: p.x, y: 2 * hy - p.y };
    default:
      return p;
  }
};

const SIDES = ["left", "right", "top", "bottom"] as const;

// Check the line segment from `from` to `to` is clear of blocking balls.
// A ball blocks if its center is within 2*BALL_RADIUS of the segment.
// `skipIds` contains balls that are intentionally on the path (cue, target).
const isPathClear = (
  from: Vec2,
  to: Vec2,
  live: Ball[],
  skipIds: Set<number>,
): boolean => {
  const d = sub(to, from);
  const len = mag(d);
  if (len < 1e-9) return true;
  const dn = { x: d.x / len, y: d.y / len };
  const minDist = 2 * BALL_RADIUS + 0.004; // 4mm margin for physics tolerance
  for (const b of live) {
    if (skipIds.has(b.id)) continue;
    const v = sub(b.pos, from);
    const t = v.x * dn.x + v.y * dn.y;
    if (t < 0 || t > len) continue;
    const perp2 = (v.x - t * dn.x) ** 2 + (v.y - t * dn.y) ** 2;
    if (perp2 < minDist * minDist) return false;
  }
  return true;
};

// Enumerate candidates for one shooter given the legal target ids.
export const generateCandidates = (
  balls: Ball[],
  table: Table,
  targets: number[],
): Candidate[] => {
  const cue = balls.find((b) => b.id === CUE_ID);
  if (!cue || cue.pocketed) return [];
  const cuePos = cue.pos;
  const live = balls.filter((b) => !b.pocketed);
  const out: Candidate[] = [];

  // Max total path length for a 2-cushion route: beyond this the shot is
  // almost impossible to execute at the required precision.
  const MAX_DOUBLE_BANK_PATH = table.length * 3.5;

  for (const tid of targets) {
    const obj = live.find((b) => b.id === tid);
    if (!obj) continue;

    const skipCueAndTarget = new Set([CUE_ID, tid]);

    for (const pk of table.pockets) {
      // --- Direct pot -------------------------------------------------------
      const ghost = ghostBall(obj.pos, pk.center);
      const cutOk = isReachable(cuePos, ghost, obj.pos, pk.center);
      if (cutOk && isPathClear(cuePos, ghost, live, skipCueAndTarget)) {
        const dist = mag(sub(pk.center, obj.pos)) + mag(sub(ghost, cuePos));
        out.push({
          kind: "direct",
          target: tid,
          potId: tid,
          pocket: pk.id,
          aimPoint: ghost,
          action: aimAction(cuePos, ghost, powerFor(dist)),
          path: [obj.pos, pk.center],
          banks: 0,
        });
      }

      // --- Single-cushion bank ----------------------------------------------
      // Aim the object ball at the mirror image of the pocket across each rail:
      // the straight line to the mirror crosses the rail at the true bank point.
      for (const side of SIDES) {
        const mirror = mirrorAcross(pk.center, table, side);
        const bankPoint = railCrossing(obj.pos, mirror, table, side);
        if (!bankPoint) continue;
        const gb = ghostBall(obj.pos, bankPoint);
        if (!isReachable(cuePos, gb, obj.pos, bankPoint)) continue;
        if (!isPathClear(cuePos, gb, live, skipCueAndTarget)) continue;
        const dist =
          mag(sub(bankPoint, obj.pos)) +
          mag(sub(pk.center, bankPoint)) +
          mag(sub(gb, cuePos));
        out.push({
          kind: "bank",
          target: tid,
          potId: tid,
          pocket: pk.id,
          aimPoint: gb,
          action: aimAction(cuePos, gb, powerFor(dist)),
          path: [obj.pos, bankPoint, pk.center],
          banks: 1,
        });
      }

      // --- Double-cushion bank (2 rails) ------------------------------------
      // Mirror pocket across rail1 → m1, then m1 across rail2 → m2. Object
      // ball aims toward m2, hits side2 at B2, deflects toward m1, hits side1
      // at B1, arrives at pocket. The reflection principle ensures the straight
      // line obj→m2 unfolds to the correct two-cushion path.
      for (const side1 of SIDES) {
        for (const side2 of SIDES) {
          if (side1 === side2) continue;
          const m1 = mirrorAcross(pk.center, table, side1);
          const m2 = mirrorAcross(m1, table, side2);
          const B2 = railCrossing(obj.pos, m2, table, side2);
          if (!B2) continue;
          const B1 = railCrossing(B2, m1, table, side1);
          if (!B1) continue;
          const gb = ghostBall(obj.pos, B2);
          if (!isReachable(cuePos, gb, obj.pos, B2)) continue;
          if (!isPathClear(cuePos, gb, live, skipCueAndTarget)) continue;
          const dist =
            mag(sub(gb, cuePos)) +
            mag(sub(B2, obj.pos)) +
            mag(sub(B1, B2)) +
            mag(sub(pk.center, B1));
          if (dist > MAX_DOUBLE_BANK_PATH) continue;
          out.push({
            kind: "double-bank",
            target: tid,
            potId: tid,
            pocket: pk.id,
            aimPoint: gb,
            action: aimAction(cuePos, gb, powerFor(dist)),
            path: [obj.pos, B2, B1, pk.center],
            banks: 2,
          });
        }
      }
    }

    // --- Combo (through one intermediate ball) ----------------------------
    // Pruned to candidates where the intermediate is within COMBO_RADIUS —
    // long-range combos are rarely makeable and blow the search budget.
    const COMBO_RADIUS = table.length * 0.45;
    for (const mid of live) {
      if (mid.id === tid || mid.id === CUE_ID) continue;
      if (mag(sub(mid.pos, obj.pos)) > COMBO_RADIUS) continue;
      for (const pk of table.pockets) {
        const midGhost = ghostBall(mid.pos, pk.center);
        const objGhost = ghostBall(obj.pos, midGhost);
        if (!isReachable(cuePos, objGhost, obj.pos, midGhost)) continue;
        if (!isReachable(obj.pos, midGhost, mid.pos, pk.center)) continue;
        if (!isPathClear(cuePos, objGhost, live, skipCueAndTarget)) continue;
        const dist =
          mag(sub(objGhost, cuePos)) +
          mag(sub(mid.pos, obj.pos)) +
          mag(sub(pk.center, mid.pos));
        out.push({
          kind: "combo",
          target: tid,
          potId: mid.id,
          pocket: pk.id,
          aimPoint: objGhost,
          action: aimAction(cuePos, objGhost, powerFor(dist)),
          path: [obj.pos, mid.pos, pk.center],
          banks: 0,
        });
      }

      // --- Rail-plus-combination (combo, intermediate ball banks one rail) --
      // Same combo geometry, but the intermediate ball's leg to the pocket
      // reflects off one cushion instead of running straight — the pocket is
      // mirrored across a rail (as in the single-cushion bank above) and the
      // intermediate ball is aimed at the bank point instead of the pocket
      // directly. This is a genuinely distinct, harder route (two balls'
      // worth of aiming precision plus a cushion), not a relabeled combo.
      for (const pk of table.pockets) {
        for (const side of SIDES) {
          const mirror = mirrorAcross(pk.center, table, side);
          const bankPoint = railCrossing(mid.pos, mirror, table, side);
          if (!bankPoint) continue;
          const midGhost = ghostBall(mid.pos, bankPoint);
          const objGhost = ghostBall(obj.pos, midGhost);
          if (!isReachable(cuePos, objGhost, obj.pos, midGhost)) continue;
          if (!isReachable(obj.pos, midGhost, mid.pos, bankPoint)) continue;
          if (!isPathClear(cuePos, objGhost, live, skipCueAndTarget)) continue;
          const dist =
            mag(sub(objGhost, cuePos)) +
            mag(sub(mid.pos, obj.pos)) +
            mag(sub(bankPoint, mid.pos)) +
            mag(sub(pk.center, bankPoint));
          out.push({
            kind: "rail-combo",
            target: tid,
            potId: mid.id,
            pocket: pk.id,
            aimPoint: objGhost,
            action: aimAction(cuePos, objGhost, powerFor(dist)),
            path: [obj.pos, mid.pos, bankPoint, pk.center],
            banks: 1,
          });
        }
      }
    }
  }

  // Cap per candidate kind so seeding cost (1 physics sim + rolloutsPerEval
  // rollouts per candidate) stays within ~600 WASM calls on an open table.
  // Shorter paths are more makeable, so sort by total path distance and keep
  // the top N of each kind. The search still considers all shot types; it just
  // prunes the least-promising geometric variants before UCB search begins.
  const pathLen = (c: Candidate): number => {
    let d = 0;
    for (let i = 1; i < c.path.length; i++) d += mag(sub(c.path[i], c.path[i - 1]));
    return d;
  };
  const capByKind = (kind: CandidateKind, n: number): Candidate[] =>
    out
      .filter((c) => c.kind === kind)
      .sort((a, b) => pathLen(a) - pathLen(b))
      .slice(0, n);

  return [
    ...capByKind("direct", 12),
    ...capByKind("bank", 24),
    ...capByKind("double-bank", 4),
    ...capByKind("combo", 4),
    ...capByKind("rail-combo", 4),
  ];
};

// Is the ghost-ball contact roughly in front of the object ball relative to the
// pocket (i.e. not a physically impossible "cut from behind the pocket line"),
// and is the cue on the correct side to make that contact? A coarse geometric
// feasibility filter — the physics sim is the final arbiter.
const isReachable = (
  cue: Vec2,
  ghost: Vec2,
  obj: Vec2,
  aimTarget: Vec2,
): boolean => {
  // The cue travels from its position *into* the ghost point, then the object
  // sets off toward the target. For a makeable cut those two directions must be
  // roughly aligned (dot > 0); a dot <= 0 means the cue would have to strike the
  // far side of the object (cut beyond 90 degrees), which is impossible.
  const toTarget = norm(sub(aimTarget, obj));
  const cueToGhost = norm(sub(ghost, cue));
  return dotv(toTarget, cueToGhost) > 0.05;
};

const dotv = (a: Vec2, b: Vec2): number => a.x * b.x + a.y * b.y;

// Where the segment obj->mirror crosses the given rail line (the bank point on
// the actual rail). Returns null if it doesn't cross within the rail extent.
const railCrossing = (
  obj: Vec2,
  mirror: Vec2,
  table: Table,
  side: string,
): Vec2 | null => {
  const hx = table.length / 2 - BALL_RADIUS;
  const hy = table.width / 2 - BALL_RADIUS;
  const d = sub(mirror, obj);
  let tCross: number;
  let cx: number;
  let cy: number;
  if (side === "left" || side === "right") {
    const railX = side === "left" ? -hx : hx;
    if (Math.abs(d.x) < 1e-9) return null;
    tCross = (railX - obj.x) / d.x;
    if (tCross <= 0 || tCross >= 1) return null;
    cx = railX;
    cy = obj.y + d.y * tCross;
    if (cy < -hy || cy > hy) return null;
  } else {
    const railY = side === "bottom" ? -hy : hy;
    if (Math.abs(d.y) < 1e-9) return null;
    tCross = (railY - obj.y) / d.y;
    if (tCross <= 0 || tCross >= 1) return null;
    cy = railY;
    cx = obj.x + d.x * tCross;
    if (cx < -hx || cx > hx) return null;
  }
  return add({ x: 0, y: 0 }, { x: cx, y: cy });
};
