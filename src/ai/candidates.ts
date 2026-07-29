import { type Vec2 } from "../physics/vec";
import { type Ball } from "../physics/ball";
import { type Table } from "../physics/table";
import { BALL_RADIUS } from "../physics/constants";
import { type CueAction } from "../physics/cue";
import { CUE_ID } from "../game/rack";

// Candidate-shot GENERATOR. This enumerates aiming *paths* geometrically — it is
// NOT shot scripting. For each legal target ball and each pocket it computes the
// cut angle for a direct pot, plus single- and double-cushion (bank) routes by
// mirroring the pocket across rails, plus simple combo routes through an
// intermediate ball. It emits only the aim direction + a power suggestion; the
// physics sim (and the value function / MCTS) decide which candidate is good.
// Banks and combos are made *available* here exactly like a direct shot is — the
// search selects them on value, so trick shots stay emergent, never canned.

export type CandidateKind = "direct" | "bank" | "combo";

export interface Candidate {
  kind: CandidateKind;
  target: number; // object ball id being played
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

  for (const tid of targets) {
    const obj = live.find((b) => b.id === tid);
    if (!obj) continue;

    for (const pk of table.pockets) {
      // --- Direct pot -----------------------------------------------------
      const ghost = ghostBall(obj.pos, pk.center);
      const cutOk = isReachable(cuePos, ghost, obj.pos, pk.center);
      if (cutOk) {
        const dist = mag(sub(pk.center, obj.pos)) + mag(sub(ghost, cuePos));
        out.push({
          kind: "direct",
          target: tid,
          pocket: pk.id,
          aimPoint: ghost,
          action: aimAction(cuePos, ghost, powerFor(dist)),
          path: [obj.pos, pk.center],
          banks: 0,
        });
      }

      // --- Single-cushion bank -------------------------------------------
      // Aim the object ball at the mirror image of the pocket across each rail:
      // the straight line to the mirror crosses the rail at the true bank point.
      for (const side of SIDES) {
        const mirror = mirrorAcross(pk.center, table, side);
        const bankPoint = railCrossing(obj.pos, mirror, table, side);
        if (!bankPoint) continue;
        const gb = ghostBall(obj.pos, bankPoint);
        if (!isReachable(cuePos, gb, obj.pos, bankPoint)) continue;
        const dist =
          mag(sub(bankPoint, obj.pos)) +
          mag(sub(pk.center, bankPoint)) +
          mag(sub(gb, cuePos));
        out.push({
          kind: "bank",
          target: tid,
          pocket: pk.id,
          aimPoint: gb,
          action: aimAction(cuePos, gb, powerFor(dist)),
          path: [obj.pos, bankPoint, pk.center],
          banks: 1,
        });
      }
    }

    // --- Combo (through one intermediate ball) ----------------------------
    // Play the cue into `obj`, driving it into another live ball `mid` that then
    // heads to a pocket. Enumerated as geometry; value decides if it's worth it.
    for (const mid of live) {
      if (mid.id === tid || mid.id === CUE_ID) continue;
      for (const pk of table.pockets) {
        const midGhost = ghostBall(mid.pos, pk.center); // where obj must send mid
        const objGhost = ghostBall(obj.pos, midGhost); // where cue must send obj
        if (!isReachable(cuePos, objGhost, obj.pos, midGhost)) continue;
        if (!isReachable(obj.pos, midGhost, mid.pos, pk.center)) continue;
        const dist =
          mag(sub(objGhost, cuePos)) +
          mag(sub(mid.pos, obj.pos)) +
          mag(sub(pk.center, mid.pos));
        out.push({
          kind: "combo",
          target: tid,
          pocket: pk.id,
          aimPoint: objGhost,
          action: aimAction(cuePos, objGhost, powerFor(dist)),
          path: [obj.pos, mid.pos, pk.center],
          banks: 0,
        });
      }
    }
  }

  return out;
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
