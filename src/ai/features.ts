import { type Vec2, dot, mag, normalize, sub } from "../physics/vec";
import { BALL_RADIUS } from "../physics/constants";
import { type Table } from "../physics/table";
import { type GameState } from "../game/state";
import { CUE_ID, EIGHT_ID } from "../game/rack";
import { type Candidate } from "./candidates";

// Feature extraction for candidate ranking. One function, used identically by
// the training pipeline (to build the dataset) and the in-game ranker (at
// inference) — so the model can never see a different representation at play
// time than it was trained on.
//
// All features are pure geometry of (state, candidate); none of them peeks at
// a simulation result. Normalisations keep values roughly in [0, 1].

export const FEATURE_NAMES = [
  "kind_direct",
  "kind_bank",
  "kind_kick",
  "kind_combo",
  "rails_planned",
  "cue_path_len",
  "object_path_len",
  "cut_cos",
  "pocket_approach_cos",
  "cue_path_clearance",
  "object_path_clearance",
  "target_is_eight",
  "power",
] as const;

export const FEATURE_DIM = FEATURE_NAMES.length;

// Distance from point p to segment a-b.
const segDist = (p: Vec2, a: Vec2, b: Vec2): number => {
  const ab = sub(b, a);
  const len2 = ab.x * ab.x + ab.y * ab.y;
  if (len2 < 1e-12) return mag(sub(p, a));
  let t = ((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / len2;
  t = Math.max(0, Math.min(1, t));
  return mag(sub(p, { x: a.x + ab.x * t, y: a.y + ab.y * t }));
};

// Minimum clearance margin of a polyline against every ball not taking part in
// the shot. 1 = wide open, 0 = a ball sits directly on the path. A passing
// ball collides when centres come within one diameter, so the margin measures
// how much beyond that the corridor offers.
const pathClearance = (
  path: Vec2[],
  g: GameState,
  exclude: number[],
): number => {
  let minD = Infinity;
  for (const b of g.balls) {
    if (b.pocketed || exclude.includes(b.id)) continue;
    for (let i = 1; i < path.length; i++) {
      const d = segDist(b.pos, path[i - 1], path[i]);
      if (d < minD) minD = d;
    }
  }
  if (!isFinite(minD)) return 1;
  const margin = (minD - 2 * BALL_RADIUS) / (2 * BALL_RADIUS);
  return Math.max(0, Math.min(1, margin));
};

export const featuresOf = (
  g: GameState,
  table: Table,
  cand: Candidate,
): number[] => {
  const cuePathLen = pathLen(cand.cuePath);
  const objPathLen = pathLen(cand.objectPath);

  // Cut quality at the (first) impact: how straight the striker drives through
  // the ghost point. For combos, take the worse of the two stages.
  const cutCos = cutQuality(cand);

  // Approach alignment into the pocket mouth. The table origin is the centre,
  // so a pocket's centre direction doubles as its mouth direction: corner
  // pockets open along the diagonal, side pockets straight into the rail.
  const lastLeg = normalize(
    sub(
      cand.objectPath[cand.objectPath.length - 1],
      cand.objectPath[cand.objectPath.length - 2],
    ),
  );
  const pocket = table.pockets.find((p) => p.id === cand.pocketId);
  const mouthDir = pocket ? normalize(pocket.center) : lastLeg;
  const approachCos = Math.max(0, dot(lastLeg, mouthDir));

  // Path clearances exclude the balls that are supposed to be struck.
  const involved = [CUE_ID, cand.targetBall];
  if (cand.comboBall !== undefined) involved.push(cand.comboBall);
  const cueClear = pathClearance(cand.cuePath, g, involved);
  const objClear = pathClearance(cand.objectPath, g, involved);

  return [
    cand.kind === "direct" ? 1 : 0,
    cand.kind === "bank" ? 1 : 0,
    cand.kind === "kick" ? 1 : 0,
    cand.kind === "combo" ? 1 : 0,
    cand.railsPlanned / 2,
    Math.min(1, cuePathLen / 2.2),
    Math.min(1, objPathLen / 2.2),
    cutCos,
    approachCos,
    cueClear,
    objClear,
    cand.targetBall === EIGHT_ID ? 1 : 0,
    cand.action.power,
  ];
};

const pathLen = (pts: Vec2[]): number => {
  let L = 0;
  for (let i = 1; i < pts.length; i++) L += mag(sub(pts[i], pts[i - 1]));
  return L;
};

const cutQuality = (cand: Candidate): number => {
  // Striker's incoming direction at the ghost point vs the intended object
  // direction. cuePath's last leg approaches the ghost; objectPath's first leg
  // is the intended object direction.
  const n = cand.cuePath.length;
  const inDir = normalize(sub(cand.cuePath[n - 1], cand.cuePath[n - 2]));
  const outDir = normalize(sub(cand.objectPath[1], cand.objectPath[0]));
  const c1 = Math.max(0, dot(inDir, outDir));
  if (cand.kind !== "combo") return c1;
  // Combo second stage: objectPath is [A, ghostT, T, P]; the A->ghostT leg
  // must line up with T->P.
  const stage2In = normalize(sub(cand.objectPath[1], cand.objectPath[0]));
  const stage2Out = normalize(sub(cand.objectPath[3], cand.objectPath[2]));
  const c2 = Math.max(0, dot(stage2In, stage2Out));
  return Math.min(c1, c2);
};
