// Candidate-ranker feature encoding — the ONE encoder used by all three
// consumers: the Node/tsx dataset generator (training/ranker/gen_dataset.ts),
// the browser at inference time (onnx.ts's evaluateCandidates), and the tests.
// There is deliberately no second (Python) re-implementation of this function
// — see docs/repair/showboat-ml/ARCHITECTURE_DECISION.md's "versioned feature
// contract" section. Python only ever consumes already-encoded rows written
// by the dataset generator; it never re-derives features from raw state.

import schema from "./schema.json";
import { type Candidate } from "../candidates";
import { type Table } from "../../physics/table";
import { type Ball } from "../../physics/ball";
import { type Vec2 } from "../../physics/vec";
import { BALL_RADIUS } from "../../physics/constants";
import { encodeObservation, OBS_DIM, OBS_BALLS } from "../onnx";
import { CUE_ID } from "../../game/rack";

export const SCHEMA_VERSION: string = schema.schema_version;
export const BOARD_DIM: number = schema.board_dim;
export const CANDIDATE_DIM: number = schema.candidate_dim;
export const TOTAL_DIM: number = schema.total_dim;
export const CANDIDATE_FEATURE_NAMES: readonly string[] = schema.candidate_features;

// Board-dim sanity check happens at module load, not buried in a test only —
// if onnx.ts's OBS_DIM ever changes, this throws immediately instead of
// silently misaligning every downstream ranker row.
if (BOARD_DIM !== OBS_DIM) {
  throw new Error(
    `ranker schema.board_dim (${BOARD_DIM}) no longer matches onnx.ts OBS_DIM (${OBS_DIM}) — update schema.json deliberately, this is not a drift to patch around.`,
  );
}

const POCKET_IDS = ["bl", "tl", "br", "tr", "sb", "st"] as const;
const KINDS = ["direct", "bank", "double-bank", "combo", "rail-combo"] as const;

const sub = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y });
const mag = (a: Vec2): number => Math.hypot(a.x, a.y);

/**
 * Minimum perpendicular clearance (in ball radii) between the candidate's
 * intended object-ball path and any other live ball — a continuous version
 * of the same geometry `candidates.ts`'s isPathClear already checks as a
 * boolean. Larger = safer/more open shot. Balls on the path intentionally
 * (the target itself, the cue) are excluded. Returns a large sentinel
 * (not Infinity, to stay finite in a training tensor) when the path has
 * fewer than 2 points or nothing is close.
 *
 * KNOWN DEFECT (v2 schema, deliberately NOT fixed here — see
 * `encodeDefects.test.ts` and docs/repair/product-proof-sprint/showboat-live/
 * REPORT.md §6): `skip` omits `candidate.potId`. For a combo/rail-combo the
 * intermediate ball IS `path[1]`, so its perpendicular distance to the path is
 * exactly 0 and this function returns 0 for every combo and rail-combo
 * candidate — measured at 100% of both kinds across the baseline dataset.
 * Clearance is the single highest-weight feature in the transparent heuristic
 * baseline (+1.47), so those two kinds are effectively blind to obstruction.
 *
 * It is not patched in place because the shipped artifact was trained on rows
 * produced by this exact function: changing it would break train/inference
 * parity silently. The fix belongs to a coordinated schema v3 + dataset
 * regeneration + retrain (a v2 artifact then fails loudly on the version check
 * instead of mis-encoding), which is the training track's call, not a
 * runtime-side edit.
 */
function minClearanceBallRadii(candidate: Candidate, balls: Ball[]): number {
  const path = candidate.path;
  if (path.length < 2) return 8; // sentinel: "no path to obstruct"
  let minPerp = Infinity;
  const skip = new Set<number>([CUE_ID, candidate.target]);
  for (let i = 0; i < path.length - 1; i++) {
    const from = path[i];
    const to = path[i + 1];
    const d = sub(to, from);
    const len = mag(d);
    if (len < 1e-9) continue;
    const dn = { x: d.x / len, y: d.y / len };
    for (const b of balls) {
      if (skip.has(b.id) || b.pocketed) continue;
      const v = sub(b.pos, from);
      const t = v.x * dn.x + v.y * dn.y;
      if (t < 0 || t > len) continue;
      const perp = Math.hypot(v.x - t * dn.x, v.y - t * dn.y);
      minPerp = Math.min(minPerp, perp);
    }
  }
  if (!Number.isFinite(minPerp)) return 8; // nothing found near the path
  return minPerp / BALL_RADIUS;
}

function pathLength(path: Vec2[]): number {
  let len = 0;
  for (let i = 0; i < path.length - 1; i++) len += mag(sub(path[i + 1], path[i]));
  return len;
}

/**
 * Encode one candidate's own features (not the board). Length = CANDIDATE_DIM.
 *
 * ---------------------------------------------------------------------------
 * KNOWN REPRESENTATION CEILING — two-cushion (`double-bank`) geometry.
 *
 * Recorded here, at the encoder, because this is where the information is lost.
 * Diagnosed in the prior sprint (docs/repair/product-proof-sprint/
 * showboat-live/REPORT.md §6, `eval/double_bank_analysis.py`) and extended in
 * docs/repair/release-candidate/showboat/REPORT.md §6. **Not fixed** — fixing
 * it requires schema v3 + dataset regeneration + a retrain, all out of scope.
 *
 * `candidates.ts` builds a double-bank as `path = [obj, B2, B1, pocket]`, where
 * B2 and B1 are the two rail contact points produced by the mirror
 * construction, chosen from an ordered pair of distinct cushions
 * (`side1`, `side2`). This block then collapses that entire route into:
 *
 *   - `banks_norm` = min(banks/4, 1) — a CONSTANT 0.5 for every double-bank, so
 *     it carries exactly zero within-kind information;
 *   - `path_length_norm` — one scalar for the whole three-segment polyline;
 *   - `obstruction_margin_norm` — one global minimum clearance over all
 *     segments.
 *
 * So the following are **not expressible**, and no model reading these 20 dims
 * can rank on them, however large:
 *
 *   1. WHICH two cushions were used. `(side1, side2)` is discarded entirely.
 *      Long-then-short and short-then-long routes to the same pocket with the
 *      same total length are literally the same input vector.
 *   2. The rail contact points B1, B2 themselves.
 *   3. The angle of incidence at each cushion — the quantity that actually
 *      governs cushion speed loss and throw, i.e. whether a two-railer is
 *      makeable at all.
 *   4. The per-segment split of the route. `|obj-B2| + |B2-B1| + |B1-pocket|`
 *      is one number; a short final approach and a long one are
 *      indistinguishable at equal total.
 *   5. Per-segment clearance. A ball blocking the first segment and a ball
 *      blocking the last collapse to the same global minimum.
 *
 * Consistent with that, the measured within-state Spearman of the best single
 * candidate feature against the label, restricted to double-bank, is 0.094 —
 * against 0.27-0.40 for every other kind. The Phase 2E relational model reaches
 * 0.237 on test (vs the MLP's 0.103) by re-deriving mirror geometry from the
 * BOARD block inside its own graph, but it has to do so for all four cushions
 * generically, because which pair this candidate uses is not in the input. That
 * is the residual ceiling.
 *
 * A future schema v3 would need, at minimum: `side1`/`side2` one-hots, the two
 * contact points, incidence cos/sin at each cushion, per-segment lengths, and
 * per-segment clearances — roughly candidate_dim 20 -> ~40. The version bump is
 * what makes it safe: a v2 artifact then fails loudly on the schema check
 * rather than silently mis-encoding.
 * ---------------------------------------------------------------------------
 */
export function encodeCandidateFeatures(
  candidate: Candidate,
  table: Table,
  balls: Ball[],
): Float32Array {
  const f = new Float32Array(CANDIDATE_DIM);
  let i = 0;
  f[i++] = Math.sin(candidate.action.phi);
  f[i++] = Math.cos(candidate.action.phi);
  f[i++] = candidate.action.power;
  f[i++] = candidate.action.sideSpin;
  f[i++] = candidate.action.topSpin;
  for (const k of KINDS) f[i++] = candidate.kind === k ? 1 : 0;
  f[i++] = Math.min(candidate.banks / 4, 1);
  f[i++] = Math.min(candidate.target / (OBS_BALLS - 1), 1);
  for (const p of POCKET_IDS) f[i++] = candidate.pocket === p ? 1 : 0;
  const diag = Math.hypot(table.length, table.width);
  f[i++] = Math.min(pathLength(candidate.path) / diag, 1);
  f[i++] = Math.min(minClearanceBallRadii(candidate, balls) / 8, 1);
  // Float32Array silently drops out-of-range writes and leaves under-filled
  // tail slots at 0 — a KINDS/POCKET_IDS length change wouldn't otherwise
  // surface as an error, just quietly wrong training data. Fail loudly instead.
  if (i !== CANDIDATE_DIM) {
    throw new Error(
      `encodeCandidateFeatures wrote ${i} floats, schema.json declares candidate_dim=${CANDIDATE_DIM} — update schema.json deliberately if KINDS/POCKET_IDS changed.`,
    );
  }
  return f;
}

/** Full model input row for one candidate: board block + candidate block. Length = TOTAL_DIM. */
export function encodeRow(balls: Ball[], table: Table, candidate: Candidate): Float32Array {
  const board = encodeObservation(balls, table.length / 2, table.width / 2);
  const cand = encodeCandidateFeatures(candidate, table, balls);
  const out = new Float32Array(TOTAL_DIM);
  out.set(board, 0);
  out.set(cand, BOARD_DIM);
  return out;
}
