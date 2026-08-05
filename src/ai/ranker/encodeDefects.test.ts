// Characterization test for a REAL, CONFIRMED defect in the v2 feature
// encoder — pinned deliberately, not endorsed.
//
// Found during the live-integration sprint by
// `eval/double_bank_analysis.py`: `obstruction_margin_norm` is exactly 0.0 for
// 100% of `combo` and `rail-combo` rows in the baseline dataset (measured:
// 2,151 and 2,394 within-state candidate pairs, clearance mean 0.0, sentinel
// share 0.0). The cause is in `minClearanceBallRadii` (encode.ts): it excludes
// the cue and `candidate.target` from the obstruction scan, but for a
// combo/rail-combo the *intermediate* ball `candidate.potId` sits exactly on
// the intended path (it IS `path[1]`), and it is not excluded — so its
// perpendicular distance to the path is 0, and the minimum clearance is 0 for
// every such candidate regardless of the actual board.
//
// Why this test pins the bug instead of fixing it:
//
// The shipped artifact (public/model/ranker/) was TRAINED on rows produced by
// this exact encoder. Changing `minClearanceBallRadii` would silently break
// train/inference parity — the browser would feed the model a feature
// distribution it has never seen, for two of five candidate kinds, with no
// error anywhere. That is strictly worse than the current, known,
// consistently-wrong-in-both-places behaviour.
//
// The correct fix is a coordinated one owned by the training track: exclude
// `candidate.potId` from the scan, bump `schema.json` to v3 (which makes every
// v2 artifact fail loudly on dimension/version check rather than mis-encode),
// regenerate the dataset, and retrain. Until that lands, this test exists so
// the defect cannot be "cleaned up" by accident in a refactor, and so anyone
// reading the per-kind metrics knows why clearance carries no signal for
// combo/rail-combo.

import { describe, it, expect } from "vitest";
import { makeTable } from "../../physics/table";
import { makeBall } from "../../physics/ball";
import { CUE_ID } from "../../game/rack";
import { generateCandidates } from "../candidates";
import { encodeCandidateFeatures, CANDIDATE_FEATURE_NAMES } from "./encode";

const table = makeTable();
const CLEARANCE_IDX = CANDIDATE_FEATURE_NAMES.indexOf("obstruction_margin_norm");

describe("encode.ts v2 known defect: clearance is degenerate for combo kinds", () => {
  it("obstruction_margin_norm is 0 for every combo / rail-combo candidate, on an open board", () => {
    // A deliberately open board: nothing is actually obstructing anything, so a
    // correct clearance feature would report the large-margin sentinel (1.0
    // after normalisation), not 0.
    const balls = [
      makeBall(CUE_ID, -0.8, 0.0),
      makeBall(1, -0.2, 0.05),
      makeBall(2, 0.3, 0.1),
      makeBall(3, 0.6, -0.2),
    ];
    const targets = [1, 2, 3];
    const candidates = generateCandidates(balls, table, targets);

    const comboLike = candidates.filter(
      (c) => c.kind === "combo" || c.kind === "rail-combo",
    );
    expect(comboLike.length).toBeGreaterThan(0);
    for (const c of comboLike) {
      const f = encodeCandidateFeatures(c, table, balls);
      expect(f[CLEARANCE_IDX]).toBe(0);
    }

    // Meanwhile direct/bank candidates on the same open board DO report the
    // large-clearance sentinel — proving the zero above is specific to the
    // combo path shape, not a property of the board.
    const straight = candidates.filter((c) => c.kind === "direct" || c.kind === "bank");
    expect(straight.length).toBeGreaterThan(0);
    expect(straight.some((c) => encodeCandidateFeatures(c, table, balls)[CLEARANCE_IDX] > 0)).toBe(
      true,
    );
  });

  it("the cause is specifically that candidate.potId is on the path and not skipped", () => {
    const balls = [
      makeBall(CUE_ID, -0.8, 0.0),
      makeBall(1, -0.2, 0.05),
      makeBall(2, 0.3, 0.1),
      makeBall(3, 0.6, -0.2),
    ];
    const combo = generateCandidates(balls, table, [1, 2, 3]).find((c) => c.kind === "combo");
    expect(combo).toBeDefined();
    // potId is a different ball from target (that is what makes it a combo),
    // and it lies exactly on the encoded path.
    expect(combo!.potId).not.toBe(combo!.target);
    const mid = balls.find((b) => b.id === combo!.potId)!;
    expect(combo!.path.some((p) => p.x === mid.pos.x && p.y === mid.pos.y)).toBe(true);
  });
});
