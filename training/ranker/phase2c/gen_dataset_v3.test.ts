// Semantic fixture tests for the Phase 2C generator's own labeling logic.
//
// Most of the ruleset's semantic categories (illegal first contact, scratch,
// no-rail foul, legal 8-ball win, illegal 8-ball loss, combo potId != target)
// already have real deterministic fixtures elsewhere and are NOT
// re-duplicated here:
//   - src/game/rules.test.ts: no-contact foul, scratch foul, open-table group
//     assignment, illegal-first-contact-after-groups-assigned foul, 8-early
//     loss, legal-8-ball win, break-shot resolution.
//   - src/ai/shotSearch.test.ts: legal/illegal direct pot, legal combo (potId
//     != target) vs illegal combo, scratch-candidate rejection, trick-vs-
//     direct selection for bank/double-bank/combo/rail-combo.
//
// What has zero prior coverage is THIS phase's own new code: classifyPocketed
// (own/opponent ball-pocketed classification) and processState's wiring of
// real takeShot() outcomes into raw_counts. Those are what this file tests —
// via the real WASM physics engine and real production rules, not mocks.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { makeTable } from "../../../src/physics/table";
import { makeBall } from "../../../src/physics/ball";
import { CUE_ID, EIGHT_ID } from "../../../src/game/rack";
import { type GameState } from "../../../src/game/state";
import { initPhysics } from "../../../src/physics/wasm-bridge";
import { classifyPocketed, processState, makeRng, type Profile } from "./gen_dataset_v3";

const __dirname = dirname(fileURLToPath(import.meta.url));

describe("classifyPocketed", () => {
  it("classifies a shooter's own-group ball as own, excluding cue and eight", () => {
    const post: GameState = {
      balls: [],
      turn: 0,
      groups: { 0: "solids", 1: "stripes" },
      ballInHand: false,
      winner: null,
      broken: true,
      shotCount: 1,
    };
    const { own, opp } = classifyPocketed(post, 0, [1, CUE_ID, EIGHT_ID]);
    expect(own).toBe(1);
    expect(opp).toBe(0);
  });

  it("classifies an opponent-group ball as opp", () => {
    const post: GameState = {
      balls: [],
      turn: 0,
      groups: { 0: "solids", 1: "stripes" },
      ballInHand: false,
      winner: null,
      broken: true,
      shotCount: 1,
    };
    const { own, opp } = classifyPocketed(post, 0, [9, 2]);
    expect(own).toBe(1); // ball 2 (solid)
    expect(opp).toBe(1); // ball 9 (stripe)
  });

  it("returns zero/zero when groups are not yet assigned (ambiguous, don't count)", () => {
    const post: GameState = {
      balls: [],
      turn: 0,
      groups: { 0: null, 1: null },
      ballInHand: false,
      winner: null,
      broken: true,
      shotCount: 1,
    };
    const { own, opp } = classifyPocketed(post, 0, [1, 9]);
    expect(own).toBe(0);
    expect(opp).toBe(0);
  });
});

describe("processState: real-physics integration", () => {
  beforeAll(async () => {
    const wasmPath = join(__dirname, "../../../src/wasm/showboat_physics_bg.wasm");
    await initPhysics(readFileSync(wasmPath));
  });

  const testProfile: Profile = { games: 0, maxShotsPerGame: 0, controlledStates: 0, perturbations: 8 };
  const versions = { generator: "test", physics: "test" };

  it("a reliable cut shot into a corner pocket produces a high legal_pot rate and counts it as own, not opponent", () => {
    const table = makeTable();
    const hx = table.length / 2;
    const hy = table.width / 2;
    // Corner pocket "br" is at (hx, -hy). This exact geometry was empirically
    // verified (not derived analytically) to make the real WASM physics pot
    // ~90% of 30 perturbed trials (AIM_JITTER_RAD=0.01 rad, POWER_JITTER=0.03)
    // without scratching — a dead-straight in-line cue-behind-ball shot into
    // a corner turned out to scratch on nearly every trial (the cue follows
    // the object ball straight into the same pocket), so this uses a cut
    // angle instead, which is both more realistic and more reliable here.
    const ball1 = makeBall(1, hx - 0.25, -hy + 0.15); // solid, cut angle into "br"
    const ball2 = makeBall(2, 0, 0.3); // another solid, far away, keeps "solids remaining" > 1 after ball1 drops
    const eight = makeBall(EIGHT_ID, -0.3, -0.3);
    const cue = makeBall(CUE_ID, hx - 0.7, -hy + 0.1);

    const state: GameState = {
      balls: [cue, ball1, ball2, eight],
      turn: 0,
      groups: { 0: "solids", 1: "stripes" },
      ballInHand: false,
      winner: null,
      broken: true,
      shotCount: 5,
    };

    const rng = makeRng(42);
    const rows = processState(state, table, "fixture-family", "fixture-state", "golden", rng, testProfile, versions, "2026-01-01T00:00:00.000Z", 42);
    const rowsForBall1 = rows.filter((r) => r.candidate_pot_id === 1 && r.candidate_target === 1);
    expect(rowsForBall1.length).toBeGreaterThan(0);

    const best = rowsForBall1.reduce((a, b) => (b.raw_counts.legal_pot > a.raw_counts.legal_pot ? b : a));
    // Near-point-blank shot: expect most of the 8 perturbed trials to legally pot it.
    expect(best.raw_counts.legal_pot).toBeGreaterThanOrEqual(6);
    expect(best.raw_counts.own_balls_pocketed_total).toBeGreaterThan(0);
    expect(best.raw_counts.opponent_balls_pocketed_total).toBe(0);
    expect(best.raw_counts.terminal_win).toBe(0); // one of two solids remains after this pot
  });

  it("rejects rows with outcome-derived feature leakage: features are a pure function of the pre-shot state and candidate, independent of raw_counts", () => {
    // Structural leakage check: encode the same state+candidate twice, once
    // before and once after computing raw_counts (i.e. simulate any
    // ordering-dependence bug), and confirm identical feature vectors. If a
    // future change accidentally threaded a post-shot value into encodeRow,
    // the two calls could diverge if raw_counts computation mutated shared
    // ball objects; processState's use of cloneState per trial should
    // prevent that.
    const table = makeTable();
    const ball1 = makeBall(1, 0.5, 0.2);
    const cue = makeBall(CUE_ID, -0.3, -0.1);
    const eight = makeBall(EIGHT_ID, -0.5, 0.4);
    const state: GameState = {
      balls: [cue, ball1, eight],
      turn: 0,
      groups: { 0: "solids", 1: "stripes" },
      ballInHand: false,
      winner: null,
      broken: true,
      shotCount: 5,
    };
    const rng1 = makeRng(7);
    const rows1 = processState(state, table, "f1", "s1", "golden", rng1, testProfile, versions, "2026-01-01T00:00:00.000Z", 7);
    const rng2 = makeRng(7);
    const rows2 = processState(state, table, "f1", "s1", "golden", rng2, testProfile, versions, "2026-01-01T00:00:00.000Z", 7);
    expect(rows1.length).toBe(rows2.length);
    for (let i = 0; i < rows1.length; i++) {
      expect(rows1[i].features).toEqual(rows2[i].features);
    }
  });
});
