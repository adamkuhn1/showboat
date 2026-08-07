// The hybrid's SEARCH POLICY: how a fixed physics budget is split across
// candidate kinds after the model has ranked them.
//
// These tests deliberately use synthetic prior score vectors rather than the
// model's own output. The properties asserted here must hold for ANY vector the
// model could ever emit — including adversarial ones — so testing them against
// one artifact's actual scores would prove much less. The model's real
// influence is proven separately, against the real artifact, in
// `neural/hybrid.test.ts`.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../physics/table";
import { makeBall, type Ball } from "../physics/ball";
import { CUE_ID } from "../game/rack";
import { initPhysics } from "../physics/wasm-bridge";
import { generateCandidates } from "./candidates";
import {
  searchCandidates,
  searchWithLegacySelection,
  defaultConfig,
  DEFAULT_PRIOR_RESERVE,
  type SearchConfig,
} from "./shotSearch";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../..");
const table = makeTable();

const BUDGET = 60;
const KEEP_TOP = 8;
/**
 * The seeding clock, off.
 *
 * `defaultConfig` pins `searchTimeoutMs: Infinity` so a real search in a test
 * depends on the physics budget rather than on how loaded the machine is. It
 * leaves `seedTimeoutMs` at its 2,000 ms default, which is the same hazard one
 * layer down — and it is not hypothetical: this sprint reproduced it twice,
 * once in `overlayTruthfulness` and once in `rankerIntegration`, both only
 * under CPU contention and both passing in isolation.
 *
 * Live play is unaffected: `withinDeadline` (ai/brain.ts) folds the turn's real
 * remaining time in with `Math.min`, so `Infinity` never reaches a played turn.
 */
const config: SearchConfig = {
  ...defaultConfig,
  simulations: BUDGET,
  seed: 20260805,
  seedTimeoutMs: Number.POSITIVE_INFINITY,
};

// A layout that generates both direct and non-direct candidates in quantity,
// so pruning genuinely has to choose between kinds.
const BALLS: Ball[] = [
  makeBall(CUE_ID, -0.6, -0.1),
  makeBall(1, 0.3, 0.02),
  makeBall(2, 0.38, 0.1),
  makeBall(4, 0.1, -0.25),
  makeBall(5, -0.2, 0.3),
  makeBall(11, 0.62, -0.05),
];
const targets = BALLS.filter((b) => b.id !== CUE_ID).map((b) => b.id);

const run = (scores: number[], reserve?: Record<string, number>) =>
  searchCandidates(generateCandidates(BALLS, table, targets), BALLS, targets, {
    ...config,
    prior: { scores, keepTop: KEEP_TOP, source: "synthetic-prior", reserve },
  });

let candidates: ReturnType<typeof generateCandidates>;

beforeAll(async () => {
  await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
  candidates = generateCandidates(BALLS, table, targets);
  // Guard the fixture itself: these assertions are meaningless if the board
  // doesn't actually produce a mix of kinds with more candidates than slots.
  expect(candidates.length).toBeGreaterThan(KEEP_TOP);
  expect(candidates.some((c) => c.kind === "direct")).toBe(true);
  expect(candidates.filter((c) => c.kind !== "direct").length).toBeGreaterThan(KEEP_TOP);
}, 60_000);

/** A prior that ranks every non-direct candidate above every direct one — the
 *  exact pathology the reserve exists for, made maximal. */
const directsLastScores = (): number[] =>
  candidates.map((c) => (c.kind === "direct" ? 0.01 : 0.99));

describe("kind reserve inside the learned prior's top-K", () => {
  it("is budget-neutral: the same number of candidates survive pruning", () => {
    const scores = directsLastScores();
    const without = run(scores);
    const with2 = run(scores, { direct: 2 });
    expect(with2.trace!.candidatesConsidered).toBe(without.trace!.candidatesConsidered);
    expect(with2.trace!.prunedByPrior).toBe(without.trace!.prunedByPrior);
    // And the hard guarantee that matters: never more than keepTop.
    expect(with2.trace!.candidatesConsidered).toBeLessThanOrEqual(KEEP_TOP);
  });

  it("spends no more physics than the un-reserved policy", () => {
    const scores = directsLastScores();
    expect(run(scores, { direct: 2 }).trace!.physicsCalls).toBeLessThanOrEqual(BUDGET);
    expect(run(scores).trace!.physicsCalls).toBeLessThanOrEqual(BUDGET);
  });

  it("rescues directs the global top-K would have dropped", () => {
    const scores = directsLastScores();
    const without = run(scores);
    const with2 = run(scores, { direct: 2 });

    const directsVerified = (r: ReturnType<typeof run>) =>
      r.trace!.verifiedIndices.filter((i) => candidates[i].kind === "direct").length;

    expect(directsVerified(without)).toBe(0);
    expect(directsVerified(with2)).toBe(2);
    expect(with2.trace!.reservePromotions).toBe(2);
  });

  it("costs nothing when the model already ranks directs highly", () => {
    // Inverse pathology: directs on top. The reserve must be a no-op.
    const scores = candidates.map((c) => (c.kind === "direct" ? 0.99 : 0.01));
    const without = run(scores);
    const with2 = run(scores, { direct: 2 });
    expect(with2.trace!.reservePromotions).toBe(0);
    expect(with2.trace!.verifiedIndices).toEqual(without.trace!.verifiedIndices);
  });

  it("reports the model's own global rank, not the post-reserve order", () => {
    const scores = directsLastScores();
    const r = run(scores, { direct: 2 });
    // Directs were scored last by this prior, so their reported rank must be
    // in the tail even though the reserve kept them. The overlay must not
    // imply the model liked a candidate it didn't.
    const nDirect = candidates.filter((c) => c.kind === "direct").length;
    const rescued = r.stats.filter((s) => s.candidate.kind === "direct");
    for (const s of rescued) {
      expect(s.priorRank!).toBeGreaterThan(candidates.length - nDirect);
    }
  });

  it("cannot promote a candidate past physics verification", () => {
    // Reserve a direct AND rank everything so that direct is the model's
    // favourite. Physics still decides `potsTarget`, and the selection rule
    // still only picks from candidates the simulation sanctioned.
    //
    // Runs through the LEGACY mixed policy on purpose: this test is about the
    // search-policy/reserve interaction, which is shared by both policies, and
    // the legacy selector is the one that still returns a `best` to inspect.
    // The live trick-only policy's own guarantees live in policy/*.test.ts.
    const scores = directsLastScores();
    const r = searchWithLegacySelection(
      generateCandidates(BALLS, table, targets),
      BALLS,
      targets,
      { ...config, prior: { scores, keepTop: KEEP_TOP, source: "synthetic-prior", reserve: { direct: 2 } } },
    );
    for (const s of r.stats) {
      // Every stat the selection can see was really simulated.
      expect(s.verified).toBe(true);
      expect(s.visits).toBeGreaterThan(0);
    }
    if (r.best) {
      expect(r.best.verified).toBe(true);
      // A chosen trick still had to clear the physics-derived threshold.
      if (r.trace!.selectionReason === "trick-qualified") {
        expect(r.best.candidate.kind).not.toBe("direct");
        expect(r.best.potsTarget).toBe(true);
      }
    }
  });

  it("degrades to the un-reserved policy when the kind is absent", () => {
    // A two-ball board: one cue, one object ball, so no candidate can involve
    // an intermediate ball and `combo` cannot be generated at all.
    const twoBall: Ball[] = [makeBall(CUE_ID, -0.6, -0.1), makeBall(1, 0.3, 0.02)];
    const t = [1];
    const cands = generateCandidates(twoBall, table, t);
    expect(cands.some((c) => c.kind === "combo")).toBe(false);

    const scores = cands.map((_, i) => 1 / (i + 1));
    const go = (reserve?: Record<string, number>) =>
      searchCandidates(cands, twoBall, t, {
        ...config,
        prior: { scores, keepTop: KEEP_TOP, source: "synthetic-prior", reserve },
      });

    // Asking to reserve a kind that doesn't exist must be a no-op, not an
    // error and not a stolen slot.
    const a = go({ combo: 3 });
    const b = go();
    expect(a.trace!.reservePromotions).toBe(0);
    expect(a.trace!.verifiedIndices).toEqual(b.trace!.verifiedIndices);
  });

  it("never reserves more slots than keepTop", () => {
    const scores = directsLastScores();
    const greedy = run(scores, { direct: 99, bank: 99, "double-bank": 99 } as Record<string, number>);
    expect(greedy.trace!.candidatesConsidered).toBeLessThanOrEqual(KEEP_TOP);
    expect(greedy.trace!.physicsCalls).toBeLessThanOrEqual(BUDGET);
  });

  it("the shipped default reserves exactly two directs", () => {
    // The production value is asserted here so a silent change to it fails a
    // test rather than quietly altering the agent's behaviour.
    expect(DEFAULT_PRIOR_RESERVE).toEqual({ direct: 2 });
  });
});
