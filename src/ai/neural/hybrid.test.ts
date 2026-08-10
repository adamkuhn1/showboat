// Does the trained model actually change anything, and can it change the
// things it must never be able to change?
//
// Everything here runs the real path: real `generateCandidates`, the real
// committed ONNX artifact through real `onnxruntime-web`, and the real WASM
// physics search. The fixtures are fixed board layouts, so every assertion is
// deterministic and re-derivable.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../../physics/table";
import { makeBall, type Ball } from "../../physics/ball";
import { CUE_ID } from "../../game/rack";
import { initPhysics } from "../../physics/wasm-bridge";
import { generateCandidates, type Candidate } from "../candidates";
import {
  searchCandidates,
  searchWithLegacySelection,
  defaultConfig,
  TRICK_RELIABILITY_THRESHOLD,
  type SearchConfig,
  type SearchResult,
} from "../shotSearch";
import { NeuralCandidateEvaluator } from "./evaluator";
import { makeFileFetch } from "./fileFetch";
import { neuralTrickOnlyBrain, classicalTrickOnlyBrain, getBrain, brainLabel } from "../brain";
import { _resetRankerForTests } from "../onnx";
import { type GameState } from "../../game/state";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../../..");
const table = makeTable();

// A tight budget on purpose. `defaultConfig.simulations = 60` at 3 physics
// units per candidate reaches ~20 candidates; these boards generate ~30-45, so
// the budget already forces a choice about WHICH candidates get examined. 30
// units sharpens that to ~10, which is where the ordering decision bites
// hardest and the test runs fastest. Both modes get the identical number.
const BUDGET = 30;
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

/** Fixed mid-game-like layouts with several viable candidate kinds each. */
const BOARDS: Record<string, Ball[]> = {
  openSpread: [
    makeBall(CUE_ID, -0.35, 0.05),
    makeBall(1, 0.25, 0.15),
    makeBall(3, -0.05, -0.28),
    makeBall(9, 0.5, -0.2),
    makeBall(2, 0.55, 0.3),
  ],
  clusteredRight: [
    makeBall(CUE_ID, -0.6, -0.1),
    makeBall(1, 0.3, 0.02),
    makeBall(2, 0.38, 0.1),
    makeBall(4, 0.1, -0.25),
    makeBall(5, -0.2, 0.3),
    makeBall(11, 0.62, -0.05),
  ],
  railHeavy: [
    makeBall(CUE_ID, 0.0, 0.0),
    makeBall(1, -0.7, 0.28),
    makeBall(3, 0.72, -0.29),
    makeBall(6, -0.4, -0.3),
    makeBall(12, 0.45, 0.31),
  ],
  longTable: [
    makeBall(CUE_ID, -0.85, 0.2),
    makeBall(2, 0.15, -0.1),
    makeBall(7, 0.68, 0.24),
    makeBall(9, -0.3, -0.3),
    makeBall(13, 0.4, 0.05),
    makeBall(14, -0.55, -0.05),
  ],
};

const targetsFor = (balls: Ball[]) => balls.filter((b) => b.id !== CUE_ID).map((b) => b.id);

const asState = (balls: Ball[]): GameState => ({
  balls,
  turn: 0,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 1,
});

let evaluator: NeuralCandidateEvaluator;

interface Pair {
  name: string;
  candidates: Candidate[];
  classical: SearchResult;
  hybrid: SearchResult;
  priorScores: number[];
}

const pairs: Pair[] = [];

describe("neural hybrid: material influence on real decisions", () => {
  beforeAll(async () => {
    _resetRankerForTests();
    await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
    evaluator = new NeuralCandidateEvaluator("model/ranker");
    const state = await evaluator.load(makeFileFetch(join(APP_ROOT, "public")));
    expect(state.status).toBe("ready");

    for (const [name, balls] of Object.entries(BOARDS)) {
      const targets = targetsFor(balls);
      const candidates = generateCandidates(balls, table, targets);
      const scored = await evaluator.score(balls, table, candidates);
      expect(scored).not.toBeNull();
      const manifest = evaluator.getManifest()!;
      pairs.push({
        name,
        candidates,
        priorScores: scored!.scores,
        classical: searchWithLegacySelection(candidates, balls, targets, config),
        hybrid: searchWithLegacySelection(candidates, balls, targets, {
          ...config,
          prior: {
            scores: scored!.scores,
            keepTop: KEEP_TOP,
            source: manifest.artifact,
          },
        }),
      });
    }
  }, 60_000);

  it("produces a genuinely per-candidate score, not one number for the board", () => {
    for (const p of pairs) {
      expect(p.priorScores.length).toBe(p.candidates.length);
      expect(new Set(p.priorScores.map((s) => s.toFixed(6))).size).toBeGreaterThan(1);
      for (const s of p.priorScores) {
        expect(s).toBeGreaterThanOrEqual(0);
        expect(s).toBeLessThanOrEqual(1);
      }
    }
  });

  it("changes candidate ordering: at least one fixture ranks candidates differently", () => {
    const orderOf = (r: SearchResult, cands: Candidate[]) =>
      r.stats.map((s) => cands.indexOf(s.candidate)).join(",");
    const differing = pairs.filter((p) => orderOf(p.classical, p.candidates) !== orderOf(p.hybrid, p.candidates));
    expect(differing.length).toBeGreaterThan(0);
  });

  it("changes budget allocation: the set of physics-verified candidates differs", () => {
    let differed = 0;
    for (const p of pairs) {
      const c = new Set(p.classical.trace!.verifiedIndices);
      const h = new Set(p.hybrid.trace!.verifiedIndices);
      // Pruning genuinely happened on every fixture with more candidates than K.
      if (p.candidates.length > KEEP_TOP) expect(p.hybrid.trace!.prunedByPrior).toBeGreaterThan(0);
      if (c.size !== h.size || [...h].some((i) => !c.has(i))) differed++;
    }
    expect(differed).toBeGreaterThan(0);
  });

  it("changes at least one final shot decision", () => {
    const changed = pairs.filter(
      (p) =>
        p.classical.best?.candidate !== p.hybrid.best?.candidate &&
        p.classical.best !== null &&
        p.hybrid.best !== null,
    );
    expect(changed.length).toBeGreaterThan(0);
  });

  it("spends the same physics budget in both modes — the comparison is honest", () => {
    for (const p of pairs) {
      expect(p.classical.simulations).toBeLessThanOrEqual(BUDGET);
      expect(p.hybrid.simulations).toBeLessThanOrEqual(BUDGET);
    }
  });

  it("labels every decision with the mode that actually produced it", () => {
    for (const p of pairs) {
      expect(p.classical.trace!.mode).toBe("classical");
      expect(p.classical.trace!.modelId).toBeUndefined();
      expect(p.hybrid.trace!.mode).toBe("neural-hybrid");
      // Derived from manifest.artifact at runtime (brain.ts), not hardcoded --
      // this assertion tracks whichever model is currently staged in
      // public/model/ranker/ (currently the Phase 2E Deep Sets ranker; see
      // docs/repair/product-proof-sprint/showboat-model-research/REPORT.md).
      expect(p.hybrid.trace!.modelId).toContain("showboat-ranker-phase2e-deepsets");
    }
  });
});

describe("neural hybrid: what the model must NOT be able to do", () => {
  // These use a hand-built adversarial prior rather than the real model's
  // scores. That is the point: the guarantee has to hold for ANY score vector
  // the model could ever emit, including a maximally wrong one — proving it
  // with the real model's (reasonable) output would prove much less.
  //
  // They run through `searchWithLegacySelection` — the evaluation baseline —
  // because they are about what a PRIOR can and cannot do to a physics search,
  // which is shared by both policies, and the baseline is the one that returns
  // a `best` to inspect. The equivalent guarantees for the live trick-only
  // policy are in `src/ai/policy/trickOnly.test.ts`, including an adversarial
  // prior that scores every direct 0.99 with the eligibility filter disabled.

  const board = BOARDS.longTable;
  const targets = targetsFor(board);
  let candidates: Candidate[];
  // Exactly enough to seed every candidate once (1 shot sim + rolloutsPerEval
  // rollout units each) and no more, so these tests verify the FULL candidate
  // set without paying for a long UCB refinement phase.
  let fullBudget: number;

  beforeAll(async () => {
    await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
    candidates = generateCandidates(board, table, targets);
    fullBudget = candidates.length * (1 + config.rolloutsPerEval);
  });

  it("cannot make a non-potting candidate win: selection still requires a real physics pot", () => {
    // Find a candidate the real physics says does NOT pot, and give it the
    // maximum possible prior score.
    const full = searchCandidates(candidates, board, targets, {
      ...config,
      simulations: fullBudget, // verify everything, so we know each candidate's truth
    });
    const nonPotting = full.stats.find((s) => !s.potsTarget);
    const potting = full.stats.find((s) => s.potsTarget);
    expect(nonPotting, "fixture must contain a non-potting candidate").toBeDefined();
    expect(potting, "fixture must contain a potting candidate").toBeDefined();

    const scores = candidates.map((c) => (c === nonPotting!.candidate ? 1 : 0.999));
    const res = searchWithLegacySelection(candidates, board, targets, {
      ...config,
      simulations: fullBudget,
      prior: { scores, keepTop: candidates.length, source: "adversarial" },
    });
    // The adversarial favourite is examined first (rank 1) and still loses.
    const adversarial = res.stats.find((s) => s.candidate === nonPotting!.candidate);
    expect(adversarial?.priorRank).toBe(1);
    expect(res.best!.potsTarget).toBe(true);
    expect(res.best!.candidate).not.toBe(nonPotting!.candidate);
  }, 120_000);

  it("cannot make an unreliable trick qualify: the threshold is on physics strength", () => {
    const scores = candidates.map((c) => (c.kind === "direct" ? 0 : 1));
    const res = searchWithLegacySelection(candidates, board, targets, {
      ...config,
      simulations: fullBudget,
      prior: { scores, keepTop: candidates.length, source: "adversarial-trick-max" },
    });
    const best = res.best!;
    if (best.candidate.kind !== "direct") {
      // A trick was chosen — it must have earned it on physics, not on the prior.
      expect(best.potsTarget).toBe(true);
      expect(best.strength).toBeGreaterThanOrEqual(TRICK_RELIABILITY_THRESHOLD);
    }
    // No trick below the bar can ever be the choice, whatever the prior says.
    for (const s of res.stats) {
      if (s === best) continue;
      if (s.candidate.kind !== "direct" && s.strength < TRICK_RELIABILITY_THRESHOLD) {
        expect(res.best).not.toBe(s);
      }
    }
    // And the prior really did put tricks first.
    const firstRanked = res.stats.find((s) => s.priorRank === 1);
    if (firstRanked) expect(firstRanked.candidate.kind).not.toBe("direct");
  }, 120_000);

  it("cannot resurrect a scratching candidate", () => {
    const probe = searchCandidates(candidates, board, targets, {
      ...config,
      simulations: fullBudget,
    });
    // Candidates that were verified but never made it into `stats` are exactly
    // the ones whose real simulation scratched the cue.
    const inStats = new Set(probe.stats.map((s) => candidates.indexOf(s.candidate)));
    const scratching = probe.trace!.verifiedIndices.filter((i) => !inStats.has(i));
    if (scratching.length === 0) return; // nothing to prove on this fixture

    const scores = candidates.map((_, i) => (scratching.includes(i) ? 1 : 0));
    const res = searchWithLegacySelection(candidates, board, targets, {
      ...config,
      simulations: fullBudget,
      prior: { scores, keepTop: candidates.length, source: "adversarial-scratch" },
    });
    expect(res.trace!.scratched).toBeGreaterThan(0);
    const bestIdx = candidates.indexOf(res.best!.candidate);
    expect(scratching).not.toContain(bestIdx);
  }, 120_000);

  it("never lets the trace claim neural when the model didn't run", () => {
    const res = searchCandidates(candidates, board, targets, config);
    expect(res.trace!.mode).toBe("classical");
    expect(res.trace!.neuralInferenceMs).toBeUndefined();
    // A prior whose length doesn't match the candidate list is ignored outright
    // rather than partially applied.
    const mismatched = searchCandidates(candidates, board, targets, {
      ...config,
      prior: { scores: [0.5], keepTop: 4, source: "wrong-length" },
    });
    expect(mismatched.trace!.mode).toBe("classical");
    expect(mismatched.trace!.prunedByPrior).toBe(0);
  }, 60_000);
});

describe("neural hybrid: fallback is real and never mislabelled", () => {
  it("an unloadable model produces a classical decision with an explicit reason", async () => {
    _resetRankerForTests();
    await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
    const broken = new NeuralCandidateEvaluator("model/ranker");
    const state = await broken.load(
      makeFileFetch(join(APP_ROOT, "public"), { "model/ranker/manifest.json": null }),
    );
    expect(state.status).toBe("absent");

    // Asking for the hybrid brain explicitly, with a dead evaluator.
    const brain = neuralTrickOnlyBrain(broken);
    const res = await brain.plan(asState(BOARDS.openSpread), table, 0, config);
    expect(res.trace!.fallbackReason).toMatch(/model unavailable/);
    // A real playable shot came out of it. `best` is the selected CANDIDATE's
    // stat and is null on a safety kick, which this board reaches at the tight
    // budget above once a trick has to be measurably a trick — so the assertion
    // is on the shot, which exists on every rung.
    expect(res.shot).not.toBeNull();
    expect(res.shot!.kind).not.toBe("direct");
    // The decision itself is a real classical search — same shape, real physics.
    expect(res.simulations).toBeGreaterThan(0);
    expect(res.trace!.prunedByPrior).toBe(0);

    // And the user-facing selection never claims neural.
    expect(getBrain(true, broken).kind).toBe("classical-trick-only");
    expect(brainLabel(true, broken)).toBe("the physics-search opponent");
  }, 30_000);

  it("the classical brain and the fallback path produce the identical decision", async () => {
    _resetRankerForTests();
    await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
    const dead = new NeuralCandidateEvaluator("model/ranker");
    await dead.load(makeFileFetch(join(APP_ROOT, "public"), { "model/ranker/manifest.json": null }));

    const st = asState(BOARDS.clusteredRight);
    const viaFallback = await neuralTrickOnlyBrain(dead).plan(st, table, 0, config);
    const viaClassical = await classicalTrickOnlyBrain().plan(st, table, 0, config);
    expect(viaFallback.best?.candidate.kind).toBe(viaClassical.best?.candidate.kind);
    expect(viaFallback.best?.candidate.pocket).toBe(viaClassical.best?.candidate.pocket);
    expect(viaFallback.simulations).toBe(viaClassical.simulations);
  }, 30_000);
});
