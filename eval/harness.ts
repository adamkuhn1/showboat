// Equal-physics-budget evaluation: classical search vs. the neural hybrid.
//
// The comparison is paired and budget-matched by construction. Both modes see
// the SAME fixture states, the SAME `generateCandidates` output, and the SAME
// `config.simulations` ceiling (which `searchCandidates` enforces with a
// running counter, so "equal budget" is measured, not assumed). The only
// difference is the order candidates are handed to physics in and how many are
// handed over at all.
//
// Every outcome number here comes from executing the chosen shot through the
// real `takeShot` + WASM simulator and reading the real rules outcome. Nothing
// is inferred from the search's own estimates.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable, type Table } from "../src/physics/table";
import { type Ball, cloneBall } from "../src/physics/ball";
import { CUE_ID } from "../src/game/rack";
import { type GameState } from "../src/game/state";
import { makeGame, takeShot, placeCueBall, cloneState } from "../src/game/game";
import { initPhysics, simulateShotWasm, separateOverlaps } from "../src/physics/wasm-bridge";
import { generateCandidates, type Candidate, type CandidateKind } from "../src/ai/candidates";
import {
  searchWithLegacySelection,
  isLegalPot,
  defaultConfig,
  DEFAULT_PRIOR_KEEP_TOP,
  DEFAULT_PRIOR_RESERVE,
  type SearchResult,
  type SearchConfig,
  type SelectionReason,
} from "../src/ai/shotSearch";
import { legalTargets } from "../src/ai/turn";
import { NeuralCandidateEvaluator } from "../src/ai/neural/evaluator";
import { makeFileFetch } from "../src/ai/neural/fileFetch";
import { randomControlledState } from "../training/ranker/phase2c/controlled";

const __dirname = dirname(fileURLToPath(import.meta.url));
export const APP_ROOT = join(__dirname, "..");

export const KINDS: CandidateKind[] = ["direct", "bank", "double-bank", "combo", "rail-combo"];

/** Deterministic PRNG, so a fixture set is reproducible from its seed alone. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export async function bootstrap(): Promise<{ table: Table; evaluator: NeuralCandidateEvaluator }> {
  await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
  const evaluator = new NeuralCandidateEvaluator("model/ranker");
  const state = await evaluator.load(makeFileFetch(join(APP_ROOT, "public")));
  if (state.status !== "ready") {
    throw new Error(`neural evaluator not ready: ${state.status} — ${state.reason}`);
  }
  return { table: makeTable(), evaluator };
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

export interface Fixture {
  id: string;
  balls: Ball[];
  targets: number[];
}

/**
 * A fixed set of decision fixtures. Two sources, both deterministic:
 *  - `controlled`: `randomControlledState` from the Phase 2C generator, the
 *    same routine that produced part of the training data's state
 *    distribution (different seeds — these are not training states).
 *  - `selfplay`: pre-shot states pulled out of real games played by the
 *    classical agent, so the evaluation is not limited to synthetic layouts.
 */
export function makeFixtures(table: Table, count: number, seed: number): Fixture[] {
  const rng = mulberry32(seed);
  const out: Fixture[] = [];

  const nControlled = Math.ceil(count * 0.6);
  for (let i = 0; out.length < nControlled; i++) {
    const balls = randomControlledState(table, rng);
    const targets = balls.filter((b) => b.id !== CUE_ID && b.id !== 8).map((b) => b.id);
    if (targets.length === 0) continue;
    if (generateCandidates(balls, table, targets).length < 6) continue;
    out.push({ id: `controlled-${i}`, balls, targets });
  }

  let guard = 0;
  while (out.length < count && guard++ < 40) {
    for (const st of classicalSelfPlayStates(table, mulberry32(seed + 1000 + guard), 12)) {
      const targets = legalTargets(st, st.turn);
      if (targets.length === 0) continue;
      if (generateCandidates(st.balls, table, targets).length < 6) continue;
      out.push({ id: `selfplay-${guard}-${out.length}`, balls: st.balls, targets });
      if (out.length >= count) break;
    }
  }
  return out;
}

/** Pre-shot states from real classical-agent games. */
function classicalSelfPlayStates(table: Table, rng: () => number, maxShots: number): GameState[] {
  const { state: initial } = makeGame();
  let state = initial;
  const seen: GameState[] = [];
  const cfg: SearchConfig = { simulations: 16, rolloutDepth: 1, rolloutsPerEval: 2, seed: 4242 };
  for (let shot = 0; shot < maxShots && state.winner === null; shot++) {
    if (state.ballInHand !== false) {
      state = placeCueBall(state, -table.length / 4, (rng() - 0.5) * table.width * 0.5);
    }
    if (shot === 0) {
      const report = takeShot(
        state,
        table,
        { phi: 0.02 * (rng() - 0.5), power: 0.95, sideSpin: 0, topSpin: 0 },
        simulateShotWasm,
      );
      state = report.next;
      continue;
    }
    seen.push(cloneState(state));
    const targets = legalTargets(state, state.turn);
    if (targets.length === 0) break;
    const res = searchWithLegacySelection(
      generateCandidates(state.balls, table, targets),
      state.balls,
      targets,
      cfg,
    );
    if (!res.best) break;
    state = takeShot(state, table, res.best.candidate.action, simulateShotWasm).next;
  }
  return seen;
}

// ---------------------------------------------------------------------------
// Oracle: exhaustive, unbudgeted physics over every candidate
// ---------------------------------------------------------------------------

export interface OracleEntry {
  index: number;
  kind: CandidateKind;
  legalPot: boolean;
  scratch: boolean;
}

/**
 * Runs the real WASM simulator once on EVERY candidate, ignoring the search
 * budget. This is the ground truth both modes are scored against — it is never
 * available to either agent, it only defines what "the right answer was".
 */
export function oracle(balls: Ball[], candidates: Candidate[]): OracleEntry[] {
  const work = balls.map(cloneBall);
  separateOverlaps(work);
  return candidates.map((c, index) => {
    const sim = simulateShotWasm(work.map(cloneBall), c.action);
    const scratch = sim.pocketed.includes(CUE_ID);
    return { index, kind: c.kind, legalPot: !scratch && isLegalPot(sim, c), scratch };
  });
}

// ---------------------------------------------------------------------------
// One decision, one mode
// ---------------------------------------------------------------------------

export interface DecisionRecord {
  fixtureId: string;
  mode: "classical" | "hybrid";
  chosenKind: CandidateKind | null;
  chosenIndex: number | null;
  /** Real outcome of executing the chosen shot through takeShot + WASM. */
  legalPot: boolean;
  foul: boolean;
  scratch: boolean;
  isTrickAttempt: boolean;
  /** 1 when a potting candidate existed and the agent didn't choose one. */
  regret: number;
  hadPottingCandidate: boolean;
  hadPottingTrick: boolean;
  physicsCalls: number;
  decisionMs: number;
  neuralMs: number;
  neuralEncodeMs: number;
  neuralRunMs: number;
  candidatesGenerated: number;
  /** Candidates the agent actually spent a physics simulation on. */
  verifiedCount: number;
  /** Of the oracle's potting candidates, how many survived to be verified. */
  recallHits: number;
  recallTotal: number;
  recallByKind: Record<string, { hits: number; total: number }>;
  // --- direct-fallback preservation (the search-policy repair's target) ---
  /** Oracle-potting `direct` candidates that existed at all in this state. */
  pottingDirects: number;
  /** Of those, how many the agent actually spent a physics simulation on. */
  verifiedPottingDirects: number;
  /** Any `direct` candidate at all (potting or not) that reached physics. */
  verifiedDirects: number;
  /** Candidates the kind reserve rescued from pruning. 0 in classical mode. */
  reservePromotions: number;
  /** Which branch of the selection rule fired. */
  selectionReason: SelectionReason | null;
}

/**
 * EXPLORATORY prior transform, evaluation-only — deliberately not in the
 * production search.
 *
 * The headline run showed the hybrid's one clear regression: pruning to the
 * model's global top-K crowds `direct` candidates out (recall 83.0% vs the
 * classical order's 99.1%), because the generator emits 24 banks to 12 directs
 * and the model's scores don't separate them by enough to compensate. A
 * per-kind floor guarantees each kind's own top-N candidates survive pruning,
 * by lifting their score above every un-floored candidate while preserving the
 * model's ordering *within* the floored set. `floor = 0` is the identity.
 *
 * This is a search-policy question (how to spend a fixed budget across kinds),
 * not a model change, which is why it can be tested purely as a score
 * transform without touching `searchCandidates`.
 */
export function applyPerKindFloor(
  scores: number[],
  candidates: Candidate[],
  floor: number,
): number[] {
  if (floor <= 0) return scores;
  const out = scores.slice();
  for (const k of KINDS) {
    const idx = candidates
      .map((c, i) => ({ c, i }))
      .filter(({ c }) => c.kind === k)
      .sort((a, b) => scores[b.i] - scores[a.i])
      .slice(0, floor)
      .map(({ i }) => i);
    for (const i of idx) out[i] = scores[i] + 1;
  }
  return out;
}

export async function runDecision(
  fixture: Fixture,
  table: Table,
  mode: "classical" | "hybrid",
  evaluator: NeuralCandidateEvaluator,
  config: SearchConfig,
  oracleEntries: OracleEntry[],
  keepTop = DEFAULT_PRIOR_KEEP_TOP,
  perKindFloor = 0,
  reserve: Partial<Record<CandidateKind, number>> | undefined = DEFAULT_PRIOR_RESERVE,
): Promise<DecisionRecord> {
  const candidates = generateCandidates(fixture.balls, table, fixture.targets);
  const t0 = performance.now();
  let neuralMs = 0;
  let neuralEncodeMs = 0;
  let neuralRunMs = 0;
  let result: SearchResult;

  if (mode === "hybrid") {
    const scored = await evaluator.score(fixture.balls, table, candidates);
    if (!scored) throw new Error("hybrid mode requested but the evaluator returned no scores");
    neuralMs = scored.inferenceMs;
    neuralEncodeMs = scored.encodeMs;
    neuralRunMs = scored.runMs;
    const manifest = evaluator.getManifest()!;
    result = searchWithLegacySelection(candidates, fixture.balls, fixture.targets, {
      ...config,
      prior: {
        scores: applyPerKindFloor(scored.scores, candidates, perKindFloor),
        keepTop,
        reserve,
        source: manifest.artifact,
        inferenceMs: scored.inferenceMs,
      },
    });
  } else {
    result = searchWithLegacySelection(candidates, fixture.balls, fixture.targets, config);
  }
  const decisionMs = performance.now() - t0;

  // The exact set the search ran a real simulation on, straight from the
  // trace — including candidates whose simulation scratched, which `stats`
  // drops.
  const verifiedIdx = new Set(result.trace?.verifiedIndices ?? []);
  const verifiedTotal = verifiedIdx.size;

  const potting = oracleEntries.filter((o) => o.legalPot);
  const recallByKind: Record<string, { hits: number; total: number }> = {};
  for (const k of KINDS) recallByKind[k] = { hits: 0, total: 0 };
  let recallHits = 0;
  for (const o of potting) {
    recallByKind[o.kind].total++;
    if (verifiedIdx.has(o.index)) {
      recallHits++;
      recallByKind[o.kind].hits++;
    }
  }

  const chosenIndex = result.best ? candidates.indexOf(result.best.candidate) : null;
  const chosenKind = result.best ? result.best.candidate.kind : null;

  // Execute the chosen shot for real.
  let legalPot = false;
  let foul = false;
  let scratch = false;
  if (result.best) {
    const gs: GameState = {
      balls: fixture.balls.map(cloneBall),
      turn: 0,
      groups: { 0: null, 1: null },
      ballInHand: false,
      winner: null,
      broken: true,
      shotCount: 1,
    };
    const report = takeShot(gs, table, result.best.candidate.action, simulateShotWasm);
    scratch = report.sim.pocketed.includes(CUE_ID);
    foul = report.outcome.foul;
    legalPot = !scratch && isLegalPot(report.sim, result.best.candidate);
  }

  return {
    fixtureId: fixture.id,
    mode,
    chosenKind,
    chosenIndex,
    legalPot,
    foul,
    scratch,
    isTrickAttempt: chosenKind !== null && chosenKind !== "direct",
    hadPottingCandidate: potting.length > 0,
    hadPottingTrick: potting.some((o) => o.kind !== "direct"),
    regret: potting.length > 0 && !legalPot ? 1 : 0,
    physicsCalls: result.simulations,
    decisionMs,
    neuralMs,
    neuralEncodeMs,
    neuralRunMs,
    candidatesGenerated: candidates.length,
    verifiedCount: verifiedTotal,
    recallHits,
    recallTotal: potting.length,
    recallByKind,
    pottingDirects: potting.filter((o) => o.kind === "direct").length,
    verifiedPottingDirects: potting.filter((o) => o.kind === "direct" && verifiedIdx.has(o.index))
      .length,
    verifiedDirects: [...verifiedIdx].filter((i) => candidates[i]?.kind === "direct").length,
    reservePromotions: result.trace?.reservePromotions ?? 0,
    selectionReason: result.trace?.selectionReason ?? null,
  };
}

// ---------------------------------------------------------------------------
// Full games
// ---------------------------------------------------------------------------

export interface GameRecord {
  seed: number;
  /** Which player index the hybrid agent controlled. */
  hybridPlayer: 0 | 1;
  winner: 0 | 1 | null;
  shots: number;
  hybridShots: number;
  classicalShots: number;
  hybridPots: number;
  classicalPots: number;
  hybridTrickAttempts: number;
  hybridTrickSuccesses: number;
  classicalTrickAttempts: number;
  classicalTrickSuccesses: number;
}

export async function playGame(
  table: Table,
  evaluator: NeuralCandidateEvaluator,
  seed: number,
  hybridPlayer: 0 | 1,
  config: SearchConfig,
  maxShots = 90,
  perKindFloor = 0,
  reserve: Partial<Record<CandidateKind, number>> | undefined = DEFAULT_PRIOR_RESERVE,
): Promise<GameRecord> {
  const rng = mulberry32(seed);
  const { state: initial } = makeGame();
  let state = initial;
  const rec: GameRecord = {
    seed,
    hybridPlayer,
    winner: null,
    shots: 0,
    hybridShots: 0,
    classicalShots: 0,
    hybridPots: 0,
    classicalPots: 0,
    hybridTrickAttempts: 0,
    hybridTrickSuccesses: 0,
    classicalTrickAttempts: 0,
    classicalTrickSuccesses: 0,
  };

  for (let shot = 0; shot < maxShots && state.winner === null; shot++) {
    if (state.ballInHand !== false) {
      state = placeCueBall(state, -table.length / 4, (rng() - 0.5) * table.width * 0.4);
    }
    if (shot === 0) {
      state = takeShot(
        state,
        table,
        { phi: 0.02 * (rng() - 0.5), power: 0.95, sideSpin: 0, topSpin: 0 },
        simulateShotWasm,
      ).next;
      rec.shots++;
      continue;
    }

    const player = state.turn;
    const isHybrid = player === hybridPlayer;
    const targets = legalTargets(state, player);
    if (targets.length === 0) break;
    const candidates = generateCandidates(state.balls, table, targets);
    if (candidates.length === 0) break;

    let result: SearchResult;
    if (isHybrid) {
      const scored = await evaluator.score(state.balls, table, candidates);
      const manifest = evaluator.getManifest()!;
      result = searchWithLegacySelection(candidates, state.balls, targets, {
        ...config,
        seed: config.seed + shot,
        prior: scored
          ? {
              scores: applyPerKindFloor(scored.scores, candidates, perKindFloor),
              keepTop: DEFAULT_PRIOR_KEEP_TOP,
              reserve,
              source: manifest.artifact,
            }
          : undefined,
      });
    } else {
      result = searchWithLegacySelection(candidates, state.balls, targets, {
        ...config,
        seed: config.seed + shot,
      });
    }
    if (!result.best) break;

    const cand = result.best.candidate;
    const report = takeShot(state, table, cand.action, simulateShotWasm);
    const potted = !report.sim.pocketed.includes(CUE_ID) && isLegalPot(report.sim, cand);
    const isTrick = cand.kind !== "direct";
    if (isHybrid) {
      rec.hybridShots++;
      if (potted) rec.hybridPots++;
      if (isTrick) rec.hybridTrickAttempts++;
      if (isTrick && potted) rec.hybridTrickSuccesses++;
    } else {
      rec.classicalShots++;
      if (potted) rec.classicalPots++;
      if (isTrick) rec.classicalTrickAttempts++;
      if (isTrick && potted) rec.classicalTrickSuccesses++;
    }
    state = report.next;
    rec.shots++;
  }
  rec.winner = state.winner;
  return rec;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

export const mean = (xs: number[]): number => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);

/** Paired mean difference with a normal-approximation 95% CI. */
export function pairedDiff(a: number[], b: number[]): { diff: number; ci: [number, number]; n: number } {
  const d = a.map((x, i) => x - b[i]);
  const m = mean(d);
  if (d.length < 2) return { diff: m, ci: [NaN, NaN], n: d.length };
  const sd = Math.sqrt(d.reduce((s, x) => s + (x - m) ** 2, 0) / (d.length - 1));
  const se = sd / Math.sqrt(d.length);
  return { diff: m, ci: [m - 1.96 * se, m + 1.96 * se], n: d.length };
}

const lnGamma = (z: number): number => {
  // Lanczos approximation, g=7, n=9. Accurate to ~1e-13 for the range used here.
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z);
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
};

const lnChoose = (n: number, k: number): number =>
  lnGamma(n + 1) - lnGamma(k + 1) - lnGamma(n - k + 1);

/**
 * Exact McNemar test on paired binary outcomes (`a` = hybrid, `b` = classical).
 *
 * Reported alongside the normal-approximation CI because for a rate near 0.9 at
 * n=210 the two can disagree at the margin, and the discordant-pair counts are
 * the transparent version of the same evidence: concordant pairs carry no
 * information about a difference, so only `aOnly`/`bOnly` do.
 *
 * Two-sided exact binomial p-value under H0: a discordant pair is equally
 * likely to fall either way.
 */
export function mcnemarExact(
  a: number[],
  b: number[],
): { aOnly: number; bOnly: number; nDiscordant: number; p: number } {
  let aOnly = 0;
  let bOnly = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] > b[i]) aOnly++;
    else if (a[i] < b[i]) bOnly++;
  }
  const n = aOnly + bOnly;
  if (n === 0) return { aOnly, bOnly, nDiscordant: 0, p: 1 };
  const k = Math.min(aOnly, bOnly);
  let tail = 0;
  for (let i = 0; i <= k; i++) tail += Math.exp(lnChoose(n, i) + n * Math.log(0.5));
  return { aOnly, bOnly, nDiscordant: n, p: Math.min(1, 2 * tail) };
}

export const median = (xs: number[]): number => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((x, y) => x - y);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};

export const quantile = (xs: number[], q: number): number => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
};

export { defaultConfig };
