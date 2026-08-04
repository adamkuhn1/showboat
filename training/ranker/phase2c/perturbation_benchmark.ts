// Stage A closure item #3: benchmark 8 vs. 16 vs. 32 perturbation rollouts
// per candidate, using NESTED deterministic trials — one 32-trial run per
// candidate, where the 8-rollout estimate is trials[0:8]'s mean, the
// 16-rollout estimate is trials[0:16]'s mean, and 32 is all of them. This
// makes "the first 8 are identical across all three conditions" true by
// construction (a single seeded RNG stream, not three separately-reseeded
// runs), exactly as the brief requires, with no extra bookkeeping.
//
// Usage: npx tsx training/ranker/phase2c/perturbation_benchmark.ts

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { makeTable } from "../../../src/physics/table";
import { type GameState } from "../../../src/game/state";
import { cloneState, takeShot } from "../../../src/game/game";
import { legalTargets } from "../../../src/ai/turn";
import { generateCandidates, type Candidate } from "../../../src/ai/candidates";
import { initPhysics, simulateShotWasm } from "../../../src/physics/wasm-bridge";
import { isLegalPot } from "../../../src/ai/shotSearch";
import { randomControlledState } from "./controlled";
import { playSelfPlayGame } from "./selfplay";
import { makeRng } from "./gen_dataset_v3";

const __dirname = dirname(fileURLToPath(import.meta.url));
const MAX_ROLLOUTS = 32;
const AIM_JITTER_RAD = 0.01;
const POWER_JITTER = 0.03;
const SEED = 20260804;

interface CandidateTrialRecord {
  kind: Candidate["kind"];
  stateId: string;
  candidateId: number;
  legalPotTrials: boolean[]; // length 32, nested-prefix-valid
}

function rateAt(trials: boolean[], n: number): number {
  return trials.slice(0, n).filter(Boolean).length / n;
}

// Wilson score interval (better-behaved than normal-approx at small n / extreme p).
function wilsonInterval(successes: number, n: number): [number, number] {
  if (n === 0) return [0, 1];
  const z = 1.96;
  const p = successes / n;
  const denom = 1 + (z * z) / n;
  const center = p + (z * z) / (2 * n);
  const margin = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return [(center - margin) / denom, (center + margin) / denom];
}

function pooledSpearman(a: number[], b: number[]): number {
  if (a.length < 3) return NaN;
  const rank = (arr: number[]) => {
    const idx = arr.map((v, i) => [v, i] as const).sort((x, y) => x[0] - y[0]);
    const ranks = new Array(arr.length);
    idx.forEach(([, i], r) => (ranks[i] = r));
    return ranks;
  };
  const ra = rank(a);
  const rb = rank(b);
  const n = a.length;
  const meanR = (n - 1) / 2;
  let num = 0,
    da = 0,
    db = 0;
  for (let i = 0; i < n; i++) {
    num += (ra[i] - meanR) * (rb[i] - meanR);
    da += (ra[i] - meanR) ** 2;
    db += (rb[i] - meanR) ** 2;
  }
  return da === 0 || db === 0 ? NaN : num / Math.sqrt(da * db);
}

async function main() {
  const wasmPath = join(__dirname, "../../../src/wasm/showboat_physics_bg.wasm");
  await initPhysics(readFileSync(wasmPath));
  const table = makeTable();
  const rng = makeRng(SEED);

  // Representative states: a handful of self-play trajectory states (to
  // cover mid-game/assigned-group geometry) plus controlled states (to
  // guarantee enough combo/rail-combo-viable layouts — those kinds require
  // two well-placed balls on a usable line, which is rarer in random
  // self-play than a deliberately dense controlled board).
  const states: { state: GameState; id: string }[] = [];
  for (let g = 0; g < 2; g++) {
    const { trajectory } = playSelfPlayGame(table, rng, { maxShots: 15 });
    trajectory.forEach((ts, i) => states.push({ state: ts.state, id: `traj-${g}-s${i}` }));
  }
  for (let c = 0; c < 14; c++) {
    const balls = randomControlledState(table, rng);
    states.push({
      state: { balls, turn: 0, groups: { 0: null, 1: null }, ballInHand: false, winner: null, broken: true, shotCount: 1 },
      id: `controlled-${c}`,
    });
  }

  console.error(`[perturbation-benchmark] ${states.length} representative states`);

  const records: CandidateTrialRecord[] = [];
  const t0 = Date.now();

  for (const { state, id } of states) {
    const targets = legalTargets(state, state.turn);
    if (targets.length === 0) continue;
    const candidates = generateCandidates(state.balls, table, targets);
    candidates.forEach((candidate, candidateId) => {
      const trials: boolean[] = [];
      for (let p = 0; p < MAX_ROLLOUTS; p++) {
        const trialPre = cloneState(state);
        const jittered = {
          ...candidate.action,
          phi: candidate.action.phi + (rng() * 2 - 1) * AIM_JITTER_RAD,
          power: Math.min(1, Math.max(0.05, candidate.action.power + (rng() * 2 - 1) * POWER_JITTER)),
        };
        const report = takeShot(trialPre, table, jittered, simulateShotWasm);
        trials.push(!report.outcome.foul && isLegalPot(report.sim, candidate));
      }
      records.push({ kind: candidate.kind, stateId: id, candidateId, legalPotTrials: trials });
    });
  }

  const elapsedFull = (Date.now() - t0) / 1000;
  console.error(`[perturbation-benchmark] ${records.length} candidates x ${MAX_ROLLOUTS} trials in ${elapsedFull.toFixed(1)}s`);

  // --- Per-record nested estimates at 8/16/32 ---
  const at8 = records.map((r) => rateAt(r.legalPotTrials, 8));
  const at16 = records.map((r) => rateAt(r.legalPotTrials, 16));
  const at32 = records.map((r) => rateAt(r.legalPotTrials, 32));

  const meanAbsDiff = (a: number[], b: number[]) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0) / a.length;
  const mad_8_32 = meanAbsDiff(at8, at32);
  const mad_16_32 = meanAbsDiff(at16, at32);
  const corr_8_32 = pooledSpearman(at8, at32);
  const corr_16_32 = pooledSpearman(at16, at32);

  // Threshold-crossing frequency (0.5 reliability qualification).
  const crosses8v32 = at8.filter((v, i) => (v >= 0.5) !== (at32[i] >= 0.5)).length;
  const crosses16v32 = at16.filter((v, i) => (v >= 0.5) !== (at32[i] >= 0.5)).length;

  // Trick-qualification flips (kind != direct AND reliability >= 0.5).
  const qualifies = (rate: number, kind: string) => kind !== "direct" && rate >= 0.5;
  let trickFlips8 = 0,
    trickFlips16 = 0;
  records.forEach((r, i) => {
    if (qualifies(at8[i], r.kind) !== qualifies(at32[i], r.kind)) trickFlips8++;
    if (qualifies(at16[i], r.kind) !== qualifies(at32[i], r.kind)) trickFlips16++;
  });

  // Uncertainty interval width (Wilson) at each rollout count, using the
  // successes actually observed at that count (not the 32-trial ground truth).
  const meanIntervalWidth = (n: number) => {
    const widths = records.map((r) => {
      const successes = r.legalPotTrials.slice(0, n).filter(Boolean).length;
      const [lo, hi] = wilsonInterval(successes, n);
      return hi - lo;
    });
    return widths.reduce((a, b) => a + b, 0) / widths.length;
  };

  // Within-state top-candidate stability: does argmax candidate (by
  // reliability) within a state change between 8 and 32 rollouts?
  const byState = new Map<string, number[]>(); // stateId -> record indices
  records.forEach((r, i) => {
    if (!byState.has(r.stateId)) byState.set(r.stateId, []);
    byState.get(r.stateId)!.push(i);
  });
  let topStable = 0,
    topTotal = 0;
  for (const idxs of byState.values()) {
    if (idxs.length < 2) continue;
    topTotal++;
    const top8 = idxs.reduce((a, b) => (at8[b] > at8[a] ? b : a));
    const top32 = idxs.reduce((a, b) => (at32[b] > at32[a] ? b : a));
    if (top8 === top32) topStable++;
  }

  // Within-state ranking stability (mean per-state Spearman, 8 vs 32).
  let rankingStabilitySum = 0,
    rankingStabilityCount = 0;
  for (const idxs of byState.values()) {
    if (idxs.length < 3) continue;
    const a = idxs.map((i) => at8[i]);
    const b = idxs.map((i) => at32[i]);
    if (new Set(a).size === 1 || new Set(b).size === 1) continue;
    const rho = pooledSpearman(a, b);
    if (!Number.isNaN(rho)) {
      rankingStabilitySum += rho;
      rankingStabilityCount++;
    }
  }

  // Per-kind breakdown.
  const kinds = [...new Set(records.map((r) => r.kind))];
  const byKind: Record<string, { n: number; mad_8_32: number; mad_16_32: number; mean32: number }> = {};
  for (const kind of kinds) {
    const idxs = records.map((_, i) => i).filter((i) => records[i].kind === kind);
    byKind[kind] = {
      n: idxs.length,
      mad_8_32: meanAbsDiff(idxs.map((i) => at8[i]), idxs.map((i) => at32[i])),
      mad_16_32: meanAbsDiff(idxs.map((i) => at16[i]), idxs.map((i) => at32[i])),
      mean32: idxs.reduce((s, i) => s + at32[i], 0) / idxs.length,
    };
  }

  // Per-reliability-tier breakdown (tiers from the 32-rollout "ground truth").
  const tiers: Record<string, number[]> = { low: [], medium: [], high: [] };
  records.forEach((_, i) => {
    const tier = at32[i] < 0.33 ? "low" : at32[i] < 0.67 ? "medium" : "high";
    tiers[tier].push(i);
  });
  const byTier: Record<string, { n: number; mad_8_32: number }> = {};
  for (const [tier, idxs] of Object.entries(tiers)) {
    byTier[tier] = { n: idxs.length, mad_8_32: idxs.length ? meanAbsDiff(idxs.map((i) => at8[i]), idxs.map((i) => at32[i])) : NaN };
  }

  const result = {
    n_states: states.length,
    n_candidates: records.length,
    total_wall_seconds: elapsedFull,
    runtime_multiplier_8_to_32: 4, // by construction: same trials, just truncated -- real cost multiplier if run standalone
    mean_absolute_diff: { "8_vs_32": mad_8_32, "16_vs_32": mad_16_32 },
    correlation: { "8_vs_32": corr_8_32, "16_vs_32": corr_16_32 },
    threshold_crossing_frequency_0_5: {
      "8_vs_32": crosses8v32 / records.length,
      "16_vs_32": crosses16v32 / records.length,
    },
    trick_qualification_flip_frequency: {
      "8_vs_32": trickFlips8 / records.length,
      "16_vs_32": trickFlips16 / records.length,
    },
    mean_wilson_interval_width: { "8": meanIntervalWidth(8), "16": meanIntervalWidth(16), "32": meanIntervalWidth(32) },
    within_state_top_candidate_stability_8_vs_32: topTotal ? topStable / topTotal : NaN,
    within_state_ranking_stability_8_vs_32: rankingStabilityCount ? rankingStabilitySum / rankingStabilityCount : NaN,
    by_candidate_kind: byKind,
    by_reliability_tier_ground_truth_32: byTier,
  };

  console.error(JSON.stringify(result, null, 2));
  const outPath = join(__dirname, "perturbation_benchmark_results.json");
  writeFileSync(outPath, JSON.stringify(result, null, 2));
  console.error(`[perturbation-benchmark] wrote ${outPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
