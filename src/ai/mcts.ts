import { type Ball, cloneBall } from "../physics/ball";
import { type Table } from "../physics/table";
import { type CueAction } from "../physics/cue";
import { type Candidate, generateCandidates } from "./candidates";
import { rolloutValueWasm, separateOverlaps, simulateShotWasm } from "../physics/wasm-bridge";
import { railsBeforePot } from "./trace";

// Flat UCB bandit over the candidate shot set. One level of MCTS is enough for
// a pool turn: a shot is a single continuous action, so the branching factor
// is already handled by candidate generation. Physics rollouts give the value.

export interface CandidateStat {
  candidate: Candidate;
  visits: number;
  value: number;
  winProb: number;
  rails: number;
  potsTarget: boolean;
  styleScore: number;
}

export interface SearchResult {
  best: CandidateStat | null;
  stats: CandidateStat[];
  simulations: number;
}

const UCB_C = 1.2;

// Map mean-balls-pocketed (~0..a few) to a displayable win probability.
const squash = (v: number): number => 1 - Math.exp(-v);

export interface SearchConfig {
  simulations: number;
  rolloutDepth: number;
  rolloutsPerEval: number;
  seed: number;
  // When the trained net is loaded, it supplies a position value here so the
  // seeding phase can skip per-candidate rollouts and spend the whole budget
  // on UCB refinement instead.
  netSeedValue?: number;
}

export const defaultConfig: SearchConfig = {
  simulations: 60,
  rolloutDepth: 1,
  rolloutsPerEval: 2,
  seed: 12345,
};

export const searchBaseline = (
  balls: Ball[],
  table: Table,
  targets: number[],
  config: SearchConfig = defaultConfig,
): SearchResult => {
  const candidates = generateCandidates(balls, table, targets);
  if (candidates.length === 0) {
    return { best: null, stats: [], simulations: 0 };
  }

  // Separate any overlapping balls before handing to Rust — the TS animation
  // engine can leave balls at exact contact distance.
  const workBalls = balls.map(cloneBall);
  separateOverlaps(workBalls);

  const stats: CandidateStat[] = candidates.map((c) => ({
    candidate: c,
    visits: 0,
    value: 0,
    winProb: 0,
    rails: c.banks,
    potsTarget: false,
    styleScore: 0,
  }));

  let seed = config.seed >>> 0;
  const nextSeed = (): number => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed;
  };

  let sims = 0;
  const useNetSeed = config.netSeedValue !== undefined;
  const SEED_TIMEOUT_MS = 2000;
  const seedStart = performance.now();

  for (let ci = 0; ci < stats.length; ci++) {
    if (performance.now() - seedStart > SEED_TIMEOUT_MS) break;
    const s = stats[ci];
    const copy = workBalls.map(cloneBall);
    const sim = simulateShotWasm(copy, s.candidate.action);
    // Cue ball id is always 0. A scratch is a foul regardless of what else was
    // pocketed — skip the candidate entirely so it can't win UCB selection.
    if (sim.pocketed.includes(0)) continue;
    s.potsTarget = sim.pocketed.includes(s.candidate.target);
    s.rails = railsBeforePot(sim);
    const isComboLike = s.candidate.kind === "combo" || s.candidate.kind === "double-bank";
    s.styleScore = s.rails + (isComboLike ? 1 : 0);

    if (useNetSeed) {
      s.value = config.netSeedValue!;
      s.winProb = squash(config.netSeedValue!);
      s.visits = 1;
      sims += 1;
    } else {
      const v = rolloutValueWasm(
        workBalls,
        s.candidate.action,
        targets,
        config.rolloutDepth,
        config.rolloutsPerEval,
        nextSeed(),
      );
      s.visits = 1;
      s.value = v;
      s.winProb = squash(v);
      sims += config.rolloutsPerEval;
    }
  }

  const seedCost = useNetSeed ? 0 : stats.length * config.rolloutsPerEval;
  const nRounds = Math.floor(Math.max(0, config.simulations - seedCost) / config.rolloutsPerEval);
  const seeded = stats.filter(s => s.visits > 0);

  for (let r = 0; r < nRounds && seeded.length > 0; r++) {
    const totalVisits = seeded.reduce((a, s) => a + s.visits, 0);
    let pick = seeded[0];
    let bestUcb = -Infinity;
    for (const s of seeded) {
      const ucb = s.value + UCB_C * Math.sqrt(Math.log(totalVisits + 1) / s.visits);
      if (ucb > bestUcb) { bestUcb = ucb; pick = s; }
    }
    const v = rolloutValueWasm(
      workBalls, pick.candidate.action, targets,
      config.rolloutDepth, config.rolloutsPerEval, nextSeed(),
    );
    pick.value = (pick.value * pick.visits + v) / (pick.visits + 1);
    pick.visits += 1;
    pick.winProb = squash(pick.value);
    sims += config.rolloutsPerEval;
  }

  const sorted = [...stats]
    .filter(s => s.visits > 0)
    .sort((a, b) => b.visits - a.visits || b.value - a.value);

  const best = sorted.find(s => s.potsTarget) ?? sorted[0] ?? null;
  return { best, stats: sorted, simulations: sims };
};

export const chooseShot = (
  balls: Ball[],
  table: Table,
  targets: number[],
  config?: SearchConfig,
): CueAction | null => {
  const res = searchBaseline(balls, table, targets, config);
  return res.best ? res.best.candidate.action : null;
};
