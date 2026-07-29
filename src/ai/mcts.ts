import { type Ball, cloneBall } from "../physics/ball";
import { type Table } from "../physics/table";
import { type CueAction } from "../physics/cue";
import { type Candidate, generateCandidates } from "./candidates";
import { rolloutValueWasm, simulateShotWasm } from "../physics/wasm-bridge";
import { railsBeforePot } from "./trace";

// Client-side MCTS over the candidate-shot set, using the Rust→WASM physics for
// simulation and the native rollout hot loop for value estimation.
//
// This is the pure-search BASELINE opponent (milestone 3): priors are uniform
// and the leaf value is the physics rollout — there is NO learned network here.
// It is explicitly the win-rate benchmark, NOT the shipped trained AI. When the
// ONNX net lands (milestone 5) it replaces the uniform prior + rollout value.
//
// Search is a flat "bandit over candidates" MCTS: because a pool turn is a
// single continuous action refined into discrete candidate paths, one level of
// UCB selection over candidates with physics rollouts is the honest baseline and
// keeps the per-decision cost bounded for real-time play.

export interface CandidateStat {
  candidate: Candidate;
  visits: number;
  value: number; // running mean rollout value
  winProb: number; // squashed value in [0,1] for the overlay
  rails: number; // cushions before the pot in the *actual* simulated shot
  potsTarget: boolean; // did the simulated shot pocket the intended ball?
}

export interface SearchResult {
  best: CandidateStat | null;
  stats: CandidateStat[]; // all candidates, sorted by visits desc (overlay data)
  simulations: number;
}

const UCB_C = 1.2;

// Squash a rollout value (mean target-balls pocketed, ~0..a few) into [0,1] so
// the overlay can show a "win-prob"-style number. Monotonic, honest transform.
const squash = (v: number): number => 1 - Math.exp(-v);

export interface SearchConfig {
  simulations: number; // total rollout budget across the turn
  rolloutDepth: number; // continuation shots per rollout
  rolloutsPerEval: number; // playouts averaged per candidate visit
  seed: number;
}

export const defaultConfig: SearchConfig = {
  simulations: 240,
  rolloutDepth: 2,
  rolloutsPerEval: 6,
  seed: 12345,
};

// Run the baseline search for the current shooter. `targets` are the legal
// object-ball ids (their group, or the 8 when cleared).
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

  const stats: CandidateStat[] = candidates.map((c) => ({
    candidate: c,
    visits: 0,
    value: 0,
    winProb: 0,
    rails: c.banks,
    potsTarget: false,
  }));

  // Seed: simulate each candidate once to record what the shot *actually* does
  // (real event trace → rails-before-pot, did-it-pot) for honest overlay data.
  let seed = config.seed >>> 0;
  const nextSeed = (): number => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed;
  };

  let sims = 0;
  for (const s of stats) {
    const copy = balls.map(cloneBall);
    const sim = simulateShotWasm(copy, s.candidate.action);
    s.potsTarget = sim.pocketed.includes(s.candidate.target);
    s.rails = railsBeforePot(sim);
    const v = rolloutValueWasm(
      balls,
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

  // UCB rounds: spend the remaining budget on the most promising candidates.
  const rounds = Math.max(0, config.simulations - stats.length * config.rolloutsPerEval);
  const perRound = config.rolloutsPerEval;
  const nRounds = Math.floor(rounds / perRound);

  for (let r = 0; r < nRounds; r++) {
    const totalVisits = stats.reduce((a, s) => a + s.visits, 0);
    let pick = stats[0];
    let bestUcb = -Infinity;
    for (const s of stats) {
      const exploit = s.value;
      const explore = UCB_C * Math.sqrt(Math.log(totalVisits + 1) / s.visits);
      const ucb = exploit + explore;
      if (ucb > bestUcb) {
        bestUcb = ucb;
        pick = s;
      }
    }
    const v = rolloutValueWasm(
      balls,
      pick.candidate.action,
      targets,
      config.rolloutDepth,
      perRound,
      nextSeed(),
    );
    // Incremental mean.
    pick.value = (pick.value * pick.visits + v) / (pick.visits + 1);
    pick.visits += 1;
    pick.winProb = squash(pick.value);
    sims += perRound;
  }

  const sorted = [...stats].sort((a, b) => b.visits - a.visits || b.value - a.value);
  // The chosen shot is the most-visited candidate (standard MCTS move rule),
  // tie-broken by value. Prefer a candidate that actually pots its target.
  const best =
    sorted.find((s) => s.potsTarget) ?? sorted[0] ?? null;

  return { best, stats: sorted, simulations: sims };
};

// Convenience: pick just the action to play.
export const chooseShot = (
  balls: Ball[],
  table: Table,
  targets: number[],
  config?: SearchConfig,
): CueAction | null => {
  const res = searchBaseline(balls, table, targets, config);
  return res.best ? res.best.candidate.action : null;
};
