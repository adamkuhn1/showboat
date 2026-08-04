import { type Ball, cloneBall } from "../physics/ball";
import { type Table } from "../physics/table";
import { type CueAction } from "../physics/cue";
import { type SimResult } from "../physics/engine";
import { type Candidate, generateCandidates } from "./candidates";
import { rolloutValueWasm, separateOverlaps, simulateShotWasm } from "../physics/wasm-bridge";
import { railsBeforePot } from "./trace";

/**
 * A legal pot requires: the cue's first contact matches the candidate's
 * intended legal ball (`sim.firstContact` — an obstruction the geometry check
 * missed, or a deflection, can make the cue strike something else first,
 * which is an illegal first contact regardless of what ends up pocketed),
 * the ball actually meant to drop (`candidate.potId` — for a combo/
 * rail-combo this is the driven intermediate ball, NOT `candidate.target`,
 * the first-contact ball; see `Candidate.potId`'s doc comment in
 * candidates.ts) is the one that's pocketed, AND — when `potId !== target`
 * (combo/rail-combo, where a second object ball is involved) — that the
 * intended intermediate ball actually struck the pocketed ball, not just
 * that both events happened to occur somewhere in the same shot.
 *
 * The third condition closes a real endpoint-only labeling gap (found during
 * Phase 2C Stage A closure): checking only `firstContact` + final `pocketed`
 * set cannot distinguish "the combo worked as intended" from "the cue hit
 * `target` first, and `potId` was *separately* pocketed by some unrelated
 * contact chain in the same shot" — both produce an identical
 * (firstContact, pocketed) endpoint. `SimResult.events` (from the real WASM
 * simulator) records every ball-ball collision in order, which is exactly
 * the information needed to tell these apart. Direct/bank/double-bank never
 * need this check (`potId === target`, one object ball throughout — no
 * intermediate contact to misattribute; a ball reaching a pocket via more or
 * fewer cushion bounces than the candidate generator planned is still a
 * completely legal pot under real 8-ball rules, not a labeling error).
 *
 * Extracted as its own function (rather than inlined in the seeding loop) so
 * it's directly unit-testable against hand-built SimResult fixtures without
 * needing real physics to organically produce an illegal-first-contact case.
 */
export function isLegalPot(sim: Pick<SimResult, "firstContact" | "pocketed" | "events">, candidate: Candidate): boolean {
  const legalFirstContact = sim.firstContact === candidate.target;
  if (!legalFirstContact || !sim.pocketed.includes(candidate.potId)) return false;
  if (candidate.potId === candidate.target) return true;
  return sim.events.some(
    (e) => e.kind === "ball-ball" && e.balls.includes(candidate.target) && e.balls.includes(candidate.potId),
  );
}

// Flat UCB bandit over the candidate shot set. One level of MCTS is enough for
// a pool turn: a shot is a single continuous action, so the branching factor
// is already handled by candidate generation. Physics rollouts give the value.

export interface CandidateStat {
  candidate: Candidate;
  visits: number;
  value: number;
  // A monotonic, displayable transform of `value` in [0,1) — NOT a
  // calibrated win/pot probability. Nothing has ever measured this against
  // real outcome frequencies (see docs/repair/showboat-ml/EVALUATION_SPEC.md
  // on what "probability" is allowed to mean once that measurement exists).
  // Named `strength` deliberately, not `winProb`/`confidence`, so no UI
  // consumer is tempted to render it with a "%" as if it were one.
  strength: number;
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

// Map mean-balls-pocketed (~0..a few) to a bounded [0,1) display score. This
// is an ad-hoc monotonic squash of the rollout/model value, not a calibrated
// probability — see `strength`'s doc comment above.
const toStrength = (v: number): number => 1 - Math.exp(-v);

export interface SearchConfig {
  simulations: number;
  rolloutDepth: number;
  rolloutsPerEval: number;
  seed: number;
  // When the trained net is loaded, it supplies a position value here so the
  // seeding phase can skip per-candidate rollouts and spend the whole budget
  // on UCB refinement instead.
  netSeedValue?: number;
  // Phase 2A candidate ranker: one score per candidate, indexed identically
  // to `generateCandidates`'s output order. Distinct from `netSeedValue`
  // (a single whole-board scalar applied to every candidate, from the older
  // parked policy/value net) — this is the per-candidate signal that was
  // missing before (see docs/repair/showboat-ml/ARCHITECTURE_DECISION.md).
  // Takes priority over `netSeedValue` when both are present.
  netSeedScores?: number[];
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
    strength: 0,
    rails: c.banks,
    potsTarget: false,
    styleScore: 0,
  }));

  let seed = config.seed >>> 0;
  const nextSeed = (): number => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed;
  };

  // Strict budget accounting: `sims` counts every physics-engine call as it
  // actually happens (a single simulateShotWasm = 1 unit; a rolloutValueWasm
  // call = rolloutsPerEval units, since it runs that many continuations
  // internally), and the seed+refinement loop together are hard-capped at
  // `config.simulations` total units. The previous version estimated seeding
  // cost analytically from `stats.length * rolloutsPerEval` *after* the loop
  // ran — which silently diverged from what actually happened whenever
  // SEED_TIMEOUT_MS truncated the loop early or a scratch skipped a
  // candidate (both make the real seeded count less than `stats.length`),
  // and it didn't count the always-run simulateShotWasm call in the
  // net-seeded path at all — so total physics calls could exceed
  // `config.simulations` without the code ever knowing. Tracking real cost
  // as it's spent, against one shared ceiling, makes that structurally
  // impossible instead of relying on an estimate matching reality.
  let sims = 0;
  const BUDGET = config.simulations;
  const netScores = config.netSeedScores;
  const useNetScores = netScores !== undefined && netScores.length === candidates.length;
  const useNetSeed = useNetScores || config.netSeedValue !== undefined;
  const SEED_TIMEOUT_MS = 2000;
  const seedStart = performance.now();

  for (let ci = 0; ci < stats.length; ci++) {
    if (performance.now() - seedStart > SEED_TIMEOUT_MS) break;
    if (sims + 1 > BUDGET) break; // can't even afford the seeding shot sim
    const s = stats[ci];
    const copy = workBalls.map(cloneBall);
    const sim = simulateShotWasm(copy, s.candidate.action);
    sims += 1;
    // Cue ball id is always 0. A scratch is a foul regardless of what else was
    // pocketed — skip the candidate entirely so it can't win UCB selection.
    if (sim.pocketed.includes(0)) continue;
    s.potsTarget = isLegalPot(sim, s.candidate);
    s.rails = railsBeforePot(sim);
    const isComboLike =
      s.candidate.kind === "combo" ||
      s.candidate.kind === "double-bank" ||
      s.candidate.kind === "rail-combo";
    s.styleScore = s.rails + (isComboLike ? 1 : 0);

    if (useNetSeed) {
      // Per-candidate score when available (Phase 2A ranker); otherwise the
      // old flat whole-board scalar (parked policy/value net) as a fallback.
      // No additional physics cost beyond the simulateShotWasm call above —
      // the net score itself is not a physics rollout.
      const v = useNetScores ? netScores![ci] : config.netSeedValue!;
      s.value = v;
      s.strength = toStrength(v);
      s.visits = 1;
    } else {
      if (sims + config.rolloutsPerEval > BUDGET) continue; // leave unseeded (visits=0); can't afford it
      const v = rolloutValueWasm(
        workBalls,
        s.candidate.action,
        targets,
        config.rolloutDepth,
        config.rolloutsPerEval,
        nextSeed(),
      );
      sims += config.rolloutsPerEval;
      s.visits = 1;
      s.value = v;
      s.strength = toStrength(v);
    }
  }

  const seeded = stats.filter(s => s.visits > 0);

  while (seeded.length > 0 && sims + config.rolloutsPerEval <= BUDGET) {
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
    sims += config.rolloutsPerEval;
    pick.value = (pick.value * pick.visits + v) / (pick.visits + 1);
    pick.visits += 1;
    pick.strength = toStrength(pick.value);
  }

  const sorted = [...stats]
    .filter(s => s.visits > 0)
    .sort((a, b) => b.visits - a.visits || b.value - a.value);

  const best = selectBest(sorted);
  return { best, stats: sorted, simulations: sims };
};

// A trick candidate must clear this strength score to be considered reliable
// enough to attempt over a safe direct pot. Tunable — this is the one number
// that encodes "how much risk Showboat is willing to take for style." 0.5
// means the search's own rollout-estimated success value must be at least as
// likely to work as not, before style preference is allowed to override
// safety. Exported so tests and the overlay can reference the same bar
// rather than a second copy of the number.
export const TRICK_RELIABILITY_THRESHOLD = 0.5;

// How much a style point (see `styleScore` — rails + a combo/double-bank
// bonus) is worth relative to strength when ranking *qualifying* trick
// candidates against each other. Small on purpose: style breaks ties and
// nudges toward flashier routes among comparably-reliable options, it does
// not let a wildly less reliable trick beat a more reliable one outright —
// that job belongs to `TRICK_RELIABILITY_THRESHOLD` above.
const STYLE_WEIGHT = 0.12;

const trickUtility = (s: CandidateStat): number => s.strength + STYLE_WEIGHT * s.styleScore;

/**
 * Showboat's actual selection objective, not just "highest search value":
 * a trick shot (anything other than a direct pot — bank, double-bank, combo,
 * rail-combo) is preferred whenever at least one qualifies, i.e. actually
 * pots its target *and* clears `TRICK_RELIABILITY_THRESHOLD`. Among
 * qualifying tricks, the one maximizing strength + style wins. A direct shot
 * is only selected when no trick candidate qualifies (including "rules make
 * every trick candidate impossible," which shows up here as an empty
 * qualifying set) — matching the product requirement that Showboat is not
 * merely an optimizer for the easiest pot. This is the ONE place shot choice
 * is decided; the overlay's "chosen" highlight and `chooseShot` both read
 * this same result, so the reasoning display can never disagree with it.
 */
export function selectBest(sorted: CandidateStat[]): CandidateStat | null {
  const qualifyingTricks = sorted.filter(
    (s) => s.potsTarget && s.candidate.kind !== "direct" && s.strength >= TRICK_RELIABILITY_THRESHOLD,
  );
  if (qualifyingTricks.length > 0) {
    return qualifyingTricks.reduce((a, b) => (trickUtility(b) > trickUtility(a) ? b : a));
  }
  return sorted.find((s) => s.potsTarget) ?? sorted[0] ?? null;
}

export const chooseShot = (
  balls: Ball[],
  table: Table,
  targets: number[],
  config?: SearchConfig,
): CueAction | null => {
  const res = searchBaseline(balls, table, targets, config);
  return res.best ? res.best.candidate.action : null;
};
