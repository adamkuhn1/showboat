// Pre-registered evaluation of the trick-only policy.
//
//   npx tsx eval/trickOnlyEval.ts --fixtures 400 --games 60 --seed 61903477 \
//     --out trickonly_61903477.json
//
// Pre-registration: docs/repair/personal-authorship-sprint/showboat/TRICK_ONLY_GATE.md
// (frozen and committed before this was run).
//
// THREE ARMS, paired by fixture, identical `generateCandidates` output and
// identical `config.simulations` ceiling:
//
//   A  the previous opponent  — neural prior + the LEGACY mixed policy
//                               (`selectBestWithReason`), reserve {direct: 2}.
//                               This is what shipped.
//   B  trick-only Showboat    — neural prior + `selectTrickOnly`, reserve {},
//                               `eligible: isTrickCandidate`. Same artifact,
//                               same sha256, no retraining.
//   C  classical trick-only   — no prior. Descriptive only; it exists to show
//                               the model-failure fallback is playable.
//
// Every outcome number comes from executing the chosen shot through the real
// `takeShot` + WASM simulator and reading the real `applyShotRules` outcome.
// Nothing is inferred from the search's own estimates.
//
// The win-rate comparison is deliberately NOT a gate criterion. Trick-only is a
// product identity decision; a gate that could veto it on win rate would be a
// gate designed to relitigate the decision. See the pre-registration.

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import {
  APP_ROOT,
  bootstrap,
  makeFixtures,
  oracle,
  pairedDiff,
  mean,
  median,
  quantile,
  defaultConfig,
  mulberry32,
  type Fixture,
  type OracleEntry,
} from "./harness";
import { makeGame, takeShot, placeCueBall } from "../src/game/game";
import { cloneBall } from "../src/physics/ball";
import { type Table } from "../src/physics/table";
import { type GameState } from "../src/game/state";
import { CUE_ID } from "../src/game/rack";
import { simulateShotWasm } from "../src/physics/wasm-bridge";
import { generateCandidates, type CandidateKind } from "../src/ai/candidates";
import {
  DEFAULT_PRIOR_KEEP_TOP,
  DEFAULT_PRIOR_RESERVE,
  isLegalPot,
  searchCandidates,
  searchWithLegacySelection,
  type SearchConfig,
} from "../src/ai/shotSearch";
import { legalTargets } from "../src/ai/turn";
import { isTrickCandidate, selectTrickOnly } from "../src/ai/policy/trickOnly";
import { type SelectionRung } from "../src/ai/trace/contract";
import { type NeuralCandidateEvaluator } from "../src/ai/neural/evaluator";

const arg = (name: string, fallback: number): number => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? Number(process.argv[i + 1]) : fallback;
};
const strArg = (name: string, fallback: string): string => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
};

const N_FIXTURES = arg("fixtures", 400);
const N_GAMES = arg("games", 60);
const SEED = arg("seed", 61903477);
const BUDGET = arg("budget", defaultConfig.simulations);
const OUT_NAME = strArg("out", "trickonly.json");

export type Arm = "A-previous" | "B-trickonly" | "C-classical-trickonly";
type ChosenKind = CandidateKind | "safety-kick";

// `seedTimeoutMs`/`searchTimeoutMs: Infinity` in every arm, for the same reason
// `hybridEval.ts` does it: the production wall-clock guards truncate the search
// by elapsed time, which makes results depend on machine load rather than on
// search policy. The physics budget stays the binding constraint, identically
// in all three arms, so budget parity is preserved exactly. Both guards are
// disabled together — disabling only the seeding one would leave the refinement
// loop timed and reintroduce exactly the load-dependence this avoids.
const CONFIG: SearchConfig = {
  ...defaultConfig,
  simulations: BUDGET,
  seed: 20260101,
  seedTimeoutMs: Infinity,
  searchTimeoutMs: Infinity,
};

/**
 * Fixture state for the DECISION arms. Open table — `groups: null` — which is
 * exactly what `eval/harness.ts:runDecision` uses, and it has to be: the
 * fixture's `targets` are the open-table target set (`makeFixtures` returns
 * every non-8 ball for a controlled fixture), so assigning groups here would
 * score every legal shot at a stripe as a foul and make the numbers
 * incomparable with the corrected gate's measured 92.0% / 4.0% baseline.
 *
 * The cost, stated: on an open table `applyShotRules` skips the first-contact
 * check entirely, so a shot that struck the 8 first is not scored as a foul.
 * That applies identically to all three arms, so the paired comparison is
 * unaffected — but it means the decision-level foul rate is a floor, not a
 * ceiling. First-contact legality IS enforced in the full-game series below,
 * which uses real `GameState`s with real group assignments.
 */
const asState = (balls: GameState["balls"]): GameState => ({
  balls: balls.map(cloneBall),
  turn: 0,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 1,
});

/** Cushion contacts by ANY ball across the whole executed shot. */
const cushionsIn = (events: { kind: string }[]): number =>
  events.filter((e) => e.kind === "ball-cushion").length;

export interface TrickDecisionRecord {
  fixtureId: string;
  arm: Arm;
  chosenKind: ChosenKind | null;
  chosenIndex: number | null;
  rung: SelectionRung | null;
  /** The primary product assertion. Must be 0 across arm B. */
  isDirect: boolean;
  isTrick: boolean;
  isSafety: boolean;
  legalPot: boolean;
  foul: boolean;
  foulReason: string | null;
  scratch: boolean;
  cushions: number;
  physicsCalls: number;
  safetySims: number;
  decisionMs: number;
  /** Did the ORACLE find a makeable direct pot in this state? */
  hadPottingDirect: boolean;
  hadPottingTrick: boolean;
  /** 1 when a potting candidate existed and the executed shot didn't pot. */
  regret: number;
}

async function runDecision(
  fixture: Fixture,
  table: Table,
  arm: Arm,
  evaluator: NeuralCandidateEvaluator,
  oracleEntries: OracleEntry[],
): Promise<TrickDecisionRecord> {
  const candidates = generateCandidates(fixture.balls, table, fixture.targets);
  const t0 = performance.now();

  let chosenKind: ChosenKind | null = null;
  let chosenIndex: number | null = null;
  let rung: SelectionRung | null = null;
  let action: { phi: number; power: number; sideSpin: number; topSpin: number } | null = null;
  let potCandidate: (typeof candidates)[number] | null = null;
  let physicsCalls = 0;
  let safetySims = 0;

  if (arm === "A-previous") {
    const scored = await evaluator.score(fixture.balls, table, candidates);
    if (!scored) throw new Error("arm A requires the model");
    const manifest = evaluator.getManifest()!;
    const res = searchWithLegacySelection(candidates, fixture.balls, fixture.targets, {
      ...CONFIG,
      prior: {
        scores: scored.scores,
        logits: scored.logits,
        keepTop: DEFAULT_PRIOR_KEEP_TOP,
        reserve: DEFAULT_PRIOR_RESERVE,
        source: manifest.artifact,
        inferenceMs: scored.inferenceMs,
      },
    });
    physicsCalls = res.simulations;
    if (res.best) {
      chosenKind = res.best.candidate.kind;
      chosenIndex = candidates.indexOf(res.best.candidate);
      action = res.best.candidate.action;
      potCandidate = res.best.candidate;
    }
  } else {
    let prior: SearchConfig["prior"];
    if (arm === "B-trickonly") {
      const scored = await evaluator.score(fixture.balls, table, candidates);
      if (!scored) throw new Error("arm B requires the model");
      const manifest = evaluator.getManifest()!;
      prior = {
        scores: scored.scores,
        logits: scored.logits,
        keepTop: DEFAULT_PRIOR_KEEP_TOP,
        reserve: {},
        source: manifest.artifact,
        inferenceMs: scored.inferenceMs,
      };
    }
    const outcome = searchCandidates(candidates, fixture.balls, fixture.targets, {
      ...CONFIG,
      prior,
      eligible: isTrickCandidate,
    });
    physicsCalls = outcome.simulations;
    const d = selectTrickOnly(outcome.allStats, outcome.verifications, {
      state: asState(fixture.balls),
      table,
      targets: fixture.targets,
      simulate: simulateShotWasm,
    });
    safetySims = d.safetySimsSpent;
    rung = d.rung;
    if (d.shot) {
      chosenKind = d.shot.kind;
      chosenIndex = d.shot.candidateIndex;
      action = d.shot.action;
      potCandidate = chosenIndex === null ? null : candidates[chosenIndex];
    }
  }
  const decisionMs = performance.now() - t0;

  let legalPot = false;
  let foul = false;
  let foulReason: string | null = null;
  let scratch = false;
  let cushions = 0;
  if (action) {
    const report = takeShot(asState(fixture.balls), table, action, simulateShotWasm);
    scratch = report.sim.pocketed.includes(CUE_ID);
    foul = report.outcome.foul;
    foulReason = report.outcome.foulReason;
    cushions = cushionsIn(report.sim.events);
    legalPot = !scratch && potCandidate !== null && isLegalPot(report.sim, potCandidate);
  }

  const potting = oracleEntries.filter((o) => o.legalPot);
  return {
    fixtureId: fixture.id,
    arm,
    chosenKind,
    chosenIndex,
    rung,
    isDirect: chosenKind === "direct",
    isTrick: chosenKind !== null && chosenKind !== "direct" && chosenKind !== "safety-kick",
    isSafety: chosenKind === "safety-kick",
    legalPot,
    foul,
    foulReason,
    scratch,
    cushions,
    physicsCalls,
    safetySims,
    decisionMs,
    hadPottingDirect: potting.some((o) => o.kind === "direct"),
    hadPottingTrick: potting.some((o) => o.kind !== "direct"),
    regret: potting.length > 0 && !legalPot ? 1 : 0,
  };
}

// ---------------------------------------------------------------------------
// Full games
// ---------------------------------------------------------------------------

export interface TrickGameRecord {
  seed: number;
  opponent: Agent;
  /** Which player index trick-only Showboat controlled. */
  trickPlayer: 0 | 1;
  winner: 0 | 1 | null;
  shots: number;
  trickShots: number;
  oppShots: number;
  trickPots: number;
  oppPots: number;
  trickFouls: number;
  oppFouls: number;
  trickDirects: number;
  trickSafeties: number;
  trickCushions: number;
  byKind: Record<string, { n: number; pots: number }>;
}

/**
 * The three agents that can sit at a table here.
 *
 *   `trickonly`       — the product. Neural prior + `selectTrickOnly`.
 *   `previous`        — what shipped. Neural prior + the legacy mixed policy.
 *   `classical-mixed` — no prior, legacy mixed policy. This is the exact
 *                       opponent the corrected gate's 63.3% figure was measured
 *                       against, so reusing it keeps the two comparable.
 */
export type Agent = "trickonly" | "previous" | "classical-mixed";

type CueActionLike = { phi: number; power: number; sideSpin: number; topSpin: number };
interface Plan {
  action: CueActionLike | null;
  kind: ChosenKind | null;
  potCandidate: ReturnType<typeof generateCandidates>[number] | null;
}

const NO_PLAN: Plan = { action: null, kind: null, potCandidate: null };

async function planFor(
  agent: Agent,
  state: GameState,
  table: Table,
  targets: number[],
  evaluator: NeuralCandidateEvaluator,
  cfg: SearchConfig,
): Promise<Plan> {
  const candidates = generateCandidates(state.balls, table, targets);

  const neuralPrior = async (
    reserve: Partial<Record<CandidateKind, number>>,
  ): Promise<SearchConfig["prior"]> => {
    if (candidates.length === 0) return undefined;
    const scored = await evaluator.score(state.balls, table, candidates);
    const manifest = evaluator.getManifest();
    if (!scored || !manifest) return undefined;
    return {
      scores: scored.scores,
      logits: scored.logits,
      keepTop: DEFAULT_PRIOR_KEEP_TOP,
      reserve,
      source: manifest.artifact,
    };
  };

  if (agent === "previous" || agent === "classical-mixed") {
    if (candidates.length === 0) return NO_PLAN;
    const prior = agent === "previous" ? await neuralPrior(DEFAULT_PRIOR_RESERVE) : undefined;
    const res = searchWithLegacySelection(candidates, state.balls, targets, { ...cfg, prior });
    return res.best
      ? { action: res.best.candidate.action, kind: res.best.candidate.kind, potCandidate: res.best.candidate }
      : NO_PLAN;
  }

  const outcome = searchCandidates(candidates, state.balls, targets, {
    ...cfg,
    prior: await neuralPrior({}),
    eligible: isTrickCandidate,
  });
  const d = selectTrickOnly(outcome.allStats, outcome.verifications, {
    state,
    table,
    targets,
    simulate: simulateShotWasm,
  });
  if (!d.shot) return NO_PLAN;
  return {
    action: d.shot.action,
    kind: d.shot.kind,
    potCandidate: d.shot.candidateIndex === null ? null : candidates[d.shot.candidateIndex],
  };
}

async function playGame(
  table: Table,
  evaluator: NeuralCandidateEvaluator,
  seed: number,
  trickPlayer: 0 | 1,
  opponent: Agent,
  maxShots = 90,
): Promise<TrickGameRecord> {
  const rng = mulberry32(seed);
  const { state: initial } = makeGame();
  let state = initial;
  const rec: TrickGameRecord = {
    seed,
    opponent,
    trickPlayer,
    winner: null,
    shots: 0,
    trickShots: 0,
    oppShots: 0,
    trickPots: 0,
    oppPots: 0,
    trickFouls: 0,
    oppFouls: 0,
    trickDirects: 0,
    trickSafeties: 0,
    trickCushions: 0,
    byKind: {},
  };

  for (let shot = 0; shot < maxShots && state.winner === null; shot++) {
    if (state.ballInHand !== false) {
      state = placeCueBall(state, -table.length / 4, (rng() - 0.5) * table.width * 0.4, table);
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
    const isTrickAgent = player === trickPlayer;
    const targets = legalTargets(state, player);
    if (targets.length === 0) break;

    const cfg = { ...CONFIG, seed: CONFIG.seed + shot };
    const plan = await planFor(
      isTrickAgent ? "trickonly" : opponent,
      state,
      table,
      targets,
      evaluator,
      cfg,
    );
    if (!plan.action) break;

    const report = takeShot(state, table, plan.action, simulateShotWasm);
    const potted =
      !report.sim.pocketed.includes(CUE_ID) &&
      plan.potCandidate !== null &&
      isLegalPot(report.sim, plan.potCandidate);
    if (isTrickAgent) {
      rec.trickShots++;
      rec.trickCushions += cushionsIn(report.sim.events);
      if (potted) rec.trickPots++;
      if (report.outcome.foul) rec.trickFouls++;
      if (plan.kind === "direct") rec.trickDirects++;
      if (plan.kind === "safety-kick") rec.trickSafeties++;
      const k = plan.kind ?? "none";
      rec.byKind[k] = rec.byKind[k] ?? { n: 0, pots: 0 };
      rec.byKind[k].n++;
      if (potted) rec.byKind[k].pots++;
    } else {
      rec.oppShots++;
      if (potted) rec.oppPots++;
      if (report.outcome.foul) rec.oppFouls++;
    }
    state = report.next;
    rec.shots++;
  }
  rec.winner = state.winner;
  return rec;
}

// ---------------------------------------------------------------------------

const rate = (rs: TrickDecisionRecord[], f: (d: TrickDecisionRecord) => boolean) =>
  rs.length ? rs.filter(f).length / rs.length : NaN;

const summarize = (rs: TrickDecisionRecord[]) => ({
  n: rs.length,
  direct_selections: rs.filter((d) => d.isDirect).length,
  trick_selection_rate: rate(rs, (d) => d.isTrick),
  safety_rate: rate(rs, (d) => d.isSafety),
  no_shot_rate: rate(rs, (d) => d.chosenKind === null),
  legal_pot_rate: rate(rs, (d) => d.legalPot),
  legal_shot_rate: rate(rs, (d) => !d.foul),
  foul_rate: rate(rs, (d) => d.foul),
  scratch_rate: rate(rs, (d) => d.scratch),
  mean_regret: mean(rs.map((d) => d.regret)),
  mean_cushions: mean(rs.map((d) => d.cushions)),
  mean_physics_calls: mean(rs.map((d) => d.physicsCalls)),
  mean_safety_sims: mean(rs.map((d) => d.safetySims)),
  max_safety_sims: rs.length ? Math.max(...rs.map((d) => d.safetySims)) : 0,
  decision_ms_median: median(rs.map((d) => d.decisionMs)),
  decision_ms_p95: quantile(rs.map((d) => d.decisionMs), 0.95),
  decision_ms_p99: quantile(rs.map((d) => d.decisionMs), 0.99),
  decision_ms_max: rs.length ? Math.max(...rs.map((d) => d.decisionMs)) : 0,
});

const byChosenKind = (rs: TrickDecisionRecord[]) => {
  const out: Record<string, { n: number; legal_pot_rate: number; foul_rate: number; mean_cushions: number }> = {};
  for (const k of ["direct", "bank", "double-bank", "combo", "rail-combo", "safety-kick", "none"]) {
    const sub = rs.filter((d) => (d.chosenKind ?? "none") === k);
    out[k] = {
      n: sub.length,
      legal_pot_rate: sub.length ? sub.filter((d) => d.legalPot).length / sub.length : NaN,
      foul_rate: sub.length ? sub.filter((d) => d.foul).length / sub.length : NaN,
      mean_cushions: mean(sub.map((d) => d.cushions)),
    };
  }
  return out;
};

const byRung = (rs: TrickDecisionRecord[]) => {
  const out: Record<string, number> = {};
  for (const d of rs) out[d.rung ?? "none"] = (out[d.rung ?? "none"] ?? 0) + 1;
  return out;
};

async function main() {
  const { table, evaluator } = await bootstrap();
  const manifest = evaluator.getManifest()!;
  console.log(
    `[trick-only eval] model ${manifest.artifact} sha256 ${manifest.onnx_sha256.slice(0, 16)}…\n` +
      `[trick-only eval] seed ${SEED}, ${N_FIXTURES} fixtures x 3 arms, ${N_GAMES} games x 2 opponents, ` +
      `budget ${BUDGET} physics units/turn\n`,
  );

  const fixtures = makeFixtures(table, N_FIXTURES, SEED);
  console.log(`[trick-only eval] built ${fixtures.length} fixtures`);

  const arms: Arm[] = ["A-previous", "B-trickonly", "C-classical-trickonly"];
  const decisions: TrickDecisionRecord[] = [];
  let i = 0;
  for (const f of fixtures) {
    const cands = generateCandidates(f.balls, table, f.targets);
    const oracleEntries = oracle(f.balls, cands);
    for (const arm of arms) {
      decisions.push(await runDecision(f, table, arm, evaluator, oracleEntries));
    }
    process.stdout.write(`\r[trick-only eval] fixtures ${++i}/${fixtures.length}`);
  }
  process.stdout.write("\n");

  const A = decisions.filter((d) => d.arm === "A-previous");
  const B = decisions.filter((d) => d.arm === "B-trickonly");
  const C = decisions.filter((d) => d.arm === "C-classical-trickonly");

  const paired = (f: (d: TrickDecisionRecord) => number) => pairedDiff(B.map(f), A.map(f));

  // ---- games -------------------------------------------------------------
  // Two series. `vs-previous` is the head-to-head the product decision is
  // judged on. `vs-classical-mixed` reproduces the arm structure of the
  // corrected gate's 63.3% figure, so the two numbers are comparable.
  const games: TrickGameRecord[] = [];
  for (const opponent of ["previous", "classical-mixed"] as const) {
    for (let g = 0; g < N_GAMES; g++) {
      const trickPlayer: 0 | 1 = g % 2 === 0 ? 1 : 0;
      games.push(await playGame(table, evaluator, SEED + g, trickPlayer, opponent));
      process.stdout.write(`\r[trick-only eval] games vs ${opponent} ${g + 1}/${N_GAMES}   `);
    }
    process.stdout.write("\n");
  }

  const gameSummary = (opponent: Agent) => {
    const gs = games.filter((g) => g.opponent === opponent);
    const decided = gs.filter((g) => g.winner !== null);
    const wins = decided.filter((g) => g.winner === g.trickPlayer).length;
    const n = decided.length;
    const p = n ? wins / n : NaN;
    // Wald 95% interval; at n=60 the half-width around 0.5 is ~12.6pp, which is
    // why the win rate is reported and not gated.
    const half = n ? 1.96 * Math.sqrt((p * (1 - p)) / n) : NaN;
    return {
      opponent,
      n_games: gs.length,
      n_decided: n,
      trickonly_wins: wins,
      opponent_wins: n - wins,
      trickonly_win_rate: p,
      win_rate_ci95: [p - half, p + half],
      ci_half_width_pp: half * 100,
      mean_shots_per_game: mean(gs.map((g) => g.shots)),
      trickonly_shots_to_win: mean(
        decided.filter((g) => g.winner === g.trickPlayer).map((g) => g.trickShots),
      ),
      trickonly_direct_selections: gs.reduce((a, g) => a + g.trickDirects, 0),
      trickonly_safety_rate: (() => {
        const s = gs.reduce((a, g) => a + g.trickShots, 0);
        return s ? gs.reduce((a, g) => a + g.trickSafeties, 0) / s : NaN;
      })(),
      trickonly_pot_rate: (() => {
        const s = gs.reduce((a, g) => a + g.trickShots, 0);
        return s ? gs.reduce((a, g) => a + g.trickPots, 0) / s : NaN;
      })(),
      opponent_pot_rate: (() => {
        const s = gs.reduce((a, g) => a + g.oppShots, 0);
        return s ? gs.reduce((a, g) => a + g.oppPots, 0) / s : NaN;
      })(),
      trickonly_foul_rate: (() => {
        const s = gs.reduce((a, g) => a + g.trickShots, 0);
        return s ? gs.reduce((a, g) => a + g.trickFouls, 0) / s : NaN;
      })(),
      opponent_foul_rate: (() => {
        const s = gs.reduce((a, g) => a + g.oppShots, 0);
        return s ? gs.reduce((a, g) => a + g.oppFouls, 0) / s : NaN;
      })(),
      trickonly_cushions_per_shot: (() => {
        const s = gs.reduce((a, g) => a + g.trickShots, 0);
        return s ? gs.reduce((a, g) => a + g.trickCushions, 0) / s : NaN;
      })(),
      trickonly_by_kind: (() => {
        const out: Record<string, { n: number; pots: number }> = {};
        for (const g of gs) {
          for (const [k, v] of Object.entries(g.byKind)) {
            out[k] = out[k] ?? { n: 0, pots: 0 };
            out[k].n += v.n;
            out[k].pots += v.pots;
          }
        }
        return out;
      })(),
    };
  };

  const results = {
    generated_at: new Date().toISOString(),
    preregistration: "docs/repair/personal-authorship-sprint/showboat/TRICK_ONLY_GATE.md",
    config: { fixtures: fixtures.length, games: N_GAMES, seed: SEED, budget: BUDGET },
    model: { artifact: manifest.artifact, sha256: manifest.onnx_sha256, retrained: false },
    decisions: {
      "A-previous": summarize(A),
      "B-trickonly": summarize(B),
      "C-classical-trickonly": summarize(C),
    },
    outcome_by_chosen_kind: {
      "A-previous": byChosenKind(A),
      "B-trickonly": byChosenKind(B),
      "C-classical-trickonly": byChosenKind(C),
    },
    rung_distribution: { "B-trickonly": byRung(B), "C-classical-trickonly": byRung(C) },
    paired_differences_B_minus_A: {
      legal_pot: paired((d) => (d.legalPot ? 1 : 0)),
      legal_shot: paired((d) => (d.foul ? 0 : 1)),
      foul: paired((d) => (d.foul ? 1 : 0)),
      scratch: paired((d) => (d.scratch ? 1 : 0)),
      regret: paired((d) => d.regret),
      cushions: paired((d) => d.cushions),
      physics_calls: paired((d) => d.physicsCalls),
      decision_ms: paired((d) => d.decisionMs),
    },
    // The primary product assertion, restated as a number.
    direct_selections: {
      "A-previous": A.filter((d) => d.isDirect).length,
      "B-trickonly": B.filter((d) => d.isDirect).length,
      "C-classical-trickonly": C.filter((d) => d.isDirect).length,
    },
    states_with_makeable_direct: A.filter((d) => d.hadPottingDirect).length,
    budget: {
      max_physics_calls: Math.max(...decisions.map((d) => d.physicsCalls)),
      over_budget_decisions: decisions.filter((d) => d.physicsCalls > BUDGET).length,
      max_safety_sims: Math.max(...decisions.map((d) => d.safetySims)),
    },
    games: {
      "vs-previous": gameSummary("previous"),
      "vs-classical-mixed": gameSummary("classical-mixed"),
    },
    raw_games: games,
    raw_decisions: decisions,
  };

  const outDir = join(APP_ROOT, "eval/results");
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, OUT_NAME), `${JSON.stringify(results, null, 2)}\n`);

  const pct = (x: number) => (Number.isFinite(x) ? `${(x * 100).toFixed(2)}%` : "n/a");
  const num = (x: number, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "n/a");
  console.log("\n=== decision-level, equal budget, paired by fixture ===");
  const rows: [string, (s: ReturnType<typeof summarize>) => string][] = [
    ["DIRECT selections", (s) => String(s.direct_selections)],
    ["trick selection rate", (s) => pct(s.trick_selection_rate)],
    ["safety rate", (s) => pct(s.safety_rate)],
    ["no shot produced", (s) => pct(s.no_shot_rate)],
    ["legal-pot rate", (s) => pct(s.legal_pot_rate)],
    ["legal-shot rate", (s) => pct(s.legal_shot_rate)],
    ["foul rate", (s) => pct(s.foul_rate)],
    ["scratch rate", (s) => pct(s.scratch_rate)],
    ["mean regret", (s) => num(s.mean_regret, 3)],
    ["cushions / shot", (s) => num(s.mean_cushions, 2)],
    ["physics calls / turn", (s) => num(s.mean_physics_calls, 1)],
    ["safety sims / turn", (s) => num(s.mean_safety_sims, 2)],
    ["max safety sims", (s) => String(s.max_safety_sims)],
    ["decision ms median", (s) => num(s.decision_ms_median, 1)],
    ["decision ms p95", (s) => num(s.decision_ms_p95, 1)],
    ["decision ms p99", (s) => num(s.decision_ms_p99, 1)],
    ["decision ms max", (s) => num(s.decision_ms_max, 1)],
  ];
  console.log(`${"metric".padEnd(24)}${"A previous".padEnd(16)}${"B trick-only".padEnd(16)}C classical-TO`);
  for (const [label, f] of rows) {
    console.log(
      `${label.padEnd(24)}${f(results.decisions["A-previous"]).padEnd(16)}` +
        `${f(results.decisions["B-trickonly"]).padEnd(16)}${f(results.decisions["C-classical-trickonly"])}`,
    );
  }

  console.log("\n=== paired differences (B trick-only - A previous), 95% CI ===");
  for (const [k, v] of Object.entries(results.paired_differences_B_minus_A)) {
    console.log(`${k.padEnd(20)}${num(v.diff, 4)}  [${num(v.ci[0], 4)}, ${num(v.ci[1], 4)}]  n=${v.n}`);
  }

  console.log("\n=== completion by chosen kind (arm B) ===");
  for (const [k, v] of Object.entries(results.outcome_by_chosen_kind["B-trickonly"])) {
    if (v.n === 0) continue;
    console.log(
      `${k.padEnd(14)}n=${String(v.n).padEnd(6)}pot ${pct(v.legal_pot_rate).padEnd(10)}` +
        `foul ${pct(v.foul_rate).padEnd(10)}cushions ${num(v.mean_cushions, 2)}`,
    );
  }

  console.log("\n=== rung distribution (arm B) ===");
  console.log(JSON.stringify(results.rung_distribution["B-trickonly"], null, 2));

  console.log("\n=== full games ===");
  console.log(JSON.stringify(results.games, null, 2));
  console.log(`\n[trick-only eval] wrote ${join(outDir, OUT_NAME)}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
