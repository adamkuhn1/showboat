// Equal-physics-budget evaluation runner: classical vs. neural hybrid.
//
//   npm run eval:hybrid -- --fixtures 120 --games 24 --seed 20260805
//
// Writes eval/results/hybrid_eval.json and prints a readable summary. Every
// number is measured; nothing here is estimated.

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import {
  APP_ROOT,
  KINDS,
  bootstrap,
  makeFixtures,
  oracle,
  runDecision,
  playGame,
  pairedDiff,
  mean,
  median,
  quantile,
  defaultConfig,
  type DecisionRecord,
  type GameRecord,
} from "./harness";
import { generateCandidates, type CandidateKind } from "../src/ai/candidates";
import { DEFAULT_PRIOR_RESERVE } from "../src/ai/shotSearch";

const arg = (name: string, fallback: number): number => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? Number(process.argv[i + 1]) : fallback;
};

const N_FIXTURES = arg("fixtures", 120);
const N_GAMES = arg("games", 24);
const SEED = arg("seed", 20260805);
const BUDGET = arg("budget", defaultConfig.simulations);
const KEEP_TOP = arg("keepTop", 16);
// Exploratory per-kind pruning floor (see applyPerKindFloor). 0 = the shipped
// behaviour that the headline run measured.
const FLOOR = arg("floor", 0);
// Direct-kind reserve slots inside keepTop (see DEFAULT_PRIOR_RESERVE).
// `--directReserve 0` reproduces the pre-fix pure-global-top-K policy, which is
// how the ablation in the report was run.
const DIRECT_RESERVE = arg("directReserve", DEFAULT_PRIOR_RESERVE.direct ?? 0);
const RESERVE: Partial<Record<CandidateKind, number>> =
  DIRECT_RESERVE > 0 ? { direct: DIRECT_RESERVE } : {};
const OUT_NAME = (() => {
  const i = process.argv.indexOf("--out");
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : "hybrid_eval.json";
})();

async function main() {
  const { table, evaluator } = await bootstrap();
  const manifest = evaluator.getManifest()!;
  console.log(
    `[eval] model ${manifest.artifact} (seed ${manifest.provenance.selected_seed}), ` +
      `budget ${BUDGET} physics units/turn, keepTop ${KEEP_TOP}, per-kind floor ${FLOOR}, ` +
      `direct reserve ${DIRECT_RESERVE}, ${N_FIXTURES} fixtures, ${N_GAMES} games\n`,
  );

  // `seedTimeoutMs: Infinity` — the production wall-clock guard is disabled for
  // BOTH arms here, on purpose. It truncates the seeding loop by elapsed time,
  // which makes results depend on machine load rather than search policy: a
  // four-way-concurrent dev sweep produced different classical-arm numbers
  // across runs that were supposed to be identical. The physics budget
  // (`simulations`) remains the binding constraint and is unchanged, so equal
  // budget is preserved exactly.
  const config = { ...defaultConfig, simulations: BUDGET, seed: 20260101, seedTimeoutMs: Infinity };
  const fixtures = makeFixtures(table, N_FIXTURES, SEED);
  console.log(`[eval] built ${fixtures.length} fixtures`);

  const decisions: DecisionRecord[] = [];
  const oracles: Record<string, ReturnType<typeof oracle>> = {};
  for (const f of fixtures) {
    const cands = generateCandidates(f.balls, table, f.targets);
    oracles[f.id] = oracle(f.balls, cands);
    for (const mode of ["classical", "hybrid"] as const) {
      decisions.push(
        await runDecision(f, table, mode, evaluator, config, oracles[f.id], KEEP_TOP, FLOOR, RESERVE),
      );
    }
  }

  const by = (m: string) => decisions.filter((d) => d.mode === m);
  const cl = by("classical");
  const hy = by("hybrid");

  // ---- budget equality is verified, not assumed ---------------------------
  const budgetViolations = decisions.filter((d) => d.physicsCalls > BUDGET);
  const budgetEqual = cl.every((d, i) => d.physicsCalls === hy[i].physicsCalls);

  const rate = (rs: DecisionRecord[], f: (d: DecisionRecord) => boolean) =>
    rs.length ? rs.filter(f).length / rs.length : NaN;

  const summarize = (rs: DecisionRecord[]) => ({
    n: rs.length,
    legal_pot_rate: rate(rs, (d) => d.legalPot),
    foul_rate: rate(rs, (d) => d.foul),
    scratch_rate: rate(rs, (d) => d.scratch),
    trick_attempt_rate: rate(rs, (d) => d.isTrickAttempt),
    trick_success_rate: (() => {
      const t = rs.filter((d) => d.isTrickAttempt);
      return t.length ? t.filter((d) => d.legalPot).length / t.length : NaN;
    })(),
    mean_regret: mean(rs.map((d) => d.regret)),
    candidate_recall: (() => {
      const hits = rs.reduce((a, d) => a + d.recallHits, 0);
      const tot = rs.reduce((a, d) => a + d.recallTotal, 0);
      return tot ? hits / tot : NaN;
    })(),
    // --- direct-fallback preservation ---------------------------------------
    // "Of the states where a makeable direct pot existed, in how many did the
    // agent actually put at least one of them in front of the physics engine?"
    // This is the quantity the search-policy repair targets: the selection rule
    // falls back to a reliable direct pot when no trick qualifies, and it can
    // only do that if a direct survived pruning.
    direct_fallback_preserved_rate: (() => {
      const withDirect = rs.filter((d) => d.pottingDirects > 0);
      return withDirect.length
        ? withDirect.filter((d) => d.verifiedPottingDirects > 0).length / withDirect.length
        : NaN;
    })(),
    n_states_with_potting_direct: rs.filter((d) => d.pottingDirects > 0).length,
    mean_verified_directs: mean(rs.map((d) => d.verifiedDirects)),
    mean_reserve_promotions: mean(rs.map((d) => d.reservePromotions)),
    no_trick_qualified_rate: rate(rs, (d) => d.selectionReason === "no-trick-qualified"),
    no_verified_pot_rate: rate(rs, (d) => d.selectionReason === "no-verified-pot"),
    mean_physics_calls: mean(rs.map((d) => d.physicsCalls)),
    mean_candidates_generated: mean(rs.map((d) => d.candidatesGenerated)),
    mean_verified: mean(rs.map((d) => d.verifiedCount)),
    decision_ms_median: median(rs.map((d) => d.decisionMs)),
    decision_ms_p95: quantile(rs.map((d) => d.decisionMs), 0.95),
    neural_ms_median: median(rs.map((d) => d.neuralMs)),
    neural_ms_p95: quantile(rs.map((d) => d.neuralMs), 0.95),
    neural_encode_ms_median: median(rs.map((d) => d.neuralEncodeMs)),
    neural_run_ms_median: median(rs.map((d) => d.neuralRunMs)),
    neural_run_ms_p95: quantile(rs.map((d) => d.neuralRunMs), 0.95),
  });

  const perKindRecall = (rs: DecisionRecord[]) => {
    const out: Record<string, { hits: number; total: number; recall: number }> = {};
    for (const k of KINDS) {
      const hits = rs.reduce((a, d) => a + d.recallByKind[k].hits, 0);
      const tot = rs.reduce((a, d) => a + d.recallByKind[k].total, 0);
      out[k] = { hits, total: tot, recall: tot ? hits / tot : NaN };
    }
    return out;
  };

  const perChosenKind = (rs: DecisionRecord[]) => {
    const out: Record<string, { n: number; legal_pot_rate: number; scratch_rate: number }> = {};
    for (const k of [...KINDS, "none"]) {
      const sub = rs.filter((d) => (d.chosenKind ?? "none") === k);
      out[k] = {
        n: sub.length,
        legal_pot_rate: sub.length ? sub.filter((d) => d.legalPot).length / sub.length : NaN,
        scratch_rate: sub.length ? sub.filter((d) => d.scratch).length / sub.length : NaN,
      };
    }
    return out;
  };

  // ---- split by "were there viable tricks at this state?" -----------------
  const withTrick = (rs: DecisionRecord[]) => rs.filter((d) => d.hadPottingTrick);
  const withoutTrick = (rs: DecisionRecord[]) => rs.filter((d) => !d.hadPottingTrick);

  const paired = (f: (d: DecisionRecord) => number) =>
    pairedDiff(hy.map(f), cl.map(f));

  // ---- full games ---------------------------------------------------------
  const games: GameRecord[] = [];
  for (let g = 0; g < N_GAMES; g++) {
    const hybridPlayer: 0 | 1 = g % 2 === 0 ? 1 : 0; // alternate sides
    games.push(await playGame(table, evaluator, SEED + g, hybridPlayer, config, 90, FLOOR, RESERVE));
    process.stdout.write(`\r[eval] games ${g + 1}/${N_GAMES}`);
  }
  process.stdout.write("\n");

  const decided = games.filter((g) => g.winner !== null);
  const hybridWins = decided.filter((g) => g.winner === g.hybridPlayer).length;
  const gameSummary = {
    n_games: games.length,
    n_decided: decided.length,
    hybrid_wins: hybridWins,
    classical_wins: decided.length - hybridWins,
    hybrid_win_rate: decided.length ? hybridWins / decided.length : NaN,
    mean_shots_per_game: mean(games.map((g) => g.shots)),
    mean_shots_to_win_hybrid: mean(
      decided.filter((g) => g.winner === g.hybridPlayer).map((g) => g.hybridShots),
    ),
    mean_shots_to_win_classical: mean(
      decided.filter((g) => g.winner !== g.hybridPlayer).map((g) => g.classicalShots),
    ),
    hybrid_pot_rate: (() => {
      const s = games.reduce((a, g) => a + g.hybridShots, 0);
      return s ? games.reduce((a, g) => a + g.hybridPots, 0) / s : NaN;
    })(),
    classical_pot_rate: (() => {
      const s = games.reduce((a, g) => a + g.classicalShots, 0);
      return s ? games.reduce((a, g) => a + g.classicalPots, 0) / s : NaN;
    })(),
    hybrid_trick_attempt_rate: (() => {
      const s = games.reduce((a, g) => a + g.hybridShots, 0);
      return s ? games.reduce((a, g) => a + g.hybridTrickAttempts, 0) / s : NaN;
    })(),
    classical_trick_attempt_rate: (() => {
      const s = games.reduce((a, g) => a + g.classicalShots, 0);
      return s ? games.reduce((a, g) => a + g.classicalTrickAttempts, 0) / s : NaN;
    })(),
    hybrid_trick_success_rate: (() => {
      const a = games.reduce((x, g) => x + g.hybridTrickAttempts, 0);
      return a ? games.reduce((x, g) => x + g.hybridTrickSuccesses, 0) / a : NaN;
    })(),
    classical_trick_success_rate: (() => {
      const a = games.reduce((x, g) => x + g.classicalTrickAttempts, 0);
      return a ? games.reduce((x, g) => x + g.classicalTrickSuccesses, 0) / a : NaN;
    })(),
  };

  const results = {
    generated_at: new Date().toISOString(),
    config: {
      fixtures: fixtures.length,
      games: N_GAMES,
      seed: SEED,
      budget: BUDGET,
      keep_top: KEEP_TOP,
      per_kind_floor: FLOOR,
      direct_reserve: DIRECT_RESERVE,
    },
    model: {
      artifact: manifest.artifact,
      sha256: manifest.onnx_sha256,
      selected_seed: manifest.provenance.selected_seed,
    },
    budget_equality: {
      max_physics_calls_seen: Math.max(...decisions.map((d) => d.physicsCalls)),
      over_budget_decisions: budgetViolations.length,
      identical_per_fixture: budgetEqual,
    },
    decisions: {
      classical: summarize(cl),
      hybrid: summarize(hy),
      classical_with_viable_trick: summarize(withTrick(cl)),
      hybrid_with_viable_trick: summarize(withTrick(hy)),
      classical_without_viable_trick: summarize(withoutTrick(cl)),
      hybrid_without_viable_trick: summarize(withoutTrick(hy)),
    },
    paired_differences_hybrid_minus_classical: {
      legal_pot: paired((d) => (d.legalPot ? 1 : 0)),
      foul: paired((d) => (d.foul ? 1 : 0)),
      scratch: paired((d) => (d.scratch ? 1 : 0)),
      regret: paired((d) => d.regret),
      trick_attempt: paired((d) => (d.isTrickAttempt ? 1 : 0)),
      candidate_recall_per_fixture: paired((d) =>
        d.recallTotal ? d.recallHits / d.recallTotal : 0,
      ),
      direct_recall_per_fixture: pairedDiff(
        hy.filter((_, i) => cl[i].pottingDirects > 0).map((d) => d.verifiedPottingDirects / d.pottingDirects),
        cl.filter((d) => d.pottingDirects > 0).map((d) => d.verifiedPottingDirects / d.pottingDirects),
      ),
      physics_calls: paired((d) => d.physicsCalls),
      decision_ms: paired((d) => d.decisionMs),
    },
    candidate_recall_by_kind: {
      classical: perKindRecall(cl),
      hybrid: perKindRecall(hy),
    },
    outcome_by_chosen_kind: {
      classical: perChosenKind(cl),
      hybrid: perChosenKind(hy),
    },
    decisions_that_differed: cl.filter((d, i) => d.chosenIndex !== hy[i].chosenIndex).length,
    games: gameSummary,
    raw_games: games,
    // Per-decision records, so the pooled cross-seed gate check
    // (`eval/gateReport.ts`) can recompute paired statistics from the same
    // pairs this run measured, rather than averaging two sets of aggregates.
    raw_decisions: decisions,
  };

  const outDir = join(APP_ROOT, "eval/results");
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, OUT_NAME), `${JSON.stringify(results, null, 2)}\n`);

  const pct = (x: number) => (Number.isFinite(x) ? `${(x * 100).toFixed(1)}%` : "n/a");
  const num = (x: number, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "n/a");
  console.log("\n=== decision-level, equal budget ===");
  console.log(`budget equality: max calls ${results.budget_equality.max_physics_calls_seen} / ${BUDGET}, ` +
    `over-budget ${results.budget_equality.over_budget_decisions}, identical per fixture: ${budgetEqual}`);
  const rows: [string, (s: ReturnType<typeof summarize>) => string][] = [
    ["legal-pot rate", (s) => pct(s.legal_pot_rate)],
    ["foul rate", (s) => pct(s.foul_rate)],
    ["scratch rate", (s) => pct(s.scratch_rate)],
    ["trick attempt rate", (s) => pct(s.trick_attempt_rate)],
    ["trick success rate", (s) => pct(s.trick_success_rate)],
    ["mean regret", (s) => num(s.mean_regret, 3)],
    ["candidate recall", (s) => pct(s.candidate_recall)],
    ["direct fallback kept", (s) => pct(s.direct_fallback_preserved_rate)],
    ["  (n such states)", (s) => String(s.n_states_with_potting_direct)],
    ["mean verified directs", (s) => num(s.mean_verified_directs, 2)],
    ["mean reserve promos", (s) => num(s.mean_reserve_promotions, 2)],
    ["no-trick-qualified", (s) => pct(s.no_trick_qualified_rate)],
    ["no-verified-pot", (s) => pct(s.no_verified_pot_rate)],
    ["physics calls/turn", (s) => num(s.mean_physics_calls, 1)],
    ["candidates generated", (s) => num(s.mean_candidates_generated, 1)],
    ["candidates verified", (s) => num(s.mean_verified, 1)],
    ["decision ms (median)", (s) => num(s.decision_ms_median, 1)],
    ["decision ms (p95)", (s) => num(s.decision_ms_p95, 1)],
    ["neural ms (median)", (s) => num(s.neural_ms_median, 3)],
    ["neural ms (p95)", (s) => num(s.neural_ms_p95, 3)],
    ["  encode ms (median)", (s) => num(s.neural_encode_ms_median, 3)],
    ["  ort run ms (median)", (s) => num(s.neural_run_ms_median, 3)],
    ["  ort run ms (p95)", (s) => num(s.neural_run_ms_p95, 3)],
  ];
  console.log(`${"metric".padEnd(24)}${"classical".padEnd(14)}hybrid`);
  for (const [label, f] of rows) {
    console.log(
      `${label.padEnd(24)}${f(results.decisions.classical).padEnd(14)}${f(results.decisions.hybrid)}`,
    );
  }
  console.log(`\ndecisions where the chosen shot differed: ${results.decisions_that_differed}/${fixtures.length}`);

  console.log("\n=== candidate recall by kind (share of oracle-potting candidates that got physics) ===");
  console.log(`${"kind".padEnd(14)}${"n".padEnd(8)}${"classical".padEnd(14)}hybrid`);
  for (const k of KINDS) {
    const c = results.candidate_recall_by_kind.classical[k];
    const h = results.candidate_recall_by_kind.hybrid[k];
    console.log(`${k.padEnd(14)}${String(c.total).padEnd(8)}${pct(c.recall).padEnd(14)}${pct(h.recall)}`);
  }

  console.log("\n=== outcome by chosen candidate kind ===");
  console.log(`${"kind".padEnd(14)}${"classical n / pot".padEnd(24)}hybrid n / pot`);
  for (const k of [...KINDS, "none"]) {
    const c = results.outcome_by_chosen_kind.classical[k];
    const h = results.outcome_by_chosen_kind.hybrid[k];
    console.log(
      `${k.padEnd(14)}${`${c.n} / ${pct(c.legal_pot_rate)}`.padEnd(24)}${h.n} / ${pct(h.legal_pot_rate)}`,
    );
  }

  console.log("\n=== paired differences (hybrid - classical), 95% CI ===");
  for (const [k, v] of Object.entries(results.paired_differences_hybrid_minus_classical)) {
    console.log(
      `${k.padEnd(30)}${num(v.diff, 4)}  [${num(v.ci[0], 4)}, ${num(v.ci[1], 4)}]  n=${v.n}`,
    );
  }

  console.log("\n=== full games (alternating sides, equal budget) ===");
  console.log(JSON.stringify(gameSummary, null, 2));
  console.log(`\n[eval] wrote ${join(outDir, OUT_NAME)}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
