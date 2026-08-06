// Mechanical application of the frozen trick-only gate criteria T1-T6.
//
//   npx tsx eval/trickOnlyGateReport.ts eval/results/trickonly_61903477.json
//
// Pre-registration: docs/repair/personal-authorship-sprint/showboat/TRICK_ONLY_GATE.md
//
// The thresholds are read from this file, not from the results, and the
// arithmetic is done here rather than in the runner so the pass/fail call is
// re-derivable from the committed raw JSON by a reader who does not trust the
// runner's own summary.
//
// Note what is NOT here: any criterion involving win rate. Trick-only is a
// product identity decision; a gate able to veto it on a win-rate regression
// would create an incentive to shade that number. The win rate is reported in
// the result document, ungated, alongside the interval that shows n=60 cannot
// resolve it.

import { readFileSync } from "node:fs";
import { pairedDiff, quantile } from "./harness";

interface Rec {
  arm: string;
  chosenKind: string | null;
  isDirect: boolean;
  foul: boolean;
  scratch: boolean;
  physicsCalls: number;
  safetySims: number;
  decisionMs: number;
}

const path = process.argv[2];
if (!path) {
  console.error("usage: tsx eval/trickOnlyGateReport.ts <results.json>");
  process.exit(2);
}
const r = JSON.parse(readFileSync(path, "utf8"));
const decisions: Rec[] = r.raw_decisions;
const A = decisions.filter((d) => d.arm === "A-previous");
const B = decisions.filter((d) => d.arm === "B-trickonly");

const results: { id: string; name: string; pass: boolean; detail: string }[] = [];
const check = (id: string, name: string, pass: boolean, detail: string) =>
  results.push({ id, name, pass, detail });

// T1 — zero direct selections, fixtures AND games.
const gameDirects =
  (r.games["vs-previous"].trickonly_direct_selections as number) +
  (r.games["vs-classical-mixed"].trickonly_direct_selections as number);
const fixtureDirects = B.filter((d) => d.isDirect).length;
check(
  "T1",
  "zero direct selections",
  fixtureDirects === 0 && gameDirects === 0,
  `${fixtureDirects} in ${B.length} fixtures, ${gameDirects} across ${
    r.games["vs-previous"].n_games + r.games["vs-classical-mixed"].n_games
  } games`,
);

// T2 — a selection on every decision that had a legal target.
const noShot = B.filter((d) => d.chosenKind === null).length;
check("T2", "no shot is ever unplayable", noShot === 0, `${noShot} decisions produced no shot`);

// T3 — non-hanging.
const p99A = quantile(A.map((d) => d.decisionMs), 0.99);
const p99B = quantile(B.map((d) => d.decisionMs), 0.99);
const maxMs = Math.max(...B.map((d) => d.decisionMs));
const maxSafety = Math.max(...B.map((d) => d.safetySims));
check(
  "T3",
  "non-hanging",
  p99B <= 1.5 * p99A && maxMs <= 5000 && maxSafety <= 6,
  `p99 B ${p99B.toFixed(0)} ms vs 1.5x A ${(1.5 * p99A).toFixed(0)} ms; max ${maxMs.toFixed(0)} ms; max safety sims ${maxSafety}`,
);

// T4 — budget parity.
const overBudget = decisions.filter((d) => d.physicsCalls > r.config.budget).length;
const callDiff = pairedDiff(B.map((d) => d.physicsCalls), A.map((d) => d.physicsCalls));
check(
  "T4",
  "budget parity",
  overBudget === 0 && Math.abs(callDiff.diff) <= 1.0,
  `${overBudget} over budget; mean physics-call diff (B-A) ${callDiff.diff.toFixed(3)}`,
);

// T5 — scratch ceiling.
const scratchDiff = pairedDiff(B.map((d) => (d.scratch ? 1 : 0)), A.map((d) => (d.scratch ? 1 : 0)));
check(
  "T5",
  "scratch ceiling",
  scratchDiff.ci[1] < 0.02,
  `diff ${scratchDiff.diff.toFixed(4)} CI [${scratchDiff.ci[0].toFixed(4)}, ${scratchDiff.ci[1].toFixed(4)}] < +0.02`,
);

// T6 — foul ceiling, deliberately loose (see the pre-registration).
const foulDiff = pairedDiff(B.map((d) => (d.foul ? 1 : 0)), A.map((d) => (d.foul ? 1 : 0)));
check(
  "T6",
  "foul ceiling (deliberately loose)",
  foulDiff.ci[1] < 0.1,
  `diff ${foulDiff.diff.toFixed(4)} CI [${foulDiff.ci[0].toFixed(4)}, ${foulDiff.ci[1].toFixed(4)}] < +0.10`,
);

console.log(`\ntrick-only gate — ${path}`);
console.log(`seed ${r.config.seed}, ${r.config.fixtures} fixtures, ${r.config.games} games/opponent`);
console.log(`model ${r.model.artifact} sha256 ${r.model.sha256.slice(0, 16)}… retrained: ${r.model.retrained}\n`);
for (const t of results) {
  console.log(`${t.pass ? "PASS" : "FAIL"}  ${t.id}  ${t.name.padEnd(34)}${t.detail}`);
}
const failed = results.filter((t) => !t.pass);
console.log(`\n${results.length - failed.length}/${results.length} criteria pass.`);

console.log("\n--- reported, NOT gated ---");
const pct = (x: number) => `${(x * 100).toFixed(2)}%`;
for (const key of ["vs-previous", "vs-classical-mixed"] as const) {
  const g = r.games[key];
  console.log(
    `win rate ${key.padEnd(20)} ${pct(g.trickonly_win_rate)} ` +
      `(${g.trickonly_wins}-${g.opponent_wins} of ${g.n_decided} decided), ` +
      `95% CI +/-${g.ci_half_width_pp.toFixed(1)}pp`,
  );
}
const legal = pairedDiff(
  B.map((d) => ((d as unknown as { legalPot: boolean }).legalPot ? 1 : 0)),
  A.map((d) => ((d as unknown as { legalPot: boolean }).legalPot ? 1 : 0)),
);
console.log(
  `legal-pot rate  A ${pct(r.decisions["A-previous"].legal_pot_rate)}  ->  B ${pct(
    r.decisions["B-trickonly"].legal_pot_rate,
  )}   diff ${(legal.diff * 100).toFixed(2)}pp CI [${(legal.ci[0] * 100).toFixed(2)}, ${(legal.ci[1] * 100).toFixed(2)}]`,
);
console.log(
  `foul rate       A ${pct(r.decisions["A-previous"].foul_rate)}  ->  B ${pct(r.decisions["B-trickonly"].foul_rate)}`,
);
console.log(
  `cushions/shot   A ${r.decisions["A-previous"].mean_cushions.toFixed(2)}  ->  B ${r.decisions[
    "B-trickonly"
  ].mean_cushions.toFixed(2)}`,
);

process.exit(failed.length === 0 ? 0 : 1);
