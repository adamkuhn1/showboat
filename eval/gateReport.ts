// Pooled cross-seed evaluation of the FROZEN release-candidate decision gate.
//
//   npx tsx eval/gateReport.ts <result.json> [<result.json> ...]
//
// Reads the per-decision records `hybridEval.ts` writes, pools the paired
// fixtures across seeds, and mechanically evaluates every criterion in
// docs/repair/release-candidate/showboat/DECISION_GATE.md §6.
//
// The verdict is computed here rather than read off a table by hand so the
// thresholds are applied exactly as frozen, with no room for a favourable
// reading. Thresholds are duplicated as literals below and cross-referenced to
// the gate document by letter; if they ever diverge, that is a bug in this file,
// not a licence to reinterpret the gate.

import { readFileSync } from "node:fs";
import { pairedDiff, mcnemarExact, median, quantile, mean } from "./harness";
import type { DecisionRecord } from "./harness";

interface RunFile {
  config: { seed: number; fixtures: number; budget: number; keep_top: number; direct_reserve: number };
  model: { artifact: string; sha256: string };
  budget_equality: { over_budget_decisions: number; max_physics_calls_seen: number };
  raw_decisions: DecisionRecord[];
}

const files = process.argv.slice(2);
if (files.length === 0) {
  console.error("usage: tsx eval/gateReport.ts <result.json> [...]");
  process.exit(1);
}

const runs: RunFile[] = files.map((f) => JSON.parse(readFileSync(f, "utf8")));

// --- pool paired decisions -------------------------------------------------
const cl: DecisionRecord[] = [];
const hy: DecisionRecord[] = [];
let overBudget = 0;
let maxCalls = 0;
for (const r of runs) {
  const c = r.raw_decisions.filter((d) => d.mode === "classical");
  const h = r.raw_decisions.filter((d) => d.mode === "hybrid");
  if (c.length !== h.length) throw new Error(`${r.config.seed}: unpaired decision records`);
  for (let i = 0; i < c.length; i++) {
    if (c[i].fixtureId !== h[i].fixtureId) throw new Error("fixture pairing mismatch");
  }
  cl.push(...c);
  hy.push(...h);
  overBudget += r.budget_equality.over_budget_decisions;
  maxCalls = Math.max(maxCalls, r.budget_equality.max_physics_calls_seen);
}

const n = cl.length;

// --- determinism cross-check ------------------------------------------------
// The classical arm cannot depend on the hybrid's reserve setting, so two runs
// over the same seed must produce byte-identical classical decisions. This
// catches the wall-clock-truncation nondeterminism that invalidated an earlier
// sweep, instead of trusting that it stays fixed.
const bySeed = new Map<number, RunFile[]>();
for (const r of runs) {
  const list = bySeed.get(r.config.seed) ?? [];
  list.push(r);
  bySeed.set(r.config.seed, list);
}
for (const [seed, list] of bySeed) {
  if (list.length < 2) continue;
  const ref = list[0].raw_decisions.filter((d) => d.mode === "classical");
  for (const other of list.slice(1)) {
    const cmp = other.raw_decisions.filter((d) => d.mode === "classical");
    const same = ref.every(
      (d, i) =>
        d.fixtureId === cmp[i].fixtureId &&
        d.chosenIndex === cmp[i].chosenIndex &&
        d.legalPot === cmp[i].legalPot &&
        d.physicsCalls === cmp[i].physicsCalls,
    );
    console.log(
      `[determinism] seed ${seed}: classical arms identical across runs — ${same ? "YES" : "NO (results are not reproducible)"}`,
    );
  }
}
const bin = (rs: DecisionRecord[], f: (d: DecisionRecord) => boolean) => rs.map((d) => (f(d) ? 1 : 0));
const rateOf = (xs: number[]) => mean(xs);

const legalC = bin(cl, (d) => d.legalPot);
const legalH = bin(hy, (d) => d.legalPot);
const foulC = bin(cl, (d) => d.foul);
const foulH = bin(hy, (d) => d.foul);
const scrC = bin(cl, (d) => d.scratch);
const scrH = bin(hy, (d) => d.scratch);
const trickC = bin(cl, (d) => d.isTrickAttempt);
const trickH = bin(hy, (d) => d.isTrickAttempt);
const trickDoneC = bin(cl, (d) => d.isTrickAttempt && d.legalPot);
const trickDoneH = bin(hy, (d) => d.isTrickAttempt && d.legalPot);
const regretC = cl.map((d) => d.regret);
const regretH = hy.map((d) => d.regret);
const callsC = cl.map((d) => d.physicsCalls);
const callsH = hy.map((d) => d.physicsCalls);

const dLegal = pairedDiff(legalH, legalC);
const dFoul = pairedDiff(foulH, foulC);
const dScratch = pairedDiff(scrH, scrC);
const dRegret = pairedDiff(regretH, regretC);
const dTrick = pairedDiff(trickH, trickC);
const dTrickDone = pairedDiff(trickDoneH, trickDoneC);
const dCalls = pairedDiff(callsH, callsC);

// Direct-fallback preservation, restricted to states where a makeable direct existed.
const withDirect = cl.map((d, i) => i).filter((i) => cl[i].pottingDirects > 0);
const dfC = withDirect.filter((i) => cl[i].verifiedPottingDirects > 0).length / (withDirect.length || 1);
const dfH = withDirect.filter((i) => hy[i].verifiedPottingDirects > 0).length / (withDirect.length || 1);

const decMsC = cl.map((d) => d.decisionMs);
const decMsH = hy.map((d) => d.decisionMs);
const neuralMs = hy.map((d) => d.neuralMs);

const p95C = quantile(decMsC, 0.95);
const p95H = quantile(decMsH, 0.95);
const neuralMed = median(neuralMs);
const neuralP95 = quantile(neuralMs, 0.95);

// --- criteria (letters match DECISION_GATE.md §6) --------------------------
interface Crit { id: string; what: string; detail: string; pass: boolean }
const crits: Crit[] = [
  {
    id: "A",
    what: "legal-pot non-inferiority (CI lower bound > -0.03)",
    detail: `diff ${dLegal.diff.toFixed(4)} [${dLegal.ci[0].toFixed(4)}, ${dLegal.ci[1].toFixed(4)}]`,
    pass: dLegal.ci[0] > -0.03,
  },
  {
    id: "B",
    what: "regret non-inferiority (CI upper bound < +0.03)",
    detail: `diff ${dRegret.diff.toFixed(4)} [${dRegret.ci[0].toFixed(4)}, ${dRegret.ci[1].toFixed(4)}]`,
    pass: dRegret.ci[1] < 0.03,
  },
  {
    id: "C",
    what: "foul CI upper < +0.02 and scratch CI upper < +0.01",
    detail: `foul ${dFoul.diff.toFixed(4)} [${dFoul.ci[0].toFixed(4)}, ${dFoul.ci[1].toFixed(4)}] · ` +
      `scratch ${dScratch.diff.toFixed(4)} [${dScratch.ci[0].toFixed(4)}, ${dScratch.ci[1].toFixed(4)}]`,
    pass: dFoul.ci[1] < 0.02 && dScratch.ci[1] < 0.01,
  },
  {
    id: "D",
    what: "trick benefit >= +0.05 with CI excluding zero (attempts OR completions)",
    detail:
      `attempts ${dTrick.diff.toFixed(4)} [${dTrick.ci[0].toFixed(4)}, ${dTrick.ci[1].toFixed(4)}] · ` +
      `completions ${dTrickDone.diff.toFixed(4)} [${dTrickDone.ci[0].toFixed(4)}, ${dTrickDone.ci[1].toFixed(4)}]`,
    pass:
      (dTrick.diff >= 0.05 && dTrick.ci[0] > 0) ||
      (dTrickDone.diff >= 0.05 && dTrickDone.ci[0] > 0),
  },
  {
    id: "E",
    what: "equal budget: 0 over-budget, |mean call diff| <= 1.0",
    detail: `over-budget ${overBudget}, max calls ${maxCalls}, mean diff ${dCalls.diff.toFixed(4)}`,
    pass: overBudget === 0 && Math.abs(dCalls.diff) <= 1.0,
  },
  {
    id: "F",
    what: "latency: hybrid p95 <= 1.25x classical; neural median <= 10ms, p95 <= 50ms",
    detail:
      `decision p95 ${p95H.toFixed(0)}ms vs ${p95C.toFixed(0)}ms (ratio ${(p95H / p95C).toFixed(2)}) · ` +
      `neural median ${neuralMed.toFixed(2)}ms p95 ${neuralP95.toFixed(2)}ms`,
    pass: p95H <= 1.25 * p95C && neuralMed <= 10 && neuralP95 <= 50,
  },
  {
    id: "G",
    what: "direct-fallback preservation >= classical - 0.05",
    detail: `hybrid ${(dfH * 100).toFixed(1)}% vs classical ${(dfC * 100).toFixed(1)}% (n=${withDirect.length})`,
    pass: dfH >= dfC - 0.05,
  },
];

// --- output ----------------------------------------------------------------
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
console.log(`\n=== pooled gate report ===`);
console.log(`runs: ${runs.map((r) => `seed ${r.config.seed} (${r.raw_decisions.length / 2} fixtures)`).join(", ")}`);
console.log(`model: ${runs[0].model.artifact} sha256 ${runs[0].model.sha256.slice(0, 16)}…`);
console.log(`direct reserve: ${runs.map((r) => r.config.direct_reserve).join("/")} · keepTop ${runs[0].config.keep_top} · budget ${runs[0].config.budget}`);
console.log(`pooled paired fixtures: n=${n}\n`);

console.log(`${"metric".padEnd(28)}${"classical".padEnd(12)}hybrid`);
const row = (label: string, c: string, h: string) => console.log(`${label.padEnd(28)}${c.padEnd(12)}${h}`);
row("legal-pot rate", pct(rateOf(legalC)), pct(rateOf(legalH)));
row("mean regret", rateOf(regretC).toFixed(3), rateOf(regretH).toFixed(3));
row("foul rate", pct(rateOf(foulC)), pct(rateOf(foulH)));
row("scratch rate", pct(rateOf(scrC)), pct(rateOf(scrH)));
row("trick attempt rate", pct(rateOf(trickC)), pct(rateOf(trickH)));
row("trick completion rate", pct(rateOf(trickDoneC)), pct(rateOf(trickDoneH)));
row("direct fallback kept", pct(dfC), pct(dfH));
row("physics calls/turn", mean(callsC).toFixed(2), mean(callsH).toFixed(2));
row("decision ms median", median(decMsC).toFixed(0), median(decMsH).toFixed(0));
row("decision ms p95", p95C.toFixed(0), p95H.toFixed(0));

console.log(`\n=== exact McNemar on discordant pairs ===`);
for (const [name, a, b] of [
  ["legal_pot", legalH, legalC],
  ["foul", foulH, foulC],
  ["scratch", scrH, scrC],
  ["trick_attempt", trickH, trickC],
  ["trick_completion", trickDoneH, trickDoneC],
] as [string, number[], number[]][]) {
  const m = mcnemarExact(a, b);
  console.log(
    `${name.padEnd(20)}hybrid-only ${String(m.aOnly).padStart(3)} · classical-only ${String(m.bOnly).padStart(3)} · ` +
      `discordant ${String(m.nDiscordant).padStart(3)} · p=${m.p.toFixed(4)}`,
  );
}

console.log(`\n=== FROZEN GATE (DECISION_GATE.md section 6) ===`);
let allPass = true;
for (const c of crits) {
  if (!c.pass) allPass = false;
  console.log(`${c.pass ? "PASS" : "FAIL"}  ${c.id}. ${c.what}\n        ${c.detail}`);
}
console.log(
  `\nCriteria A-G (measured here): ${allPass ? "ALL PASS" : "FAILED"}.\n` +
    `Criteria H (browser model load) and I (build health) are verified outside this script; ` +
    `see the report.\n` +
    `Verdict on the measurable criteria: neural default ${allPass ? "PERMITTED" : "REFUSED"}.\n`,
);
