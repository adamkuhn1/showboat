import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  classicalScore,
  neuralScore,
  validateModel,
} from "../src/ai/ranker";
import { splitOf } from "./split";

// Held-out evaluation of the SHIPPED model through the SHIPPED inference path,
// on the TEST split (see split.ts) -- positions train.ts never reads, not
// even for early stopping. This deliberately imports neuralScore/
// validateModel from src/ai/ranker.ts and reads src/ai/weights.json — the
// same code and file the app bundles — so a training/export mismatch cannot
// pass unnoticed.
//
// The gate is predetermined (decided before this script was ever run against
// real test numbers, see training/README or the project README): the neural
// ranker earns its place only if it beats the classical scorer on the same
// held-out test positions, on calibration (BCE) and both ranking metrics
// (AUC + mean per-position Spearman) -- not a cherry-picked one of the three.
// If it fails, this script says FAIL loudly; the honest response is to ship
// classical as the default, not to relabel the heuristic or weaken the gate.

const here = dirname(fileURLToPath(import.meta.url));
const DATA = join(here, "data", "dataset.jsonl");
const WEIGHTS = join(here, "..", "src", "ai", "weights.json");
const METRICS = join(here, "metrics.json");

interface Row {
  pos: number;
  kind: string;
  feats: number[];
  label: number;
}

const rows: Row[] = readFileSync(DATA, "utf8")
  .split("\n")
  .filter((l) => l && !l.startsWith("#"))
  .map((l) => JSON.parse(l));
const val = rows.filter((r) => splitOf(r.pos) === "test"); // held-out TEST split, see split.ts

const model = validateModel(JSON.parse(readFileSync(WEIGHTS, "utf8")));
if (!model) {
  console.error("weights.json failed validation — nothing to evaluate.");
  process.exit(1);
}

const neural = val.map((r) => neuralScore(model, r.feats));
const classical = val.map((r) => classicalScore(r.feats));
const labels = val.map((r) => r.label);

// --- metrics -----------------------------------------------------------------

const bce = (ps: number[]): number => {
  let s = 0;
  for (let i = 0; i < ps.length; i++) {
    const q = Math.min(1 - 1e-7, Math.max(1e-7, ps[i]));
    s += -(labels[i] * Math.log(q) + (1 - labels[i]) * Math.log(1 - q));
  }
  return s / ps.length;
};

// AUC via Mann-Whitney on label > 0.5 ("mostly succeeds under jitter").
const auc = (ps: number[]): number => {
  const pos: number[] = [];
  const neg: number[] = [];
  for (let i = 0; i < ps.length; i++) {
    (labels[i] > 0.5 ? pos : neg).push(ps[i]);
  }
  if (pos.length === 0 || neg.length === 0) return NaN;
  let wins = 0;
  for (const p of pos) {
    for (const n of neg) wins += p > n ? 1 : p === n ? 0.5 : 0;
  }
  return wins / (pos.length * neg.length);
};

const rank = (xs: number[]): number[] => {
  const idx = xs.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
  const out = new Array(xs.length).fill(0);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) out[idx[k][1]] = r;
    i = j + 1;
  }
  return out;
};

const spearman = (a: number[], b: number[]): number => {
  const ra = rank(a);
  const rb = rank(b);
  const ma = ra.reduce((s, x) => s + x, 0) / ra.length;
  const mb = rb.reduce((s, x) => s + x, 0) / rb.length;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < ra.length; i++) {
    num += (ra[i] - ma) * (rb[i] - mb);
    da += (ra[i] - ma) ** 2;
    db += (rb[i] - mb) ** 2;
  }
  const den = Math.sqrt(da * db);
  return den > 1e-12 ? num / den : NaN;
};

// Per-position ranking quality + how often the two rankers disagree on top-1.
const positions = [...new Set(val.map((r) => r.pos))];
let spN: number[] = [];
let spC: number[] = [];
let top1Disagree = 0;
let top1Compared = 0;
let neuralTopLabel = 0;
let classicalTopLabel = 0;
for (const p of positions) {
  const ids = val.map((r, i) => ({ r, i })).filter((x) => x.r.pos === p);
  if (ids.length < 5) continue;
  const l = ids.map((x) => labels[x.i]);
  if (new Set(l).size < 2) continue; // no variance, rank undefined
  const n = ids.map((x) => neural[x.i]);
  const c = ids.map((x) => classical[x.i]);
  const sn = spearman(n, l);
  const sc = spearman(c, l);
  if (!Number.isNaN(sn)) spN.push(sn);
  if (!Number.isNaN(sc)) spC.push(sc);
  const argmax = (xs: number[]): number =>
    xs.indexOf(Math.max(...xs));
  const tn = argmax(n);
  const tc = argmax(c);
  top1Compared++;
  if (tn !== tc) top1Disagree++;
  neuralTopLabel += l[tn];
  classicalTopLabel += l[tc];
}
const mean = (xs: number[]): number =>
  xs.reduce((s, x) => s + x, 0) / (xs.length || 1);

const report = {
  heldOutRows: val.length,
  heldOutPositions: positions.length,
  neural: { bce: bce(neural), auc: auc(neural), meanSpearman: mean(spN) },
  classical: { bce: bce(classical), auc: auc(classical), meanSpearman: mean(spC) },
  top1DisagreementRate: top1Disagree / (top1Compared || 1),
  meanLabelOfTop1: {
    neural: neuralTopLabel / (top1Compared || 1),
    classical: classicalTopLabel / (top1Compared || 1),
  },
  modelMeta: model.meta,
};

console.log(JSON.stringify(report, null, 2));

const pass =
  report.neural.auc > report.classical.auc &&
  report.neural.meanSpearman > report.classical.meanSpearman &&
  report.neural.bce < report.classical.bce;
console.log(
  pass
    ? "\nGATE: PASS — neural beats the classical baseline on held-out BCE, AUC and Spearman."
    : "\nGATE: FAIL — neural does NOT beat the classical baseline. Ship classical as default and say so.",
);
writeFileSync(METRICS, JSON.stringify({ ...report, gate: pass ? "PASS" : "FAIL" }, null, 2));
console.log(`metrics written to ${METRICS}`);

// Record the gate result IN the shipped artifact, not just in this training
// report — src/ai/ranker.ts reads meta.gatePassed to decide the app's
// default. Otherwise a failed-gate model would still validate its shape and
// ship as "neural" by default, silently contradicting this script's own
// verdict.
const shipped = JSON.parse(readFileSync(WEIGHTS, "utf8"));
shipped.meta.gatePassed = pass;
writeFileSync(WEIGHTS, JSON.stringify(shipped, null, 2));
console.log(`weights.json meta.gatePassed = ${pass}`);
