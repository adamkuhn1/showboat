import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { classicalScore, neuralScore, validateModel } from "../src/ai/ranker";
import { DIRECT_ORDER_DISCOUNT, VERIFY_TOP_K } from "../src/ai/agent";
import { makeRng } from "../src/ai/rollout";
import { splitOf } from "./split";

// Confidence intervals for the held-out gate metrics, by a position-level
// bootstrap: resample the 26 held-out TEST positions with replacement (all of
// a position's candidate rows travel together, matching how the split was
// made), recompute every metric on each resample, take the 2.5th/97.5th
// percentiles. Neural and classical are computed on the SAME resample, so the
// neural − classical difference interval is paired.
//
// Read-only with respect to the model: per-row predictions are regenerated
// here from the shipped src/ai/weights.json through the shipped forward pass
// (exactly what evaluate.ts does), and nothing is written back to the weights.
// The point estimates are checked against training/metrics.json first; if
// they don't reproduce, the script stops rather than bootstrap the wrong thing.
//
// Also reports top-10 recall (see below) and folds in every
// training/selfplay-<games>.json (written by selfplay_eval.ts) so
// training/ci.json holds every number.
//
// Env: B (resamples, default 5000), SEED (default 20261001).

const B = Number(process.env.B ?? 5000);
const SEED = Number(process.env.SEED ?? 20261001);

const here = dirname(fileURLToPath(import.meta.url));
const DATA = join(here, "data", "dataset.jsonl");
const WEIGHTS = join(here, "..", "src", "ai", "weights.json");
const METRICS = join(here, "metrics.json");
const OUT = join(here, "ci.json");

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
const test = rows.filter((r) => splitOf(r.pos) === "test");

const model = validateModel(JSON.parse(readFileSync(WEIGHTS, "utf8")));
if (!model) {
  console.error("weights.json failed validation — nothing to evaluate.");
  process.exit(1);
}

// Per-position groups of held-out rows with both rankers' predictions.
interface PosGroup {
  pos: number;
  labels: number[];
  neural: number[];
  classical: number[];
  direct: boolean[];
}
const positions = [...new Set(test.map((r) => r.pos))].sort((a, b) => a - b);
const groups: PosGroup[] = positions.map((pos) => {
  const rs = test.filter((r) => r.pos === pos);
  return {
    pos,
    labels: rs.map((r) => r.label),
    neural: rs.map((r) => neuralScore(model, r.feats)),
    classical: rs.map((r) => classicalScore(r.feats)),
    direct: rs.map((r) => r.feats[0] === 1), // kind_direct
  };
});

// --- metrics (same definitions as evaluate.ts) ---------------------------------

const bce = (ps: number[], ys: number[]): number => {
  let s = 0;
  for (let i = 0; i < ps.length; i++) {
    const q = Math.min(1 - 1e-7, Math.max(1e-7, ps[i]));
    s += -(ys[i] * Math.log(q) + (1 - ys[i]) * Math.log(1 - q));
  }
  return s / ps.length;
};

// Average ranks (ties share the mean rank), 1-based.
const rank = (xs: number[]): number[] => {
  const idx = xs.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
  const out = new Array<number>(xs.length).fill(0);
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

// AUC with positives = label > 0.5 (evaluate.ts's definition). Computed from
// rank sums (Mann-Whitney U), which equals evaluate.ts's pairwise count with
// ties scored 0.5 — just O(n log n), so thousands of resamples stay fast.
const auc = (ps: number[], ys: number[]): number => {
  const r = rank(ps);
  let nPos = 0;
  let rankSum = 0;
  for (let i = 0; i < ps.length; i++) {
    if (ys[i] > 0.5) {
      nPos++;
      rankSum += r[i];
    }
  }
  const nNeg = ps.length - nPos;
  if (nPos === 0 || nNeg === 0) return NaN;
  return (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
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

// Per-position Spearman, with evaluate.ts's eligibility rule: at least 5
// rows and some label variance. Precomputed once per position — a resample
// only changes which positions (and how many copies) enter the mean.
const spearmanOf = (g: PosGroup, scores: number[]): number | null => {
  if (g.labels.length < 5 || new Set(g.labels).size < 2) return null;
  const s = spearman(scores, g.labels);
  return Number.isNaN(s) ? null : s;
};
const spN = groups.map((g) => spearmanOf(g, g.neural));
const spC = groups.map((g) => spearmanOf(g, g.classical));

// Top-10 recall: does at least one "truly potting" candidate land in the
// first VERIFY_TOP_K the agent would verify? The order is the agent's real
// verification order (ranker prior, direct shots × DIRECT_ORDER_DISCOUNT).
// "Truly potting" = label > 0.5 (pots in most of its 6 jittered rollouts,
// the same positive class as AUC); label > 0 (pots at least once) is
// reported too. Denominator: positions that have at least one such
// candidate. Also reported: the chance a RANDOM top-10 would contain one
// (hypergeometric), because each position has at most 40 sampled candidates,
// so top 10 is already a quarter of them.
const hitAtK = (g: PosGroup, scores: number[], thr: number): boolean | null => {
  if (!g.labels.some((l) => l > thr)) return null;
  const order = scores
    .map((s, i) => ({ s: g.direct[i] ? s * DIRECT_ORDER_DISCOUNT : s, i }))
    .sort((a, b) => b.s - a.s)
    .slice(0, VERIFY_TOP_K);
  return order.some((o) => g.labels[o.i] > thr);
};
const randomHitAtK = (g: PosGroup, thr: number): number | null => {
  const n = g.labels.length;
  const k = g.labels.filter((l) => l > thr).length;
  if (k === 0) return null;
  const draws = Math.min(VERIFY_TOP_K, n);
  // P(no positive in `draws` draws without replacement)
  let pNone = 1;
  for (let i = 0; i < draws; i++) pNone *= (n - k - i) / (n - i);
  return 1 - Math.max(0, pNone);
};
const hits = {
  neural: groups.map((g) => hitAtK(g, g.neural, 0.5)),
  classical: groups.map((g) => hitAtK(g, g.classical, 0.5)),
  random: groups.map((g) => randomHitAtK(g, 0.5)),
  neuralAny: groups.map((g) => hitAtK(g, g.neural, 0)),
  classicalAny: groups.map((g) => hitAtK(g, g.classical, 0)),
  randomAny: groups.map((g) => randomHitAtK(g, 0)),
};

const meanOf = (xs: (number | boolean | null)[]): number => {
  const v = xs.filter((x): x is number | boolean => x !== null).map(Number);
  return v.reduce((s, x) => s + x, 0) / (v.length || 1);
};

// All metrics for a multiset of position indices.
const metricsFor = (idx: number[]) => {
  const ys: number[] = [];
  const pn: number[] = [];
  const pc: number[] = [];
  for (const i of idx) {
    ys.push(...groups[i].labels);
    pn.push(...groups[i].neural);
    pc.push(...groups[i].classical);
  }
  const pick = <T>(xs: T[]): T[] => idx.map((i) => xs[i]);
  const m = {
    neural: {
      auc: auc(pn, ys),
      bce: bce(pn, ys),
      meanSpearman: meanOf(pick(spN)),
      top10Recall: meanOf(pick(hits.neural)),
      top10RecallAnyPot: meanOf(pick(hits.neuralAny)),
    },
    classical: {
      auc: auc(pc, ys),
      bce: bce(pc, ys),
      meanSpearman: meanOf(pick(spC)),
      top10Recall: meanOf(pick(hits.classical)),
      top10RecallAnyPot: meanOf(pick(hits.classicalAny)),
    },
    random: {
      top10Recall: meanOf(pick(hits.random)),
      top10RecallAnyPot: meanOf(pick(hits.randomAny)),
    },
  };
  return {
    ...m,
    diff: {
      auc: m.neural.auc - m.classical.auc,
      bce: m.neural.bce - m.classical.bce,
      meanSpearman: m.neural.meanSpearman - m.classical.meanSpearman,
      top10Recall: m.neural.top10Recall - m.classical.top10Recall,
    },
  };
};

// --- point estimates must reproduce evaluate.ts --------------------------------

const all = groups.map((_, i) => i);
const point = metricsFor(all);
const published = JSON.parse(readFileSync(METRICS, "utf8"));
const checks: [string, number, number][] = [
  ["neural.auc", point.neural.auc, published.neural.auc],
  ["neural.bce", point.neural.bce, published.neural.bce],
  ["neural.meanSpearman", point.neural.meanSpearman, published.neural.meanSpearman],
  ["classical.auc", point.classical.auc, published.classical.auc],
  ["classical.bce", point.classical.bce, published.classical.bce],
  ["classical.meanSpearman", point.classical.meanSpearman, published.classical.meanSpearman],
];
for (const [name, mine, theirs] of checks) {
  if (Math.abs(mine - theirs) > 1e-9) {
    console.error(`point estimate ${name} = ${mine} does not reproduce metrics.json (${theirs})`);
    process.exit(1);
  }
}
console.log("point estimates reproduce training/metrics.json exactly.");

// --- bootstrap -------------------------------------------------------------------

const rng = makeRng(SEED);
type Flat = Record<string, number>;
const flatten = (m: ReturnType<typeof metricsFor>): Flat => {
  const out: Flat = {};
  for (const [side, vals] of Object.entries(m)) {
    for (const [k, v] of Object.entries(vals)) out[`${side}.${k}`] = v;
  }
  return out;
};
const samples: Record<string, number[]> = {};
const t0 = Date.now();
for (let b = 0; b < B; b++) {
  const idx = all.map(() => Math.floor(rng() * all.length));
  for (const [k, v] of Object.entries(flatten(metricsFor(idx)))) {
    (samples[k] ??= []).push(v);
  }
}
const pct = (xs: number[], q: number): number => {
  const s = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  const h = (s.length - 1) * q;
  const lo = Math.floor(h);
  return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (h - lo);
};

const pointFlat = flatten(point);
const ci: Record<string, { point: number; lo: number; hi: number }> = {};
for (const k of Object.keys(pointFlat)) {
  ci[k] = { point: pointFlat[k], lo: pct(samples[k], 0.025), hi: pct(samples[k], 0.975) };
}
const share = (k: string, pred: (x: number) => boolean): number =>
  samples[k].filter(pred).length / samples[k].length;

const r4 = (x: number): number => Math.round(x * 1e4) / 1e4;
const interval = (k: string) => ({
  point: r4(ci[k].point),
  ci95: [r4(ci[k].lo), r4(ci[k].hi)],
});

const eligibleSpearman = spN.filter((x) => x !== null).length;
const top10Positions = hits.neural.filter((x) => x !== null).length;
const top10PositionsAny = hits.neuralAny.filter((x) => x !== null).length;

const selfPlay = readdirSync(here)
  .filter((f) => /^selfplay-\d+\.json$/.test(f))
  .map((f) => JSON.parse(readFileSync(join(here, f), "utf8")))
  .sort((a, b) => a.games - b.games);

const out = {
  generatedAt: new Date().toISOString(),
  method:
    `Percentile bootstrap over held-out TEST positions (resampled with replacement, ` +
    `all candidate rows of a position kept together), B=${B}, seed ${SEED}. ` +
    `Neural and classical scored on the same resample (paired differences).`,
  heldOut: {
    positions: groups.length,
    rows: test.length,
    spearmanEligiblePositions: eligibleSpearman,
    positiveRows: test.filter((r) => r.label > 0.5).length,
  },
  classicalBaseline:
    "Hand-written multiplicative pot-probability heuristic over the same 13 features " +
    "(cut angle, path lengths, pocket approach, corridor clearance, per-rail and combo " +
    "penalties, power) — src/ai/ranker.ts classicalScore; no learned parameters.",
  neural: {
    auc: interval("neural.auc"),
    bce: interval("neural.bce"),
    meanSpearman: interval("neural.meanSpearman"),
  },
  classical: {
    auc: interval("classical.auc"),
    bce: interval("classical.bce"),
    meanSpearman: interval("classical.meanSpearman"),
  },
  neuralMinusClassical: {
    auc: { ...interval("diff.auc"), shareOfResamplesNeuralBetter: r4(share("diff.auc", (x) => x > 0)) },
    bce: { ...interval("diff.bce"), shareOfResamplesNeuralBetter: r4(share("diff.bce", (x) => x < 0)) },
    meanSpearman: {
      ...interval("diff.meanSpearman"),
      shareOfResamplesNeuralBetter: r4(share("diff.meanSpearman", (x) => x > 0)),
    },
  },
  top10Recall: {
    definition:
      `Share of held-out positions where at least one candidate with label > 0.5 ` +
      `(pots in most of its 6 jittered rollouts) is in the first ${VERIFY_TOP_K} of the agent's ` +
      `verification order (prior, direct × ${DIRECT_ORDER_DISCOUNT}). Ranked among the ` +
      `<= 40 candidates sampled per position in the dataset, not every generated candidate.`,
    positions: top10Positions,
    neural: interval("neural.top10Recall"),
    classical: interval("classical.top10Recall"),
    randomTop10: interval("random.top10Recall"),
    neuralMinusClassical: interval("diff.top10Recall"),
    anyPot: {
      definition: "Same, with 'truly potting' = label > 0 (pots in at least one rollout).",
      positions: top10PositionsAny,
      neural: interval("neural.top10RecallAnyPot"),
      classical: interval("classical.top10RecallAnyPot"),
      randomTop10: interval("random.top10RecallAnyPot"),
    },
  },
  selfPlay,
};

writeFileSync(OUT, JSON.stringify(out, null, 2) + "\n");
console.log(JSON.stringify(out, null, 2));
console.log(`\n${B} resamples in ${((Date.now() - t0) / 1000).toFixed(1)}s · written to ${OUT}`);
if (selfPlay.length === 0) {
  console.log("(no training/selfplay-*.json yet — run `npm run train:selfplay` first to include it)");
}
