import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { FEATURE_DIM, FEATURE_NAMES } from "../src/ai/features";
import { makeRng } from "../src/ai/rollout";
import { splitOf } from "./split";

// Train the ranker MLP on the rollout-labelled dataset. Plain TypeScript —
// the network is small enough (13 -> 20 -> 12 -> 1, ~540 parameters) that a
// hand-written Adam loop is clearer and more auditable than dragging in a
// framework, and the exported weights are consumed by the exact same forward
// pass the app ships (src/ai/ranker.ts:neuralScore).
//
// Split discipline: held-out POSITIONS, not held-out rows (see split.ts).
// This script only ever sees train + validation; test positions are not
// loaded here at all, so the final gate (evaluate.ts) can't leak into any
// decision made during training.
//
// Loss: binary cross-entropy against soft labels (the rollout success
// fraction). Early stopping on validation BCE; the best epoch's weights are
// what get exported.

const here = dirname(fileURLToPath(import.meta.url));
const DATA = join(here, "data", "dataset.jsonl");
const OUT = join(here, "..", "src", "ai", "weights.json");

const SEED = Number(process.env.SEED ?? 7);
const EPOCHS = Number(process.env.EPOCHS ?? 300);
const PATIENCE = Number(process.env.PATIENCE ?? 30);
const LR = Number(process.env.LR ?? 3e-3);
const BATCH = Number(process.env.BATCH ?? 128);
const HIDDEN = [20, 12];

interface Row {
  pos: number;
  feats: number[];
  label: number;
}

const allRows: Row[] = readFileSync(DATA, "utf8")
  .split("\n")
  .filter((l) => l && !l.startsWith("#"))
  .map((l) => JSON.parse(l));

// Test positions are excluded here entirely -- not read, not touched, not
// used for any decision this script makes. rows/train/val below are already
// test-free.
const rows = allRows.filter((r) => splitOf(r.pos) !== "test");
const train = rows.filter((r) => splitOf(r.pos) === "train");
const val = rows.filter((r) => splitOf(r.pos) === "val");
const testExcluded = allRows.length - rows.length;
console.log(
  `dataset: ${allRows.length} rows total (${testExcluded} held out as test, unread beyond this line) · ` +
    `train ${train.length} (${new Set(train.map((r) => r.pos)).size} positions) · ` +
    `val ${val.length} (${new Set(val.map((r) => r.pos)).size} positions)`,
);

// --- standardisation (train split only) -------------------------------------
const mean = new Array(FEATURE_DIM).fill(0);
const std = new Array(FEATURE_DIM).fill(0);
for (const r of train) for (let i = 0; i < FEATURE_DIM; i++) mean[i] += r.feats[i];
for (let i = 0; i < FEATURE_DIM; i++) mean[i] /= train.length;
for (const r of train)
  for (let i = 0; i < FEATURE_DIM; i++)
    std[i] += (r.feats[i] - mean[i]) ** 2;
for (let i = 0; i < FEATURE_DIM; i++)
  std[i] = Math.sqrt(std[i] / train.length) || 1;

const norm = (f: number[]): number[] => f.map((v, i) => (v - mean[i]) / std[i]);
const Xtr = train.map((r) => norm(r.feats));
const ytr = train.map((r) => r.label);
const Xva = val.map((r) => norm(r.feats));
const yva = val.map((r) => r.label);

// --- network ----------------------------------------------------------------
const rng = makeRng(SEED);
const sizes = [FEATURE_DIM, ...HIDDEN, 1];

interface Layer {
  W: number[][];
  b: number[];
  // Adam moments
  mW: number[][];
  vW: number[][];
  mb: number[];
  vb: number[];
}

const layers: Layer[] = [];
for (let l = 0; l < sizes.length - 1; l++) {
  const nin = sizes[l];
  const nout = sizes[l + 1];
  const scale = Math.sqrt(2 / (nin + nout)); // Xavier
  layers.push({
    W: Array.from({ length: nout }, () =>
      Array.from({ length: nin }, () => (rng() * 2 - 1) * scale),
    ),
    b: new Array(nout).fill(0),
    mW: Array.from({ length: nout }, () => new Array(nin).fill(0)),
    vW: Array.from({ length: nout }, () => new Array(nin).fill(0)),
    mb: new Array(nout).fill(0),
    vb: new Array(nout).fill(0),
  });
}

const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

// Forward with caches for backprop. Hidden: tanh; output: sigmoid.
const forward = (x: number[]): { acts: number[][]; out: number } => {
  const acts: number[][] = [x];
  let a = x;
  for (let l = 0; l < layers.length; l++) {
    const { W, b } = layers[l];
    const z = new Array<number>(W.length);
    for (let j = 0; j < W.length; j++) {
      let s = b[j];
      for (let i = 0; i < a.length; i++) s += W[j][i] * a[i];
      z[j] = l < layers.length - 1 ? Math.tanh(s) : sigmoid(s);
    }
    acts.push(z);
    a = z;
  }
  return { acts, out: a[0] };
};

const bce = (p: number, y: number): number => {
  const q = Math.min(1 - 1e-7, Math.max(1e-7, p));
  return -(y * Math.log(q) + (1 - y) * Math.log(1 - q));
};

const valBce = (): number => {
  let s = 0;
  for (let i = 0; i < Xva.length; i++) s += bce(forward(Xva[i]).out, yva[i]);
  return s / Xva.length;
};

// One Adam step for a mini-batch.
let adamT = 0;
const B1 = 0.9;
const B2 = 0.999;
const EPSA = 1e-8;

const step = (idx: number[]): number => {
  // Accumulate gradients.
  const gW = layers.map((l) => l.W.map((row) => row.map(() => 0)));
  const gb = layers.map((l) => l.b.map(() => 0));
  let loss = 0;

  for (const k of idx) {
    const { acts, out } = forward(Xtr[k]);
    loss += bce(out, ytr[k]);
    // dL/dz_out for sigmoid+BCE = (p - y).
    let delta = [out - ytr[k]];
    for (let l = layers.length - 1; l >= 0; l--) {
      const aPrev = acts[l];
      const layer = layers[l];
      for (let j = 0; j < layer.W.length; j++) {
        gb[l][j] += delta[j];
        for (let i = 0; i < aPrev.length; i++) {
          gW[l][j][i] += delta[j] * aPrev[i];
        }
      }
      if (l > 0) {
        // acts[l] are the tanh outputs of layer l-1 (aPrev aliases them), so
        // tanh'(z) = 1 - a^2 applies directly.
        const prevDelta = new Array<number>(aPrev.length).fill(0);
        for (let i = 0; i < aPrev.length; i++) {
          let s = 0;
          for (let j = 0; j < layer.W.length; j++) s += layer.W[j][i] * delta[j];
          prevDelta[i] = s * (1 - aPrev[i] * aPrev[i]);
        }
        delta = prevDelta;
      }
    }
  }

  adamT++;
  const n = idx.length;
  for (let l = 0; l < layers.length; l++) {
    const L = layers[l];
    for (let j = 0; j < L.W.length; j++) {
      for (let i = 0; i < L.W[j].length; i++) {
        const g = gW[l][j][i] / n;
        L.mW[j][i] = B1 * L.mW[j][i] + (1 - B1) * g;
        L.vW[j][i] = B2 * L.vW[j][i] + (1 - B2) * g * g;
        const mh = L.mW[j][i] / (1 - B1 ** adamT);
        const vh = L.vW[j][i] / (1 - B2 ** adamT);
        L.W[j][i] -= (LR * mh) / (Math.sqrt(vh) + EPSA);
      }
      const g = gb[l][j] / n;
      L.mb[j] = B1 * L.mb[j] + (1 - B1) * g;
      L.vb[j] = B2 * L.vb[j] + (1 - B2) * g * g;
      const mh = L.mb[j] / (1 - B1 ** adamT);
      const vh = L.vb[j] / (1 - B2 ** adamT);
      L.b[j] -= (LR * mh) / (Math.sqrt(vh) + EPSA);
    }
  }
  return loss / n;
};

// --- training loop ------------------------------------------------------------
let best = { bce: Infinity, epoch: -1, weights: "" };
const order = Xtr.map((_, i) => i);

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  // Shuffle (seeded).
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  let trLoss = 0;
  let batches = 0;
  for (let s = 0; s < order.length; s += BATCH) {
    trLoss += step(order.slice(s, s + BATCH));
    batches++;
  }
  const vb = valBce();
  if (vb < best.bce) {
    best = {
      bce: vb,
      epoch,
      weights: JSON.stringify(layers.map((l) => ({ W: l.W, b: l.b }))),
    };
  }
  if (epoch % 10 === 0 || epoch === EPOCHS - 1) {
    console.log(
      `epoch ${epoch} · train BCE ${(trLoss / batches).toFixed(4)} · ` +
        `val BCE ${vb.toFixed(4)}${best.epoch === epoch ? " *" : ""}`,
    );
  }
  if (epoch - best.epoch > PATIENCE) {
    console.log(`early stop at epoch ${epoch} (best ${best.epoch})`);
    break;
  }
}

// --- export -------------------------------------------------------------------
// datasetRows/heldOutRows describe the FULL dataset (train+val+test), so the
// numbers this file exports and the numbers evaluate.ts reports about the
// same file agree. "heldOut" here means the held-out validation split used
// for early stopping in THIS script; evaluate.ts's report and metrics.json
// separately describe the held-out TEST split, which this script never
// reads.
const model = {
  meta: {
    trainedAt: new Date().toISOString(),
    datasetRows: allRows.length,
    trainRows: train.length,
    valRows: val.length,
    featureNames: [...FEATURE_NAMES],
    normalization: { mean, std },
    arch: sizes.join("-"),
    seed: SEED,
    heldOut: { bce: best.bce, bestEpoch: best.epoch },
  },
  layers: JSON.parse(best.weights),
};
writeFileSync(OUT, JSON.stringify(model));
console.log(
  `exported best epoch ${best.epoch} (val BCE ${best.bce.toFixed(4)}) -> ${OUT}`,
);
console.log("run `npm run train:evaluate` for the honest held-out report.");
