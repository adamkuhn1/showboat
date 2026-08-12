import { FEATURE_DIM } from "./features";
import weightsJson from "./weights.json";

// Candidate ranking: a small trained MLP with a classical geometric scorer as
// the honest fallback. Both expose the same interface — a score in [0, 1]
// intended to approximate P(shot succeeds under execution noise), the exact
// quantity the training labels measure (see rollout.ts).
//
// The neural ranker is REAL trained ML: weights.json is produced by
// training/train.ts from rollout-labelled data generated against this same
// physics engine (training/generate.ts), and evaluated on a held-out split
// (training/evaluate.ts) through this very forward pass. If the bundled
// weights fail shape validation, OR failed evaluate.ts's held-out gate
// (meta.gatePassed), the app defaults to the classical scorer and says so in
// the UI — it never silently ships a model that didn't earn its place.

export interface RankerModel {
  meta: {
    trainedAt: string;
    datasetRows: number;
    featureNames: string[];
    normalization: { mean: number[]; std: number[] };
    heldOut: Record<string, number>;
    gatePassed?: boolean;
  };
  // Dense layers, applied in order. Activation: tanh for hidden layers,
  // sigmoid on the final single-unit layer.
  layers: { W: number[][]; b: number[] }[];
}

export type RankerName = "neural" | "classical";

export interface Ranker {
  name: RankerName;
  score: (features: number[]) => number;
}

// --- neural inference --------------------------------------------------------

const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

export const neuralScore = (m: RankerModel, features: number[]): number => {
  const { mean, std } = m.meta.normalization;
  let a = features.map((v, i) => (v - mean[i]) / (std[i] || 1));
  for (let li = 0; li < m.layers.length; li++) {
    const { W, b } = m.layers[li];
    const out = new Array<number>(W.length);
    for (let j = 0; j < W.length; j++) {
      let s = b[j];
      const row = W[j];
      for (let i = 0; i < row.length; i++) s += row[i] * a[i];
      out[j] = li < m.layers.length - 1 ? Math.tanh(s) : sigmoid(s);
    }
    a = out;
  }
  return a[0];
};

// Strict shape validation of whatever was bundled. Returns null (-> classical
// fallback) rather than trusting a malformed file.
export const validateModel = (json: unknown): RankerModel | null => {
  const m = json as RankerModel;
  try {
    if (!m || !Array.isArray(m.layers) || m.layers.length < 2) return null;
    if (
      !m.meta ||
      !Array.isArray(m.meta.normalization?.mean) ||
      m.meta.normalization.mean.length !== FEATURE_DIM ||
      m.meta.normalization.std.length !== FEATURE_DIM
    ) {
      return null;
    }
    let width: number = FEATURE_DIM;
    for (const layer of m.layers) {
      if (!Array.isArray(layer.W) || !Array.isArray(layer.b)) return null;
      if (layer.W.length !== layer.b.length) return null;
      for (const row of layer.W) {
        if (!Array.isArray(row) || row.length !== width) return null;
        if (!row.every((x) => typeof x === "number" && isFinite(x))) return null;
      }
      width = layer.W.length;
    }
    if (width !== 1) return null;
    return m;
  } catch {
    return null;
  }
};

// --- classical fallback ------------------------------------------------------

// An interpretable estimate of pot probability from the same features. Every
// factor is an explicit, explainable penalty: cut angle, travel distance,
// blocked corridors, energy lost per rail, combo difficulty. This is also the
// baseline the neural model must beat on held-out data to earn its place
// (see training/evaluate.ts).
export const classicalScore = (f: number[]): number => {
  const [
    ,
    kindBank,
    kindKick,
    kindCombo,
    railsPlanned,
    cueLen,
    objLen,
    cutCos,
    approachCos,
    cueClear,
    objClear,
    targetIsEight,
    power,
  ] = f;
  let p = 1.0;
  p *= Math.pow(Math.max(0.02, cutCos), 1.6); // thin cuts are hard
  p *= 1 - 0.45 * cueLen; // long cue travel adds angular error
  p *= 1 - 0.35 * objLen; // long object travel amplifies it
  p *= 0.35 + 0.65 * approachCos; // pocket mouth alignment
  p *= 0.25 + 0.75 * cueClear; // blocked cue corridor
  p *= 0.25 + 0.75 * objClear; // blocked object corridor
  p *= Math.pow(0.55, railsPlanned * 2); // each planned rail costs accuracy
  if (kindBank || kindKick) p *= 0.85;
  if (kindCombo) p *= 0.4; // two contacts, compounding error
  if (targetIsEight) p *= 0.95;
  p *= 1 - 0.15 * Math.max(0, power - 0.7) * 2; // hard strikes are wilder
  return Math.max(0.001, Math.min(0.999, p));
};

// --- construction ------------------------------------------------------------

export const makeRanker = (force?: RankerName): Ranker => {
  const model = validateModel(weightsJson);
  // Default: only ship neural if it validated AND beat the classical
  // baseline on held-out data (evaluate.ts's gate). ?ranker=neural forces it
  // on anyway for demo/comparison purposes even after a failed gate;
  // ?ranker=classical always forces the fallback.
  const wantNeural = force === "neural" || (force !== "classical" && model?.meta.gatePassed === true);
  if (wantNeural && model) {
    return { name: "neural", score: (f) => neuralScore(model, f) };
  }
  return { name: "classical", score: classicalScore };
};

export const loadedModelMeta = (): RankerModel["meta"] | null =>
  validateModel(weightsJson)?.meta ?? null;
