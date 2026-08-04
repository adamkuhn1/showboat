// ONNX inference wiring (milestone 5).
//
// The trained policy/value network is exported from the Python LightZero
// self-play pipeline (see training/) to ONNX and loaded here with
// onnxruntime-web (WebGPU with WASM fallback). Until the multi-hour Colab
// training run lands, there is no model file, and this module reports that
// truthfully so the UI keeps saying "search baseline" — we never dress the
// baseline up as the trained net.
//
// The model, when present, lives at `public/model/showboat.onnx` (Vite serves
// `public/` at the site root). A tiny smoke-trained net proves this plumbing end
// to end before the full run.

// onnxruntime-web is a ~26MB WASM runtime. We import it *dynamically* and only
// after confirming a model file actually exists, so a first-time visitor with no
// trained model present never downloads the runtime. This keeps the baseline
// build lean while leaving the trained-net path fully wired.
type Ort = typeof import("onnxruntime-web");
type InferenceSession = import("onnxruntime-web").InferenceSession;

export type ModelStatus = "absent" | "loading" | "loaded" | "error";

const MODEL_URL = "model/showboat.onnx";

let status: ModelStatus = "absent";
let ort: Ort | null = null;
let session: InferenceSession | null = null;
let loadPromise: Promise<void> | null = null;

// Observation layout the network expects, kept in one place so training and
// inference agree: normalized (x,y) for the 16 balls + a pocketed flag each =
// 48 floats. (The Python env exports the same layout; documented in training.)
export const OBS_BALLS = 16;
export const OBS_DIM = OBS_BALLS * 3;

// Attempt to load the model. Safe to call repeatedly; resolves whether or not a
// model exists. Never throws — a missing model is an expected state, not an
// error, so the baseline can run.
export const tryLoadModel = async (): Promise<void> => {
  if (loadPromise) return loadPromise;
  loadPromise = (async () => {
    status = "loading";
    try {
      // Probe for the model. A dev/SPA server answers a missing file with an
      // index.html fallback (200 + text/html), so a bare `res.ok` isn't enough:
      // we also require a non-HTML content-type and the ONNX/protobuf magic so a
      // missing model is cleanly "absent" (→ baseline) rather than a parse error.
      const res = await fetch(MODEL_URL, { method: "GET" });
      if (!res.ok) {
        status = "absent";
        return;
      }
      const ct = res.headers.get("content-type") ?? "";
      if (ct.includes("text/html")) {
        status = "absent"; // SPA fallback: no model file present.
        return;
      }
      const buf = await res.arrayBuffer();
      // ONNX files are protobuf; the very first byte of our exported graphs is
      // field-tag 0x08 (ir_version). An HTML page starts with '<' (0x3C).
      const first = new Uint8Array(buf.slice(0, 1))[0];
      if (first === 0x3c) {
        status = "absent";
        return;
      }
      // Only now pull in the heavy runtime.
      ort = await import("onnxruntime-web");
      session = await ort.InferenceSession.create(buf, {
        executionProviders: ["webgpu", "wasm"],
        graphOptimizationLevel: "all",
      });
      status = "loaded";
    } catch {
      // No model, or a runtime that can't load it: fall back to baseline.
      status = session ? "loaded" : "absent";
      if (status !== "loaded") status = "absent";
    }
  })();
  return loadPromise;
};

export const trainedModelStatus = (): ModelStatus => status;
export const hasTrainedModel = (): boolean => status === "loaded" && session !== null;

// Evaluate the policy/value net on an observation. Returns null when no model is
// loaded so callers transparently use the baseline value. Output contract:
//   value: scalar in [-1,1] (win expectation)
//   policy: per-candidate logits (length matches the candidate encoding)
export interface NetOutput {
  value: number;
  policy: Float32Array;
}

export const evaluate = async (obs: Float32Array): Promise<NetOutput | null> => {
  if (!hasTrainedModel() || !session || !ort) return null;
  const input = new ort.Tensor("float32", obs, [1, OBS_DIM]);
  const feeds: Record<string, import("onnxruntime-web").Tensor> = {};
  feeds[session.inputNames[0]] = input;
  const out = await session.run(feeds);
  // Convention: first output = value, second = policy. Robust to naming by
  // falling back to output order.
  const names = session.outputNames;
  const valueT = out[names[0]];
  const policyT = names.length > 1 ? out[names[1]] : undefined;
  const value = (valueT.data as Float32Array)[0] ?? 0;
  const policy = policyT
    ? (policyT.data as Float32Array)
    : new Float32Array(0);
  return { value, policy };
};

// ---------------------------------------------------------------------------
// Candidate-ranker model (Phase 2A) — a second, independent model slot.
//
// This is deliberately NOT merged with the whole-board policy/value net above.
// That net (never yet trained — see docs/repair/showboat-ml/01-current-ml-audit.md)
// outputs one scalar for the entire board; this one scores each candidate
// individually, which is the capability the audit found missing. Kept
// separate so the parked net's fallback behavior is untouched. Same
// lazy-load / magic-byte-sniff / never-throws pattern as tryLoadModel above,
// on purpose — that pattern is already correct.
//
// Per Phase 2A's acceptance criteria, this slot is not wired into
// `brainLabel()`'s public "trained AI" string and is not loaded from the
// default `public/model/` path by App.tsx — proof that it works lives in
// `ranker/rankerIntegration.test.ts`, not in a live UI claim, until Phase 2F.
// ---------------------------------------------------------------------------

export type RankerStatus = "absent" | "loading" | "loaded" | "error";

let rankerStatus: RankerStatus = "absent";
let rankerOrt: Ort | null = null;
let rankerSession: InferenceSession | null = null;
let rankerLoadPromise: Promise<void> | null = null;
let rankerLoadedUrl: string | null = null;

export const tryLoadRankerModel = async (url: string): Promise<void> => {
  if (rankerLoadPromise && rankerLoadedUrl === url) return rankerLoadPromise;
  rankerLoadedUrl = url;
  rankerLoadPromise = (async () => {
    rankerStatus = "loading";
    try {
      const res = await fetch(url, { method: "GET" });
      if (!res.ok) {
        rankerStatus = "absent";
        return;
      }
      const ct = res.headers.get("content-type") ?? "";
      if (ct.includes("text/html")) {
        rankerStatus = "absent";
        return;
      }
      const buf = await res.arrayBuffer();
      const first = new Uint8Array(buf.slice(0, 1))[0];
      if (first === 0x3c) {
        rankerStatus = "absent";
        return;
      }
      rankerOrt = ort ?? (await import("onnxruntime-web"));
      rankerSession = await rankerOrt.InferenceSession.create(buf, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      rankerStatus = "loaded";
    } catch {
      rankerStatus = rankerSession ? "loaded" : "absent";
      if (rankerStatus !== "loaded") rankerStatus = "absent";
    }
  })();
  return rankerLoadPromise;
};

export const rankerModelStatus = (): RankerStatus => rankerStatus;
export const hasRankerModel = (): boolean => rankerStatus === "loaded" && rankerSession !== null;

/**
 * Score a batch of candidate rows in one session.run() call (not N calls —
 * see docs/repair/showboat-ml/04-browser-integration.md §8 on per-call
 * overhead). `rows` is `count` rows of `dim` floats each, row-major. Returns
 * one score per row (raw logit; caller applies sigmoid if a probability-style
 * display is wanted — see EVALUATION_SPEC.md on not doing that before
 * calibration is measured).
 */
export const evaluateCandidateRows = async (
  rows: Float32Array,
  count: number,
  dim: number,
): Promise<Float32Array | null> => {
  if (!hasRankerModel() || !rankerSession || !rankerOrt) return null;
  const input = new rankerOrt.Tensor("float32", rows, [count, dim]);
  const feeds: Record<string, import("onnxruntime-web").Tensor> = {};
  feeds[rankerSession.inputNames[0]] = input;
  const out = await rankerSession.run(feeds);
  return out[rankerSession.outputNames[0]].data as Float32Array;
};

// Encode a ball list into the observation vector the net expects. Positions are
// normalized to [-1,1] by half the table dimensions; pocketed balls report 0s.
export const encodeObservation = (
  balls: { id: number; pos: { x: number; y: number }; pocketed: boolean }[],
  halfLen: number,
  halfWid: number,
): Float32Array => {
  const obs = new Float32Array(OBS_DIM);
  for (const b of balls) {
    if (b.id >= OBS_BALLS) continue;
    const o = b.id * 3;
    if (b.pocketed) {
      obs[o] = 0;
      obs[o + 1] = 0;
      obs[o + 2] = 1; // pocketed flag
    } else {
      obs[o] = b.pos.x / halfLen;
      obs[o + 1] = b.pos.y / halfWid;
      obs[o + 2] = 0;
    }
  }
  return obs;
};
