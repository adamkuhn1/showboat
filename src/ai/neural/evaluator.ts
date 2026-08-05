// NeuralCandidateEvaluator — the one boundary between the trained Phase 2D
// ranker and the rest of Showboat.
//
// Responsibilities, and deliberately nothing else:
//   1. Fetch + validate the production manifest, then load + integrity-check
//      the ONNX artifact through onnx.ts's `tryLoadRankerModel`.
//   2. Encode a candidate list with the SAME `encodeRow` the training data
//      generator used, batch it into ONE session.run(), and return a
//      calibrated make-estimate per candidate.
//
// What it explicitly does NOT do: decide anything. It produces a prior.
// `shotSearch.ts` still runs the authoritative physics for every candidate it
// keeps, still derives `potsTarget` from `isLegalPot()` on a real simulation,
// and still applies the trick-reliability threshold to a physics-derived
// strength. The model reorders and prunes work; it never substitutes for it.

import { type Ball } from "../../physics/ball";
import { type Table } from "../../physics/table";
import { type Candidate } from "../candidates";
import { encodeRow, TOTAL_DIM } from "../ranker/encode";
import {
  evaluateCandidateRows,
  hasRankerModel,
  rankerLoadError,
  rankerHashWasVerified,
  rankerModelStatus,
  tryLoadRankerModel,
  type RankerStatus,
} from "../onnx";
import {
  calibratedMakeEstimate,
  validateManifest,
  type ProductionModelManifest,
} from "./manifest";

export const DEFAULT_MODEL_DIR = "model/ranker";

export type EvaluatorState =
  | { status: "absent"; reason: string }
  | { status: "invalid"; reason: string }
  | { status: "ready"; manifest: ProductionModelManifest; hashVerified: boolean };

export interface CandidateScores {
  /** Calibrated make-estimate in [0,1], index-aligned to the input candidates. */
  scores: number[];
  /** Raw model logits, before Platt scaling and per-kind blending. */
  logits: number[];
  /** Wall-clock ms for encode + the single batched session.run(). */
  inferenceMs: number;
  /** Of that, the feature-encoding half (pure TS, no ONNX). */
  encodeMs: number;
  /** Of that, the `session.run()` half. Reported separately because Phase 2D's
   *  latency measurement only covered this part, and the two differ. */
  runMs: number;
  /** How many candidates were in the batch. */
  batchSize: number;
}

export class NeuralCandidateEvaluator {
  private state: EvaluatorState = { status: "absent", reason: "not loaded yet" };
  private loading: Promise<EvaluatorState> | null = null;
  private preflighting: Promise<{ ok: true; manifest: ProductionModelManifest } | { ok: false; reason: string }> | null =
    null;

  constructor(private readonly dir: string = DEFAULT_MODEL_DIR) {}

  getState(): EvaluatorState {
    return this.state;
  }

  isReady(): boolean {
    return this.state.status === "ready" && hasRankerModel();
  }

  /** The artifact's own manifest, or null when it isn't loaded and validated. */
  getManifest(): ProductionModelManifest | null {
    return this.state.status === "ready" ? this.state.manifest : null;
  }

  /**
   * Cheap startup check: fetch and validate the manifest (~5 KB) and hash the
   * artifact bytes (~14 KB), WITHOUT importing onnxruntime-web or creating a
   * session. That's the ~27 MB WASM runtime deferred until the model is
   * actually going to be used, while still catching a missing / corrupted /
   * schema-mismatched artifact loudly at startup.
   *
   * Returns the validated manifest, or a reason it could not be validated.
   * Never throws.
   */
  async preflight(
    fetchImpl: typeof fetch = fetch,
  ): Promise<{ ok: true; manifest: ProductionModelManifest } | { ok: false; reason: string }> {
    if (this.preflighting) return this.preflighting;
    this.preflighting = (async () => {
      const read = await this.readManifest(fetchImpl);
      if (!read.ok) return { ok: false as const, reason: read.reason };
      const manifest = read.manifest;
      try {
        const res = await fetchImpl(`${this.dir}/${manifest.artifact}`, { method: "GET" });
        if (!res.ok) return { ok: false as const, reason: `HTTP ${res.status} fetching ${manifest.artifact}` };
        const buf = await res.arrayBuffer();
        if (new Uint8Array(buf.slice(0, 1))[0] === 0x3c) {
          return { ok: false as const, reason: `${manifest.artifact} returned HTML, not an ONNX graph` };
        }
        if (buf.byteLength !== manifest.bytes) {
          return {
            ok: false as const,
            reason: `${manifest.artifact} is ${buf.byteLength} bytes, manifest declares ${manifest.bytes}`,
          };
        }
        const subtle = globalThis.crypto?.subtle;
        if (subtle) {
          const digest = await subtle.digest("SHA-256", buf);
          const hex = Array.from(new Uint8Array(digest))
            .map((b) => b.toString(16).padStart(2, "0"))
            .join("");
          if (hex !== manifest.onnx_sha256) {
            return { ok: false as const, reason: `${manifest.artifact} sha256 ${hex} != manifest ${manifest.onnx_sha256}` };
          }
        }
        return { ok: true as const, manifest };
      } catch (e) {
        return { ok: false as const, reason: e instanceof Error ? e.message : String(e) };
      }
    })();
    return this.preflighting;
  }

  private async readManifest(
    fetchImpl: typeof fetch,
  ): Promise<{ ok: true; manifest: ProductionModelManifest } | { ok: false; reason: string; kind: "absent" | "invalid" }> {
    const manifestUrl = `${this.dir}/manifest.json`;
    let raw: unknown;
    try {
      const res = await fetchImpl(manifestUrl, { method: "GET" });
      if (!res.ok) return { ok: false, kind: "absent", reason: `HTTP ${res.status} fetching ${manifestUrl}` };
      const text = await res.text();
      if (text.trimStart().startsWith("<")) {
        return { ok: false, kind: "absent", reason: `${manifestUrl} returned HTML — no manifest at that path` };
      }
      raw = JSON.parse(text);
    } catch (e) {
      return {
        ok: false,
        kind: "absent",
        reason: `could not read ${manifestUrl}: ${e instanceof Error ? e.message : String(e)}`,
      };
    }
    const validated = validateManifest(raw);
    if (!validated.ok) return { ok: false, kind: "invalid", reason: validated.error };
    return { ok: true, manifest: validated.manifest };
  }

  /**
   * Load the manifest, then the artifact and an onnxruntime-web session.
   * Idempotent. Never throws: every failure resolves to an `absent`/`invalid`
   * state carrying a human-readable reason, which the UI is required to display
   * rather than silently degrade.
   */
  async load(fetchImpl: typeof fetch = fetch): Promise<EvaluatorState> {
    if (this.loading) return this.loading;
    this.loading = (async (): Promise<EvaluatorState> => {
      const read = await this.readManifest(fetchImpl);
      if (!read.ok) return (this.state = { status: read.kind, reason: read.reason });
      const manifest = read.manifest;

      await tryLoadRankerModel(`${this.dir}/${manifest.artifact}`, {
        expectedSha256: manifest.onnx_sha256,
        expectedBytes: manifest.bytes,
        expectedInputDim: manifest.total_dim,
        fetchImpl,
      });

      const status: RankerStatus = rankerModelStatus();
      if (status !== "loaded" || !hasRankerModel()) {
        const reason = rankerLoadError() ?? `ranker artifact status "${status}"`;
        return (this.state = status === "absent" ? { status: "absent", reason } : { status: "invalid", reason });
      }
      return (this.state = { status: "ready", manifest, hashVerified: rankerHashWasVerified() });
    })();
    return this.loading;
  }

  /**
   * Score every candidate in one batched inference. Returns null when the
   * model isn't ready — callers must treat null as "run classical", never as
   * "score 0".
   */
  async score(balls: Ball[], table: Table, candidates: Candidate[]): Promise<CandidateScores | null> {
    if (!this.isReady() || candidates.length === 0) return null;
    const manifest = (this.state as Extract<EvaluatorState, { status: "ready" }>).manifest;

    const t0 = performance.now();
    const rows = new Float32Array(candidates.length * TOTAL_DIM);
    for (let i = 0; i < candidates.length; i++) {
      rows.set(encodeRow(balls, table, candidates[i]), i * TOTAL_DIM);
    }
    const t1 = performance.now();
    const out = await evaluateCandidateRows(rows, candidates.length, TOTAL_DIM);
    const t2 = performance.now();
    const inferenceMs = t2 - t0;
    if (!out || out.length !== candidates.length) return null;

    const logits: number[] = [];
    const scores: number[] = [];
    for (let i = 0; i < candidates.length; i++) {
      const logit = out[i];
      logits.push(logit);
      scores.push(calibratedMakeEstimate(logit, candidates[i].kind, manifest));
    }
    return {
      scores,
      logits,
      inferenceMs,
      encodeMs: t1 - t0,
      runMs: t2 - t1,
      batchSize: candidates.length,
    };
  }
}

/** Process-wide singleton — one artifact, one session, loaded once. */
export const neuralEvaluator = new NeuralCandidateEvaluator();
