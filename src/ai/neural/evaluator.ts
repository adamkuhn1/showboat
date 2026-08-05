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
  /** How many candidates were in the batch. */
  batchSize: number;
}

export class NeuralCandidateEvaluator {
  private state: EvaluatorState = { status: "absent", reason: "not loaded yet" };
  private loading: Promise<EvaluatorState> | null = null;

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
   * Load the manifest, then the artifact. Idempotent. Never throws: every
   * failure resolves to an `absent`/`invalid` state carrying a human-readable
   * reason, which the UI is required to display rather than silently degrade.
   */
  async load(fetchImpl: typeof fetch = fetch): Promise<EvaluatorState> {
    if (this.loading) return this.loading;
    this.loading = (async (): Promise<EvaluatorState> => {
      const manifestUrl = `${this.dir}/manifest.json`;
      let raw: unknown;
      try {
        const res = await fetchImpl(manifestUrl, { method: "GET" });
        if (!res.ok) {
          return (this.state = { status: "absent", reason: `HTTP ${res.status} fetching ${manifestUrl}` });
        }
        const text = await res.text();
        if (text.trimStart().startsWith("<")) {
          return (this.state = { status: "absent", reason: `${manifestUrl} returned HTML — no manifest at that path` });
        }
        raw = JSON.parse(text);
      } catch (e) {
        return (this.state = {
          status: "absent",
          reason: `could not read ${manifestUrl}: ${e instanceof Error ? e.message : String(e)}`,
        });
      }

      const validated = validateManifest(raw);
      if (!validated.ok) {
        return (this.state = { status: "invalid", reason: validated.error });
      }
      const manifest = validated.manifest;

      await tryLoadRankerModel(`${this.dir}/${manifest.artifact}`, {
        expectedSha256: manifest.onnx_sha256,
        expectedBytes: manifest.bytes,
        expectedInputDim: manifest.total_dim,
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
    const out = await evaluateCandidateRows(rows, candidates.length, TOTAL_DIM);
    const inferenceMs = performance.now() - t0;
    if (!out || out.length !== candidates.length) return null;

    const logits: number[] = [];
    const scores: number[] = [];
    for (let i = 0; i < candidates.length; i++) {
      const logit = out[i];
      logits.push(logit);
      scores.push(calibratedMakeEstimate(logit, candidates[i].kind, manifest));
    }
    return { scores, logits, inferenceMs, batchSize: candidates.length };
  }
}

/** Process-wide singleton — one artifact, one session, loaded once. */
export const neuralEvaluator = new NeuralCandidateEvaluator();
