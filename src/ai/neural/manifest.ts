// The production model manifest: what `public/model/ranker/manifest.json`
// contains, and what the runtime must verify about it before it is allowed to
// influence a single shot.
//
// The manifest is generated mechanically by `scripts/stage-production-model.mjs`
// from the Phase 2D training artifacts — nothing in it is hand-typed. This
// module is the consumer side of that contract.

import { SCHEMA_VERSION, TOTAL_DIM, BOARD_DIM, CANDIDATE_DIM } from "../ranker/encode";
import { type CandidateKind } from "../candidates";

export interface PlattCalibration {
  a: number;
  b: number;
  fit_on: string;
}

export interface ProductionModelManifest {
  artifact: string;
  onnx_sha256: string;
  bytes: number;
  schema_version: string;
  total_dim: number;
  board_dim: number;
  candidate_dim: number;
  input_name: string;
  output_name: string;
  platt_calibration: PlattCalibration;
  calibration_evidence: { measured: boolean; note: string };
  provenance: Record<string, unknown> & { selected_seed: number; phase: string };
  metrics: Record<string, number>;
  known_limitations: string[];
  kind_priors: { note: string; means: Record<string, number> };
  kind_confidence: Record<string, number | string>;
}

/**
 * Everything that must be true before the artifact is trusted. Any failure is
 * a hard, *loud* failure — the caller reports it, logs it, and falls back to
 * the classical search with an honest label. A silently-skipped model is the
 * failure mode this function exists to make impossible.
 */
export function validateManifest(m: unknown): { ok: true; manifest: ProductionModelManifest } | { ok: false; error: string } {
  if (typeof m !== "object" || m === null) return { ok: false, error: "manifest is not an object" };
  const man = m as Partial<ProductionModelManifest>;

  const required: (keyof ProductionModelManifest)[] = [
    "artifact",
    "onnx_sha256",
    "bytes",
    "schema_version",
    "total_dim",
    "input_name",
    "output_name",
    "platt_calibration",
    "kind_priors",
    "kind_confidence",
  ];
  for (const k of required) {
    if (man[k] === undefined) return { ok: false, error: `manifest is missing required field "${k}"` };
  }

  if (man.schema_version !== SCHEMA_VERSION) {
    return {
      ok: false,
      error:
        `manifest schema_version "${man.schema_version}" != encoder schema "${SCHEMA_VERSION}". ` +
        `An artifact trained against a different feature layout would silently mis-encode every ` +
        `candidate (this is exactly what the v1->v2 rail-combo bump broke) — refusing to load.`,
    };
  }
  if (man.total_dim !== TOTAL_DIM || man.board_dim !== BOARD_DIM || man.candidate_dim !== CANDIDATE_DIM) {
    return {
      ok: false,
      error:
        `manifest dims (total=${man.total_dim}, board=${man.board_dim}, candidate=${man.candidate_dim}) ` +
        `!= encoder dims (total=${TOTAL_DIM}, board=${BOARD_DIM}, candidate=${CANDIDATE_DIM})`,
    };
  }
  if (!/^[0-9a-f]{64}$/.test(String(man.onnx_sha256))) {
    return { ok: false, error: `onnx_sha256 "${man.onnx_sha256}" is not a 64-hex-char sha256` };
  }
  const platt = man.platt_calibration as PlattCalibration | undefined;
  if (!platt || typeof platt.a !== "number" || typeof platt.b !== "number") {
    return { ok: false, error: "platt_calibration must have numeric a and b" };
  }
  return { ok: true, manifest: man as ProductionModelManifest };
}

/**
 * Calibrated make-estimate for one candidate: Platt-scaled sigmoid of the raw
 * logit, then blended toward the training-split candidate-kind mean by the
 * manifest's per-kind confidence weight.
 *
 * For the shipped artifact the blend does nothing. Every kind carries a
 * confidence of 1.0, so `calibratedMakeEstimate` returns the Platt-scaled
 * probability unchanged and returns early before any mean is fetched. The
 * mechanism is kept because it is the mitigation a weaker checkpoint needs:
 * the previous Phase 2D MLP ranked double-bank candidates barely above chance
 * and was held at 0.5. This checkpoint's per-kind numbers did not warrant it,
 * which the manifest says in its own `kind_confidence.note`.
 *
 * A retrain that regresses one kind sets that kind's weight below 1.0 in the
 * manifest and the blend starts running again; nothing here needs to change.
 */
export function calibratedMakeEstimate(
  logit: number,
  kind: CandidateKind,
  manifest: ProductionModelManifest,
): number {
  const { a, b } = manifest.platt_calibration;
  const p = 1 / (1 + Math.exp(-(a * logit + b)));
  const wRaw = manifest.kind_confidence[kind];
  const w = typeof wRaw === "number" ? wRaw : 1;
  if (w >= 1) return p;
  const prior = manifest.kind_priors.means[kind] ?? p;
  return w * p + (1 - w) * prior;
}
