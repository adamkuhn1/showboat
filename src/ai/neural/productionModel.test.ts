// Production model delivery: the artifact in public/model/ranker/ is the
// reviewed Phase 2D checkpoint, it is byte-identical to what the training
// manifest recorded, and every way it can go wrong fails loudly instead of
// silently disabling the model.
//
// These tests run against the real committed files and the real loader — the
// only injected piece is a directory-backed `fetch` (fileFetch.ts), because
// Node has no origin to resolve `model/ranker/manifest.json` against.

import { describe, it, expect, beforeEach } from "vitest";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { SCHEMA_VERSION, TOTAL_DIM, BOARD_DIM, CANDIDATE_DIM } from "../ranker/encode";
import { NeuralCandidateEvaluator } from "./evaluator";
import { validateManifest, calibratedMakeEstimate } from "./manifest";
import { _resetRankerForTests, rankerModelStatus, rankerLoadError } from "../onnx";
import { getBrain, brainLabel } from "../brain";
import { makeFileFetch } from "./fileFetch";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../../..");
const PUBLIC_DIR = join(APP_ROOT, "public");
const MODEL_DIR = join(PUBLIC_DIR, "model/ranker");
const TRAINING_MANIFEST = join(
  APP_ROOT,
  "training/ranker/phase2d/results/artifact/MANIFEST.json",
);

const manifestJson = JSON.parse(readFileSync(join(MODEL_DIR, "manifest.json"), "utf8"));
const artifactBytes = readFileSync(join(MODEL_DIR, manifestJson.artifact));
const sha = (b: Buffer | Uint8Array) => createHash("sha256").update(b).digest("hex");

describe("production artifact: integrity + provenance (static, no runtime)", () => {
  it("the committed artifact hashes to the value its own manifest declares", () => {
    expect(artifactBytes.length).toBe(manifestJson.bytes);
    expect(sha(artifactBytes)).toBe(manifestJson.onnx_sha256);
  });

  it("that hash is the one the reviewed Phase 2D training manifest recorded", () => {
    const training = JSON.parse(readFileSync(TRAINING_MANIFEST, "utf8"));
    expect(manifestJson.onnx_sha256).toBe(training.onnx_sha256);
    expect(manifestJson.provenance.selected_seed).toBe(training.selected_seed);
    expect(manifestJson.provenance.source_checkpoint_sha256).toBe(
      training.source_checkpoint_sha256,
    );
    // The Platt parameters shown as a calibrated estimate in the UI must be
    // the ones this exact checkpoint was calibrated with, not a copy that
    // drifted.
    expect(manifestJson.platt_calibration.a).toBe(training.platt_calibration.a);
    expect(manifestJson.platt_calibration.b).toBe(training.platt_calibration.b);
  });

  it("the manifest's feature contract matches the encoder the browser actually runs", () => {
    expect(manifestJson.schema_version).toBe(SCHEMA_VERSION);
    expect(manifestJson.total_dim).toBe(TOTAL_DIM);
    expect(manifestJson.board_dim).toBe(BOARD_DIM);
    expect(manifestJson.candidate_dim).toBe(CANDIDATE_DIM);
    expect(validateManifest(manifestJson).ok).toBe(true);
  });

  it("known limitations are carried through verbatim, not quietly softened", () => {
    const training = JSON.parse(readFileSync(TRAINING_MANIFEST, "utf8"));
    expect(manifestJson.known_limitations).toEqual(training.known_limitations);
    expect(manifestJson.known_limitations[0]).toContain("double-bank");
  });
});

describe("production artifact: validation rejects every way it can be wrong", () => {
  const bad = (m: Record<string, unknown>) => validateManifest({ ...manifestJson, ...m });

  it("rejects a schema-version mismatch by name", () => {
    const r = bad({ schema_version: "showboat-ranker-v1" });
    expect(r.ok).toBe(false);
    expect(!r.ok && r.error).toMatch(/schema_version/);
  });

  it("rejects a dimension mismatch", () => {
    expect(bad({ total_dim: 67 }).ok).toBe(false);
    expect(bad({ board_dim: 50 }).ok).toBe(false);
  });

  it("rejects a malformed hash", () => {
    expect(bad({ onnx_sha256: "not-a-hash" }).ok).toBe(false);
  });

  it("rejects a missing required field", () => {
    const { platt_calibration: _drop, ...rest } = manifestJson;
    expect(validateManifest(rest).ok).toBe(false);
  });
});

describe("production artifact: the real loader, end to end", () => {
  beforeEach(() => _resetRankerForTests());

  it("loads the committed artifact and reports the hash as verified", async () => {
    const evaluator = new NeuralCandidateEvaluator("model/ranker");
    const state = await evaluator.load(makeFileFetch(PUBLIC_DIR));
    expect(state.status).toBe("ready");
    expect(state.status === "ready" && state.hashVerified).toBe(true);
    expect(evaluator.isReady()).toBe(true);
    expect(rankerModelStatus()).toBe("loaded");
    // getBrain must now genuinely offer the hybrid path.
    expect(getBrain(true, evaluator).kind).toBe("neural-hybrid");
    expect(brainLabel(true, evaluator)).toContain("neural");
  });

  it("a corrupted artifact is 'invalid', not 'absent', and never loads", async () => {
    const corrupted = new Uint8Array(artifactBytes);
    corrupted[corrupted.length - 1] ^= 0xff; // same length, different bytes
    const evaluator = new NeuralCandidateEvaluator("model/ranker");
    const state = await evaluator.load(
      makeFileFetch(PUBLIC_DIR, { [`model/ranker/${manifestJson.artifact}`]: corrupted }),
    );
    expect(state.status).toBe("invalid");
    expect(state.status === "invalid" && state.reason).toMatch(/sha256/);
    expect(evaluator.isReady()).toBe(false);
    expect(rankerLoadError()).toMatch(/refusing to load/);
  });

  it("a truncated artifact fails on byte length before any parse is attempted", async () => {
    const evaluator = new NeuralCandidateEvaluator("model/ranker");
    const state = await evaluator.load(
      makeFileFetch(PUBLIC_DIR, {
        [`model/ranker/${manifestJson.artifact}`]: new Uint8Array(artifactBytes.subarray(0, 100)),
      }),
    );
    expect(state.status).toBe("invalid");
    expect(state.status === "invalid" && state.reason).toMatch(/bytes/);
  });

  it("an incompatible schema version in the manifest fails loudly before loading the graph", async () => {
    const evaluator = new NeuralCandidateEvaluator("model/ranker");
    const state = await evaluator.load(
      makeFileFetch(PUBLIC_DIR, {
        "model/ranker/manifest.json": JSON.stringify({
          ...manifestJson,
          schema_version: "showboat-ranker-v99",
        }),
      }),
    );
    expect(state.status).toBe("invalid");
    expect(state.status === "invalid" && state.reason).toMatch(/showboat-ranker-v99/);
    // Nothing was loaded, so nothing can claim to be neural.
    expect(getBrain(true, evaluator).kind).toBe("classical");
    expect(brainLabel(true, evaluator)).toBe("the physics-search opponent");
  });

  it("a missing manifest is 'absent' and downgrades the brain honestly", async () => {
    const evaluator = new NeuralCandidateEvaluator("model/ranker");
    const state = await evaluator.load(
      makeFileFetch(PUBLIC_DIR, { "model/ranker/manifest.json": null }),
    );
    expect(state.status).toBe("absent");
    expect(evaluator.isReady()).toBe(false);
    expect(getBrain(true, evaluator).kind).toBe("classical");
    expect(brainLabel(true, evaluator)).not.toContain("neural");
  });

  it("an SPA index.html served in place of the manifest is 'absent', not a crash", async () => {
    const evaluator = new NeuralCandidateEvaluator("model/ranker");
    const state = await evaluator.load(
      makeFileFetch(PUBLIC_DIR, { "model/ranker/manifest.json": "<!doctype html><html></html>" }),
    );
    expect(state.status).toBe("absent");
  });
});

describe("calibration is applied exactly as the manifest describes", () => {
  it("a logit of 0 maps to sigmoid(b) for a full-confidence kind", () => {
    const { a, b } = manifestJson.platt_calibration;
    const expected = 1 / (1 + Math.exp(-(a * 0 + b)));
    expect(calibratedMakeEstimate(0, "direct", manifestJson)).toBeCloseTo(expected, 10);
  });

  it("double-bank is blended halfway to the training kind mean — the documented mitigation", () => {
    const { a, b } = manifestJson.platt_calibration;
    const logit = 2.0;
    const p = 1 / (1 + Math.exp(-(a * logit + b)));
    const prior = manifestJson.kind_priors.means["double-bank"];
    const w = manifestJson.kind_confidence["double-bank"];
    expect(w).toBe(0.5);
    expect(calibratedMakeEstimate(logit, "double-bank", manifestJson)).toBeCloseTo(
      w * p + (1 - w) * prior,
      10,
    );
    // The blend must actually pull a confident double-bank estimate down —
    // otherwise the mitigation is cosmetic.
    expect(calibratedMakeEstimate(logit, "double-bank", manifestJson)).toBeLessThan(p);
  });

  it("every other kind is used unmodified", () => {
    for (const kind of ["direct", "bank", "combo", "rail-combo"] as const) {
      const { a, b } = manifestJson.platt_calibration;
      const p = 1 / (1 + Math.exp(-(a * 1.5 + b)));
      expect(calibratedMakeEstimate(1.5, kind, manifestJson)).toBeCloseTo(p, 12);
    }
  });
});
