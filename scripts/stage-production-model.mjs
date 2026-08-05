#!/usr/bin/env node
// Stage the Phase 2D ranker artifact into `public/model/ranker/` as the
// PRODUCTION model, and derive its production manifest mechanically from the
// training-side artifacts rather than hand-typing any of it.
//
// Why this exists (delivery-strategy decision, see
// docs/repair/product-proof-sprint/showboat-live/REPORT.md §2):
// `training/ranker/.gitignore`'s "reproducible on demand, not committed"
// policy is right for *experiment* weights but is not a delivery mechanism
// for a browser feature that actually ships. A fresh clone with no Python
// environment must be able to `npm ci && npm run build` and get a working
// neural-enabled bundle. The artifact is 14,185 bytes, so committing it is
// the honest, boring answer — no external URL to rot, no multi-minute
// build-time training step, no CI Python dependency.
//
// This script is how the committed copy is (re)produced, so the provenance
// chain is: phase2d checkpoint -> export_onnx.py -> training MANIFEST.json
// -> this script -> public/model/ranker/. Every hop is hash-checked; the
// vitest suite `src/ai/neural/productionModel.test.ts` re-verifies the last
// hop on every test run, and `scripts/verify-build-model.mjs` re-verifies it
// again in the built `dist/` output.
//
// Usage (only needed when a NEW reviewed artifact is adopted):
//   node scripts/stage-production-model.mjs \
//     --source ../../apps/showboat/training/ranker/phase2d/results/artifact
// Defaults to the in-repo training artifact directory.

import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = resolve(__dirname, "..");

const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};

const sourceDir = resolve(
  argOf("--source", join(APP_ROOT, "training/ranker/phase2d/results/artifact")),
);
const resultsDir = resolve(argOf("--results", join(sourceDir, "..")));
const outDir = join(APP_ROOT, "public/model/ranker");

const SOURCE_ONNX = "showboat_ranker_phase2d.onnx";
const OUT_ONNX = "showboat-ranker-phase2d.onnx";

const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");

// Python's json.dump emits bare `NaN` for float('nan'), which is valid
// Python-flavoured JSON but NOT valid RFC-8259 JSON — `JSON.parse` rejects it
// outright. `phase2d_results.json` and `phase2d_test_results.json` each
// contain 32 of them (the state-only ablation's group-aware Spearman is
// legitimately undefined — see ABLATIONS.md). Rather than silently rewriting
// those committed files (they're Team B's territory), read them through this
// narrow, explicit sanitiser: bare NaN/Infinity become `null`. Only applied to
// values, never keys, and only to these Python-produced metric summaries.
const readPythonJson = (path) => {
  const raw = readFileSync(path, "utf8");
  return JSON.parse(raw.replace(/:\s*(-?Infinity|NaN)\b/g, ": null"));
};

const die = (msg) => {
  console.error(`[stage-model] FATAL: ${msg}`);
  process.exit(1);
};

const trainingManifestPath = join(sourceDir, "MANIFEST.json");
if (!existsSync(trainingManifestPath)) die(`no training manifest at ${trainingManifestPath}`);
const training = JSON.parse(readFileSync(trainingManifestPath, "utf8"));

const onnxPath = join(sourceDir, SOURCE_ONNX);
if (!existsSync(onnxPath)) {
  die(
    `no ONNX artifact at ${onnxPath}.\n` +
      `  The .onnx is gitignored on the training side; regenerate it with\n` +
      `    cd apps/showboat/training/ranker/phase2d && python export_onnx.py --results results --out results/artifact\n` +
      `  or pass --source pointing at a directory that has it. The already-staged\n` +
      `  production copy in public/model/ranker/ is the committed one and does not\n` +
      `  need this script unless a new reviewed artifact is being adopted.`,
  );
}
const onnxBytes = readFileSync(onnxPath);
const onnxHash = sha256(onnxBytes);
if (onnxHash !== training.onnx_sha256) {
  die(
    `source ONNX hash ${onnxHash} does not match the training MANIFEST.json's ` +
      `onnx_sha256 ${training.onnx_sha256}. Refusing to stage an artifact whose ` +
      `provenance cannot be established.`,
  );
}

// The schema is the single source of truth shared by encode.ts (browser),
// schema.py (training) and this manifest — never re-typed.
const schema = JSON.parse(readFileSync(join(APP_ROOT, "src/ai/ranker/schema.json"), "utf8"));
if (training.input_contract.shape[1] !== schema.total_dim) {
  die(
    `training manifest input dim ${training.input_contract.shape[1]} != schema total_dim ${schema.total_dim}`,
  );
}

// Candidate-kind means from the *training* split only, used at runtime as the
// documented fallback prior for `double-bank` (the artifact's own known
// limitation — see MANIFEST.json known_limitations[0]). Read from the
// committed metric JSON, never hand-copied.
const resultsPath = join(resultsDir, "phase2d_results.json");
if (!existsSync(resultsPath)) die(`no phase2d_results.json at ${resultsPath}`);
const results = readPythonJson(resultsPath);
const kindMeans = results.baseline_candidate_kind_mean?.kind_means;
if (!kindMeans) die("phase2d_results.json has no baseline_candidate_kind_mean.kind_means");

const manifest = {
  // --- identity / integrity -------------------------------------------
  artifact: OUT_ONNX,
  onnx_sha256: onnxHash,
  bytes: onnxBytes.length,
  // --- contract the runtime validates before trusting the graph -------
  schema_version: schema.schema_version,
  total_dim: schema.total_dim,
  board_dim: schema.board_dim,
  candidate_dim: schema.candidate_dim,
  input_name: training.input_contract.name,
  output_name: training.output_contract.name,
  output_note: training.output_contract.note,
  // --- calibration ------------------------------------------------------
  // Platt parameters for THIS seed, fit on the validation split only.
  // TRAINING_REPORT.md's calibration section: the raw logit was already
  // near-calibrated (a in [0.89,1.01], b in [-0.15,0.02]; val ECE 0.011-0.017
  // pre, 0.010-0.017 post), so this transform barely moves anything — it is
  // applied anyway because it is the parameterisation the model was evaluated
  // under, not because it rescues a miscalibrated output.
  platt_calibration: training.platt_calibration,
  calibration_evidence: {
    measured: true,
    note:
      "ECE (10 equal-width bins) measured pre- and post-Platt on validation AND on the " +
      "held-out test split with validation-fit parameters. Test ECE 0.0089-0.0150 pre, " +
      "0.0067-0.0155 post, across 5 seeds. The output is therefore allowed to be shown " +
      "as a probability-style 'make estimate' in UI copy; see TRAINING_REPORT.md.",
  },
  // --- provenance -------------------------------------------------------
  provenance: {
    phase: "2D",
    selected_seed: training.selected_seed,
    selection_rule: training.selection_rule,
    source_checkpoint: training.source_checkpoint,
    source_checkpoint_sha256: training.source_checkpoint_sha256,
    training_manifest: "training/ranker/phase2d/results/artifact/MANIFEST.json",
    protocol: training.protocol,
    dataset: "training/ranker/phase2c/data/baseline (3,076 states / 120,444 rows / 1,800 families, family-split)",
    target: training.target,
    architecture: training.architecture,
    license:
      "Trained from scratch on data generated by this repository's own physics engine " +
      "(apps/showboat/physics-core, MIT-licensed with the rest of this repo). No third-party " +
      "pretrained weights, no external dataset, no third-party license obligations.",
  },
  metrics: {
    val_bce: training.validation_metrics.val_bce,
    val_group_aware_spearman: training.validation_metrics.val_group_aware_spearman,
    test_bce_mean_5seed: training.test_metrics_all_seeds_mean.test_bce_mean,
    test_group_aware_spearman_mean_5seed:
      training.test_metrics_all_seeds_mean.test_group_aware_spearman_mean,
  },
  // --- runtime policy derived from the artifact's own known limitations --
  // Copied verbatim from the training manifest — they describe the artifact
  // and must not be quietly edited to look better once it ships.
  known_limitations: training.known_limitations,
  known_limitations_disposition: [
    "double-bank: mitigated at runtime, not ignored — kind_confidence['double-bank']=0.5 blends " +
      "the model's calibrated score halfway toward the training-split kind mean, so a double-bank " +
      "candidate can still be ranked and still be played, but the model alone cannot push one to " +
      "the top of the list on its own confidence. Physics verification and the trick-reliability " +
      "threshold apply unchanged either way.",
    "no hyperparameter search: unchanged and still true. The runtime treats this artifact as one " +
      "prior over candidates, never as the final arbiter — physics decides.",
    "'not wired into live gameplay': superseded. That sentence was accurate for Phase 2D, whose " +
      "protocol explicitly excluded live integration. This manifest exists because the artifact IS " +
      "now wired into live gameplay (see docs/repair/product-proof-sprint/showboat-live/REPORT.md). " +
      "The sentence is kept above verbatim rather than rewritten, because the training manifest is " +
      "a record of what Phase 2D concluded, not a live status field.",
  ],
  kind_priors: {
    note:
      "Candidate-kind mean legal_pot_rate over the TRAINING split only (from " +
      "phase2d_results.json's baseline_candidate_kind_mean). Used at runtime to blend " +
      "down the model's confidence for kinds it is measurably weak on — see " +
      "kind_confidence below.",
    means: kindMeans,
  },
  kind_confidence: {
    note:
      "Per-kind weight on the model's calibrated score vs. the kind-mean prior. " +
      "double-bank is 0.5 because this exported seed's own double-bank group-aware " +
      "Spearman is 0.0600 (val) / 0.0447 (test) — the artifact's first documented " +
      "known limitation. Every other kind is 1.0 (model used as-is). This is a " +
      "runtime mitigation of a measured weakness, not a tuned hyperparameter.",
    direct: 1,
    bank: 1,
    "double-bank": 0.5,
    combo: 1,
    "rail-combo": 1,
  },
  staged_by: "apps/showboat/scripts/stage-production-model.mjs",
};

mkdirSync(outDir, { recursive: true });
copyFileSync(onnxPath, join(outDir, OUT_ONNX));
writeFileSync(join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);

console.log(`[stage-model] staged ${OUT_ONNX} (${onnxBytes.length} bytes, sha256=${onnxHash})`);
console.log(`[stage-model] wrote ${join(outDir, "manifest.json")}`);
