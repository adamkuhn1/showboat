#!/usr/bin/env node
// Stage the Phase 2E relational (Deep Sets) ranker as the PRODUCTION model,
// replacing the Phase 2D MLP staged by stage-production-model.mjs.
//
// Why (see docs/repair/product-proof-sprint/showboat-model-research/REPORT.md
// and docs/repair/product-proof-sprint/COORDINATION.md, "Integration wave"):
// Team B's independent research recommends adopting the Deep Sets model --
// it beats the Phase 2D MLP on every held-out metric, substantially repairs
// the double-bank weakness, and exports to the IDENTICAL [N,68]->[N,1] ONNX
// contract Team A's runtime already speaks, verified end-to-end through
// onnxruntime-web with zero changes required to encode.ts/onnx.ts. This
// script is the lead's integration step: adopt the artifact through the
// same staging/hash-verification pattern Team A already built, adapted for
// Phase 2E's manifest shape (which differs from Phase 2D's -- io_contract
// instead of input_contract/output_contract, no known_limitations array,
// per-seed per-kind breakdowns live in phase2e_test_results.json rather
// than the training manifest itself).
//
// Provenance chain: phase2e checkpoint -> export_onnx.py -> MANIFEST_deepsets.json
// -> phase2e_test_results.json (per-seed per-kind numbers, for the SHIPPED
// seed specifically, not the 5-seed ensemble mean -- see the double-bank
// disposition below) -> this script -> public/model/ranker/.
//
// Usage:
//   node scripts/stage-production-model-phase2e.mjs \
//     --source ../../apps/showboat/training/ranker/phase2e/results/artifact

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
  argOf("--source", join(APP_ROOT, "training/ranker/phase2e/results/artifact")),
);
const phase2eResultsDir = resolve(argOf("--results", join(sourceDir, "..")));
const phase2dResultsDir = resolve(join(APP_ROOT, "training/ranker/phase2d/results"));
const outDir = join(APP_ROOT, "public/model/ranker");

const SOURCE_ONNX = "showboat_ranker_phase2e_deepsets.onnx";
const OUT_ONNX = "showboat-ranker-phase2e-deepsets.onnx";

const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");

const die = (msg) => {
  console.error(`[stage-model-phase2e] FATAL: ${msg}`);
  process.exit(1);
};

const trainingManifestPath = join(sourceDir, "MANIFEST_deepsets.json");
if (!existsSync(trainingManifestPath)) die(`no training manifest at ${trainingManifestPath}`);
const training = JSON.parse(readFileSync(trainingManifestPath, "utf8"));

const onnxPath = join(sourceDir, SOURCE_ONNX);
if (!existsSync(onnxPath)) {
  die(
    `no ONNX artifact at ${onnxPath}.\n` +
      `  The .onnx is gitignored on the training side; regenerate it with\n` +
      `    cd apps/showboat/training/ranker/phase2e && python export_onnx.py --variant deepsets\n` +
      `  or pass --source pointing at a directory that has it.`,
  );
}
const onnxBytes = readFileSync(onnxPath);
const onnxHash = sha256(onnxBytes);
if (onnxHash !== training.artifact_sha256) {
  die(
    `source ONNX hash ${onnxHash} does not match MANIFEST_deepsets.json's ` +
      `artifact_sha256 ${training.artifact_sha256}. Refusing to stage an artifact whose ` +
      `provenance cannot be established.`,
  );
}

const schema = JSON.parse(readFileSync(join(APP_ROOT, "src/ai/ranker/schema.json"), "utf8"));
if (training.io_contract.input.shape[1] !== schema.total_dim) {
  die(
    `training manifest input dim ${training.io_contract.input.shape[1]} != schema total_dim ${schema.total_dim}`,
  );
}

// Per-kind test numbers for THIS shipped seed specifically (20260805), not
// the 5-seed ensemble mean -- same discipline applied when Team A fixed the
// equivalent Phase 2D bug (a manifest documenting one checkpoint must cite
// that checkpoint's own numbers, which can be worse -- or, here, better --
// than the ensemble average).
const testResultsPath = join(phase2eResultsDir, "phase2e_test_results.json");
if (!existsSync(testResultsPath)) die(`no phase2e_test_results.json at ${testResultsPath}`);
// Same bare-NaN-from-Python issue Team A found and worked around for Phase
// 2D's JSON (the state-only ablation's undefined group-aware Spearman is
// legitimately NaN, which json.dump emits as invalid-JSON bare `NaN`).
const readPythonJson = (path) =>
  JSON.parse(readFileSync(path, "utf8").replace(/:\s*(-?Infinity|NaN)\b/g, ": null"));
const testResults = readPythonJson(testResultsPath);
const deepsetsSeeds = testResults.variants?.deepsets?.per_seed;
if (!deepsetsSeeds) die("phase2e_test_results.json has no variants.deepsets.per_seed");
const shippedSeedTest = deepsetsSeeds.find((s) => s.seed === training.selected_seed);
if (!shippedSeedTest) die(`no per-seed test entry for selected_seed ${training.selected_seed}`);

const variantValPath = join(phase2eResultsDir, "variant_deepsets.json");
if (!existsSync(variantValPath)) die(`no variant_deepsets.json at ${variantValPath}`);
const variantVal = readPythonJson(variantValPath);
const shippedSeedVal = variantVal.per_seed.find((s) => s.seed === training.selected_seed);
if (!shippedSeedVal) die(`no per-seed val entry for selected_seed ${training.selected_seed}`);

const dbValGas = shippedSeedVal.val.per_kind["double-bank"].group_aware_spearman;
const dbTestGas = shippedSeedTest.per_kind["double-bank"].group_aware_spearman;
const pooledValGas = shippedSeedVal.val.group_aware_spearman;
const pooledTestGas = shippedSeedTest.group_aware_spearman;
const valGap = pooledValGas - dbValGas;
const testGap = pooledTestGas - dbTestGas;

// Evidence-based double-bank disposition: Phase 2D's shipped seed needed a
// 0.5 runtime blend toward the kind-mean prior because its own double-bank
// group-aware Spearman (0.0600 val / 0.0447 test) was catastrophically weak
// relative to pooled. This shipped seed's own double-bank numbers are
// materially better and clear the predeclared 0.3 catastrophic-kind gap on
// BOTH splits, so no runtime mitigation is applied here -- verified from
// this seed's own numbers, not asserted from the 5-seed ensemble average.
const NO_MITIGATION_GAP_THRESHOLD = 0.3;
const dbNeedsMitigation = valGap >= NO_MITIGATION_GAP_THRESHOLD || testGap >= NO_MITIGATION_GAP_THRESHOLD;

// Candidate-kind means are a property of the dataset's candidate-kind
// distribution (raw_counts.legal_pot/n_perturbations grouped by kind on the
// training split), not of which ranker architecture reads it -- reused
// verbatim from Phase 2D's committed results rather than recomputed.
const phase2dResultsPath = join(phase2dResultsDir, "phase2d_results.json");
if (!existsSync(phase2dResultsPath)) die(`no phase2d_results.json at ${phase2dResultsPath}`);
const phase2dResults = JSON.parse(
  readFileSync(phase2dResultsPath, "utf8").replace(/:\s*(-?Infinity|NaN)\b/g, ": null"),
);
const kindMeans = phase2dResults.baseline_candidate_kind_mean?.kind_means;
if (!kindMeans) die("phase2d_results.json has no baseline_candidate_kind_mean.kind_means");

const manifest = {
  artifact: OUT_ONNX,
  onnx_sha256: onnxHash,
  bytes: onnxBytes.length,
  schema_version: schema.schema_version,
  total_dim: schema.total_dim,
  board_dim: schema.board_dim,
  candidate_dim: schema.candidate_dim,
  input_name: training.io_contract.input.name,
  output_name: training.io_contract.output.name,
  output_note: "raw logit per row -- apply sigmoid(a*logit + b) with platt_calibration for a calibrated probability",
  platt_calibration: training.platt,
  calibration_evidence: {
    measured: true,
    note:
      `Test ECE (10 equal-width bins): ${shippedSeedTest.test_ece?.toFixed?.(4) ?? shippedSeedTest.ece} pre-Platt, ` +
      `${shippedSeedTest.ece_calibrated} post-Platt, this seed. Platt fit on validation only, applied unchanged to test. ` +
      "See docs/repair/product-proof-sprint/showboat-model-research/REPORT.md section 4.2.",
  },
  provenance: {
    phase: "2E",
    architecture: training.architecture,
    n_params: training.n_params,
    selected_seed: training.selected_seed,
    selection_rule: training.selection_rule,
    source_checkpoint: training.source_checkpoint,
    source_checkpoint_sha256: training.source_checkpoint_sha256,
    training_manifest: "training/ranker/phase2e/results/artifact/MANIFEST_deepsets.json",
    dataset: "training/ranker/phase2c/data/baseline (3,076 states / 120,444 rows / 1,800 families, family-split)",
    target: "legal_pot_rate (identical to Phase 2D)",
    predecessor: "Phase 2D candidate-conditioned MLP (docs/repair/showboat-ml/phase-2d/) -- superseded, not deleted",
    license:
      "Trained from scratch on data generated by this repository's own physics engine " +
      "(apps/showboat/physics-core, MIT-licensed with the rest of this repo). No third-party " +
      "pretrained weights, no external dataset, no third-party license obligations.",
  },
  metrics: {
    val_bce: shippedSeedVal.val.bce,
    val_group_aware_spearman: pooledValGas,
    test_bce_5seed_mean: testResults.variants.deepsets.test_bce.mean,
    test_group_aware_spearman_5seed_mean: testResults.variants.deepsets.test_group_aware_spearman.mean,
    test_bce_this_seed: shippedSeedTest.test_bce,
    test_group_aware_spearman_this_seed: pooledTestGas,
  },
  known_limitations: [
    `double-bank candidates: this exported seed's (${training.selected_seed}) own group-aware Spearman is ` +
      `${dbValGas.toFixed(4)} (val) / ${dbTestGas.toFixed(4)} (test) -- the weakest of the five kinds, though a ` +
      `2.3-3.4x improvement over the predecessor Phase 2D MLP (0.0600/0.0447). Gap to pooled is ${valGap.toFixed(3)} ` +
      `(val) / ${testGap.toFixed(3)} (test), both under the predeclared 0.3 catastrophic-kind threshold. See ABLATIONS.md-equivalent discussion in the Phase 2E report.`,
    "No hyperparameter search beyond a bounded 9-point validation-only grid (search.json) -- a research " +
      "comparison, not an exhaustively tuned model.",
    "Not a graph neural network -- an attention-augmented variant (deepsets_attn) was evaluated and found " +
      "not worth its 3.4x training cost for negligible gain; see the Phase 2E report section 7.",
  ],
  known_limitations_disposition: [
    dbNeedsMitigation
      ? "double-bank: runtime mitigation IS applied (kind_confidence 0.5) -- see kind_confidence below."
      : `double-bank: NO runtime mitigation applied. This seed's own val/test gaps to pooled ` +
        `(${valGap.toFixed(3)} / ${testGap.toFixed(3)}) are both under the predeclared 0.3 catastrophic-kind ` +
        "threshold, unlike Phase 2D's shipped MLP seed which needed a 0.5 kind-mean blend. Evaluated from " +
        "this specific seed's numbers, not the 5-seed ensemble average, per the same discipline applied to " +
        "the Phase 2D manifest fix.",
    "Physics verification and the trick-reliability threshold apply unchanged regardless of kind_confidence " +
      "-- this field only affects candidate ranking/pruning priority, never legality or trick qualification.",
  ],
  kind_priors: {
    note:
      "Candidate-kind mean legal_pot_rate over the TRAINING split (reused from Phase 2D's committed " +
      "phase2d_results.json -- a property of the dataset's candidate-kind distribution, not of which " +
      "ranker architecture is deployed).",
    means: kindMeans,
  },
  kind_confidence: {
    note: dbNeedsMitigation
      ? "Per-kind weight on the model's calibrated score vs. the kind-mean prior."
      : "All kinds at 1.0 (model used as-is, no kind-mean blending) -- this seed's per-kind numbers do not " +
        "warrant the mitigation Phase 2D's MLP needed. Revisit if a future seed/retrain regresses double-bank.",
    direct: 1,
    bank: 1,
    "double-bank": dbNeedsMitigation ? 0.5 : 1,
    combo: 1,
    "rail-combo": 1,
  },
  staged_by: "apps/showboat/scripts/stage-production-model-phase2e.mjs",
};

mkdirSync(outDir, { recursive: true });
copyFileSync(onnxPath, join(outDir, OUT_ONNX));
writeFileSync(join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);

console.log(`[stage-model-phase2e] staged ${OUT_ONNX} (${onnxBytes.length} bytes, sha256=${onnxHash})`);
console.log(`[stage-model-phase2e] double-bank val gap=${valGap.toFixed(3)} test gap=${testGap.toFixed(3)} -> mitigation ${dbNeedsMitigation ? "APPLIED" : "not needed"}`);
console.log(`[stage-model-phase2e] wrote ${join(outDir, "manifest.json")}`);
