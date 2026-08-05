#!/usr/bin/env node
// Post-build gate: the production bundle must actually contain a valid,
// hash-matching ranker artifact.
//
// `vite build` copies `public/` into `dist/` without inspecting it, so a
// deleted, truncated or swapped artifact would produce a perfectly successful
// build that silently ships an app with no model. This script makes that a
// build failure instead. It runs as part of `npm run build`.
//
// It checks exactly what the browser will check at runtime, against the same
// manifest, so a green build means the runtime's validation will also pass:
//   - dist/model/ranker/manifest.json parses and declares the current schema
//   - the artifact it names exists, is the declared byte length, and hashes to
//     the declared sha256
//   - that sha256 still matches the Phase 2D training manifest's own
//     onnx_sha256, so provenance is unbroken end to end

import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const APP_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
// `--dist <dir>` exists so `scripts/verifyBuildModel.test.ts` can point this
// exact script (not a reimplementation of it) at fixture directories that are
// deliberately broken, and assert it rejects them.
const distArgIndex = process.argv.indexOf("--dist");
const DIST_MODEL_DIR =
  distArgIndex >= 0 && process.argv[distArgIndex + 1]
    ? resolve(process.argv[distArgIndex + 1])
    : join(APP_ROOT, "dist/model/ranker");

const fail = (msg) => {
  console.error(`\n[verify-build-model] BUILD REJECTED: ${msg}\n`);
  process.exit(1);
};

const manifestPath = join(DIST_MODEL_DIR, "manifest.json");
if (!existsSync(manifestPath)) {
  fail(
    `${manifestPath} is missing. The neural-enabled build requires the staged model.\n` +
      `  Fix: node scripts/stage-production-model.mjs (see public/model/ranker/README.md).`,
  );
}

let manifest;
try {
  manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
} catch (e) {
  fail(`${manifestPath} is not valid JSON: ${e.message}`);
}

const artifactPath = join(DIST_MODEL_DIR, manifest.artifact);
if (!existsSync(artifactPath)) fail(`manifest names "${manifest.artifact}" but ${artifactPath} does not exist`);

const bytes = readFileSync(artifactPath);
if (bytes.length !== manifest.bytes) {
  fail(`${manifest.artifact} is ${bytes.length} bytes, manifest declares ${manifest.bytes}`);
}
const hash = createHash("sha256").update(bytes).digest("hex");
if (hash !== manifest.onnx_sha256) {
  fail(`${manifest.artifact} sha256 ${hash} != manifest ${manifest.onnx_sha256}`);
}

// Provenance chain back to the training phase that produced and reviewed it.
const trainingManifestPath = join(
  APP_ROOT,
  "training/ranker/phase2d/results/artifact/MANIFEST.json",
);
if (!existsSync(trainingManifestPath)) {
  fail(`training manifest ${trainingManifestPath} is missing — provenance cannot be established`);
}
const training = JSON.parse(readFileSync(trainingManifestPath, "utf8"));
if (training.onnx_sha256 !== hash) {
  fail(
    `shipped artifact sha256 ${hash} != Phase 2D training manifest onnx_sha256 ` +
      `${training.onnx_sha256}. The bundle would ship a model that is not the reviewed one.`,
  );
}

// Schema contract, read from the one canonical schema file both TS and Python use.
const schema = JSON.parse(readFileSync(join(APP_ROOT, "src/ai/ranker/schema.json"), "utf8"));
if (manifest.schema_version !== schema.schema_version) {
  fail(
    `manifest schema_version "${manifest.schema_version}" != schema.json ` +
      `"${schema.schema_version}" — the artifact was trained against a different feature layout`,
  );
}
if (manifest.total_dim !== schema.total_dim) {
  fail(`manifest total_dim ${manifest.total_dim} != schema.json total_dim ${schema.total_dim}`);
}

console.log(
  `[verify-build-model] OK — ${manifest.artifact} (${bytes.length} bytes, sha256 ${hash.slice(0, 16)}…) ` +
    `matches the manifest, the Phase 2D training manifest, and schema ${schema.schema_version}.`,
);
