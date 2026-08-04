// Phase 2C data-quality validator. Reads a generated dataset directory
// (shards + manifest.jsonl + dataset_manifest.json) and checks every
// acceptance-criteria failure mode from the task brief that is observable
// from the dataset-record schema itself. See DATASET_DESIGN.md's
// "Data-quality validation" section for the full checklist and which items
// are instead covered by static/source-level checks (outcome-leakage is
// enforced structurally in encode.ts + a dedicated vitest, not here, since
// a dataset row has no raw post-shot state to check against).
//
// Usage: RANKER_PROFILE=smoke npx tsx training/ranker/phase2c/validate.ts
// Exits 1 and prints every failure found (does not stop at the first one).

import { readFileSync, readdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { TOTAL_DIM, SCHEMA_VERSION as FEATURE_SCHEMA_VERSION } from "../../../src/ai/ranker/encode";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATASET_SCHEMA_VERSION = "showboat-dataset-v1";
const KNOWN_KINDS = new Set(["direct", "bank", "double-bank", "combo", "rail-combo"]);
const KNOWN_SPLITS = new Set(["train", "val", "test"]);
const KNOWN_ORIGINS = new Set(["trajectory", "controlled", "golden"]);

interface DatasetRow {
  dataset_schema_version: string;
  feature_schema_version: string;
  example_id: string;
  family_id: string;
  state_id: string;
  state_origin: string;
  split: string | null;
  candidate_id: number;
  candidate_kind: string;
  candidate_target: number;
  candidate_pot_id: number;
  candidate_pocket: string;
  features: number[];
  n_perturbations: number;
  raw_counts: {
    legal_pot: number;
    any_pot: number;
    foul: number;
    scratch: number;
    own_balls_pocketed_total: number;
    opponent_balls_pocketed_total: number;
    terminal_win: number;
    terminal_loss: number;
  };
}

function fail(errors: string[], msg: string) {
  errors.push(msg);
}

function validateDir(dir: string): { errors: string[]; warnings: string[]; nRows: number } {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!existsSync(dir)) {
    return { errors: [`dataset directory does not exist: ${dir}`], warnings, nRows: 0 };
  }

  const manifestPath = join(dir, "dataset_manifest.json");
  if (!existsSync(manifestPath)) fail(errors, `missing dataset_manifest.json in ${dir}`);

  const shardManifestPath = join(dir, "manifest.jsonl");
  const shardIds = new Set<string>();
  if (existsSync(shardManifestPath)) {
    const lines = readFileSync(shardManifestPath, "utf8").trim().split("\n").filter(Boolean);
    for (const line of lines) {
      let entry: { shard_id: string; sha256: string; n_rows: number };
      try {
        entry = JSON.parse(line);
      } catch {
        fail(errors, `corrupt manifest.jsonl line (invalid JSON): ${line.slice(0, 80)}`);
        continue;
      }
      if (shardIds.has(entry.shard_id)) fail(errors, `duplicate shard_id in manifest.jsonl: ${entry.shard_id}`);
      shardIds.add(entry.shard_id);
      const shardPath = join(dir, `${entry.shard_id}.ndjson`);
      if (!existsSync(shardPath)) fail(errors, `manifest references missing shard file: ${shardPath}`);
    }
  } else {
    warnings.push(`no manifest.jsonl found in ${dir} (skipping shard-completeness cross-check)`);
  }

  const shardFiles = readdirSync(dir).filter((f) => f.endsWith(".ndjson") && !f.endsWith(".tmp"));
  const tmpFiles = readdirSync(dir).filter((f) => f.endsWith(".tmp"));
  for (const t of tmpFiles) warnings.push(`leftover .tmp shard (incomplete/interrupted write): ${t}`);

  const exampleIds = new Set<string>();
  const familySplit = new Map<string, string>();
  const stateFamily = new Map<string, string>();
  let nRows = 0;

  for (const shardFile of shardFiles.sort()) {
    const shardPath = join(dir, shardFile);
    const lines = readFileSync(shardPath, "utf8").trim().split("\n").filter(Boolean);
    lines.forEach((line, lineIdx) => {
      let row: DatasetRow;
      try {
        row = JSON.parse(line);
      } catch {
        fail(errors, `${shardFile}:${lineIdx + 1}: corrupt JSON`);
        return;
      }
      nRows++;
      const where = `${shardFile}:${lineIdx + 1} (example_id=${row.example_id})`;

      if (row.dataset_schema_version !== DATASET_SCHEMA_VERSION) {
        fail(errors, `${where}: unknown dataset_schema_version ${row.dataset_schema_version}`);
      }
      if (row.feature_schema_version !== FEATURE_SCHEMA_VERSION) {
        fail(errors, `${where}: feature_schema_version ${row.feature_schema_version} != current encoder's ${FEATURE_SCHEMA_VERSION}`);
      }
      if (!KNOWN_KINDS.has(row.candidate_kind)) fail(errors, `${where}: unknown candidate_kind ${row.candidate_kind}`);
      if (!KNOWN_ORIGINS.has(row.state_origin)) fail(errors, `${where}: unknown state_origin ${row.state_origin}`);
      if (row.split !== null && !KNOWN_SPLITS.has(row.split)) fail(errors, `${where}: invalid split ${row.split}`);
      if (row.state_origin === "golden" && row.split !== null) {
        fail(errors, `${where}: golden-origin row was assigned split ${row.split} (golden must never be split-assigned)`);
      }
      if (row.candidate_pot_id === undefined || row.candidate_pot_id === null) {
        fail(errors, `${where}: missing candidate_pot_id`);
      } else if (row.candidate_pot_id < 0 || row.candidate_pot_id > 15) {
        fail(errors, `${where}: candidate_pot_id ${row.candidate_pot_id} out of valid ball-id range [0,15]`);
      }
      if (row.candidate_target < 0 || row.candidate_target > 15) {
        fail(errors, `${where}: candidate_target ${row.candidate_target} out of valid ball-id range [0,15]`);
      }

      if (!Array.isArray(row.features) || row.features.length !== TOTAL_DIM) {
        fail(errors, `${where}: features length ${row.features?.length} != expected TOTAL_DIM ${TOTAL_DIM}`);
      } else {
        for (let i = 0; i < row.features.length; i++) {
          const v = row.features[i];
          if (Number.isNaN(v)) fail(errors, `${where}: features[${i}] is NaN`);
          else if (!Number.isFinite(v)) fail(errors, `${where}: features[${i}] is Inf`);
          else if (Math.abs(v) > 3) fail(errors, `${where}: features[${i}]=${v} outside plausible normalized range [-3,3]`);
        }
      }

      const rc = row.raw_counts;
      if (!rc || row.n_perturbations === undefined) {
        fail(errors, `${where}: missing raw_counts or n_perturbations`);
      } else {
        const n = row.n_perturbations;
        if (n <= 0) fail(errors, `${where}: n_perturbations must be > 0, got ${n}`);
        for (const [k, v] of Object.entries(rc)) {
          if (v < 0) fail(errors, `${where}: raw_counts.${k}=${v} is negative`);
          if (k !== "own_balls_pocketed_total" && k !== "opponent_balls_pocketed_total" && v > n) {
            fail(errors, `${where}: raw_counts.${k}=${v} exceeds n_perturbations=${n}`);
          }
        }
        if (rc.legal_pot > rc.any_pot) {
          fail(errors, `${where}: legal_pot=${rc.legal_pot} > any_pot=${rc.any_pot} (legal pots are a subset of any-pot events)`);
        }
      }

      if (exampleIds.has(row.example_id)) fail(errors, `${where}: duplicate example_id`);
      exampleIds.add(row.example_id);

      if (row.state_origin !== "golden") {
        const priorSplit = familySplit.get(row.family_id);
        if (priorSplit === undefined) familySplit.set(row.family_id, row.split ?? "null");
        else if (priorSplit !== (row.split ?? "null")) {
          fail(errors, `${where}: family_id ${row.family_id} spans multiple splits (${priorSplit} and ${row.split})`);
        }
        const priorFamily = stateFamily.get(row.state_id);
        if (priorFamily === undefined) stateFamily.set(row.state_id, row.family_id);
        else if (priorFamily !== row.family_id) {
          fail(errors, `${where}: state_id ${row.state_id} claimed by multiple families (${priorFamily} and ${row.family_id})`);
        }
      }
    });
  }

  return { errors, warnings, nRows };
}

function main() {
  const profileName = process.env.RANKER_PROFILE ?? "smoke";
  const dir = join(__dirname, "data", profileName);
  process.stderr.write(`[validate] checking ${dir}\n`);
  const { errors, warnings, nRows } = validateDir(dir);

  for (const w of warnings) process.stderr.write(`[validate] WARNING: ${w}\n`);
  for (const e of errors) process.stderr.write(`[validate] FAIL: ${e}\n`);

  process.stderr.write(
    `[validate] checked ${nRows} rows, ${warnings.length} warnings, ${errors.length} errors\n`,
  );
  if (errors.length > 0) {
    process.stderr.write(`[validate] RESULT: FAIL\n`);
    process.exit(1);
  }
  process.stderr.write(`[validate] RESULT: PASS\n`);
}

main();
