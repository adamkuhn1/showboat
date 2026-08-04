// Standalone leakage audit against already-written shards — independent of
// the in-process check gen_dataset_v3.ts already runs before writing
// (defense in depth: this reads the actual files on disk, so it also
// catches corruption introduced after generation, e.g. a hand-edited shard
// or a partial resume that skipped the in-process check).
//
// Usage: RANKER_PROFILE=pilot npx tsx training/ranker/phase2c/split_check.ts

import { readFileSync, readdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { checkLeakage, type Split } from "./split";

const __dirname = dirname(fileURLToPath(import.meta.url));

function main() {
  const profileName = process.env.RANKER_PROFILE ?? "smoke";
  const dir = join(__dirname, "data", profileName);
  if (!existsSync(dir)) {
    process.stderr.write(`[split-check] no such directory: ${dir}\n`);
    process.exit(1);
  }

  const shardFiles = readdirSync(dir).filter((f) => f.endsWith(".ndjson") && !f.endsWith(".tmp"));
  const rows: { family_id: string; state_id: string; split: Split; state_origin: "trajectory" | "controlled" | "golden" }[] = [];

  for (const shardFile of shardFiles) {
    const lines = readFileSync(join(dir, shardFile), "utf8").trim().split("\n").filter(Boolean);
    for (const line of lines) {
      const row = JSON.parse(line);
      rows.push({
        family_id: row.family_id,
        state_id: row.state_id,
        split: row.split,
        state_origin: row.state_origin,
      });
    }
  }

  process.stderr.write(`[split-check] loaded ${rows.length} rows from ${shardFiles.length} shards in ${dir}\n`);
  const report = checkLeakage(rows);
  process.stderr.write(`[split-check] family counts by split: ${JSON.stringify(report.familyCountBySplit)}\n`);
  process.stderr.write(`[split-check] state counts by split: ${JSON.stringify(report.stateCountBySplit)}\n`);
  if (!report.ok) {
    for (const issue of report.issues) process.stderr.write(`[split-check] FAIL: ${issue}\n`);
    process.stderr.write(`[split-check] RESULT: FAIL (${report.issues.length} issues)\n`);
    process.exit(1);
  }
  process.stderr.write(`[split-check] RESULT: PASS\n`);
}

main();
