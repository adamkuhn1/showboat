// Stage A closure item #6: worker_threads parallel generation.
//
// Safe because every unit is independently seeded (see resume.ts's module
// comment and gen_dataset_v3.ts's `processUnit`) — a worker processing unit
// X produces byte-identical output to the main thread processing unit X,
// regardless of which other units run concurrently in other workers. Each
// worker gets its OWN WASM instance (confirmed safe in
// docs/repair/showboat-ml/phase-2c/02-throughput-design.md §3: independent
// V8 isolates, no shared-state hazard). Shard writes are per-unit files, so
// concurrent workers never write the same file; manifest.jsonl and
// progress.json updates are serialized through the main thread (workers only
// report results via postMessage) to avoid concurrent-append races.
//
// Usage: RANKER_PROFILE=pilot RANKER_WORKERS=4 npx tsx training/ranker/phase2c/parallel.ts

import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createHash } from "node:crypto";

import { makeTable } from "../../../src/physics/table";
import { initPhysics } from "../../../src/physics/wasm-bridge";
import { checkLeakage, type Split } from "./split";
import { loadProgress, markUnitComplete, appendShardManifestEntry, listShardFiles, type VersionKey, type ShardManifestEntry } from "./resume";
import {
  PROFILES,
  DATASET_SCHEMA_VERSION,
  gitCommit,
  sha256File,
  processUnit,
  type Unit,
} from "./gen_dataset_v3";
import { SCHEMA_VERSION as FEATURE_SCHEMA_VERSION } from "../../../src/ai/ranker/encode";

const __dirname = dirname(fileURLToPath(import.meta.url));

interface WorkerTask {
  units: Unit[];
  profileName: string;
  masterSeed: number;
  versions: { generator: string; physics: string };
  runTimestamp: string;
  outDir: string;
}

interface WorkerResult {
  unitId: string;
  gamesPlayed: number;
  statesUsed: number;
  unitRowCount: number;
  logLine: string;
  manifestEntry: ShardManifestEntry;
}

async function runWorker() {
  const task = workerData as WorkerTask;
  const cfg = PROFILES[task.profileName];
  const wasmPath = join(__dirname, "../../../src/wasm/showboat_physics_bg.wasm");
  await initPhysics(readFileSync(wasmPath)); // own WASM instance, independent of the main thread's
  const table = makeTable();

  for (const unit of task.units) {
    const result = processUnit(unit, table, cfg, task.versions, task.runTimestamp, task.masterSeed, task.outDir);
    const msg: WorkerResult = { unitId: unit.id, ...result };
    parentPort!.postMessage(msg);
  }
  parentPort!.postMessage({ done: true });
}

async function runOrchestrator() {
  const profileName = process.env.RANKER_PROFILE ?? "smoke";
  const cfg = PROFILES[profileName];
  if (!cfg) throw new Error(`Unknown RANKER_PROFILE ${profileName}`);
  const MASTER_SEED = Number(process.env.RANKER_SEED ?? 20260804);
  const numWorkers = Number(process.env.RANKER_WORKERS ?? 4);
  const restartClean = process.argv.includes("--restart-clean");

  const wasmPath = join(__dirname, "../../../src/wasm/showboat_physics_bg.wasm");
  const versions = { generator: gitCommit(), physics: sha256File(wasmPath) };
  const runTimestamp = new Date().toISOString();
  const outDir = join(__dirname, "data", profileName);
  mkdirSync(outDir, { recursive: true });

  const versionKey: VersionKey = {
    dataset_schema_version: DATASET_SCHEMA_VERSION,
    feature_schema_version: FEATURE_SCHEMA_VERSION,
    generator_version: versions.generator,
    physics_version: versions.physics,
    master_seed: MASTER_SEED,
    profile: profileName,
  };
  let progress = loadProgress(outDir, versionKey, restartClean);
  const completedSet = new Set(progress.completed_units);

  const allUnits: Unit[] = [];
  for (let g = 0; g < cfg.games; g++) allUnits.push({ id: `trajectory-${g}`, kind: "trajectory", index: g });
  for (let c = 0; c < cfg.controlledStates; c++) allUnits.push({ id: `controlled-${c}`, kind: "controlled", index: c });
  const todo = allUnits.filter((u) => !completedSet.has(u.id));

  process.stderr.write(
    `[parallel] profile=${profileName} workers=${numWorkers} total_units=${allUnits.length} todo=${todo.length} already_complete=${completedSet.size}\n`,
  );
  if (todo.length === 0) {
    process.stderr.write("[parallel] nothing to do\n");
  } else {
    // Round-robin assignment (not contiguous chunks): self-play games and
    // controlled states have different per-unit cost, so round-robin keeps
    // workers more evenly loaded than splitting the list into contiguous
    // blocks (which would give one worker all the expensive units).
    const chunks: Unit[][] = Array.from({ length: numWorkers }, () => []);
    todo.forEach((u, i) => chunks[i % numWorkers].push(u));

    const t0 = Date.now();
    let doneWorkers = 0;
    let unitsCompleted = 0;
    await new Promise<void>((resolve, reject) => {
      const workers: Worker[] = [];
      for (let w = 0; w < numWorkers; w++) {
        if (chunks[w].length === 0) {
          doneWorkers++;
          continue;
        }
        const task: WorkerTask = { units: chunks[w], profileName, masterSeed: MASTER_SEED, versions, runTimestamp, outDir };
        const worker = new Worker(new URL(import.meta.url), { workerData: task });
        workers.push(worker);
        worker.on("message", (msg: WorkerResult | { done: true }) => {
          if ("done" in msg) {
            doneWorkers++;
            if (doneWorkers === numWorkers) resolve();
            return;
          }
          unitsCompleted++;
          // The main thread is the SOLE writer of manifest.jsonl/progress.json:
          // workers only compute results and report them via postMessage, they
          // never touch these files themselves (see processUnit's comment in
          // gen_dataset_v3.ts) — avoids a concurrent-append race that existed
          // here previously (found by review; workers used to write
          // manifest.jsonl directly, unserialized).
          appendShardManifestEntry(outDir, msg.manifestEntry);
          progress = markUnitComplete(outDir, progress, msg.unitId);
          const elapsedS = ((Date.now() - t0) / 1000).toFixed(1);
          process.stderr.write(
            `[parallel t=${elapsedS}s] (${unitsCompleted}/${todo.length}) ${msg.logLine}\n`,
          );
        });
        worker.on("error", reject);
      }
      if (doneWorkers === numWorkers) resolve(); // edge case: fewer todo units than workers, some got empty chunks
    });
    process.stderr.write(`[parallel] all ${todo.length} units complete in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
  }

  // Same final aggregation as gen_dataset_v3.ts's main() — read all shards
  // from disk, don't rely on in-memory state from any one worker.
  const shardFiles = listShardFiles(outDir).sort();
  const rows: Record<string, unknown>[] = [];
  for (const shardFile of shardFiles) {
    const content = readFileSync(join(outDir, shardFile), "utf8").trim();
    if (!content) continue;
    for (const line of content.split("\n")) rows.push(JSON.parse(line));
  }
  const leakage = checkLeakage(
    rows.map((r) => ({
      family_id: r.family_id as string,
      state_id: r.state_id as string,
      split: r.split as Split,
      state_origin: r.state_origin as "trajectory" | "controlled" | "golden",
    })),
  );
  if (!leakage.ok) {
    process.stderr.write(`[parallel] LEAKAGE CHECK FAILED:\n${leakage.issues.join("\n")}\n`);
    process.exit(1);
  }
  const shardManifestEntries = readFileSync(join(outDir, "manifest.jsonl"), "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l) as { shard_id: string; sha256: string });
  const hashByShardId = new Map(shardManifestEntries.map((e) => [e.shard_id, e.sha256]));
  const shardHashes = shardFiles.map((f) => hashByShardId.get(f.replace(/\.ndjson$/, ""))!);
  const datasetHash = createHash("sha256").update(shardHashes.join("\n")).digest("hex");

  const byKind: Record<string, number> = {};
  const bySplit: Record<string, number> = { train: 0, val: 0, test: 0 };
  const byOrigin: Record<string, number> = {};
  const distinctStates = new Set<string>();
  const distinctTrajectoryFamilies = new Set<string>();
  for (const r of rows) {
    const kind = r.candidate_kind as string;
    byKind[kind] = (byKind[kind] ?? 0) + 1;
    if (r.split) bySplit[r.split as string] = (bySplit[r.split as string] ?? 0) + 1;
    byOrigin[r.state_origin as string] = (byOrigin[r.state_origin as string] ?? 0) + 1;
    distinctStates.add(r.state_id as string);
    if (r.state_origin === "trajectory") distinctTrajectoryFamilies.add(r.family_id as string);
  }
  const datasetManifest = {
    dataset_schema_version: DATASET_SCHEMA_VERSION,
    feature_schema_version: FEATURE_SCHEMA_VERSION,
    profile: profileName,
    master_seed: MASTER_SEED,
    generator_version: versions.generator,
    physics_version: versions.physics,
    games_played: distinctTrajectoryFamilies.size,
    controlled_states_requested: cfg.controlledStates,
    states_used: distinctStates.size,
    n_rows: rows.length,
    n_shards: shardFiles.length,
    shard_hashes: shardHashes,
    dataset_hash: datasetHash,
    perturbations_per_candidate: cfg.perturbations,
    rows_by_candidate_kind: byKind,
    rows_by_split: bySplit,
    rows_by_state_origin: byOrigin,
    family_count_by_split: leakage.familyCountBySplit,
    state_count_by_split: leakage.stateCountBySplit,
    mean_legal_pot_rate:
      rows.reduce((a, r) => a + (r.raw_counts as { legal_pot: number }).legal_pot / (r.n_perturbations as number), 0) /
      (rows.length || 1),
    leakage_check: { ok: leakage.ok, issues: leakage.issues },
    generated_at: new Date().toISOString(),
    generated_with: `parallel.ts, ${numWorkers} workers`,
  };
  writeFileSync(join(outDir, "dataset_manifest.json"), JSON.stringify(datasetManifest, null, 2));
  process.stdout.write(`Wrote ${rows.length} rows (${shardFiles.length} shards) to ${outDir} using ${numWorkers} workers\n`);
  process.stdout.write(JSON.stringify(datasetManifest, null, 2) + "\n");
}

if (!isMainThread) {
  runWorker().catch((err) => {
    console.error(err);
    process.exit(1);
  });
} else {
  runOrchestrator().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
