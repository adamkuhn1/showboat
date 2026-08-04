// Stage A closure item #5: resumable shard generation.
//
// Design: each work unit (one self-play game, or one controlled state) gets
// its own shard file and is independently seeded (see gen_dataset_v3.ts's
// `unitSeed`) — NOT a shared sequential RNG stream across units. This is
// what makes resume-with-skipping produce the same dataset as an
// uninterrupted run: a shared stream would require replaying the exact same
// draw sequence for every skipped unit to stay in sync, which is fragile
// and un-parallelizable. Independent per-unit seeding makes every unit's
// output a pure function of (masterSeed, unitId), independent of processing
// order or which other units ran — trivially resumable, and incidentally
// also trivially parallelizable across workers later if that's ever needed.
//
// progress.json tracks which units are durably complete. A unit is only
// added to it AFTER its shard file is atomically written (tmp -> rename)
// and progress.json itself is rewritten the same atomic way — so a crash or
// kill at any point leaves either the old progress.json (safe, that unit
// just gets regenerated) or the new one (safe, that unit's shard is already
// on disk). There is no window where progress.json can claim a unit is done
// without its shard actually existing.

import { readFileSync, writeFileSync, existsSync, renameSync, readdirSync, rmSync, appendFileSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";

/**
 * The ONE hashing function for shard content, used both when a unit's shard
 * is first written (gen_dataset_v3.ts's `processUnit`) and when verifying
 * an existing shard on resume (below). Operates on parsed row objects, not
 * raw file text, specifically so both call sites agree on the empty-unit
 * edge case (a unit with zero rows) by construction — two independently
 * written "stringify the content, minus generated_at" implementations
 * previously disagreed on how to hash an empty shard (one produced "", the
 * other "\n"), which would have made resume permanently mis-detect every
 * zero-row unit as corrupt and endlessly regenerate it.
 */
export function hashRows<T extends object>(rows: T[]): string {
  const content = rows.length ? rows.map((r) => JSON.stringify({ ...r, generated_at: undefined })).join("\n") + "\n" : "";
  return createHash("sha256").update(content).digest("hex");
}

export interface VersionKey {
  dataset_schema_version: string;
  feature_schema_version: string;
  generator_version: string;
  physics_version: string;
  master_seed: number;
  profile: string;
}

export interface Progress {
  version_key: VersionKey;
  completed_units: string[];
}

function versionKeyEquals(a: VersionKey, b: VersionKey): boolean {
  return (
    a.dataset_schema_version === b.dataset_schema_version &&
    a.feature_schema_version === b.feature_schema_version &&
    a.generator_version === b.generator_version &&
    a.physics_version === b.physics_version &&
    a.master_seed === b.master_seed &&
    a.profile === b.profile
  );
}

function progressPath(outDir: string): string {
  return join(outDir, "progress.json");
}

function shardManifestPath(outDir: string): string {
  return join(outDir, "manifest.jsonl");
}

export interface ShardManifestEntry {
  unit_id: string;
  shard_id: string;
  n_rows: number;
  sha256: string;
  completed_at: string;
}

function readShardManifest(outDir: string): ShardManifestEntry[] {
  const p = shardManifestPath(outDir);
  if (!existsSync(p)) return [];
  return readFileSync(p, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as ShardManifestEntry);
}

/**
 * Loads existing progress for this output dir, verifying it against the
 * CURRENT run's version key. Returns an empty (fresh-start) progress object
 * if none exists. Throws if a prior run's progress exists but its version
 * key doesn't match — resuming across a schema/physics/generator/seed
 * change would silently produce an inconsistent dataset, so this refuses
 * rather than guessing; the caller must pass `restartClean` to wipe and
 * start over instead.
 */
export function loadProgress(outDir: string, currentVersion: VersionKey, restartClean: boolean): Progress {
  if (restartClean && existsSync(outDir)) {
    rmSync(outDir, { recursive: true, force: true });
  }
  const path = progressPath(outDir);
  if (!existsSync(path)) {
    return { version_key: currentVersion, completed_units: [] };
  }
  const existing: Progress = JSON.parse(readFileSync(path, "utf8"));
  if (!versionKeyEquals(existing.version_key, currentVersion)) {
    throw new Error(
      `Incompatible prior run found in ${outDir}:\n` +
        `  existing: ${JSON.stringify(existing.version_key)}\n` +
        `  current:  ${JSON.stringify(currentVersion)}\n` +
        `Refusing to silently resume across a schema/physics/generator/seed/profile change ` +
        `(this would produce an internally inconsistent dataset). Re-run with --restart-clean ` +
        `to wipe this profile's output and start fresh.`,
    );
  }

  // Verify every unit progress.json claims is complete actually has a valid,
  // hash-matching shard on disk. A unit whose shard is missing or corrupt is
  // dropped from the completed set so it gets regenerated — this is what
  // makes resume safe against a kill mid-shard-write (the atomic rename
  // means a killed write never leaves a corrupt *final* shard file, but this
  // check also protects against disk corruption or manual tampering).
  const manifestEntries = new Map(readShardManifest(outDir).map((e) => [e.unit_id, e]));
  const verifiedCompleted = existing.completed_units.filter((unitId) => {
    const entry = manifestEntries.get(unitId);
    if (!entry) return false;
    const shardPath = join(outDir, `${entry.shard_id}.ndjson`);
    if (!existsSync(shardPath)) return false;
    const content = readFileSync(shardPath, "utf8").trim();
    const rows = content ? content.split("\n").filter(Boolean).map((line) => JSON.parse(line)) : [];
    return hashRows(rows) === entry.sha256;
  });

  const dropped = existing.completed_units.length - verifiedCompleted.length;
  if (dropped > 0) {
    process.stderr.write(`[resume] ${dropped} previously-completed unit(s) failed shard verification — will regenerate\n`);
  }

  return { version_key: currentVersion, completed_units: verifiedCompleted };
}

/**
 * Atomically records that `unitId`'s shard (already written + hashed by the
 * caller) is durably complete. Rewrites the whole progress.json via
 * tmp -> rename rather than in-place mutation, so a crash mid-write leaves
 * either the fully-old or fully-new file, never a truncated/corrupt one.
 */
export function markUnitComplete(outDir: string, progress: Progress, unitId: string): Progress {
  const updated: Progress = { version_key: progress.version_key, completed_units: [...progress.completed_units, unitId] };
  const path = progressPath(outDir);
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, JSON.stringify(updated, null, 2));
  renameSync(tmp, path);
  return updated;
}

export function appendShardManifestEntry(outDir: string, entry: ShardManifestEntry): void {
  appendFileSync(shardManifestPath(outDir), JSON.stringify(entry) + "\n");
}

export function listShardFiles(outDir: string): string[] {
  if (!existsSync(outDir)) return [];
  return readdirSync(outDir).filter((f) => f.endsWith(".ndjson") && !f.endsWith(".tmp"));
}
