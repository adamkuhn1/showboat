// Headless authoritative-physics dataset generator for the Showboat candidate
// ranker (Phase 2A feasibility spike).
//
// Run with: npm run ranker:gen-dataset --workspace=apps/showboat
//
// Loads the EXACT compiled WASM artifact the browser ships
// (src/wasm/showboat_physics_bg.wasm) via initPhysics()'s Node-compatible
// path (see physics/wasm-bridge.ts), and reuses generateCandidates (candidates.ts),
// applyShotRules (game/rules.ts), and the ranker feature encoder (ai/ranker/encode.ts)
// completely unchanged from what the browser build uses. See
// docs/repair/showboat-ml/ARCHITECTURE_DECISION.md for why this path was
// chosen over a new Rust CLI or a Python/PyO3 bridge.
//
// Known Phase 2A simplification (documented, not hidden): there is no
// general legal-mid-game-state sampler in the repo yet (confirmed absent by
// docs/repair/showboat-ml/02-physics-data-path.md §8) — building one is
// Phase 2C's job. This script instead perturbs the standard rack by removing
// a random subset of object balls (simulating "some balls already potted")
// and jittering remaining ball positions with overlap rejection, then treats
// each resulting board as an open-table, single-shot decision (both groups
// unassigned) when applying rules. That is a real simplification of full
// game history, not a fabrication of physics or outcomes — every recorded
// label still comes from actually running the authoritative simulator.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { makeTable } from "../../src/physics/table";
import { makeBall, cloneBall, type Ball } from "../../src/physics/ball";
import { BALL_RADIUS } from "../../src/physics/constants";
import { CUE_ID, EIGHT_ID, SOLIDS, STRIPES } from "../../src/game/rack";
import { type GameState } from "../../src/game/state";
import { applyShotRules } from "../../src/game/rules";
import { generateCandidates, type Candidate } from "../../src/ai/candidates";
import { initPhysics, simulateShotWasm } from "../../src/physics/wasm-bridge";
import { encodeRow, TOTAL_DIM, SCHEMA_VERSION } from "../../src/ai/ranker/encode";
import { isLegalPot } from "../../src/ai/shotSearch";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ---- deterministic seeded RNG (mulberry32) so runs are reproducible --------
function makeRng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SEED = 20260803;
const N_STATES = Number(process.env.RANKER_N_STATES ?? 60);
const PERTURBATIONS_PER_CANDIDATE = Number(process.env.RANKER_N_PERTURB ?? 8);
const AIM_JITTER_RAD = 0.01; // ~0.57 degrees of aim noise
const POWER_JITTER = 0.03; // fraction of [0,1] power range

const table = makeTable();
const hx = table.length / 2 - BALL_RADIUS - 0.01;
const hy = table.width / 2 - BALL_RADIUS - 0.01;

/** A random legal-ish open-table board: random subset of balls remain, non-overlapping. */
function randomState(rng: () => number): Ball[] {
  const allIds = [...SOLIDS, EIGHT_ID, ...STRIPES];
  // Keep a random 60-100% of object balls on the table (simulating "some already potted").
  const keepFraction = 0.6 + rng() * 0.4;
  const kept = allIds.filter(() => rng() < keepFraction);
  // Always keep at least 2 non-eight balls so candidates exist.
  const nonEight = kept.filter((id) => id !== EIGHT_ID);
  if (nonEight.length < 2) {
    return randomState(rng); // resample rather than emit a degenerate state
  }
  const ids = [CUE_ID, ...kept];
  const balls: Ball[] = [];
  const MIN_DIST = 2 * BALL_RADIUS + 0.01;
  for (const id of ids) {
    let placed = false;
    for (let attempt = 0; attempt < 200 && !placed; attempt++) {
      const x = (rng() * 2 - 1) * hx;
      const y = (rng() * 2 - 1) * hy;
      if (balls.every((b) => Math.hypot(b.pos.x - x, b.pos.y - y) >= MIN_DIST)) {
        balls.push(makeBall(id, x, y));
        placed = true;
      }
    }
    if (!placed) return randomState(rng); // resample on a packed failure
  }
  return balls;
}

/**
 * Wrap a raw board as an open-table, single-shot GameState for applyShotRules.
 * Represents "neither player has claimed a group yet" — valid because
 * `applyShotRules` skips its first-contact-group-legality branch entirely
 * when `groups[shooter] === null` (see rules.ts's `pre.broken && shooterGroup
 * !== null` guard), so an open-table wrapper produces real, rules-correct
 * foul/scratch/pot labels without needing a full multi-shot game history.
 * Phase 2C replaces this with states sampled from actual played-out games.
 */
function asOpenTableState(balls: Ball[]): GameState {
  return {
    balls,
    turn: 0,
    groups: { 0: null, 1: null },
    ballInHand: false,
    winner: null,
    broken: true,
    shotCount: 1,
  };
}

interface DatasetRow {
  schema_version: string;
  state_id: number;
  candidate_kind: Candidate["kind"];
  features: number[];
  label: number; // fraction of perturbed executions that legally potted the target
  n_perturbations: number;
}

async function main() {
  const wasmPath = join(__dirname, "../../src/wasm/showboat_physics_bg.wasm");
  await initPhysics(readFileSync(wasmPath));

  const rng = makeRng(SEED);
  const rows: DatasetRow[] = [];
  let statesUsed = 0;

  for (let s = 0; s < N_STATES; s++) {
    const board = randomState(rng);
    const onTable = board.filter((b) => b.id !== CUE_ID);
    // Open table: every remaining object ball (not the 8, unless it's the only one) is a legal target.
    const nonEight = onTable.filter((b) => b.id !== EIGHT_ID);
    const targets = (nonEight.length > 0 ? nonEight : onTable).map((b) => b.id);
    if (targets.length === 0) continue;

    const candidates = generateCandidates(board, table, targets);
    if (candidates.length === 0) continue;
    statesUsed++;
    process.stderr.write(`state ${s}: ${board.length} balls, ${candidates.length} candidates\n`);

    for (const [ci, candidate] of candidates.entries()) {
      const candStart = Date.now();
      let successes = 0;
      for (let p = 0; p < PERTURBATIONS_PER_CANDIDATE; p++) {
        const simStart = Date.now();
        const trial = board.map(cloneBall);
        const jittered = {
          ...candidate.action,
          phi: candidate.action.phi + (rng() * 2 - 1) * AIM_JITTER_RAD,
          power: Math.min(1, Math.max(0.05, candidate.action.power + (rng() * 2 - 1) * POWER_JITTER)),
        };
        const pre = asOpenTableState(trial.map(cloneBall));
        const result = simulateShotWasm(trial, jittered);
        const simMs = Date.now() - simStart;
        if (simMs > 500) {
          process.stderr.write(
            `  [slow] state ${s} candidate ${ci}/${candidates.length} (${candidate.kind}) perturbation ${p}: ${simMs}ms\n`,
          );
        }
        const g = asOpenTableState(trial);
        const outcome = applyShotRules(g, pre, result);
        // Same legality check shotSearch.ts's seeding loop uses (isLegalPot),
        // plus applyShotRules' broader foul detection (rail-after-contact,
        // etc.) that isLegalPot alone doesn't cover.
        const legalPot = !outcome.foul && isLegalPot(result, candidate);
        if (legalPot) successes++;
      }
      const candMs = Date.now() - candStart;
      if (candMs > 2000) {
        process.stderr.write(`  [slow candidate] state ${s} candidate ${ci}/${candidates.length}: ${candMs}ms total\n`);
      }
      const label = successes / PERTURBATIONS_PER_CANDIDATE;
      const features = Array.from(encodeRow(board, table, candidate));
      if (features.length !== TOTAL_DIM) {
        throw new Error(`encodeRow produced ${features.length} features, expected ${TOTAL_DIM}`);
      }
      rows.push({
        schema_version: SCHEMA_VERSION,
        state_id: s,
        candidate_kind: candidate.kind,
        features,
        label,
        n_perturbations: PERTURBATIONS_PER_CANDIDATE,
      });
    }
    if ((s + 1) % 10 === 0) {
      process.stderr.write(`... ${s + 1}/${N_STATES} states, ${rows.length} rows so far\n`);
    }
  }

  const outDir = join(__dirname, "dataset");
  mkdirSync(outDir, { recursive: true });
  const outPath = join(outDir, `${SCHEMA_VERSION}.ndjson`);
  writeFileSync(outPath, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");

  const byKind: Record<string, number> = {};
  for (const r of rows) byKind[r.candidate_kind] = (byKind[r.candidate_kind] ?? 0) + 1;
  const meta = {
    schema_version: SCHEMA_VERSION,
    seed: SEED,
    n_states_requested: N_STATES,
    n_states_used: statesUsed,
    n_rows: rows.length,
    perturbations_per_candidate: PERTURBATIONS_PER_CANDIDATE,
    aim_jitter_rad: AIM_JITTER_RAD,
    power_jitter: POWER_JITTER,
    rows_by_candidate_kind: byKind,
    mean_label: rows.reduce((a, r) => a + r.label, 0) / (rows.length || 1),
    generated_at: new Date().toISOString(),
  };
  writeFileSync(join(outDir, `${SCHEMA_VERSION}.meta.json`), JSON.stringify(meta, null, 2));

  console.log(`Wrote ${rows.length} rows from ${statesUsed} states to ${outPath}`);
  console.log(JSON.stringify(meta, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
