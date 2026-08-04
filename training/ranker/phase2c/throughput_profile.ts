// Stage A closure item #6: per-stage throughput profiling. Runs a small
// representative sample (not full-scale) and reports where wall-clock time
// actually goes: self-play shot-selection search, candidate generation,
// perturbation simulation calls, serialization, and hashing — the
// breakdown the original throughput audit (02-throughput-design.md) did not
// separate (it measured the labeling loop as one lump).
//
// Usage: npx tsx training/ranker/phase2c/throughput_profile.ts

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createHash } from "node:crypto";

import { makeTable } from "../../../src/physics/table";
import { type GameState } from "../../../src/game/state";
import { cloneState, takeShot } from "../../../src/game/game";
import { legalTargets, planTurn } from "../../../src/ai/turn";
import { generateCandidates } from "../../../src/ai/candidates";
import { initPhysics, simulateShotWasm } from "../../../src/physics/wasm-bridge";
import { isLegalPot } from "../../../src/ai/shotSearch";
import { encodeRow } from "../../../src/ai/ranker/encode";
import { playSelfPlayGame, SELF_PLAY_SEARCH_CONFIG } from "./selfplay";
import { randomControlledState } from "./controlled";
import { makeRng } from "./gen_dataset_v3";

const __dirname = dirname(fileURLToPath(import.meta.url));

const timers: Record<string, number> = {};
function time<T>(bucket: string, fn: () => T): T {
  const t0 = performance.now();
  const result = fn();
  timers[bucket] = (timers[bucket] ?? 0) + (performance.now() - t0);
  return result;
}

async function main() {
  const initStart = performance.now();
  const wasmPath = join(__dirname, "../../../src/wasm/showboat_physics_bg.wasm");
  await initPhysics(readFileSync(wasmPath));
  timers.wasm_init = performance.now() - initStart;

  const table = makeTable();
  const rng = makeRng(999);

  // Self-play: separately time the game's OWN shot-selection search vs.
  // everything else (state bookkeeping, cue placement).
  for (let g = 0; g < 3; g++) {
    time("selfplay_total", () => playSelfPlayGame(table, rng, { maxShots: 12, searchConfig: SELF_PLAY_SEARCH_CONFIG }));
  }

  // Controlled-state generation cost (cheap, no search).
  const controlledStates = [];
  for (let c = 0; c < 10; c++) {
    controlledStates.push(time("controlled_state_gen", () => randomControlledState(table, rng)));
  }

  // Labeling loop, broken into its real sub-costs: candidate generation,
  // perturbation simulation, serialization, hashing.
  let nCandidates = 0;
  let nTrials = 0;
  for (const balls of controlledStates) {
    const state: GameState = { balls, turn: 0, groups: { 0: null, 1: null }, ballInHand: false, winner: null, broken: true, shotCount: 1 };
    const targets = legalTargets(state, state.turn);
    const candidates = time("candidate_generation", () => generateCandidates(state.balls, table, targets));
    nCandidates += candidates.length;
    for (const candidate of candidates) {
      time("feature_encoding", () => encodeRow(state.balls, table, candidate));
      for (let p = 0; p < 8; p++) {
        const trialPre = time("clone_state", () => cloneState(state));
        const jittered = { ...candidate.action, phi: candidate.action.phi + (rng() * 2 - 1) * 0.01, power: candidate.action.power };
        const report = time("simulation", () => takeShot(trialPre, table, jittered, simulateShotWasm));
        time("legality_check", () => isLegalPot(report.sim, candidate));
        nTrials++;
      }
    }
  }

  // Serialization + hashing cost on a representative row set.
  const sampleRows = Array.from({ length: 500 }, (_, i) => ({
    example_id: `x${i}`,
    features: Array.from({ length: 68 }, () => Math.random()),
    raw_counts: { legal_pot: 1, any_pot: 2, foul: 3, scratch: 0, own_balls_pocketed_total: 0, opponent_balls_pocketed_total: 0, terminal_win: 0, terminal_loss: 0 },
    generated_at: new Date().toISOString(),
  }));
  const serialized = time("serialization", () => sampleRows.map((r) => JSON.stringify(r)).join("\n"));
  time("hashing", () => createHash("sha256").update(serialized).digest("hex"));

  // Self-play's own shot-selection cost, isolated (re-run with search vs. without).
  const rng2 = makeRng(1000);
  const t0 = performance.now();
  playSelfPlayGame(table, rng2, { maxShots: 12, searchConfig: SELF_PLAY_SEARCH_CONFIG });
  const withSearch = performance.now() - t0;
  // planTurn call count proxy: measure search time directly via a few manual calls.
  const rng3 = makeRng(1000);
  const balls0 = randomControlledState(table, rng3);
  const state0: GameState = { balls: balls0, turn: 0, groups: { 0: null, 1: null }, ballInHand: false, winner: null, broken: true, shotCount: 1 };
  const searchStart = performance.now();
  for (let i = 0; i < 5; i++) planTurn(state0, table, SELF_PLAY_SEARCH_CONFIG);
  const searchTime5 = performance.now() - searchStart;

  const total = Object.values(timers).reduce((a, b) => a + b, 0);
  console.log(`\n=== Throughput profile (n_candidates=${nCandidates}, n_trials=${nTrials}) ===`);
  console.log(`wasm_init:           ${timers.wasm_init.toFixed(1)}ms`);
  console.log(`selfplay_total (3 games, incl. search + game logic): ${timers.selfplay_total.toFixed(1)}ms`);
  console.log(`controlled_state_gen (10 states): ${timers.controlled_state_gen.toFixed(1)}ms`);
  console.log(`candidate_generation: ${timers.candidate_generation.toFixed(1)}ms`);
  console.log(`feature_encoding:     ${timers.feature_encoding.toFixed(1)}ms`);
  console.log(`clone_state:          ${timers.clone_state.toFixed(1)}ms`);
  console.log(`simulation (${nTrials} calls): ${timers.simulation.toFixed(1)}ms (${(timers.simulation / nTrials).toFixed(2)}ms/call, ${(1000 / (timers.simulation / nTrials)).toFixed(1)} calls/sec)`);
  console.log(`legality_check:       ${timers.legality_check.toFixed(1)}ms`);
  console.log(`serialization (500 rows): ${timers.serialization.toFixed(2)}ms`);
  console.log(`hashing (500 rows):   ${timers.hashing.toFixed(2)}ms`);
  console.log(`\nlabeling-loop share of total non-init time: ${(((timers.simulation + timers.candidate_generation + timers.feature_encoding + timers.clone_state + timers.legality_check) / total) * 100).toFixed(1)}%`);
  console.log(`self-play share of total non-init time: ${((timers.selfplay_total / total) * 100).toFixed(1)}%`);
  console.log(`\nsingle-decision search cost (5x planTurn @ ${SELF_PLAY_SEARCH_CONFIG.simulations} sims): ${searchTime5.toFixed(1)}ms total, ${(searchTime5 / 5).toFixed(1)}ms/decision`);
  console.log(`one full self-play game (12 shots, incl. search): ${withSearch.toFixed(1)}ms`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
