// Trick categories attempted and completed, across full racks.
//
// The pre-registered three-arm evaluation (`eval/trickOnlyEval.ts`) is the
// authority on policy comparisons and is unchanged by this sprint. This is a
// narrower, cheaper question the sprint report has to answer directly: over
// complete games, which trick kinds does the opponent actually try, and how
// often does the real simulation say it worked?
//
// One arm — the shipped opponent, neural prior + trick-only selection. Every
// outcome comes from executing the chosen shot through the real `takeShot` and
// reading the real `applyShotRules` result; nothing is inferred from the
// search's own estimate of itself.
//
//   npx tsx qa/rack-tricks.ts [--games 12] [--seed 20260807]

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeGame, takeShot } from "../src/game/game";
import { CUE_ID } from "../src/game/rack";
import type { GameState } from "../src/game/state";
import { initPhysics, simulateShotWasm } from "../src/physics/wasm-bridge";
import { getBrain } from "../src/ai/brain";
import { NeuralCandidateEvaluator } from "../src/ai/neural/evaluator";
import { makeFileFetch } from "../src/ai/neural/fileFetch";
import { defaultConfig } from "../src/ai/shotSearch";
import { Deadline } from "../src/ai/deadline";
import { placeCueBall } from "../src/game/game";

const APP_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const arg = (n: string, d: number) => {
  const i = process.argv.indexOf(`--${n}`);
  return i >= 0 && process.argv[i + 1] ? Number(process.argv[i + 1]) : d;
};
const GAMES = arg("games", 12);
const SEED = arg("seed", 20260807);
const MAX_SHOTS = 120;

const KINDS = ["bank", "double-bank", "combo", "rail-combo", "safety-kick"] as const;

const main = async () => {
  await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
  const evaluator = new NeuralCandidateEvaluator("model/ranker");
  const st = await evaluator.load(makeFileFetch(join(APP_ROOT, "public")));
  if (st.status !== "ready") throw new Error(`model not ready: ${JSON.stringify(st)}`);

  const attempted: Record<string, number> = {};
  const potted: Record<string, number> = {};
  const fouled: Record<string, number> = {};
  const rungs: Record<string, number> = {};
  let turns = 0;
  let directsSelected = 0;
  let racksFinished = 0;
  let aiWins = 0;

  // A deterministic pseudo-random break per game, so the racks differ but the
  // run reproduces from the seed.
  let s = SEED >>> 0;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);

  for (let g = 0; g < GAMES; g++) {
    const game = makeGame();
    let state: GameState = game.state;
    // Break with the human player, then let the opponent play both sides so a
    // full rack is actually cleared rather than stalling on a human that never
    // shoots.
    state = takeShot(
      state,
      game.table,
      { phi: (rnd() - 0.5) * 0.12, power: 0.85 + rnd() * 0.15, sideSpin: 0, topSpin: 0 },
      simulateShotWasm,
    ).next;

    for (let shot = 0; shot < MAX_SHOTS && state.winner === null; shot++) {
      if (state.ballInHand !== false) state = placeCueBall(state, -game.table.length / 4, 0);
      const brain = getBrain(true, evaluator);
      const decision = await brain.plan(
        state,
        game.table,
        state.turn,
        { ...defaultConfig, seed: (SEED + shot * 7919 + g * 104729) >>> 0 },
        Deadline.none(),
      );
      const sel = decision.decision.selected;
      if (!decision.shot || !sel) break;
      turns++;
      rungs[sel.rung] = (rungs[sel.rung] ?? 0) + 1;
      if ((sel.kind as string) === "direct") directsSelected++;
      attempted[sel.kind] = (attempted[sel.kind] ?? 0) + 1;

      const report = takeShot(state, game.table, decision.shot.action, simulateShotWasm);
      const before = state.balls.filter((b) => b.pocketed && b.id !== CUE_ID).length;
      const after = report.next.balls.filter((b) => b.pocketed && b.id !== CUE_ID).length;
      if (after > before) potted[sel.kind] = (potted[sel.kind] ?? 0) + 1;
      if (report.outcome.foul) fouled[sel.kind] = (fouled[sel.kind] ?? 0) + 1;
      state = report.next;
    }
    if (state.winner !== null) {
      racksFinished++;
      if (state.winner === 1) aiWins++;
    }
  }

  const pct = (a: number, b: number) => (b === 0 ? "—" : `${((100 * a) / b).toFixed(1)}%`);
  console.log(`\ntrick categories over ${GAMES} racks (seed ${SEED}) — ${turns} opponent turns\n`);
  console.log(["kind", "attempted", "potted", "pot rate", "fouled"].map((h) => h.padStart(14)).join(""));
  for (const k of KINDS) {
    const a = attempted[k] ?? 0;
    if (a === 0) {
      console.log([k, "0", "—", "—", "—"].map((x) => x.padStart(14)).join(""));
      continue;
    }
    console.log(
      [k, String(a), String(potted[k] ?? 0), pct(potted[k] ?? 0, a), String(fouled[k] ?? 0)]
        .map((x) => x.padStart(14))
        .join(""),
    );
  }
  const attemptedTotal = Object.values(attempted).reduce((x, y) => x + y, 0);
  const pottedTotal = Object.values(potted).reduce((x, y) => x + y, 0);
  console.log(
    ["TOTAL", String(attemptedTotal), String(pottedTotal), pct(pottedTotal, attemptedTotal), ""]
      .map((x) => x.padStart(14))
      .join(""),
  );
  console.log(`\nselection rungs: ${JSON.stringify(rungs)}`);
  console.log(`directs selected: ${directsSelected}   (the trick-only guarantee: must be 0)`);
  console.log(`racks played to a winner: ${racksFinished}/${GAMES}, opponent won ${aiWins}`);
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
