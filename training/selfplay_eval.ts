import { makeGame } from "../src/game/game";
import { aiTakeTurn } from "../src/ai/agent";
import { makeRanker, type Ranker } from "../src/ai/ranker";
import { makeRng } from "../src/ai/rollout";
import { isTrickShot } from "../src/ai/classify";

// Self-play evaluation: the neural-ranked agent vs the classical-ranked agent
// playing full racks of 8-ball through the real engine. Both sides use the
// SAME candidate generator, verification and selection logic — the only
// difference is the ranker ordering candidates — so any win-rate gap is
// attributable to the ranking model.
//
// Reports: win rate, average shots-to-win, and how many of each side's potted
// shots were measured trick shots. Seeded and reproducible.
//
// Env: GAMES (default 20), MAX_SHOTS per game (default 120), SEED.

const GAMES = Number(process.env.GAMES ?? 20);
const MAX_SHOTS = Number(process.env.MAX_SHOTS ?? 120);
const SEED = Number(process.env.SEED ?? 42);

// Force "neural" regardless of the held-out gate (src/ai/ranker.ts's default
// may be classical if the gate failed) — self-play's whole purpose is
// comparing the two even when neural didn't earn default status.
const neural = makeRanker("neural");
const classical = makeRanker("classical");
if (neural.name !== "neural") {
  console.error(
    "weights.json failed shape validation — self-play would be classical vs classical. Train first.",
  );
  process.exit(1);
}

interface SideStats {
  wins: number;
  shotsToWin: number[];
  potsTotal: number;
  potsTrick: number;
}
const stats: Record<"neural" | "classical", SideStats> = {
  neural: { wins: 0, shotsToWin: [], potsTotal: 0, potsTrick: 0 },
  classical: { wins: 0, shotsToWin: [], potsTotal: 0, potsTrick: 0 },
};

const t0 = Date.now();
for (let game = 0; game < GAMES; game++) {
  const rng = makeRng(SEED + game * 7919);
  // Alternate which ranker breaks (player 0 breaks).
  const players: [Ranker, Ranker] =
    game % 2 === 0 ? [neural, classical] : [classical, neural];

  let { state, table } = makeGame();
  let shots = 0;
  while (state.winner === null && shots < MAX_SHOTS) {
    const ranker = players[state.turn];
    // eslint-disable-next-line no-await-in-loop
    const move = await aiTakeTurn(state, table, ranker, { rng });
    if (!move) break;
    const side = ranker.name as "neural" | "classical";
    const m = move.decision.selected?.measured;
    if (m?.targetPotted) {
      stats[side].potsTotal++;
      if (isTrickShot(m)) stats[side].potsTrick++;
    }
    state = move.report.next;
    shots++;
  }
  if (state.winner !== null) {
    const winner = players[state.winner].name as "neural" | "classical";
    stats[winner].wins++;
    stats[winner].shotsToWin.push(shots);
    console.log(
      `game ${game + 1}/${GAMES}: ${winner} wins in ${shots} shots ` +
        `(${((Date.now() - t0) / 1000).toFixed(0)}s elapsed)`,
    );
  } else {
    console.log(`game ${game + 1}/${GAMES}: no winner within ${MAX_SHOTS} shots`);
  }
}

const avg = (xs: number[]): string =>
  xs.length ? (xs.reduce((s, x) => s + x, 0) / xs.length).toFixed(1) : "n/a";

const decided = stats.neural.wins + stats.classical.wins;
console.log("\n=== self-play report ===");
console.log(`games: ${GAMES} (${decided} decided) · seed ${SEED}`);
for (const side of ["neural", "classical"] as const) {
  const s = stats[side];
  console.log(
    `${side}: ${s.wins}/${decided} wins (${((s.wins / (decided || 1)) * 100).toFixed(0)}%) · ` +
      `avg shots-to-win ${avg(s.shotsToWin)} · ` +
      `trick pots ${s.potsTrick}/${s.potsTotal} (${(
        (s.potsTrick / (s.potsTotal || 1)) * 100
      ).toFixed(0)}%)`,
  );
}
console.log(`wall time: ${((Date.now() - t0) / 1000).toFixed(0)}s`);
