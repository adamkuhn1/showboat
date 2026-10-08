import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
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
//
// Writes training/selfplay-<GAMES>.json (counts + a Wilson 95% interval on the
// neural side's win rate over decided games); training/bootstrap.ts folds it
// into training/ci.json.

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
const wallSeconds = (Date.now() - t0) / 1000;
console.log(`wall time: ${wallSeconds.toFixed(0)}s`);

// Wilson score interval for a binomial proportion (z = 1.96 -> 95%).
const wilson = (k: number, n: number, z = 1.96): [number, number] => {
  if (n === 0) return [0, 1];
  const p = k / n;
  const z2 = z * z;
  const centre = (p + z2 / (2 * n)) / (1 + z2 / n);
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / (1 + z2 / n);
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
};
const ci = wilson(stats.neural.wins, decided);
console.log(
  `neural win rate ${stats.neural.wins}/${decided} · Wilson 95% CI ` +
    `${(ci[0] * 100).toFixed(1)}%–${(ci[1] * 100).toFixed(1)}%`,
);

const here = dirname(fileURLToPath(import.meta.url));
const OUT = join(here, `selfplay-${GAMES}.json`);
writeFileSync(
  OUT,
  JSON.stringify(
    {
      games: GAMES,
      decided,
      seed: SEED,
      maxShots: MAX_SHOTS,
      neuralWins: stats.neural.wins,
      classicalWins: stats.classical.wins,
      neuralWinRate: stats.neural.wins / (decided || 1),
      neuralWinRateWilson95: ci,
      neural: { ...stats.neural, shotsToWin: undefined, avgShotsToWin: avg(stats.neural.shotsToWin) },
      classical: { ...stats.classical, shotsToWin: undefined, avgShotsToWin: avg(stats.classical.shotsToWin) },
      wallSeconds: Math.round(wallSeconds),
      ranAt: new Date().toISOString(),
    },
    null,
    2,
  ),
);
console.log(`summary written to ${OUT}`);
