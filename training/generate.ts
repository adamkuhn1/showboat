import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { makeTable } from "../src/physics/table";
import { BALL_RADIUS } from "../src/physics/constants";
import { makeBall } from "../src/physics/ball";
import { type GameState } from "../src/game/state";
import { generateCandidates } from "../src/ai/candidates";
import { featuresOf, FEATURE_NAMES } from "../src/ai/features";
import { makeRng, rolloutSuccess } from "../src/ai/rollout";

// Training data generation. Random mid-game positions -> candidate shots ->
// jittered physics rollouts of THIS repo's engine as labels. The label is the
// fraction of jittered executions that legally pot the intended ball — i.e.
// P(success under execution noise), the exact quantity the in-game ranker is
// asked to predict.
//
// Reproducible: a fixed seed drives ball placement, candidate subsampling and
// rollout jitter. Run `npm run train:generate` to rebuild the dataset;
// the file itself is not committed (see training/data/.gitignore).
//
// Env knobs: POSITIONS (default 260), ROLLOUTS (6), MAX_CANDS (40), SEED.

const POSITIONS = Number(process.env.POSITIONS ?? 260);
const ROLLOUTS = Number(process.env.ROLLOUTS ?? 6);
const MAX_CANDS = Number(process.env.MAX_CANDS ?? 40);
const SEED = Number(process.env.SEED ?? 20260811);

const here = dirname(fileURLToPath(import.meta.url));
const OUT = join(here, "data", "dataset.jsonl");

const table = makeTable();
const rng = makeRng(SEED);

// Random legal resting position, rejection-sampled against overlap.
const randomState = (): GameState => {
  const hx = table.length / 2 - BALL_RADIUS * 1.5;
  const hy = table.width / 2 - BALL_RADIUS * 1.5;
  const placed: { x: number; y: number }[] = [];
  const place = (): { x: number; y: number } => {
    for (let tries = 0; tries < 500; tries++) {
      const p = { x: (rng() * 2 - 1) * hx, y: (rng() * 2 - 1) * hy };
      if (
        placed.every(
          (q) => Math.hypot(q.x - p.x, q.y - p.y) > BALL_RADIUS * 2.1,
        )
      ) {
        placed.push(p);
        return p;
      }
    }
    throw new Error("could not place ball");
  };

  // Shooter is player 0 on solids. Sometimes the group is already cleared so
  // the 8 is the legal target — the model must learn those states too.
  const cleared = rng() < 0.12;
  const nOwn = cleared ? 0 : 1 + Math.floor(rng() * 7);
  const nOpp = 1 + Math.floor(rng() * 7);

  const balls = [];
  const cuePos = place();
  balls.push(makeBall(0, cuePos.x, cuePos.y));
  for (let i = 0; i < nOwn; i++) {
    const p = place();
    balls.push(makeBall(1 + i, p.x, p.y)); // solids 1..7
  }
  const p8 = place();
  balls.push(makeBall(8, p8.x, p8.y));
  for (let i = 0; i < nOpp; i++) {
    const p = place();
    balls.push(makeBall(9 + i, p.x, p.y)); // stripes 9..15
  }

  return {
    balls,
    turn: 0,
    groups: { 0: "solids", 1: "stripes" },
    ballInHand: false,
    winner: null,
    broken: true,
    shotCount: 10,
  };
};

const rows: string[] = [];
let sims = 0;
const t0 = Date.now();

for (let pos = 0; pos < POSITIONS; pos++) {
  const state = randomState();
  let cands = generateCandidates(state, table);
  // Unbiased random subsample so no kind is over-represented by construction.
  for (let i = cands.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [cands[i], cands[j]] = [cands[j], cands[i]];
  }
  cands = cands.slice(0, MAX_CANDS);

  for (const cand of cands) {
    const feats = featuresOf(state, table, cand);
    const rr = rolloutSuccess(
      state,
      table,
      cand.action,
      cand.targetBall,
      ROLLOUTS,
      rng,
    );
    sims += ROLLOUTS;
    rows.push(
      JSON.stringify({
        pos,
        kind: cand.kind,
        rails: cand.railsPlanned,
        target: cand.targetBall,
        feats,
        label: rr.successes / rr.n,
      }),
    );
  }

  if ((pos + 1) % 10 === 0) {
    const dt = (Date.now() - t0) / 1000;
    const eta = (dt / (pos + 1)) * (POSITIONS - pos - 1);
    console.log(
      `position ${pos + 1}/${POSITIONS} · rows ${rows.length} · sims ${sims} · ` +
        `${dt.toFixed(0)}s elapsed · ~${eta.toFixed(0)}s left`,
    );
  }
}

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(
  OUT,
  `# showboat ranker dataset · seed=${SEED} positions=${POSITIONS} rollouts=${ROLLOUTS} maxCands=${MAX_CANDS}\n` +
    `# features: ${FEATURE_NAMES.join(",")}\n` +
    rows.join("\n") +
    "\n",
);
const posCount = new Set(rows.map((r) => JSON.parse(r).pos)).size;
const mean =
  rows.reduce((s, r) => s + (JSON.parse(r).label as number), 0) / rows.length;
console.log(
  `wrote ${rows.length} rows over ${posCount} positions to ${OUT}\n` +
    `total physics rollouts: ${sims} · mean label ${mean.toFixed(3)} · ` +
    `${((Date.now() - t0) / 1000).toFixed(0)}s`,
);
