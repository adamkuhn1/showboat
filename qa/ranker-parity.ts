// Model parity across an onnxruntime-web entry-point change.
//
// Swapping the `onnxruntime-web` import for its `./wasm` subpath changes which
// build of the runtime executes the graph. The graph, the artifact and its
// sha256 are untouched — but "should be identical" is not evidence, so this
// prints the actual logits the production path produces for a fixed board and
// they are compared before and after.
//
// Uses `NeuralCandidateEvaluator`, i.e. the exact path a turn takes, so the
// numbers include the encoder and the calibration as well as the graph.
//
//   npx tsx qa/ranker-parity.ts > before.json
//   (make the change)
//   npx tsx qa/ranker-parity.ts > after.json
//   node -e "...diff..."

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../src/physics/table";
import { makeBall, type Ball } from "../src/physics/ball";
import { CUE_ID } from "../src/game/rack";
import { initPhysics } from "../src/physics/wasm-bridge";
import { generateCandidates } from "../src/ai/candidates";
import { NeuralCandidateEvaluator } from "../src/ai/neural/evaluator";
import { makeFileFetch } from "../src/ai/neural/fileFetch";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "..");
const table = makeTable();

const BOARDS: { name: string; balls: Ball[]; targets: number[] }[] = [
  {
    name: "open",
    balls: [
      makeBall(CUE_ID, -0.6, -0.1),
      makeBall(1, 0.3, 0.02),
      makeBall(2, 0.38, 0.1),
      makeBall(4, 0.1, -0.25),
      makeBall(5, -0.2, 0.3),
    ],
    targets: [1, 2, 4, 5],
  },
  {
    name: "tight",
    balls: [makeBall(CUE_ID, -0.75, 0.22), makeBall(3, 0.45, -0.28), makeBall(6, -0.1, 0.3)],
    targets: [3, 6],
  },
];

const main = async () => {
  await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
  const evaluator = new NeuralCandidateEvaluator("model/ranker");
  const state = await evaluator.load(makeFileFetch(join(APP_ROOT, "public")));
  if (state.status !== "ready") throw new Error(`model not ready: ${JSON.stringify(state)}`);

  const out: Record<string, unknown> = {
    manifest: evaluator.getManifest(),
    boards: {},
  };
  for (const b of BOARDS) {
    const cands = generateCandidates(b.balls, table, b.targets);
    const scored = await evaluator.score(b.balls, table, cands);
    if (!scored) throw new Error(`no scores for ${b.name}`);
    (out.boards as Record<string, unknown>)[b.name] = {
      candidates: cands.length,
      // Fixed precision so a formatting difference cannot look like a numeric one.
      logits: scored.logits.map((v) => v.toFixed(9)),
      scores: scored.scores.map((v) => v.toFixed(9)),
    };
  }
  console.log(JSON.stringify(out, null, 1));
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
