// One opponent turn: plan it, then run the authoritative shot simulation for
// it. Shared verbatim by the worker host (`planWorker.ts`) and by the
// main-thread fallback (`usePlanner.ts`), so there is exactly one
// implementation of "what the opponent does on its turn" regardless of which
// thread it runs on.
//
// The shot simulation is done here, beside the search, for two reasons. It is
// the expensive part the page used to block on; and the cushion/contact marks
// the presentation draws come from `report.sim.events` + `waypoints` — the run
// that actually decides the outcome — so the presentation needs the executed
// simulation in hand before it starts, not after.

import type { GameState, PlayerId } from "../../game/state";
import type { Table } from "../../physics/table";
import type { CueAction } from "../../physics/cue";
import { initPhysics, simulateShotWasm } from "../../physics/wasm-bridge";
import { type ShotReport } from "../../game/game";
import { getBrain } from "../../ai/brain";
import { executeAiShot } from "../../ai/policy/execute";
import { NeuralCandidateEvaluator, neuralEvaluator } from "../../ai/neural/evaluator";
import type { DecisionTraceV1 } from "../../ai/trace/contract";

export interface PlanInput {
  state: GameState;
  table: Table;
  player: PlayerId;
  useNeural: boolean;
  /**
   * ABSOLUTE URL of the model directory.
   *
   * The evaluator's default is the relative `model/ranker`, which resolves
   * against the *worker's* own URL inside a worker — `/assets/planWorker-*.js`
   * — so `model/ranker/manifest.json` became `/assets/model/ranker/...`, the
   * dev/preview server answered with index.html, and the ranker failed
   * preflight with "returned HTML". The page resolves the real URL against
   * `document.baseURI` (which honours Vite's `base: "./"` and the portfolio's
   * embed path) and hands it over.
   */
  modelDir?: string;
  /** Bytes for the physics WASM, when the host cannot use Vite's `?url`. */
  wasmSource?: BufferSource | string;
}

// One evaluator per realm. The page keeps its own for `preflight()` (a small
// JSON + sha256 check that pulls no ONNX runtime); this one is the worker's,
// and it is the only place the ~27 MB onnxruntime-web runtime is instantiated.
let scopedEvaluator: NeuralCandidateEvaluator | null = null;
let scopedDir: string | null = null;

function evaluatorFor(modelDir: string | undefined): NeuralCandidateEvaluator {
  if (!modelDir) return neuralEvaluator;
  if (!scopedEvaluator || scopedDir !== modelDir) {
    scopedEvaluator = new NeuralCandidateEvaluator(modelDir);
    scopedDir = modelDir;
  }
  return scopedEvaluator;
}

/**
 * Create the ranker session ahead of the first opponent turn. Called when the
 * page is idle, so the download and the session build are paid for while the
 * human is lining up a break rather than in the middle of a turn.
 */
export async function warmModel(modelDir: string): Promise<void> {
  await evaluatorFor(modelDir).load();
}

export interface ModelStatus {
  status: "ready" | "failed";
  reason: string | null;
  hashVerified: boolean;
}

export interface PlanHooks {
  /** Fired before the (possibly slow) first model load, so the host can say so. */
  onModelLoadStart?: () => void;
  onModelStatus?: (s: ModelStatus) => void;
}

/**
 * A turn in which the opponent shot. `action` is the action carried by the
 * branded `PlayableShot` the policy chose — it is here for the cue-stroke
 * animation only; nothing may re-execute it, and nothing can, because
 * `executeAiShot` has already produced the authoritative `report` and is the
 * only AI-side path to `takeShot`.
 */
export interface PlayedTurn {
  kind: "shot";
  trace: DecisionTraceV1;
  action: CueAction;
  report: ShotReport;
}

/**
 * A turn in which the policy returned no shot.
 *
 * The trick-only ladder's rungs 4 and 5 produce a shot whenever a legal target
 * exists, so this now means exactly one thing: **there was no legal target.**
 * It is not a search failure and it is not a fallback — there is nothing to
 * fall back to. The host rests the turn rather than sitting on "searching…".
 */
export interface RestedTurn {
  kind: "no-legal-shot";
  trace: DecisionTraceV1;
}

/**
 * A discriminated union rather than a nullable `report`, so a host that
 * forgets the resting case does not compile. That case is the one that used to
 * wedge the panel at "searching…" forever.
 */
export type PlannedTurn = PlayedTurn | RestedTurn;

export async function planTurnTraced(
  input: PlanInput,
  hooks: PlanHooks = {},
): Promise<PlannedTurn> {
  await initPhysics(input.wasmSource);

  const evaluator = evaluatorFor(input.modelDir);
  let hashVerified = false;
  if (input.useNeural) {
    if (!evaluator.isReady()) hooks.onModelLoadStart?.();
    // First neural turn pays for the onnxruntime-web session; idempotent after.
    // A failure is reported, never silently downgraded.
    const loaded = await evaluator.load();
    if (loaded.status === "ready") {
      hashVerified = loaded.hashVerified;
      hooks.onModelStatus?.({ status: "ready", reason: null, hashVerified });
    } else {
      console.error(`[showboat] neural ranker failed to load: ${loaded.reason}`);
      hooks.onModelStatus?.({ status: "failed", reason: loaded.reason, hashVerified: false });
    }
  }

  // ONE decision, ONE trace, ONE way to play it.
  //
  // `brain.plan` runs the trick-only policy and publishes `decision`, a
  // `DecisionTraceV1` whose `candidates` are already in generation order. This
  // used to regenerate the candidate list to recover that order and then
  // synthesise a trace from `SearchResult` through `planner/adaptTrace.ts`;
  // both are gone. The renderer now reads the trace the decision was actually
  // made with, not a reconstruction of it.
  const brain = getBrain(input.useNeural, evaluator);
  const result = await brain.plan(input.state, input.table, input.player);
  const trace = result.decision;

  // `result.shot` is a branded `PlayableShot` — the only thing in the codebase
  // that can be played, and something only `policy/trickOnly.ts` can mint. It
  // is null only when there was no legal target (the ladder's rungs 4-5 cover
  // every other case), and there is no local aim to fall back on: the
  // nearest-legal-ball escape hatch that used to live here is deleted.
  if (result.shot === null) return { kind: "no-legal-shot", trace };

  const report = executeAiShot(input.state, input.table, result.shot, simulateShotWasm);
  return { kind: "shot", trace, action: result.shot.action, report };
}
