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
import { BALL_RADIUS } from "../../physics/constants";
import { initPhysics, simulateShotWasm } from "../../physics/wasm-bridge";
import { takeShot, type ShotReport } from "../../game/game";
import { CUE_ID } from "../../game/rack";
import { legalTargets } from "../../ai/turn";
import { generateCandidates } from "../../ai/candidates";
import { getBrain } from "../../ai/brain";
import { neuralEvaluator } from "../../ai/neural/evaluator";
import type { DecisionTraceV1 } from "../../ai/trace/contract";
import { adaptSearchResult } from "./adaptTrace";

export interface PlanInput {
  state: GameState;
  table: Table;
  player: PlayerId;
  useNeural: boolean;
  /** Bytes for the physics WASM, when the host cannot use Vite's `?url`. */
  wasmSource?: BufferSource | string;
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

export interface PlannedTurn {
  trace: DecisionTraceV1;
  action: CueAction;
  report: ShotReport;
}

export async function planTurnTraced(
  input: PlanInput,
  hooks: PlanHooks = {},
): Promise<PlannedTurn> {
  await initPhysics(input.wasmSource);

  let hashVerified = false;
  if (input.useNeural) {
    if (!neuralEvaluator.isReady()) hooks.onModelLoadStart?.();
    // First neural turn pays for the onnxruntime-web session; idempotent after.
    // A failure is reported, never silently downgraded.
    const loaded = await neuralEvaluator.load();
    if (loaded.status === "ready") {
      hashVerified = loaded.hashVerified;
      hooks.onModelStatus?.({ status: "ready", reason: null, hashVerified });
    } else {
      console.error(`[showboat] neural ranker failed to load: ${loaded.reason}`);
      hooks.onModelStatus?.({ status: "failed", reason: loaded.reason, hashVerified: false });
    }
  }

  const targets = legalTargets(input.state, input.player);
  // Regenerated here purely to recover GENERATION ORDER, which the search's
  // `stats` destroys (it drops `visits === 0` rows and re-sorts). Generation is
  // a pure function of the same inputs and costs ~2 ms, so this list is
  // identical to the one the search built internally — a recomputation, not a
  // second opinion. When T2's `ai/trace/build.ts` lands, the search hands the
  // ordered list over directly and this call goes away with the adapter.
  const generated = generateCandidates(input.state.balls, input.table, targets);

  const brain = getBrain(input.useNeural);
  const startedAt = performance.now();
  const result = await brain.plan(input.state, input.table, input.player);
  const totalMs = performance.now() - startedAt;

  const trace = adaptSearchResult(result, {
    generated,
    player: input.player,
    shotIndex: input.state.shotCount,
    legalTargets: targets,
    totalMs,
    hashVerified,
  });

  const action = trace.selected
    ? trace.selected.action
    : lastResortAim(input.state, targets);

  const report = takeShot(input.state, input.table, action, simulateShotWasm);
  return { trace, action, report };
}

/**
 * INTERIM — remove when T2's selection ladder lands.
 *
 * This is the second of the two direct-shot escape hatches the sprint's audit
 * found (it lived at `App.tsx:379-416`). It is reproduced here rather than
 * dropped because the search can still return no selection, and an opponent
 * that declines to shoot wedges the game. T2's rungs 4–5 (a non-direct safety
 * kick, then a shortest legal contact) replace it, at which point
 * `trace.selected` is non-null whenever a legal target exists and this function
 * is deleted outright — not made unreachable, deleted.
 */
function lastResortAim(state: GameState, targets: number[]): CueAction {
  const cueBall = state.balls.find((b) => b.id === CUE_ID)!;
  const live = state.balls.filter((b) => !b.pocketed);
  const byDist = live
    .filter((b) => targets.includes(b.id))
    .sort(
      (a, b) =>
        Math.hypot(a.pos.x - cueBall.pos.x, a.pos.y - cueBall.pos.y) -
        Math.hypot(b.pos.x - cueBall.pos.x, b.pos.y - cueBall.pos.y),
    );
  const pathClearTo = (t: (typeof byDist)[0]) => {
    const dx = t.pos.x - cueBall.pos.x;
    const dy = t.pos.y - cueBall.pos.y;
    const len = Math.hypot(dx, dy);
    if (len < 1e-9) return true;
    const nx = dx / len;
    const ny = dy / len;
    for (const b of live) {
      if (b.id === CUE_ID || b.id === t.id) continue;
      const vx = b.pos.x - cueBall.pos.x;
      const vy = b.pos.y - cueBall.pos.y;
      const proj = vx * nx + vy * ny;
      if (proj <= 0 || proj >= len) continue;
      const perp2 = (vx - proj * nx) ** 2 + (vy - proj * ny) ** 2;
      if (perp2 < (2 * BALL_RADIUS) ** 2) return false;
    }
    return true;
  };
  const nearest = byDist.find(pathClearTo) ?? byDist[0];
  const phi = nearest
    ? Math.atan2(nearest.pos.y - cueBall.pos.y, nearest.pos.x - cueBall.pos.x)
    : 0;
  return { phi, power: 0.3, sideSpin: 0, topSpin: 0 };
}
