// Headless self-play trajectory collection — reuses 100% of production game
// logic (makeGame/takeShot/applyShotRules/planTurn/legalTargets/placeCueBall)
// with zero new game-logic code, only orchestration. This is what closes the
// Phase 2C audit's severity-2 finding (docs/repair/showboat-ml/phase-2c/
// 01-state-label-audit.md §1/§6/§7): the Phase 2A/2B generator's always-open-
// table synthetic states can never produce an assigned-group state or a legal
// 8-ball win, and there was no self-play loop over the authoritative engine
// at all (the only existing self-play code wraps PoolTool, a different
// physics library, on the parked RL track).

import { type CueAction } from "../../../src/physics/cue";
import { type GameState } from "../../../src/game/state";
import { type Table } from "../../../src/physics/table";
import { makeGame, cloneState, takeShot, placeCueBall } from "../../../src/game/game";
import { legalTargets, planTurn } from "../../../src/ai/turn";
import { type SearchConfig } from "../../../src/ai/shotSearch";
import { simulateShotWasm } from "../../../src/physics/wasm-bridge";
import { CUE_ID } from "../../../src/game/rack";
import { BALL_RADIUS } from "../../../src/physics/constants";

export interface TrajectoryState {
  state: GameState; // pre-shot snapshot, cloned (safe to mutate/keep)
  shotIndex: number;
}

// A much smaller budget than production's defaultConfig (60): self-play only
// needs to pick a REASONABLE, non-degenerate shot to advance the game and
// produce realistic states — not a maximally strong decision. Measured cost
// compounds fast otherwise: each shot decision costs roughly
// (simulations budget / ~41 calls/sec) wall-clock, on top of the labeling
// pass this phase already runs on every recorded state — see
// docs/repair/showboat-ml/phase-2c/DATASET_DESIGN.md's throughput section.
export const SELF_PLAY_SEARCH_CONFIG: SearchConfig = {
  simulations: 16,
  rolloutDepth: 1,
  rolloutsPerEval: 2,
  seed: 777,
};

/**
 * Ball-in-hand placement for self-play: a random legal, non-overlapping spot.
 * Real gameplay lets the player choose strategically; self-play only needs a
 * PHYSICALLY LEGAL placement to keep the game moving and produce a real,
 * playable next state — the placement itself is not a modeled decision this
 * phase (see DATASET_DESIGN.md's "What this phase deliberately does not do").
 */
function randomLegalCuePlacement(
  state: GameState,
  table: Table,
  rng: () => number,
): { x: number; y: number } {
  const hx = table.length / 2 - BALL_RADIUS - 0.01;
  const hy = table.width / 2 - BALL_RADIUS - 0.01;
  const others = state.balls.filter((b) => !b.pocketed && b.id !== CUE_ID);
  const MIN_DIST = 2 * BALL_RADIUS + 0.01;
  for (let attempt = 0; attempt < 200; attempt++) {
    const x = (rng() * 2 - 1) * hx;
    const y = (rng() * 2 - 1) * hy;
    if (others.every((b) => Math.hypot(b.pos.x - x, b.pos.y - y) >= MIN_DIST)) {
      return { x, y };
    }
  }
  return { x: 0, y: 0 }; // rare fallback: table too crowded for 200 tries
}

/**
 * Play one full self-play game via the real search (planTurn/searchBaseline),
 * recording the pre-shot state before every shot after the break. The break
 * itself (shot 0) uses a fixed, lightly-jittered power shot rather than the
 * AI search — matching real gameplay convention (App.tsx's break is not a
 * searched decision either) and avoiding wasted search budget against a
 * fully-clustered rack where isPathClear rejects almost every candidate.
 *
 * Returns the recorded trajectory states (family = one call = one game) plus
 * a terminal summary so the caller can tag terminal-win/terminal-loss labels.
 */
export function playSelfPlayGame(
  table: Table,
  rng: () => number,
  opts: { maxShots?: number; searchConfig?: SearchConfig; onShot?: (shot: number) => void } = {},
): { trajectory: TrajectoryState[]; winner: 0 | 1 | null; shotsPlayed: number } {
  const maxShots = opts.maxShots ?? 60;
  const searchConfig = opts.searchConfig ?? SELF_PLAY_SEARCH_CONFIG;
  const { state: initial } = makeGame();
  let state = initial;
  const trajectory: TrajectoryState[] = [];

  for (let shot = 0; shot < maxShots && state.winner === null; shot++) {
    opts.onShot?.(shot);
    if (state.ballInHand !== false) {
      const { x, y } = randomLegalCuePlacement(state, table, rng);
      state = placeCueBall(state, x, y, table);
    }

    const targets = legalTargets(state, state.turn);
    if (targets.length === 0) break; // shouldn't happen in a well-formed state; guard anyway

    let action: CueAction;
    if (shot === 0) {
      action = { phi: (rng() * 2 - 1) * 0.03, power: 0.9 + rng() * 0.08, sideSpin: 0, topSpin: 0 };
    } else {
      trajectory.push({ state: cloneState(state), shotIndex: shot });
      const result = planTurn(state, table, searchConfig);
      if (!result.best) break;
      action = result.best.candidate.action;
    }

    const report = takeShot(state, table, action, simulateShotWasm);
    state = report.next;
  }

  return { trajectory, winner: state.winner, shotsPlayed: state.shotCount };
}
