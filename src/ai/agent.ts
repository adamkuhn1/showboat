import { type Vec2 } from "../physics/vec";
import { type Table } from "../physics/table";
import { type CueAction } from "../physics/cue";
import { BALL_RADIUS } from "../physics/constants";
import { type GameState } from "../game/state";
import { cloneState, placeCueBall, takeShot, type ShotReport } from "../game/game";
import { CUE_ID } from "../game/rack";
import {
  generateCandidates,
  safetyAction,
  type Candidate,
  type CandidateKind,
} from "./candidates";
import { featuresOf } from "./features";
import { type Ranker } from "./ranker";
import { classifyShot, isTrickShot, type MeasuredShot } from "./classify";
import { rolloutSuccess, shotSucceeded } from "./rollout";

// The AI turn. One straight pipeline, one published result:
//
//   state -> generate candidates -> rank (neural or classical prior)
//         -> physics-verify the top K -> select a legal shot
//         -> publish ONE Decision containing THE ShotReport that will be
//            committed and played back.
//
// There is exactly one representation of the chosen shot: the ShotReport whose
// simulation frames the playback animates and whose outcome the game commits.
// The thinking overlay renders from this same Decision object — it cannot say
// anything the decision data does not contain.

export interface EvaluatedCandidate {
  cand: Candidate;
  features: number[];
  prior: number; // ranker output BEFORE any physics verification
  orderScore: number; // prior after the explicit trick-shot preference weight
  report?: ShotReport; // the verification simulation (with frames)
  measured?: MeasuredShot; // classification of what actually happened
  success?: boolean; // legal + target potted, per rollout.ts
  robustness?: { successes: number; n: number }; // jittered re-executions
}

export interface Decision {
  ranker: "neural" | "classical";
  phase: "break" | "shot";
  candidatesGenerated: number;
  byKind: Record<CandidateKind, number>;
  verified: EvaluatedCandidate[]; // in verification order
  selected: EvaluatedCandidate | null; // null -> safety fallback was used
  safety: { action: CueAction; measured: MeasuredShot } | null;
  reason: string; // composed only from measured/decision data above
  cuePlacedAt: Vec2 | null; // ball-in-hand placement, if any
}

export interface AiMove {
  stateAfterPlacement: GameState; // == input state unless ball-in-hand
  action: CueAction;
  report: ShotReport; // THE simulation: committed and played back
  decision: Decision;
}

export interface AiHooks {
  onPhase?: (phase: string) => void;
  // Fired after a ball-in-hand placement so the UI can repaint the moved cue.
  onPlaced?: (placed: GameState) => void;
  // Fired once with the real generation counts (before any verification).
  onGenerated?: (count: number, byKind: Record<CandidateKind, number>) => void;
  onCandidate?: (ec: EvaluatedCandidate, index: number, total: number) => void;
  // Pause between candidate verifications so the thinking display is legible.
  // Self-play evaluation passes 0.
  delayMs?: number;
  signal?: { cancelled: boolean };
  rng?: () => number;
}

// The AI's explicit style: trick shots (bank/kick/combo) keep their full
// ranker score; plain direct shots are discounted when ordering. Verification
// still decides legality — a discounted direct shot that is the only shot
// that works will still be chosen. This constant IS the "strongly prefers
// trick shots" behaviour; nothing else biases the choice.
export const DIRECT_ORDER_DISCOUNT = 0.5;
// How many top-ranked candidates get a full physics verification.
export const VERIFY_TOP_K = 10;
// Jittered re-executions of a verified candidate (robustness measurement).
export const ROBUSTNESS_N = 3;

const sleep = (ms: number): Promise<void> =>
  new Promise((res) => setTimeout(res, ms));

export const aiTakeTurn = async (
  input: GameState,
  table: Table,
  ranker: Ranker,
  hooks: AiHooks = {},
): Promise<AiMove | null> => {
  const rng = hooks.rng ?? Math.random;
  const cancelled = (): boolean => hooks.signal?.cancelled === true;
  let state = cloneState(input);

  // --- opening break: aim through the apex ball at full power --------------
  if (!state.broken) {
    hooks.onPhase?.("break");
    const cue = state.balls.find((b) => b.id === CUE_ID)!;
    const apex = state.balls
      .filter((b) => b.id !== CUE_ID && !b.pocketed)
      .reduce((a, b) => (a.pos.x < b.pos.x ? a : b));
    const action: CueAction = {
      phi: Math.atan2(apex.pos.y - cue.pos.y, apex.pos.x - cue.pos.x) +
        (rng() - 0.5) * 0.02,
      power: 1,
      sideSpin: 0,
      topSpin: 0,
    };
    const report = takeShot(state, table, action, { recordFrames: true });
    return {
      stateAfterPlacement: state,
      action,
      report,
      decision: {
        ranker: ranker.name,
        phase: "break",
        candidatesGenerated: 0,
        byKind: { direct: 0, bank: 0, kick: 0, combo: 0 },
        verified: [],
        selected: null,
        safety: null,
        reason: `Breaking at full power${
          report.sim.pocketed.length > 0
            ? ` — ${report.sim.pocketed.length} ball(s) down`
            : ""
        }.`,
        cuePlacedAt: null,
      },
    };
  }

  // --- ball in hand: place the cue where the ranker sees the best shot -----
  let cuePlacedAt: Vec2 | null = null;
  if (state.ballInHand !== false) {
    hooks.onPhase?.("placing");
    const spot = bestCuePlacement(state, table, ranker);
    if (spot) {
      state = placeCueBall(state, spot.x, spot.y);
      cuePlacedAt = spot;
      hooks.onPlaced?.(state);
    }
    if (cancelled()) return null;
  }

  // --- generate + rank ------------------------------------------------------
  hooks.onPhase?.("generating");
  const cands = generateCandidates(state, table);
  const byKind: Record<CandidateKind, number> = {
    direct: 0,
    bank: 0,
    kick: 0,
    combo: 0,
  };
  for (const c of cands) byKind[c.kind]++;
  hooks.onGenerated?.(cands.length, byKind);

  const evaluated: EvaluatedCandidate[] = cands.map((cand) => {
    const features = featuresOf(state, table, cand);
    const prior = ranker.score(features);
    const orderScore =
      cand.kind === "direct" ? prior * DIRECT_ORDER_DISCOUNT : prior;
    return { cand, features, prior, orderScore };
  });
  evaluated.sort((a, b) => b.orderScore - a.orderScore);

  // --- verify the top K through the real physics ---------------------------
  hooks.onPhase?.("verifying");
  const top = evaluated.slice(0, VERIFY_TOP_K);
  for (let i = 0; i < top.length; i++) {
    if (cancelled()) return null;
    const ec = top[i];
    ec.report = takeShot(state, table, ec.cand.action, { recordFrames: true });
    ec.measured = classifyShot(ec.report.sim, ec.cand.targetBall);
    ec.success = shotSucceeded(ec.report, ec.cand.targetBall);
    if (ec.success) {
      ec.robustness = rolloutSuccess(
        state,
        table,
        ec.cand.action,
        ec.cand.targetBall,
        ROBUSTNESS_N,
        rng,
      );
    }
    hooks.onCandidate?.(ec, i, top.length);
    if (hooks.delayMs) await sleep(hooks.delayMs);
  }
  if (cancelled()) return null;

  // --- select ---------------------------------------------------------------
  hooks.onPhase?.("selecting");
  const successes = top.filter((e) => e.success);
  successes.sort((a, b) => {
    // Trick shots first (measured, not intended), then robustness, then prior.
    const ta = a.measured && isTrickShot(a.measured) ? 1 : 0;
    const tb = b.measured && isTrickShot(b.measured) ? 1 : 0;
    if (ta !== tb) return tb - ta;
    const ra = a.robustness ? a.robustness.successes : 0;
    const rb = b.robustness ? b.robustness.successes : 0;
    if (ra !== rb) return rb - ra;
    return b.prior - a.prior;
  });

  const decisionBase = {
    ranker: ranker.name,
    phase: "shot" as const,
    candidatesGenerated: cands.length,
    byKind,
    verified: top,
    cuePlacedAt,
  };

  if (successes.length > 0) {
    const chosen = successes[0];
    const m = chosen.measured!;
    const rb = chosen.robustness!;
    const reason =
      `${capitalise(m.label)} on the ${ballName(chosen.cand.targetBall)} ` +
      `into the ${pocketName(m.pocketId)} — pots in simulation, ` +
      `${rb.successes}/${rb.n} under aim jitter` +
      (m.kind !== "direct" ? ", preferred as a trick shot." : ".");
    return {
      stateAfterPlacement: state,
      action: chosen.cand.action,
      report: chosen.report!,
      decision: { ...decisionBase, selected: chosen, safety: null, reason },
    };
  }

  // --- nothing pots: play safe ---------------------------------------------
  const safe = safetyAction(state, table);
  const action: CueAction = safe ?? { phi: 0, power: 0.3, sideSpin: 0, topSpin: 0 };
  const report = takeShot(state, table, action, { recordFrames: true });
  const measured = classifyShot(report.sim, -1);
  const reason =
    `None of the ${top.length} verified candidates pots a ball — ` +
    `rolling up to the ${
      report.sim.firstContact !== null
        ? ballName(report.sim.firstContact)
        : "nearest legal ball"
    } as a safety.`;
  return {
    stateAfterPlacement: state,
    action,
    report,
    decision: {
      ...decisionBase,
      selected: null,
      safety: { action, measured },
      reason,
    },
  };
};

// Sample free positions and keep the one whose best-ranked candidate scores
// highest (with the same trick preference used for ordering). Pure ranking —
// physics verification then applies to the shot taken from there.
const bestCuePlacement = (
  g: GameState,
  table: Table,
  ranker: Ranker,
): Vec2 | null => {
  const hx = table.length / 2 - BALL_RADIUS * 2;
  const hy = table.width / 2 - BALL_RADIUS * 2;
  let best: { score: number; p: Vec2 } | null = null;
  for (let ix = 0; ix < 12; ix++) {
    for (let iy = 0; iy < 6; iy++) {
      const p = {
        x: -hx + (2 * hx * ix) / 11,
        y: -hy + (2 * hy * iy) / 5,
      };
      const blocked = g.balls.some(
        (b) =>
          !b.pocketed &&
          b.id !== CUE_ID &&
          Math.hypot(b.pos.x - p.x, b.pos.y - p.y) < BALL_RADIUS * 2.2,
      );
      if (blocked) continue;
      const placed = placeCueBall(g, p.x, p.y);
      const cands = generateCandidates(placed, table);
      let score = 0;
      for (const cand of cands) {
        const s =
          ranker.score(featuresOf(placed, table, cand)) *
          (cand.kind === "direct" ? DIRECT_ORDER_DISCOUNT : 1);
        if (s > score) score = s;
      }
      if (!best || score > best.score) best = { score, p };
    }
  }
  return best?.p ?? null;
};

// --- naming helpers (shared wording with the caption layer) -----------------

const ballName = (id: number): string =>
  id === 8 ? "8-ball" : `${id}-ball`;

const pocketName = (id: string | null): string => {
  const map: Record<string, string> = {
    bl: "bottom-left pocket",
    tl: "top-left pocket",
    br: "bottom-right pocket",
    tr: "top-right pocket",
    sb: "bottom-side pocket",
    st: "top-side pocket",
  };
  return id ? map[id] ?? "pocket" : "pocket";
};

const capitalise = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);
