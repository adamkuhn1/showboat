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
  safetyCandidates,
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
//         -> physics-verify the top K -> select a legal shot (selectShot:
//            robustness-gated, trick shots preferred only above the bar)
//         -> or, if nothing pots, a simulated legal safety (chooseSafety)
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
  safety: {
    action: CueAction;
    measured: MeasuredShot;
    considered: number; // safety roll-ups simulated
    legal: number; // of those, how many were foul-free (0 -> old fallback)
  } | null;
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
// that works will still be chosen. This constant decides which candidates get
// verified; after verification, selectShot prefers a measured trick shot only
// among shots that clear the ROBUST_FRACTION bar.
export const DIRECT_ORDER_DISCOUNT = 0.5;
// How many top-ranked candidates get a full physics verification.
export const VERIFY_TOP_K = 10;
// Jittered re-executions of a verified candidate (robustness measurement).
export const ROBUSTNESS_N = 3;
// The robustness bar: a shot "clears" it when at least this fraction of its
// ROBUSTNESS_N jittered re-executions also pot (2 of 3 by default). Only
// shots that clear it get the trick-shot preference — a bank that pots once
// in three tries must not outrank a direct pot that goes in every time.
export const ROBUST_FRACTION = 2 / 3;

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
  const pick = selectShot(top);

  const decisionBase = {
    ranker: ranker.name,
    phase: "shot" as const,
    candidatesGenerated: cands.length,
    byKind,
    verified: top,
    cuePlacedAt,
  };

  if (pick) {
    const chosen = pick.chosen;
    return {
      stateAfterPlacement: state,
      action: chosen.cand.action,
      report: chosen.report!,
      decision: { ...decisionBase, selected: chosen, safety: null, reason: pick.reason },
    };
  }

  // --- nothing pots: play safe ---------------------------------------------
  hooks.onPhase?.("safety");
  const safe = chooseSafety(state, table, ranker);
  const { action, report } = safe;
  const measured = classifyShot(report.sim, -1);
  const lead = `None of the ${top.length} verified candidates pots a ball`;
  let reason: string;
  if (safe.legal === 0) {
    reason =
      `${lead}. No legal safety found; all ${safe.considered} simulated ` +
      `roll-ups foul, so rolling up to the ${
        report.sim.firstContact !== null
          ? ballName(report.sim.firstContact)
          : "nearest legal ball"
      } anyway.`;
  } else {
    const target = ballName(report.sim.firstContact!);
    reason =
      `${lead} — rolling up to the ${target} as a safety ` +
      (safe.legal === 1
        ? `(the only legal one of ${safe.considered} simulated roll-ups).`
        : `(${safe.legal} of ${safe.considered} simulated roll-ups are legal; ` +
          `this one leaves the opponent the weakest best shot).`);
  }
  return {
    stateAfterPlacement: state,
    action,
    report,
    decision: {
      ...decisionBase,
      selected: null,
      safety: { action, measured, considered: safe.considered, legal: safe.legal },
      reason,
    },
  };
};

// --- shot selection ------------------------------------------------------------

const robustFraction = (ec: EvaluatedCandidate): number =>
  ec.robustness && ec.robustness.n > 0
    ? ec.robustness.successes / ec.robustness.n
    : 0;

// Clears the bar when at least ROBUST_FRACTION of the jittered re-executions
// also pot (2 of 3 at the default ROBUSTNESS_N). The epsilon only absorbs
// float rounding in the comparison (2/3 vs 0.6666…).
export const isRobust = (ec: EvaluatedCandidate): boolean =>
  robustFraction(ec) >= ROBUST_FRACTION - 1e-9;

const isTrick = (ec: EvaluatedCandidate): boolean =>
  ec.measured !== undefined && isTrickShot(ec.measured);

const byRobustnessThenPrior = (
  a: EvaluatedCandidate,
  b: EvaluatedCandidate,
): number => {
  const d = robustFraction(b) - robustFraction(a);
  if (Math.abs(d) > 1e-9) return d;
  return b.prior - a.prior;
};

const byTrickThenRobustness = (
  a: EvaluatedCandidate,
  b: EvaluatedCandidate,
): number => {
  const ta = isTrick(a) ? 1 : 0;
  const tb = isTrick(b) ? 1 : 0;
  if (ta !== tb) return tb - ta;
  return byRobustnessThenPrior(a, b);
};

export interface ShotPick {
  chosen: EvaluatedCandidate;
  // Why it won — exactly one of these describes the decisive comparison.
  rule: "trick-preferred" | "robust" | "most-robust-below-bar";
  reason: string;
}

// The selection rule, over physics-verified candidates:
//   1. Only candidates that pot legally in the verification simulation count.
//   2. Of those, the ones that also pot in >= ROBUST_FRACTION of their jittered
//      re-executions "clear the bar". Among them a measured trick shot is
//      preferred, then higher robustness, then higher ranker prior.
//   3. If none clears the bar, take the most robust successful shot (then
//      prior) — no trick preference for fragile shots.
//   4. No successful candidate at all -> null (the caller plays safe).
// The reason text names the comparison that actually decided the pick.
export const selectShot = (verified: EvaluatedCandidate[]): ShotPick | null => {
  const successes = verified.filter((e) => e.success && e.measured && e.robustness);
  if (successes.length === 0) return null;
  const robust = successes.filter(isRobust).sort(byTrickThenRobustness);

  if (robust.length > 0) {
    const chosen = robust[0];
    // Did the trick preference decide it? Only if, ranked on robustness and
    // prior alone, a different (non-trick) shot would have come first.
    const plain = [...robust].sort(byRobustnessThenPrior)[0];
    const trickDecided = plain !== chosen && isTrick(chosen);
    let reason = shotSummary(chosen);
    if (trickDecided) {
      reason += `, preferred as a trick shot over a ${jitterOf(plain)} ${plain.measured!.label}.`;
    } else {
      // A trick shot that lost only because it was fragile (the old
      // trick-first rule would have taken it) is worth saying out loud.
      const fragileTrick = successes
        .filter((e) => isTrick(e) && !isRobust(e))
        .sort(byRobustnessThenPrior)[0];
      reason +=
        fragileTrick && !isTrick(chosen)
          ? `; passed over a ${fragileTrick.measured!.label} (${jitterOf(fragileTrick)} under jitter) ` +
            `for a ${jitterOf(chosen)} ${chosen.measured!.kind === "direct" ? "direct pot" : chosen.measured!.label}.`
          : ".";
    }
    return { chosen, rule: trickDecided ? "trick-preferred" : "robust", reason };
  }

  const chosen = [...successes].sort(byRobustnessThenPrior)[0];
  const bar = Math.ceil(ROBUST_FRACTION * chosen.robustness!.n - 1e-9);
  const reason =
    `No verified shot pots in ${bar}/${chosen.robustness!.n} under jitter; ` +
    `taking the most robust one. ${shotSummary(chosen)}.`;
  return { chosen, rule: "most-robust-below-bar", reason };
};

const jitterOf = (ec: EvaluatedCandidate): string =>
  `${ec.robustness!.successes}/${ec.robustness!.n}`;

const shotSummary = (ec: EvaluatedCandidate): string => {
  const m = ec.measured!;
  return (
    `${capitalise(m.label)} on the ${ballName(ec.cand.targetBall)} ` +
    `into the ${pocketName(m.pocketId)} — pots in simulation, ` +
    `${jitterOf(ec)} under aim jitter`
  );
};

// --- safety search ---------------------------------------------------------------

export interface SafetyChoice {
  action: CueAction;
  report: ShotReport; // with frames: this is the simulation that gets played
  considered: number; // roll-ups simulated
  legal: number; // of those, how many were foul-free
}

// Simulate every proposed roll-up with the real engine and keep only legal
// ones (no foul of any kind: wrong first contact, no rail, scratch, an early
// 8). Among the legal ones, prefer a roll-up that keeps the table (it potted
// one of ours), otherwise the one that leaves the opponent the weakest best
// shot by the same ranker the agent uses — pure ranking of the resulting
// position, no extra physics. Ties keep proposal order, so the classic
// nearest-ball roll-up wins whenever it is already legal and no worse.
// Only if nothing is legal does it fall back to the old single roll-up.
export const chooseSafety = (
  state: GameState,
  table: Table,
  ranker: Ranker,
): SafetyChoice => {
  const proposals = safetyCandidates(state, table);
  let best: { action: CueAction; score: number } | null = null;
  let legal = 0;
  for (const action of proposals) {
    const r = takeShot(state, table, action);
    if (r.outcome.foul || r.outcome.gameOver || r.sim.firstContact === null) continue;
    legal++;
    const score = r.outcome.turnPasses ? opponentBestPrior(r.next, table, ranker) : -1;
    if (!best || score < best.score - 1e-12) best = { action, score };
  }
  const action: CueAction =
    best?.action ??
    safetyAction(state, table) ?? { phi: 0, power: 0.3, sideSpin: 0, topSpin: 0 };
  const report = takeShot(state, table, action, { recordFrames: true });
  return { action, report, considered: proposals.length, legal };
};

// How good is the incoming player's best candidate in this position, per the
// ranker (with the agent's own trick-preference ordering weight)?
const opponentBestPrior = (g: GameState, table: Table, ranker: Ranker): number => {
  let best = 0;
  for (const cand of generateCandidates(g, table)) {
    const s =
      ranker.score(featuresOf(g, table, cand)) *
      (cand.kind === "direct" ? DIRECT_ORDER_DISCOUNT : 1);
    if (s > best) best = s;
  }
  return best;
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
