// ===========================================================================
// The opponent's decision, presented.
//
// This module is pure: it takes a completed `DecisionTraceV1` and a wall-clock
// offset and returns what should be on the felt at that instant. No canvas, no
// React, no timers — so the whole sequence is unit-testable, and so that the
// rules about what may be drawn live in one readable file instead of being
// smeared across a paint loop.
//
// WHAT IS REAL AND WHAT IS PRESENTATION — read this before changing anything.
//
// The search completes before the presentation begins. Every route, every
// rejection reason, every contact and every number below is read off the trace
// the opponent actually decided with; nothing is synthesised. What the
// presentation adds is *pacing*: the order and the speed at which an already-
// finished decision is revealed. That is disclosed in the panel and in the
// README, and it is the only liberty taken.
//
// The two places pacing is derived from measured work rather than chosen:
//   - VERIFYING's length is the search's real physics time
//     (`timing.physicsMs`), divided across the candidates that really were
//     simulated (`physics !== null`). It is an average, so it makes no claim
//     about any individual candidate's cost — see VERIFY_* below.
//   - SHOOTING is the authoritative simulation replayed. Its rate is a
//     presentation choice the visitor controls (`ui/playbackSpeed.ts`) and is
//     labelled as one; it reparameterises time and changes nothing else.
//
// The routes the felt shows come in two kinds and are never conflated. Until
// SELECTED they are candidate PLANS — mirror geometry, dashed. From SELECTED on
// the shot being played is drawn from `selected.executed`, the measured motion
// of the run whose outcome is committed, and is solid. See `RouteSource`.
//
// The three hold states (ENUMERATING, SELECTED, READY — and RANKING) are
// reveals of a finished set. They must never render a progress bar, a spinner,
// a climbing counter, or the word "computing".
// ===========================================================================

import type {
  DecisionTraceV1,
  ExecutedMotion,
  RejectionReason,
  TracedCandidate,
  TracedKind,
  Vec2Trace,
} from "../ai/trace/contract";

export type PresentationState =
  | "IDLE"
  | "ENUMERATING"
  | "RANKING"
  | "VERIFYING"
  | "SELECTED"
  | "READY"
  | "STROKE"
  | "SHOOTING"
  | "SETTLED";

/** The five reasoning states, in order. RANKING is absent in classical mode. */
export const REASONING_STATES = [
  "ENUMERATING",
  "RANKING",
  "VERIFYING",
  "SELECTED",
  "READY",
] as const satisfies readonly PresentationState[];

export type ReasoningState = (typeof REASONING_STATES)[number];

/**
 * The only five strings that may appear in the panel's state slot. They
 * distinguish the five things that actually happened: candidate generation,
 * the model's ordering, real physics simulation, the selection, and the
 * finished plan.
 */
export const STATE_LABEL: Record<ReasoningState, string> = {
  ENUMERATING: "evaluating",
  RANKING: "neural ranking",
  VERIFYING: "physics verification",
  SELECTED: "selected",
  READY: "ready to shoot",
};

// --- Timing ---------------------------------------------------------------
//
// Every duration is derived from the size of the trace. The bounds are the
// only literals, and each is justified by perception rather than convenience.
export const TIMING = {
  /** ~one route per frame at 60 Hz; faster reads as a single flash. */
  ENUM_PER_CANDIDATE_MS: 18,
  /** Below ~400 ms a state change is subliminal. */
  ENUM_MIN_MS: 400,
  /** A reveal with no work behind it must not outlast one that has. */
  ENUM_MAX_MS: 900,
  /** 450 ms reorder + 150 ms settle: long enough to track an item moving. */
  RANK_MS: 600,
  /**
   * Per-candidate floor/ceiling on the averaged physics cost.
   *
   * The ceiling came down from 260 ms (and the state ceiling from 3200 ms)
   * because the sequence was proportioned backwards. Measured on a production
   * build by polling the panel's state label: `physics verification` held for
   * 2.2-2.6 s while SELECTED — the one frame that shows what the opponent
   * actually chose, and the only frame whose content differs from turn to turn
   * — was on screen for 0.20-0.36 s. VERIFYING is a reveal of a set of already
   * finished simulations; it does not need two and a half seconds to be
   * legible, and the time is worth more to SELECTED.
   */
  VERIFY_MIN_ITEM_MS: 40,
  VERIFY_MAX_ITEM_MS: 80,
  VERIFY_MIN_MS: 500,
  VERIFY_MAX_MS: 1000,
  /**
   * SELECTED is now the longest beat of the sequence, by design and at every
   * decay. It is the payload: the route that is about to be played, locking in
   * against the ones that lost.
   */
  SELECT_BASE_MS: 900,
  SELECT_PER_LOSER_MS: 35,
  SELECT_MIN_MS: 1500,
  SELECT_MAX_MS: 2000,
  READY_BASE_MS: 500,
  /** ~430 wpm — a scan rate, not a read rate; the visitor is watching the table too. */
  READY_PER_WORD_MS: 28,
  READY_MIN_MS: 700,
  READY_MAX_MS: 1400,
  /** Backswing and strike. Direction and power come from the real action. */
  STROKE_MS: 350,
  /** Uniform scale over hold states only. Never applied to VERIFYING. */
  REASONING_CEILING_MS: 5600,
  /** The hold states never collapse below this fraction of their natural length. */
  MIN_HOLD_SCALE: 0.25,
  /**
   * SELECTED's own floor, and the reason it has one.
   *
   * The per-session decay exists because a visitor who has watched two full
   * sequences has learned to READ them — and that is true of the process
   * states, whose shape is identical every turn, and false of SELECTED, whose
   * content is a different shot on a different board every time. Decaying it
   * like the rest is what crushed the informative frame to a fifth of a second
   * by the third turn. It still shortens; it does not disappear.
   */
  SELECT_MIN_HOLD_SCALE: 0.75,
  /** Skipping lands in READY for long enough to read the sentence's shape. */
  SKIP_READY_MS: 250,
} as const;

/**
 * Per-session decay, applied to hold states only. A visitor who has watched
 * two full sequences has learned to read them; the third does not need to
 * re-teach. Held in memory by the caller — nothing is written to storage.
 */
export const holdScaleForTurn = (turnIndex: number): number =>
  turnIndex <= 2 ? 1 : turnIndex <= 5 ? 0.6 : 0.4;

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

export interface Segment {
  state: PresentationState;
  startMs: number;
  durationMs: number;
}

export interface PresentationSchedule {
  segments: Segment[];
  /** End of READY — i.e. when the cue starts moving. */
  reasoningEndMs: number;
  /** End of STROKE — i.e. when the balls start moving. */
  strokeEndMs: number;
  /** The scale actually applied to the hold states (decay x ceiling). */
  holdScale: number;
  /** True when the sequence was collapsed for `prefers-reduced-motion`. */
  reduced: boolean;
}

export interface ScheduleInput {
  trace: DecisionTraceV1;
  /** Words in the plain-language sentence READY has to be long enough to read. */
  sentenceWords: number;
  /** Per-session decay (see `holdScaleForTurn`). */
  decay?: number;
  /** Collapse the choreography; the ball motion is content and is untouched. */
  reducedMotion?: boolean;
}

/** Candidates that really were handed to the physics simulator. */
const verifiedOf = (trace: DecisionTraceV1): TracedCandidate[] =>
  trace.candidates.filter((c) => c.physics !== null);

export function buildSchedule(input: ScheduleInput): PresentationSchedule {
  const { trace, sentenceWords } = input;
  const decay = input.decay ?? 1;

  const strokeSeg = (startMs: number): Segment => ({
    state: "STROKE",
    startMs,
    durationMs: TIMING.STROKE_MS,
  });

  if (input.reducedMotion) {
    // One static frame: every route in its final resolved state, the winner
    // locked, the sentence written. Held for READY's natural length.
    const ready = readyMs(sentenceWords);
    return {
      segments: [{ state: "READY", startMs: 0, durationMs: ready }, strokeSeg(ready)],
      reasoningEndMs: ready,
      strokeEndMs: ready + TIMING.STROKE_MS,
      holdScale: 1,
      reduced: true,
    };
  }

  const G = trace.budget.candidatesGenerated;
  const verified = verifiedOf(trace);
  const V = verified.length;
  const selectedIndex = trace.selected?.candidateIndex ?? null;
  const losers = verified.filter((c) => c.index !== selectedIndex).length;

  const natural = {
    ENUMERATING: clamp(G * TIMING.ENUM_PER_CANDIDATE_MS, TIMING.ENUM_MIN_MS, TIMING.ENUM_MAX_MS),
    // Absent, not greyed, when no model ran. The visitor sees a structurally
    // shorter sequence, which is the most honest rendering of the toggle.
    RANKING: trace.mode === "neural-hybrid" && trace.fallback === null ? TIMING.RANK_MS : 0,
    VERIFYING: verifyMs(trace, V),
    SELECTED: clamp(
      TIMING.SELECT_BASE_MS + losers * TIMING.SELECT_PER_LOSER_MS,
      TIMING.SELECT_MIN_MS,
      TIMING.SELECT_MAX_MS,
    ),
    READY: readyMs(sentenceWords),
  };

  // The ceiling is spent on the holds, never on VERIFYING: scaling preserves
  // relative cost, truncating would misrepresent it. (VERIFYING's own clamp is
  // a bound on the state, applied before this, and is disclosed as an average
  // in `verifyMs` — it is not a scale.)
  const holdsNatural = natural.ENUMERATING + natural.RANKING + natural.SELECTED + natural.READY;
  const budgetForHolds = TIMING.REASONING_CEILING_MS - natural.VERIFYING;
  const ceilingScale =
    holdsNatural * decay > budgetForHolds
      ? clamp(budgetForHolds / (holdsNatural || 1), TIMING.MIN_HOLD_SCALE, 1)
      : 1;
  const holdScale = clamp(decay * ceilingScale, TIMING.MIN_HOLD_SCALE, 1);
  // SELECTED shrinks on its own, gentler floor. See SELECT_MIN_HOLD_SCALE.
  const selectScale = clamp(decay * ceilingScale, TIMING.SELECT_MIN_HOLD_SCALE, 1);

  const segments: Segment[] = [];
  let t = 0;
  const push = (state: ReasoningState, ms: number) => {
    if (ms <= 0) return; // absent, not zero-length
    segments.push({ state, startMs: t, durationMs: ms });
    t += ms;
  };
  push("ENUMERATING", natural.ENUMERATING * holdScale);
  push("RANKING", natural.RANKING * holdScale);
  push("VERIFYING", natural.VERIFYING);
  push("SELECTED", natural.SELECTED * selectScale);
  push("READY", natural.READY * holdScale);

  const reasoningEndMs = t;
  segments.push(strokeSeg(t));

  return {
    segments,
    reasoningEndMs,
    strokeEndMs: reasoningEndMs + TIMING.STROKE_MS,
    holdScale,
    reduced: false,
  };
}

const readyMs = (words: number): number =>
  clamp(
    TIMING.READY_BASE_MS + words * TIMING.READY_PER_WORD_MS,
    TIMING.READY_MIN_MS,
    TIMING.READY_MAX_MS,
  );

/**
 * VERIFYING's length, from measured physics time.
 *
 * `timing.physicsMs` is the wall-clock the search really spent in the WASM
 * simulator; `V` is how many candidates really went through it. The quotient
 * is an average, and the state reveals one candidate per average — so the
 * state is as long as the physics was, and no individual candidate is claimed
 * to have cost any particular amount.
 *
 * The contract carries no per-candidate `verifyMs` and no progress stream, so
 * a per-candidate pace cannot be told truthfully. If T2 ever adds either, this
 * is the one function that changes.
 */
function verifyMs(trace: DecisionTraceV1, V: number): number {
  if (V === 0) return 0; // no physics ran; the state is absent, not empty
  const perItem = clamp(
    trace.timing.physicsMs / V,
    TIMING.VERIFY_MIN_ITEM_MS,
    TIMING.VERIFY_MAX_ITEM_MS,
  );
  return clamp(perItem * V, TIMING.VERIFY_MIN_MS, TIMING.VERIFY_MAX_MS);
}

/**
 * Jump to READY from wherever the sequence is. Always safe: the trace is
 * complete before the presentation starts, so skipping can never desync the
 * display from the decision.
 */
export function withSkip(schedule: PresentationSchedule, atMs: number): PresentationSchedule {
  const start = Math.max(0, atMs);
  const reasoningEndMs = start + TIMING.SKIP_READY_MS;
  return {
    ...schedule,
    segments: [
      { state: "READY", startMs: start, durationMs: TIMING.SKIP_READY_MS },
      { state: "STROKE", startMs: reasoningEndMs, durationMs: TIMING.STROKE_MS },
    ],
    reasoningEndMs,
    strokeEndMs: reasoningEndMs + TIMING.STROKE_MS,
  };
}

export interface Placement {
  state: PresentationState;
  elapsedInStateMs: number;
  durationMs: number;
  /** 0..1 within the current state; 1 once the sequence is past its end. */
  progress: number;
}

export function stateAt(schedule: PresentationSchedule, elapsedMs: number): Placement {
  for (const seg of schedule.segments) {
    if (elapsedMs < seg.startMs) {
      // Before the first segment (only possible after a skip).
      return { state: seg.state, elapsedInStateMs: 0, durationMs: seg.durationMs, progress: 0 };
    }
    if (elapsedMs < seg.startMs + seg.durationMs) {
      const e = elapsedMs - seg.startMs;
      return {
        state: seg.state,
        elapsedInStateMs: e,
        durationMs: seg.durationMs,
        progress: seg.durationMs > 0 ? e / seg.durationMs : 1,
      };
    }
  }
  return { state: "SHOOTING", elapsedInStateMs: 0, durationMs: 0, progress: 1 };
}

// --- What to draw ---------------------------------------------------------

export type RouteRole =
  /** Generated, not yet resolved by anything. */
  | "candidate"
  /** Physics ran and it survived. */
  | "verified"
  /** Eliminated. `reason` says why, and is null when the trace did not say. */
  | "rejected"
  /** The shot about to be played. */
  | "selected";

/**
 * Where a drawn route's geometry came from. The distinction the overlay used to
 * lose: a `plan` is what the candidate generator hoped for, a `simulated` route
 * is what the authoritative run measured. They are drawn differently (dashed
 * versus solid) and described differently, and one is never presented as the
 * other.
 */
export type RouteSource = "plan" | "simulated";

/** One ball's measured route, with the simulation time of every point. */
export interface MeasuredLeg {
  ballId: number;
  points: Vec2Trace[];
  timesSec: number[];
}

/**
 * The executed motion, split into the channels the overlay draws.
 *
 * `object` is the ball that dropped where there was one, otherwise the ball the
 * cue struck first — both read off the event log, never off the plan. `others`
 * is every remaining ball that moved, which is how a combination shows its
 * middle ball instead of implying the cue reached the pocket on its own.
 */
export interface MeasuredRoute {
  cue: MeasuredLeg | null;
  object: MeasuredLeg | null;
  others: MeasuredLeg[];
  durationSec: number;
}

const legOf = (t: ExecutedMotion["trajectories"][number]): MeasuredLeg => ({
  ballId: t.ballId,
  points: t.points,
  timesSec: t.timesSec,
});

export function measuredRoute(m: ExecutedMotion): MeasuredRoute {
  const cue = m.trajectories.find((t) => t.roles.includes("cue")) ?? null;
  const objects = m.trajectories.filter((t) => !t.roles.includes("cue"));
  const object =
    objects.find((t) => t.roles.includes("potted")) ??
    objects.find((t) => t.roles.includes("first-contact")) ??
    objects[0] ??
    null;
  return {
    cue: cue ? legOf(cue) : null,
    object: object ? legOf(object) : null,
    others: objects.filter((t) => t !== object).map(legOf),
    durationSec: m.durationSec,
  };
}

export interface RouteRender {
  index: number;
  kind: TracedKind;
  /**
   * Cue-ball geometry. For a `plan` this is [cue ball, ghost-ball contact
   * point] — the leg `candidate.path` omits. For a `simulated` route it is the
   * cue ball's whole measured route, cushions and all.
   */
  cueLeg: Vec2Trace[] | null;
  /** The object ball's route: planned for a `plan`, measured for a `simulated`. */
  objectLeg: Vec2Trace[];
  source: RouteSource;
  /** Non-null exactly when `source === "simulated"`. Carries the point times. */
  measured: MeasuredRoute | null;
  /** 0..1 of the route drawn so far, for the staggered reveal. */
  reveal: number;
  /** 0..1 line prominence. Driven by real ordering, never printed as a number. */
  weight: number;
  alpha: number;
  role: RouteRole;
  /** Only ever a reason the trace carried. Never inferred here. */
  reason: RejectionReason | null;
  /** True for the single route resolving this instant, during VERIFYING. */
  resolving: boolean;
  /**
   * True for the route physics most recently finished with. This is the one
   * whose rejection reason is captioned on the felt, so each elimination is
   * legible as it happens rather than as an undifferentiated fade.
   */
  justResolved: boolean;
}

export interface PresentationFrame {
  state: PresentationState;
  /** One of the five, or null outside the reasoning sequence. */
  label: string | null;
  progress: number;
  routes: RouteRender[];
  /** Cushion/contact marks from the executed simulation may be drawn. */
  showContacts: boolean;
  /** 0..1 through the backswing-and-strike, or null when the cue is still. */
  strokeProgress: number | null;
}

export interface FrameGeometry {
  /** Cue-ball position in world coordinates, for the cue leg. */
  cuePos: Vec2Trace;
}

const easeOut = (t: number): number => 1 - (1 - t) * (1 - t);

/** Rank a candidate 0..1 by the model's own ordering; 0.5 when no model ran. */
const priorWeight = (c: TracedCandidate, n: number): number =>
  c.neural === null ? 0.5 : 1 - (c.neural.rank - 1) / Math.max(1, n);

export function frameAt(
  trace: DecisionTraceV1,
  schedule: PresentationSchedule,
  elapsedMs: number,
  geom: FrameGeometry,
): PresentationFrame {
  const place = stateAt(schedule, elapsedMs);
  const state = place.state;
  const selectedIndex = trace.selected?.candidateIndex ?? null;
  const cands = trace.candidates;
  const n = cands.length;

  const base = (c: TracedCandidate): RouteRender => ({
    index: c.index,
    kind: c.kind,
    cueLeg: [geom.cuePos, c.aimPoint],
    objectLeg: c.path,
    source: "plan",
    measured: null,
    reveal: 1,
    weight: 0.35,
    alpha: 0.5,
    role: c.index === selectedIndex ? "selected" : "candidate",
    reason: null,
    resolving: false,
    justResolved: false,
  });

  /**
   * Swap a route's plan geometry for the measured motion, once there is any.
   *
   * Applied only to the shot being played, and only from SELECTED onward: while
   * the opponent is still comparing candidates the honest picture is the set of
   * intentions, and a measured route among them would say the others had been
   * simulated to the same fidelity. When `executed` is null — a decision replay
   * of a search-only trace, a simulation with no waypoints — the route stays a
   * plan and stays labelled one.
   */
  const withMeasured = (r: RouteRender): RouteRender => {
    const executed = trace.selected?.executed ?? null;
    if (executed === null) return r;
    const m = measuredRoute(executed);
    if (m.cue === null && m.object === null) return r;
    return {
      ...r,
      source: "simulated",
      measured: m,
      cueLeg: m.cue ? m.cue.points : null,
      objectLeg: m.object ? m.object.points : [],
    };
  };

  const routes: RouteRender[] = [];

  switch (state) {
    case "ENUMERATING": {
      // Every generated route, staggered, in generation order. Directs are
      // drawn: they were generated, and showing them is what makes the later
      // "excluded by policy" rejection legible rather than an assertion.
      const front = place.progress * n;
      for (const c of cands) {
        if (c.index >= front) continue;
        const age = clamp(front - c.index, 0, 1);
        const r = base(c);
        r.role = "candidate";
        r.reveal = easeOut(age);
        r.weight = 0.25;
        r.alpha = 0.18 + 0.32 * age;
        routes.push(r);
      }
      break;
    }

    case "RANKING": {
      // Re-weight by the model's ordering; the set it pruned fades out.
      for (const c of cands) {
        const r = base(c);
        const w = priorWeight(c, n);
        const pruned = c.rejection === "pruned-by-prior";
        r.role = pruned ? "rejected" : "candidate";
        r.reason = pruned ? c.rejection : null;
        r.weight = 0.2 + 0.6 * w;
        r.alpha = pruned ? 0.5 * (1 - place.progress) : 0.25 + 0.4 * w * place.progress + 0.1;
        routes.push(r);
      }
      break;
    }

    case "VERIFYING": {
      // One route resolves per revealed step, in generation order, over the
      // real physics time. Everything the search never simulated stays faint.
      const verified = cands.filter((c) => c.physics !== null);
      const resolvedCount = Math.floor(place.progress * verified.length);
      const resolvedIds = new Set(verified.slice(0, resolvedCount).map((c) => c.index));
      const resolvingId = verified[resolvedCount]?.index ?? null;
      const justResolvedId = resolvedCount > 0 ? verified[resolvedCount - 1].index : null;

      for (const c of cands) {
        const r = base(c);
        if (c.physics === null) {
          // Never simulated. Still drawn — it was considered — but at the
          // weight of a memory, so the routes under real examination read.
          r.role = c.rejection === null ? "candidate" : "rejected";
          r.reason = c.rejection;
          r.weight = 0.15;
          r.alpha = 0.07;
        } else if (resolvedIds.has(c.index)) {
          const ok = c.physics.legalPot && !c.physics.scratched;
          r.justResolved = c.index === justResolvedId;
          r.role = ok ? "verified" : "rejected";
          r.reason = ok ? null : c.rejection;
          // Real search effort: visit share is what the bandit actually spent.
          r.weight = ok ? 0.45 + 0.35 * clamp(c.physics.strength, 0, 1) : 0.2;
          r.alpha = ok ? 0.8 : 0.22;
        } else if (c.index === resolvingId) {
          r.role = "candidate";
          r.resolving = true;
          r.weight = 0.7;
          r.alpha = 0.85;
        } else {
          r.role = "candidate";
          r.weight = 0.25;
          r.alpha = 0.3;
        }
        routes.push(r);
      }
      break;
    }

    case "SELECTED": {
      // Survivors that lost the selection dim away; the winner locks.
      for (const c of cands) {
        const r = base(c);
        if (c.index === selectedIndex) {
          r.role = "selected";
          r.weight = 1;
          r.alpha = 1;
          routes.push(withMeasured(r));
          continue;
        } else if (c.physics !== null) {
          r.role = "rejected";
          r.reason = c.rejection;
          r.weight = 0.25;
          r.alpha = 0.45 * (1 - easeOut(place.progress));
        } else {
          r.role = "rejected";
          r.reason = c.rejection;
          r.weight = 0.15;
          r.alpha = 0.07 * (1 - place.progress);
        }
        routes.push(r);
      }
      break;
    }

    case "READY":
    case "STROKE":
    case "SHOOTING": {
      const chosen = selectedIndex === null ? null : cands.find((c) => c.index === selectedIndex);
      if (chosen) {
        const r = base(chosen);
        r.role = "selected";
        r.weight = 1;
        r.alpha = state === "SHOOTING" ? 0.7 : 1;
        routes.push(withMeasured(r));
      } else if (trace.selected) {
        // A generated safety has no candidate row, so its own trace entry is
        // the route. `cuePath` is the contract's cue-ball route — for a kick,
        // [cue position, rail point, target] — and its first segment is the
        // line the white actually travels first. Only that segment is drawn:
        // the rest of the kick is cue-ball geometry, and `objectLeg` is the
        // OBJECT ball's route, which a safety plans none of (`path` is empty).
        // Putting cue geometry there would draw more of the shot at the cost of
        // saying something false about which ball goes where.
        routes.push(
          withMeasured({
            index: -1,
            kind: trace.selected.kind,
            cueLeg:
              trace.selected.cuePath.length >= 2
                ? [trace.selected.cuePath[0], trace.selected.cuePath[1]]
                : null,
            objectLeg: trace.selected.path,
            source: "plan",
            measured: null,
            reveal: 1,
            weight: 1,
            alpha: state === "SHOOTING" ? 0.7 : 1,
            role: "selected",
            reason: null,
            resolving: false,
            justResolved: false,
          }),
        );
      }
      break;
    }

    default:
      break;
  }

  return {
    state,
    label: isReasoning(state) ? STATE_LABEL[state] : null,
    progress: place.progress,
    routes,
    showContacts:
      state === "SELECTED" || state === "READY" || state === "STROKE" || state === "SHOOTING",
    strokeProgress: state === "STROKE" ? place.progress : null,
  };
}

export const isReasoning = (s: PresentationState): s is ReasoningState =>
  (REASONING_STATES as readonly string[]).includes(s);

/**
 * Short, plain phrases for the rejection reasons the contract defines. The
 * renderer may only ever print one of these, and only when the trace supplied
 * the corresponding reason — it never derives a reason of its own.
 */
export const REJECTION_TEXT: Record<RejectionReason, string> = {
  "direct-excluded-by-policy": "direct — excluded",
  "pruned-by-prior": "pruned before physics",
  "budget-exhausted": "no budget left",
  "seed-timeout": "search deadline",
  "scratched-in-simulation": "scratches",
  "illegal-first-contact": "illegal first contact",
  "did-not-pot": "did not pot",
  "below-reliability-threshold": "below the bar",
  "lower-utility-than-selected": "lower utility",
};
