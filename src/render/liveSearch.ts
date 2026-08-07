// The opponent's decision, OBSERVED.
//
// `presentation.ts` paces a finished `DecisionTraceV1`: it knows the answer
// before it draws the first frame, and its honesty rests on never claiming
// otherwise. This module is the other half. It consumes the live event stream
// from `ai/search/progress.ts` and holds exactly what the search has published
// SO FAR — so a route on the felt is there because the search has reached it,
// and a rejection caption is there because the search has just decided it.
//
// WHAT THIS MODULE MAY NOT DO, AND DOES NOT
//
//   - It may not invent an event. Every field below is written by `apply` from
//     an event the search published; there is no timer in this file, no
//     interpolation between events, and no state that advances on its own.
//   - It may not show a progress percentage. It cannot: the search does not
//     know how many candidates it will reach before the budget or the clock
//     stops it, so no honest denominator exists. Motion comes from events
//     arriving, which is the real thing.
//   - It may not drop a resolved candidate. Everything the search finishes with
//     stays on the felt in its resolved state for the rest of the search, which
//     is what makes a fast stream readable without slowing the search down.
//
// The five state labels are unchanged from the scheduled presentation, and
// deliberately so: they were already the right names for the five things that
// happen. What changed is that they are now reached by the search reaching
// them, rather than by a schedule built after the fact.

import type { RejectionReason } from "../ai/trace/contract";
import {
  SEARCH_PROGRESS_VERSION,
  type ProgressCandidate,
  type ProgressPhysics,
  type SearchProgressEvent,
} from "../ai/search/progress";
import type {
  FrameGeometry,
  PresentationFrame,
  PresentationState,
  RouteRender,
  RouteRole,
} from "./presentation";
import { STATE_LABEL, isReasoning } from "./presentation";

/** Where the observed search has got to. Each is entered by a real event. */
export type LivePhase =
  /** A search has started; no candidates published yet. */
  | "started"
  /** Candidate geometry has arrived. */
  | "enumerated"
  /** The model has returned its ordering. Never reached in classical mode. */
  | "ranked"
  /** At least one candidate has gone into the simulator. */
  | "verifying"
  /** The policy has chosen. */
  | "selected"
  /** `search-completed` has arrived. */
  | "completed";

/** One candidate, as far as the live stream has described it. */
export interface LiveCandidate {
  candidate: ProgressCandidate;
  role: RouteRole;
  reason: RejectionReason | null;
  /** The search's own bounded value. 0 until a `candidate-retained` says otherwise. */
  strength: number;
  visits: number;
  physics: ProgressPhysics | null;
  /** 1-based model rank, or null when no model ranked this search. */
  rank: number | null;
  score: number | null;
  /** In the simulator right now. */
  simulating: boolean;
  /** The `seq` of the event that last changed this candidate's role. */
  resolvedSeq: number | null;
}

export interface LiveSearch {
  version: typeof SEARCH_PROGRESS_VERSION;
  phase: LivePhase;
  mode: "classical" | "neural-hybrid" | null;
  /** Index-aligned with the candidate list; sparse until geometry arrives. */
  candidates: LiveCandidate[];
  modelId: string | null;
  inferenceMs: number | null;
  fallback: { cause: string; detail: string } | null;
  selectedIndex: number | null;
  selectedKind: string | null;
  selectedRung: string | null;
  simulatingIndex: number | null;
  /**
   * The candidate whose rejection reason is captioned on the felt, and the
   * event time at which it took the caption.
   *
   * A caption needs a minimum dwell to be readable, and the search does not
   * provide one: measured in Chrome against a real dev build, consecutive
   * simulation results arrive 12-126 ms apart (median ~77 ms over two turns).
   * Captioning the newest rejection unconditionally would strobe.
   *
   * The dwell is applied to the CAPTION only. Routes still resolve at the rate
   * the search resolves them, nothing is delayed, and the search is not slowed
   * — the only thing held back is which of several real reasons is currently
   * spelled out. `captionAtMs` comes from the events' own `atMs`, so this is
   * driven by the search's clock rather than by a timer of the renderer's.
   */
  captionIndex: number | null;
  captionAtMs: number;
  /** Real counters, from real events. Never a denominator for a percentage. */
  counts: { generated: number; simulated: number; retained: number; rejected: number };
  completed: {
    physicsUnitsSpent: number;
    physicsMs: number;
    physicsVerified: number;
    seedTimedOut: boolean;
    searchTimedOut: boolean;
  } | null;
  /**
   * Every event, in arrival order.
   *
   * Retained rather than folded away, because the fold is lossy and the stream
   * is the evidence: this is what the decision replay is built from, what a
   * test asserts ordering against, and what a future debugging surface would
   * read. A turn publishes on the order of a hundred small objects; keeping one
   * turn's worth is not a memory concern, and only the current turn is kept.
   */
  events: SearchProgressEvent[];
  /** Events whose `seq` did not follow the previous one. Should always be empty. */
  outOfOrder: number;
}

export function createLiveSearch(): LiveSearch {
  return {
    version: SEARCH_PROGRESS_VERSION,
    phase: "started",
    mode: null,
    candidates: [],
    modelId: null,
    inferenceMs: null,
    fallback: null,
    selectedIndex: null,
    selectedKind: null,
    selectedRung: null,
    simulatingIndex: null,
    captionIndex: null,
    captionAtMs: -Infinity,
    counts: { generated: 0, simulated: 0, retained: 0, rejected: 0 },
    completed: null,
    events: [],
    outOfOrder: 0,
  };
}

/**
 * Minimum time a rejection caption holds the felt before another may take it.
 *
 * 420 ms, from the measured arrival rate rather than from taste: consecutive
 * simulation results land 12-126 ms apart in a real browser, so without a dwell
 * a caption is legible only when the search happens to be slow. Two or three
 * words at a glance need roughly this long, and it is short enough that a
 * ~1.9 s search still spells out four or five distinct reasons.
 *
 * This is a display rate limit on ONE line of text. It delays nothing, hides no
 * route, and every caption it does show is a real reason for a real route.
 */
export const CAPTION_DWELL_MS = 420;

const blank = (candidate: ProgressCandidate): LiveCandidate => ({
  candidate,
  // A candidate whose geometry has arrived and which nothing has resolved is
  // exactly that: considered, undecided.
  role: "candidate",
  reason: null,
  strength: 0,
  visits: 0,
  physics: null,
  rank: null,
  score: null,
  simulating: false,
  resolvedSeq: null,
});

/**
 * Fold one event in. Mutates, and returns the same object: this is called once
 * per event on the frame path, and allocating a new tree per event would make
 * the observer the expensive part of watching a search.
 *
 * A phase never moves backwards — `selected` arriving before `search-completed`
 * is the normal order, and a late `candidate-rejected` (the policy-stage ones,
 * published after selection) must not drag the display back to "verifying".
 */
export function applyProgress(live: LiveSearch, e: SearchProgressEvent): LiveSearch {
  const prev = live.events[live.events.length - 1];
  if (prev !== undefined && e.seq !== prev.seq + 1) live.outOfOrder++;
  live.events.push(e);

  const at = (index: number): LiveCandidate | null => live.candidates[index] ?? null;
  const advance = (p: LivePhase) => {
    const order: LivePhase[] = ["started", "enumerated", "ranked", "verifying", "selected", "completed"];
    if (order.indexOf(p) > order.indexOf(live.phase)) live.phase = p;
  };

  switch (e.kind) {
    case "search-started":
      live.mode = e.mode;
      break;

    case "candidates-generated": {
      live.candidates = [];
      for (const c of e.candidates) live.candidates[c.index] = blank(c);
      live.counts.generated = e.candidates.length;
      advance("enumerated");
      break;
    }

    case "neural-scored": {
      live.modelId = e.modelId;
      live.inferenceMs = e.inferenceMs;
      for (const r of e.ranks) {
        const c = at(r.index);
        if (c) {
          c.rank = r.rank;
          c.score = r.score;
        }
      }
      advance("ranked");
      break;
    }

    case "prior-pruned":
      // The per-candidate consequence arrives as its own `candidate-rejected`
      // events; this one carries the shape of the decision, not its members.
      break;

    case "candidate-simulating": {
      const c = at(e.index);
      if (c) c.simulating = true;
      live.simulatingIndex = e.index;
      advance("verifying");
      break;
    }

    case "candidate-verified": {
      const c = at(e.index);
      if (c) {
        c.simulating = false;
        c.physics = e.physics;
      }
      if (live.simulatingIndex === e.index) live.simulatingIndex = null;
      live.counts.simulated++;
      break;
    }

    case "candidate-rejected": {
      const c = at(e.index);
      if (c) {
        c.role = "rejected";
        c.reason = e.reason;
        c.simulating = false;
        c.resolvedSeq = e.seq;
      }
      live.counts.rejected++;
      // Only a rejection the visitor can watch land gets the caption. A
      // `direct-excluded-by-policy` is published for every direct in one burst
      // before any physics runs, and captioning the last of those would put a
      // reason on the felt for a route nobody was looking at.
      if (
        e.reason !== "direct-excluded-by-policy" &&
        e.reason !== "pruned-by-prior" &&
        e.atMs - live.captionAtMs >= CAPTION_DWELL_MS
      ) {
        live.captionIndex = e.index;
        live.captionAtMs = e.atMs;
      }
      break;
    }

    case "candidate-retained": {
      const c = at(e.index);
      if (c) {
        // A refinement round republishes a candidate that is already retained;
        // it updates the numbers and does not re-announce the route.
        if (c.role !== "verified") live.counts.retained++;
        c.role = "verified";
        c.reason = null;
        c.simulating = false;
        c.strength = e.strength;
        c.visits = e.visits;
        c.resolvedSeq = e.seq;
      }
      // Deliberately does NOT take the caption. The caption spells out why a
      // route LOST; a survivor has no reason to spell out, and letting it hold
      // the slot only meant the next real rejection had to wait out a dwell it
      // had not earned.
      break;
    }

    case "fallback":
      live.fallback = { cause: e.cause, detail: e.detail };
      // The decision is being made without the model, so the display must not
      // sit in a state that says a model is ranking.
      live.mode = "classical";
      break;

    case "selected": {
      live.selectedIndex = e.index;
      live.selectedKind = e.shotKind;
      live.selectedRung = e.rung;
      if (e.index !== null) {
        const c = at(e.index);
        if (c) {
          c.role = "selected";
          c.reason = null;
        }
      }
      live.simulatingIndex = null;
      live.captionIndex = null;
      advance("selected");
      break;
    }

    case "search-completed":
      live.completed = {
        physicsUnitsSpent: e.physicsUnitsSpent,
        physicsMs: e.physicsMs,
        physicsVerified: e.physicsVerified,
        seedTimedOut: e.seedTimedOut,
        searchTimedOut: e.searchTimedOut,
      };
      live.simulatingIndex = null;
      advance("completed");
      break;

    default:
      break;
  }
  return live;
}

/**
 * The panel's state label for an observed search.
 *
 * Same five words the scheduled presentation uses, because they were already
 * the right names for these five things. `RANKING` is reachable only when a
 * model actually returned an ordering, so a classical search shows a
 * structurally shorter sequence here exactly as it does in the replay.
 */
export function liveState(live: LiveSearch): PresentationState {
  switch (live.phase) {
    case "started":
      return "IDLE";
    case "enumerated":
      return "ENUMERATING";
    case "ranked":
      return "RANKING";
    case "verifying":
      return "VERIFYING";
    case "selected":
    case "completed":
      return "SELECTED";
  }
}

/** 0..1 line prominence from the model's own ordering. 0.5 when none ranked. */
const rankWeight = (c: LiveCandidate, n: number): number =>
  c.rank === null ? 0.5 : 1 - (c.rank - 1) / Math.max(1, n);

const clamp01 = (v: number): number => Math.min(1, Math.max(0, v));

/**
 * What is on the felt right now.
 *
 * Every route drawn corresponds to a candidate whose geometry the search has
 * published, in the state the search's own events have left it. There is no
 * reveal fraction and no stagger: a route is drawn in full because it exists in
 * full — the search generated the whole thing at once, and animating it in
 * would be inventing a process that did not happen.
 */
export function liveFrame(live: LiveSearch, geom: FrameGeometry): PresentationFrame {
  const state = liveState(live);
  const n = live.counts.generated;
  const routes: RouteRender[] = [];

  for (const c of live.candidates) {
    if (!c) continue;
    const w = rankWeight(c, n);
    let weight: number;
    let alpha: number;

    if (c.role === "selected") {
      weight = 1;
      alpha = 1;
    } else if (c.simulating) {
      // In the simulator this instant. The brightest thing that is not the
      // answer, because it is the only route anything is happening to.
      weight = 0.7;
      alpha = 0.85;
    } else if (c.role === "verified") {
      // Real search effort: the value the physics produced.
      weight = 0.45 + 0.35 * clamp01(c.strength);
      alpha = 0.8;
    } else if (c.role === "rejected") {
      // Eliminated before physics versus eliminated by physics: the second one
      // cost a simulation and stays more legible than the first.
      const bySearch = c.physics !== null;
      weight = bySearch ? 0.2 : 0.15;
      alpha = bySearch ? 0.22 : 0.07;
    } else {
      weight = 0.25;
      alpha = live.phase === "enumerated" ? 0.18 + 0.32 * w : 0.3;
    }

    routes.push({
      index: c.candidate.index,
      kind: c.candidate.kind,
      cueLeg: [geom.cuePos, c.candidate.aimPoint],
      objectLeg: c.candidate.path,
      // Live routes are always PLAN geometry. The measured motion does not
      // exist until the shot is executed, which is after the search; drawing a
      // plan as a measured route is the exact conflation the contract forbids.
      source: "plan",
      measured: null,
      reveal: 1,
      weight,
      alpha,
      role: c.role,
      reason: c.reason,
      resolving: c.simulating,
      justResolved: c.candidate.index === live.captionIndex && c.reason !== null,
    });
  }

  return {
    state,
    label: isReasoning(state) ? STATE_LABEL[state] : null,
    // Deliberately constant. `PresentationFrame.progress` drives nothing on the
    // felt for a live search, and there is no honest value to put here — see
    // the header. It is not rendered as a bar, a percentage or a counter.
    progress: 0,
    routes,
    // The contact marks come from the EXECUTED simulation, which has not
    // happened yet during a live search.
    showContacts: false,
    strokeProgress: null,
  };
}
