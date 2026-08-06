// The presentation state machine is pure, so it is tested as arithmetic and as
// a set of drawing rules — no canvas, no React, no clock.
//
// The three things this file is really guarding:
//   1. RANKING is ABSENT in classical mode, not greyed. The structurally
//      shorter sequence is how the toggle explains itself.
//   2. The shot about to be played is always drawn. The overlay used to draw
//      `stats.slice(0, 6)` ordered by *visits* while selection was by *trick
//      utility*, so with enough candidates the chosen route could be missing
//      from the picture entirely. Addressing the winner by generation index
//      makes that unrepresentable.
//   3. No route ever carries a rejection reason the trace did not supply.

import { describe, it, expect } from "vitest";
import type {
  DecisionTraceV1,
  RejectionReason,
  TracedCandidate,
  TracedKind,
} from "../ai/trace/contract";
import { DECISION_TRACE_VERSION } from "../ai/trace/contract";
import {
  TIMING,
  buildSchedule,
  frameAt,
  holdScaleForTurn,
  stateAt,
  withSkip,
} from "./presentation";

const geom = { cuePos: { x: -0.6, y: -0.1 } };

function candidate(
  index: number,
  over: Partial<TracedCandidate> & { verified?: boolean; pots?: boolean } = {},
): TracedCandidate {
  const { verified, pots, ...rest } = over;
  return {
    index,
    kind: (over.kind ?? "bank") as TracedKind,
    eligible: true,
    target: 1 + (index % 7),
    potId: 1 + (index % 7),
    pocket: "tl",
    aimPoint: { x: 0.1 * index, y: 0.02 * index },
    // The contract's cue-ball route: from the cue ball's real position to the
    // ghost-ball contact point, exactly as `ai/trace/build.ts` writes it.
    cuePath: [geom.cuePos, { x: 0.1 * index, y: 0.02 * index }],
    path: [
      { x: 0.1 * index, y: 0.02 * index },
      { x: 0.4, y: 0.3 },
      { x: 0.9, y: 0.45 },
    ],
    action: { phi: 0.2, power: 0.6, sideSpin: 0, topSpin: 0 },
    neural: null,
    physics: verified
      ? {
          firstContact: 1,
          legalFirstContact: true,
          scratched: false,
          legalPot: pots ?? true,
          pocketed: pots ?? true ? [1] : [],
          railContacts: [],
          railsBeforePot: 1,
          contactSequence: [],
          strength: 0.4 + index * 0.01,
          value: 0.5,
          visits: 3,
          styleScore: 0.2,
        }
      : null,
    rejection: null,
    ...rest,
  };
}

function trace(over: Partial<DecisionTraceV1> = {}): DecisionTraceV1 {
  const candidates = over.candidates ?? [
    candidate(0, { verified: true, pots: true }),
    candidate(1, { verified: true, pots: false, rejection: "did-not-pot" }),
    candidate(2, { rejection: "budget-exhausted" }),
  ];
  return {
    version: DECISION_TRACE_VERSION,
    policy: "trick-only",
    mode: "classical-trick-only",
    turn: { player: 1, shotIndex: 4, legalTargets: [1, 2], cueBall: geom.cuePos },
    model: null,
    budget: {
      physicsUnitsAllowed: 60,
      physicsUnitsSpent: 42,
      safetySimsSpent: 0,
      candidatesGenerated: candidates.length,
      candidatesEligible: candidates.length,
      candidatesConsidered: candidates.length,
      physicsVerified: candidates.filter((c) => c.physics !== null).length,
      prunedByPrior: 0,
      reservePromotions: 0,
      seedTimedOut: false,
    },
    timing: { totalMs: 900, neuralEncodeMs: null, neuralRunMs: null, physicsMs: 880, selectionMs: 0 },
    candidates,
    selected: {
      candidateIndex: 0,
      kind: "bank",
      rung: "trick-qualified",
      action: { phi: 0.2, power: 0.6, sideSpin: 0, topSpin: 0 },
      // Guarded: a fixture may pass `candidates: []` and its own `selected`,
      // and this default is still evaluated before `...over` replaces it.
      cuePath: candidates[0]?.cuePath ?? [],
      path: candidates[0]?.path ?? [],
      utility: 0.6,
      reliabilityThreshold: 0.5,
      qualifyingTricks: 1,
      safetyQuality: null,
    },
    fallback: null,
    ...over,
  };
}

const statesIn = (t: DecisionTraceV1, words = 10) =>
  buildSchedule({ trace: t, sentenceWords: words }).segments.map((s) => s.state);

describe("the sequence is structurally shorter when no model ran", () => {
  it("classical mode has no RANKING segment at all", () => {
    expect(statesIn(trace())).toEqual([
      "ENUMERATING",
      "VERIFYING",
      "SELECTED",
      "READY",
      "STROKE",
    ]);
  });

  it("neural mode inserts RANKING between enumeration and verification", () => {
    expect(statesIn(trace({ mode: "neural-hybrid" }))).toEqual([
      "ENUMERATING",
      "RANKING",
      "VERIFYING",
      "SELECTED",
      "READY",
      "STROKE",
    ]);
  });

  it("a neural run that fell back to classical does not claim a ranking state", () => {
    const t = trace({
      mode: "neural-hybrid",
      fallback: {
        from: "neural-hybrid",
        to: "classical-trick-only",
        cause: "model-absent",
        detail: "model unavailable: HTTP 404",
      },
    });
    expect(statesIn(t)).not.toContain("RANKING");
  });

  it("VERIFYING is absent when no candidate reached physics", () => {
    const cands = [candidate(0), candidate(1)];
    const t = trace({
      candidates: cands,
      selected: null,
    });
    expect(statesIn(t)).not.toContain("VERIFYING");
  });
});

describe("timing derives from the trace, not from constants", () => {
  it("ENUMERATING scales with the number of generated candidates, within its bounds", () => {
    const few = buildSchedule({ trace: trace(), sentenceWords: 8 });
    const many = buildSchedule({
      trace: trace({
        candidates: Array.from({ length: 45 }, (_, i) => candidate(i, { verified: i < 16 })),
      }),
      sentenceWords: 8,
    });
    const enumOf = (s: typeof few) => s.segments.find((x) => x.state === "ENUMERATING")!.durationMs;
    expect(enumOf(few)).toBe(TIMING.ENUM_MIN_MS);
    expect(enumOf(many)).toBeCloseTo(45 * TIMING.ENUM_PER_CANDIDATE_MS, 5);
    expect(enumOf(many)).toBeLessThanOrEqual(TIMING.ENUM_MAX_MS);
  });

  it("VERIFYING comes from the search's real physics time", () => {
    const t = trace({
      timing: { totalMs: 2400, neuralEncodeMs: null, neuralRunMs: null, physicsMs: 2400, selectionMs: 0 },
      candidates: Array.from({ length: 16 }, (_, i) => candidate(i, { verified: true })),
    });
    const v = buildSchedule({ trace: t, sentenceWords: 8 }).segments.find(
      (s) => s.state === "VERIFYING",
    )!;
    // 2400 ms over 16 verified candidates = 150 ms each, inside the per-item
    // bounds, so the state is exactly as long as the physics was.
    expect(v.durationMs).toBeCloseTo(2400, 5);
  });

  it("READY is long enough to read the sentence, and no longer than its ceiling", () => {
    const short = buildSchedule({ trace: trace(), sentenceWords: 1 });
    const long = buildSchedule({ trace: trace(), sentenceWords: 200 });
    const readyOf = (s: typeof short) => s.segments.find((x) => x.state === "READY")!.durationMs;
    expect(readyOf(short)).toBe(TIMING.READY_MIN_MS);
    expect(readyOf(long)).toBe(TIMING.READY_MAX_MS);
  });

  it("the 6.5 s ceiling is spent on the holds and never on VERIFYING", () => {
    const t = trace({
      mode: "neural-hybrid",
      timing: { totalMs: 9000, neuralEncodeMs: null, neuralRunMs: null, physicsMs: 9000, selectionMs: 0 },
      candidates: Array.from({ length: 45 }, (_, i) => candidate(i, { verified: true })),
    });
    const s = buildSchedule({ trace: t, sentenceWords: 40 });
    const verify = s.segments.find((x) => x.state === "VERIFYING")!;
    // Clamped by VERIFY_MAX_MS, which is a bound on the state, not a scale.
    expect(verify.durationMs).toBe(TIMING.VERIFY_MAX_MS);
    expect(s.reasoningEndMs).toBeLessThanOrEqual(TIMING.REASONING_CEILING_MS + 0.001);
    expect(s.holdScale).toBeLessThan(1);
  });

  it("the per-session decay shortens the holds and leaves VERIFYING alone", () => {
    const t = trace({
      candidates: Array.from({ length: 20 }, (_, i) => candidate(i, { verified: i < 8 })),
    });
    const full = buildSchedule({ trace: t, sentenceWords: 10, decay: holdScaleForTurn(1) });
    const late = buildSchedule({ trace: t, sentenceWords: 10, decay: holdScaleForTurn(9) });
    const dur = (s: typeof full, name: string) =>
      s.segments.find((x) => x.state === name)!.durationMs;
    expect(dur(late, "ENUMERATING")).toBeLessThan(dur(full, "ENUMERATING"));
    expect(dur(late, "VERIFYING")).toBe(dur(full, "VERIFYING"));
  });
});

describe("skipping", () => {
  it("lands in READY for 250 ms from wherever it is asked", () => {
    const s = withSkip(buildSchedule({ trace: trace(), sentenceWords: 10 }), 700);
    expect(stateAt(s, 700).state).toBe("READY");
    expect(stateAt(s, 700 + TIMING.SKIP_READY_MS - 1).state).toBe("READY");
    expect(stateAt(s, 700 + TIMING.SKIP_READY_MS + 1).state).toBe("STROKE");
  });
});

describe("reduced motion", () => {
  it("collapses the choreography to one resolved frame, and keeps the stroke", () => {
    const s = buildSchedule({ trace: trace(), sentenceWords: 10, reducedMotion: true });
    expect(s.segments.map((x) => x.state)).toEqual(["READY", "STROKE"]);
    expect(s.reduced).toBe(true);
  });
});

describe("what gets drawn", () => {
  const many = Array.from({ length: 24 }, (_, i) =>
    candidate(i, { verified: i < 14, pots: i % 3 === 0 }),
  );
  // The winner sits deep in generation order and is NOT the most-visited row —
  // exactly the shape that used to drop it out of the drawn set.
  const t = trace({
    candidates: many.map((c) =>
      c.index === 19
        ? { ...c, physics: { ...candidate(19, { verified: true }).physics! } }
        : c,
    ),
    selected: {
      candidateIndex: 19,
      kind: "double-bank",
      rung: "trick-qualified",
      action: { phi: 0.1, power: 0.7, sideSpin: 0, topSpin: 0 },
      cuePath: many[19].cuePath,
      path: many[19].path,
      utility: 0.7,
      reliabilityThreshold: 0.5,
      qualifyingTricks: 2,
      safetyQuality: null,
    },
  });
  const schedule = buildSchedule({ trace: t, sentenceWords: 10 });

  it("the shot about to be played is drawn in every state from SELECTED on", () => {
    for (const at of ["SELECTED", "READY", "STROKE"] as const) {
      const seg = schedule.segments.find((s) => s.state === at)!;
      const frame = frameAt(t, schedule, seg.startMs + seg.durationMs / 2, geom);
      const selected = frame.routes.filter((r) => r.role === "selected");
      expect(selected, `no selected route drawn during ${at}`).toHaveLength(1);
      expect(selected[0].index).toBe(19);
      expect(selected[0].alpha).toBeGreaterThan(0.3);
    }
  });

  it("no route is truncated out of the enumeration", () => {
    const seg = schedule.segments.find((s) => s.state === "ENUMERATING")!;
    const end = frameAt(t, schedule, seg.startMs + seg.durationMs - 1, geom);
    expect(end.routes).toHaveLength(t.candidates.length);
  });

  it("every route is drawn cue-first", () => {
    const seg = schedule.segments.find((s) => s.state === "VERIFYING")!;
    const frame = frameAt(t, schedule, seg.startMs + 10, geom);
    for (const r of frame.routes) {
      expect(r.cueLeg![0]).toEqual(geom.cuePos);
      // The object leg still starts where the trace says the object ball is.
      expect(r.objectLeg[0]).toEqual(t.candidates[r.index].path[0]);
    }
  });

  it("no route carries a reason the trace did not supply", () => {
    const supplied = new Map<number, RejectionReason | null>(
      t.candidates.map((c) => [c.index, c.rejection]),
    );
    for (let ms = 0; ms < schedule.strokeEndMs; ms += 37) {
      for (const r of frameAt(t, schedule, ms, geom).routes) {
        if (r.reason === null) continue;
        expect(r.reason).toBe(supplied.get(r.index));
      }
    }
  });

  it("only one route resolves at a time during verification, in generation order", () => {
    const seg = schedule.segments.find((s) => s.state === "VERIFYING")!;
    const seen: number[] = [];
    for (let ms = seg.startMs; ms < seg.startMs + seg.durationMs; ms += 5) {
      const resolving = frameAt(t, schedule, ms, geom).routes.filter((r) => r.resolving);
      expect(resolving.length).toBeLessThanOrEqual(1);
      if (resolving.length === 1 && seen[seen.length - 1] !== resolving[0].index) {
        seen.push(resolving[0].index);
      }
    }
    expect(seen).toEqual([...seen].sort((a, b) => a - b));
    expect(seen.every((i) => t.candidates[i].physics !== null)).toBe(true);
  });

  it("at most one route is captioned at a time, and only ones physics finished with", () => {
    const seg = schedule.segments.find((s) => s.state === "VERIFYING")!;
    for (let ms = seg.startMs; ms < seg.startMs + seg.durationMs; ms += 7) {
      const just = frameAt(t, schedule, ms, geom).routes.filter((r) => r.justResolved);
      expect(just.length).toBeLessThanOrEqual(1);
      for (const r of just) expect(t.candidates[r.index].physics).not.toBeNull();
    }
  });

  it("a generated safety draws the cue leg the trace gave it, and no object route", () => {
    // A safety kick is not a member of the candidate list, so it has no row to
    // be addressed by index. Its route has to come from `selected.cuePath` —
    // [cue position, rail point, target] — and `selected.path` is empty,
    // because a kick plans no object-ball route at all. Before the real
    // contract landed there was no `cuePath` and this drew nothing.
    const rail = { x: 0.8, y: -0.5 };
    const safety = trace({
      candidates: [],
      selected: {
        candidateIndex: null,
        kind: "safety-kick",
        rung: "non-direct-safety",
        action: { phi: -0.4, power: 0.35, sideSpin: 0, topSpin: 0 },
        cuePath: [geom.cuePos, rail, { x: 0.2, y: 0.3 }],
        path: [],
        utility: null,
        reliabilityThreshold: 0.5,
        qualifyingTricks: 0,
        safetyQuality: "foul-free",
      },
    });
    const s = buildSchedule({ trace: safety, sentenceWords: 6 });
    const seg = s.segments.find((x) => x.state === "READY")!;
    const frame = frameAt(safety, s, seg.startMs + 1, geom);
    expect(frame.routes).toHaveLength(1);
    expect(frame.routes[0].role).toBe("selected");
    expect(frame.routes[0].kind).toBe("safety-kick");
    // The first real segment of the cue's route, not a line invented between
    // the white and wherever the shot ends up.
    expect(frame.routes[0].cueLeg).toEqual([geom.cuePos, rail]);
    expect(frame.routes[0].objectLeg).toEqual([]);
  });

  it("the label slot is empty outside the reasoning sequence", () => {
    expect(frameAt(t, schedule, schedule.strokeEndMs + 1, geom).label).toBeNull();
  });
});
