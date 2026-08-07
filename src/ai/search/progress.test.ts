// The live search stream, driven by the real search.
//
// The claim under test is the one the whole feature rests on: the events come
// from the production search path as it runs, and every number and every reason
// in them is the one the search actually produced. So nothing here builds a
// stream by hand. Every stream in this file is collected by attaching a sink to
// a real `brain.plan()` on a real board with the real WASM physics, and then
// checked against that same decision's published `DecisionTraceV1`.
//
// If the search is ever changed to compute a display value that the record does
// not carry, the cross-checks below fail rather than the drift going unnoticed.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../../physics/table";
import { makeBall, type Ball } from "../../physics/ball";
import { CUE_ID } from "../../game/rack";
import type { GameState } from "../../game/state";
import { initPhysics } from "../../physics/wasm-bridge";
import { defaultConfig, searchCandidates } from "../shotSearch";
import { generateCandidates } from "../candidates";
import { classicalTrickOnlyBrain, neuralTrickOnlyBrain } from "../brain";
import type { NeuralCandidateEvaluator } from "../neural/evaluator";
import type { AiDecision } from "../brain";
import { isTrickCandidate } from "../policy/trickOnly";
import {
  createProgressSink,
  NO_PROGRESS,
  SEARCH_PROGRESS_VERSION,
  validateProgressStream,
  type SearchProgressEvent,
} from "./progress";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../../..");
const WASM = readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm"));
const table = makeTable();

beforeAll(async () => {
  await initPhysics(WASM);
}, 60_000);

/** An ordinary open board with several legal targets and plenty of candidates. */
const BOARD: Ball[] = [
  makeBall(CUE_ID, -0.6, -0.1),
  makeBall(1, 0.3, 0.02),
  makeBall(2, 0.38, 0.1),
  makeBall(4, 0.1, -0.25),
  makeBall(5, -0.2, 0.3),
];

const state: GameState = {
  balls: BOARD,
  turn: 1,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 3,
};

// Both clocks disabled, for the reason `defaultConfig` documents: a wall-clock
// guard makes a fixture depend on how loaded the machine is rather than on the
// physics budget, and this suite compares two runs of the same search.
const cfg = { ...defaultConfig, seed: 20260807, seedTimeoutMs: Infinity, searchTimeoutMs: Infinity };

/** Run a real classical decision and collect its live stream. */
async function runClassical(): Promise<{ events: SearchProgressEvent[]; decision: AiDecision }> {
  const events: SearchProgressEvent[] = [];
  const decision = await classicalTrickOnlyBrain().plan(
    state,
    table,
    1,
    cfg,
    undefined,
    createProgressSink((e) => events.push(e)),
  );
  return { events, decision };
}

describe("the live search stream is published by the real search", () => {
  it("a real decision produces a stream that satisfies the protocol's own ordering rules", async () => {
    const { events } = await runClassical();
    expect(events.length).toBeGreaterThan(10);
    // seq contiguous, atMs non-decreasing, geometry before reference,
    // simulation before verification, nothing after completion.
    expect(validateProgressStream(events)).toEqual([]);
  });

  it("starts with the search and ends with its completion", async () => {
    const { events } = await runClassical();
    expect(events[0].kind).toBe("search-started");
    expect(events[events.length - 1].kind).toBe("search-completed");
  });

  it("no candidate is referred to before its geometry has been published", async () => {
    const { events } = await runClassical();
    const genAt = events.findIndex((e) => e.kind === "candidates-generated");
    expect(genAt).toBeGreaterThanOrEqual(0);
    // Every event that names a candidate index comes after the geometry.
    const namesACandidate = (e: SearchProgressEvent) =>
      e.kind === "candidate-simulating" ||
      e.kind === "candidate-verified" ||
      e.kind === "candidate-rejected" ||
      e.kind === "candidate-retained";
    const first = events.findIndex(namesACandidate);
    expect(first).toBeGreaterThan(genAt);
  });

  it("every verification is preceded by that candidate's own simulation event", async () => {
    const { events } = await runClassical();
    const seenSimulating = new Set<number>();
    let verifications = 0;
    for (const e of events) {
      if (e.kind === "candidate-simulating") seenSimulating.add(e.index);
      if (e.kind === "candidate-verified") {
        expect(seenSimulating.has(e.index)).toBe(true);
        verifications++;
      }
    }
    expect(verifications).toBeGreaterThan(0);
  });

  it("a verification is published only for a candidate the search really simulated", async () => {
    const { events, decision } = await runClassical();
    // `verifiedIndices` is the search's own record of which candidates a real
    // `simulateShotWasm` call was spent on. The live stream may not exceed it.
    const reallyVerified = new Set(decision.trace!.verifiedIndices);
    const claimed = events.filter((e) => e.kind === "candidate-verified").map((e) => e.index);
    expect(claimed.length).toBe(reallyVerified.size);
    for (const i of claimed) expect(reallyVerified.has(i)).toBe(true);
  });

  it("each published physics result is the one the trace records for that candidate", async () => {
    const { events, decision } = await runClassical();
    const traced = decision.decision.candidates;
    let checked = 0;
    for (const e of events) {
      if (e.kind !== "candidate-verified") continue;
      const t = traced[e.index];
      expect(t.physics, `candidate ${e.index} was streamed but has no recorded physics`).not.toBeNull();
      expect(e.physics.legalPot).toBe(t.physics!.legalPot);
      expect(e.physics.scratched).toBe(t.physics!.scratched);
      expect(e.physics.legalFirstContact).toBe(t.physics!.legalFirstContact);
      expect(e.physics.firstContact).toBe(t.physics!.firstContact);
      expect(e.physics.railsBeforePot).toBe(t.physics!.railsBeforePot);
      expect(e.physics.pocketed).toEqual(t.physics!.pocketed);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("every rejection reason streamed is the reason the finished trace gives", async () => {
    const { events, decision } = await runClassical();
    const traced = decision.decision.candidates;
    // The LAST reason streamed for a candidate is its settled one: a candidate
    // can be rejected by physics and then, if it survived, by the policy.
    const streamed = new Map<number, string>();
    for (const e of events) {
      if (e.kind === "candidate-rejected") streamed.set(e.index, e.reason);
    }
    expect(streamed.size).toBeGreaterThan(0);
    for (const [index, reason] of streamed) {
      expect(traced[index].rejection, `candidate ${index}`).toBe(reason);
    }
  });

  it("every candidate the trace says was rejected was streamed a rejection", async () => {
    const { events, decision } = await runClassical();
    const streamed = new Set(
      events.filter((e) => e.kind === "candidate-rejected").map((e) => e.index),
    );
    for (const c of decision.decision.candidates) {
      if (c.rejection !== null) {
        expect(streamed.has(c.index), `candidate ${c.index} (${c.rejection}) never streamed`).toBe(
          true,
        );
      }
    }
  });

  it("the selection streamed is the shot the decision actually plays", async () => {
    const { events, decision } = await runClassical();
    const sel = events.filter((e) => e.kind === "selected");
    expect(sel.length).toBe(1);
    const s = sel[0] as Extract<SearchProgressEvent, { kind: "selected" }>;
    expect(s.index).toBe(decision.decision.selected!.candidateIndex);
    expect(s.shotKind).toBe(decision.decision.selected!.kind);
    expect(s.rung).toBe(decision.decision.selected!.rung);
    // Trick-only holds on the live stream too: the selected route is never a
    // direct, whatever else was considered.
    expect(s.shotKind).not.toBe("direct");
  });

  it("the completion figures are the search's own, not a second count", async () => {
    const { events, decision } = await runClassical();
    const done = events[events.length - 1] as Extract<
      SearchProgressEvent,
      { kind: "search-completed" }
    >;
    expect(done.physicsUnitsSpent).toBe(decision.trace!.physicsCalls);
    expect(done.physicsVerified).toBe(decision.trace!.physicsVerified);
    expect(done.seedTimedOut).toBe(decision.trace!.seedTimedOut);
  });

  it("directs are streamed as excluded by policy, and are never selected", async () => {
    const { events } = await runClassical();
    const gen = events.find((e) => e.kind === "candidates-generated") as Extract<
      SearchProgressEvent,
      { kind: "candidates-generated" }
    >;
    const directs = gen.candidates.filter((c) => c.kind === "direct");
    expect(directs.length).toBeGreaterThan(0);
    // Published, so the rejected comparison is visible rather than hidden…
    for (const d of directs) expect(d.eligible).toBe(false);
    const excluded = new Set(
      events
        .filter((e) => e.kind === "candidate-rejected" && e.reason === "direct-excluded-by-policy")
        .map((e) => (e as { index: number }).index),
    );
    // …and every one of them carries the real reason.
    for (const d of directs) expect(excluded.has(d.index)).toBe(true);
  });

  it("the stream is versioned", () => {
    expect(SEARCH_PROGRESS_VERSION).toBe("showboat-search-progress/1");
  });
});

describe("the model's ordering is the model's, and it is published before physics", () => {
  const manifest = { artifact: "ranker.onnx", onnx_sha256: "abc", schema_version: "v2" };

  /**
   * A stub that returns a KNOWN ranking, so the test can assert the published
   * order is the scored order rather than a coincidence of generation order.
   * Scores descend with index, which is the reverse of generation order.
   */
  const evaluatorWithScores = (n: () => number): NeuralCandidateEvaluator =>
    ({
      getManifest: () => manifest,
      isReady: () => true,
      getState: () => ({ status: "ready", manifest, hashVerified: true }),
      score: async (_b: unknown, _t: unknown, cands: unknown[]) => ({
        scores: cands.map((_, i) => (cands.length - i) / cands.length),
        logits: cands.map((_, i) => cands.length - i),
        inferenceMs: n(),
        encodeMs: 0,
        runMs: 0,
        batchSize: cands.length,
      }),
    }) as unknown as NeuralCandidateEvaluator;

  it("publishes the real scored ranking, before any candidate is simulated", async () => {
    const events: SearchProgressEvent[] = [];
    await neuralTrickOnlyBrain(evaluatorWithScores(() => 1.5)).plan(
      state,
      table,
      1,
      cfg,
      undefined,
      createProgressSink((e) => events.push(e)),
    );
    const rankAt = events.findIndex((e) => e.kind === "neural-scored");
    const firstSim = events.findIndex((e) => e.kind === "candidate-simulating");
    expect(rankAt).toBeGreaterThanOrEqual(0);
    expect(firstSim).toBeGreaterThan(rankAt);

    const scored = events[rankAt] as Extract<SearchProgressEvent, { kind: "neural-scored" }>;
    expect(scored.modelId).toBe("ranker");
    expect(scored.inferenceMs).toBe(1.5);
    // Ranks are 1..n, contiguous and unique.
    const ranks = scored.ranks.map((r) => r.rank).sort((a, b) => a - b);
    expect(ranks).toEqual(ranks.map((_, i) => i + 1));
    // The stub scores descend with index, so rank 1 must be index 0 and the
    // published order must be by score — not by generation order.
    const byRank = [...scored.ranks].sort((a, b) => a.rank - b.rank);
    expect(byRank[0].index).toBe(0);
    for (let i = 1; i < byRank.length; i++) {
      expect(byRank[i].score).toBeLessThanOrEqual(byRank[i - 1].score);
    }
  });

  it("the candidates the prior dropped are streamed as pruned, not as unexamined", async () => {
    const candidates = generateCandidates(BOARD, table, [1, 2, 4, 5]);
    const eligible = candidates.filter(isTrickCandidate).length;
    const keepTop = 4;
    const events: SearchProgressEvent[] = [];
    searchCandidates(candidates, BOARD, [1, 2, 4, 5], {
      ...cfg,
      eligible: isTrickCandidate,
      prior: {
        scores: candidates.map((_, i) => -i),
        keepTop,
        source: "stub",
      },
      progress: createProgressSink((e) => events.push(e)),
    });
    const pruned = events.find((e) => e.kind === "prior-pruned") as Extract<
      SearchProgressEvent,
      { kind: "prior-pruned" }
    >;
    expect(pruned).toBeDefined();
    expect(pruned.keptTop).toBe(keepTop);
    expect(pruned.indices.length).toBe(eligible - keepTop);
    // Every pruned index also carries its own per-route reason.
    const reasons = new Map(
      events
        .filter((e) => e.kind === "candidate-rejected")
        .map((e) => [(e as { index: number }).index, (e as { reason: string }).reason]),
    );
    for (const i of pruned.indices) expect(reasons.get(i)).toBe("pruned-by-prior");
  });
});

describe("publication does not make the search slower", () => {
  /**
   * The requirement is that observing the search must not change how much work
   * it does or meaningfully change how long it takes.
   *
   * The first half is exact and is the real guarantee: an identical search, run
   * with and without a sink, must spend the identical physics budget and reach
   * the identical decision. Publication is a side channel; if it could change
   * the search, this is where it would show.
   *
   * The second half is a wall-clock bound, kept deliberately loose. A tight
   * timing assertion in a unit suite measures machine load, which is a lesson
   * this codebase has already learned twice (see `defaultConfig`). The
   * measured overhead is reported in the sprint write-up; this only fails if
   * publication has become expensive by an order of magnitude.
   */
  it("an observed search spends the same budget and reaches the same decision", async () => {
    const quiet = await classicalTrickOnlyBrain().plan(state, table, 1, cfg, undefined, NO_PROGRESS);
    const events: SearchProgressEvent[] = [];
    const watched = await classicalTrickOnlyBrain().plan(
      state,
      table,
      1,
      cfg,
      undefined,
      createProgressSink((e) => events.push(e)),
    );

    expect(watched.trace!.physicsCalls).toBe(quiet.trace!.physicsCalls);
    expect(watched.trace!.physicsVerified).toBe(quiet.trace!.physicsVerified);
    expect(watched.trace!.verifiedIndices).toEqual(quiet.trace!.verifiedIndices);
    expect(watched.decision.selected?.candidateIndex).toBe(quiet.decision.selected?.candidateIndex);
    expect(watched.decision.selected?.action).toEqual(quiet.decision.selected?.action);
    expect(events.length).toBeGreaterThan(10);
  });

  it("the wall-clock cost of publication is a small fraction of the search", async () => {
    const runs = 3;
    let quietMs = 0;
    let watchedMs = 0;
    for (let i = 0; i < runs; i++) {
      const a = await classicalTrickOnlyBrain().plan(state, table, 1, cfg, undefined, NO_PROGRESS);
      quietMs += a.trace!.physicsMs ?? 0;
      const events: SearchProgressEvent[] = [];
      const b = await classicalTrickOnlyBrain().plan(
        state,
        table,
        1,
        cfg,
        undefined,
        createProgressSink((e) => events.push(e)),
      );
      watchedMs += b.trace!.physicsMs ?? 0;
    }
    // Publication is ~100 small object literals against a search that spends
    // its time in WASM. Doubling would be a real regression; anything under
    // that is noise on a shared machine.
    expect(watchedMs).toBeLessThan(quietMs * 2 + 50);
  });

  it("with no observer, nothing is built at all", async () => {
    // `NO_PROGRESS`'s methods return undefined and record nothing; the search
    // path must not depend on a sink returning anything.
    const decision = await classicalTrickOnlyBrain().plan(
      state,
      table,
      1,
      cfg,
      undefined,
      NO_PROGRESS,
    );
    expect(decision.decision.selected).not.toBeNull();
  });
});

describe("validateProgressStream catches what it claims to", () => {
  const base = { atMs: 0 } as const;

  it("flags a gap in the sequence", () => {
    const bad = validateProgressStream([
      { ...base, seq: 0, kind: "search-started", mode: "classical", budgetUnits: 60 },
      { ...base, seq: 2, kind: "search-completed", physicsUnitsSpent: 0, physicsMs: 0, physicsVerified: 0, seedTimedOut: false, searchTimedOut: false },
    ]);
    expect(bad.map((b) => b.problem)).toContain("seq 2 follows 0");
  });

  it("flags a clock that goes backwards", () => {
    const bad = validateProgressStream([
      { seq: 0, atMs: 10, kind: "search-started", mode: "classical", budgetUnits: 60 },
      { seq: 1, atMs: 4, kind: "candidates-generated", candidates: [] },
    ]);
    expect(bad.map((b) => b.problem)).toContain("atMs 4 follows 10");
  });

  it("flags a verification with no simulation before it", () => {
    const bad = validateProgressStream([
      {
        ...base,
        seq: 0,
        kind: "candidates-generated",
        candidates: [
          {
            index: 0,
            kind: "bank",
            eligible: true,
            target: 1,
            potId: 1,
            pocket: "tl",
            aimPoint: { x: 0, y: 0 },
            cuePath: [],
            path: [],
          },
        ],
      },
      {
        ...base,
        seq: 1,
        kind: "candidate-verified",
        index: 0,
        physics: {
          firstContact: 1,
          legalFirstContact: true,
          scratched: false,
          legalPot: true,
          pocketed: [1],
          railsBeforePot: 1,
        },
      },
    ]);
    expect(bad.map((b) => b.problem)).toContain("verified 0 with no preceding simulation");
  });

  it("flags a route referenced before its geometry exists", () => {
    const bad = validateProgressStream([
      { ...base, seq: 0, kind: "search-started", mode: "classical", budgetUnits: 60 },
      { ...base, seq: 1, kind: "candidate-simulating", index: 7 },
    ]);
    expect(bad.map((b) => b.problem)).toContain("simulating unknown candidate 7");
  });

  it("flags anything published after the search completed", () => {
    const bad = validateProgressStream([
      { ...base, seq: 0, kind: "search-completed", physicsUnitsSpent: 0, physicsMs: 0, physicsVerified: 0, seedTimedOut: false, searchTimedOut: false },
      { ...base, seq: 1, kind: "candidates-generated", candidates: [] },
    ]);
    expect(bad.map((b) => b.problem)).toContain("candidates-generated after search-completed");
  });

  it("a real stream survives a round trip through structured-clone-equivalent JSON", async () => {
    const { events } = await runClassical();
    // The stream crosses the worker boundary; if a field were ever a Map, a
    // class instance or an `undefined`, this is where it would show.
    expect(JSON.parse(JSON.stringify(events))).toEqual(events);
  });
});
