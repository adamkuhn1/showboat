// What the felt shows about the shot being played must be the shot being
// played — end to end, through the real planner.
//
// Every fixture here runs `planTurnTraced`, the same function the worker and
// the main-thread fallback both call, on a real board with the real WASM
// physics. The assertions are about the seam the sprint exists to close: the
// route the overlay draws, the contacts it marks, the outcome it commits, and
// the guarantee that none of the three can come from a different simulation
// than the other two.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../../physics/table";
import { makeBall, cloneBall, type Ball } from "../../physics/ball";
import { BALL_RADIUS } from "../../physics/constants";
import { CUE_ID } from "../../game/rack";
import type { GameState } from "../../game/state";
import { initPhysics } from "../../physics/wasm-bridge";
import { computeView } from "../../render/renderer";
import { contactMarksFromExecuted } from "../../render/annotate";
import { buildSchedule, frameAt, measuredRoute } from "../../render/presentation";
import { extractExecutedMotion } from "../../ai/trace/executed";
import { buildDecisionTrace } from "../../ai/trace/build";
import { generateCandidates } from "../../ai/candidates";
import type { CandidateStat, SearchOutcome } from "../../ai/shotSearch";
import type { TrickOnlyDecision } from "../../ai/policy/trickOnly";
import type { PlannedTurn, PlayedTurn } from "./plan";
import { planTurnTraced } from "./plan";
import { planViaWorker, type PlanChannel } from "./workerPlan";
import type { PlanRequest, PlanResponse } from "./protocol";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../../..");
const table = makeTable();
const VIEW = computeView(900, 500, table);
const ROUTE_TOLERANCE_PX = 0.5;
const px = (m: number) => m * VIEW.scale;

const asState = (balls: Ball[]): GameState => ({
  balls,
  turn: 1,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 3,
});

const BOARDS: Record<string, Ball[]> = {
  openSpread: [
    makeBall(CUE_ID, -0.35, 0.05),
    makeBall(1, 0.25, 0.15),
    makeBall(3, -0.05, -0.28),
    makeBall(9, 0.5, -0.2),
  ],
  railHeavy: [
    makeBall(CUE_ID, 0.0, 0.0),
    makeBall(1, -0.7, 0.28),
    makeBall(3, 0.72, -0.29),
    makeBall(6, -0.4, -0.3),
  ],
  // The cue is screened off its only legal target: the ladder drops to the
  // safety rung, which is a shot with a route of its own to be truthful about.
  screened: [
    makeBall(CUE_ID, -0.8, 0.0),
    makeBall(8, -0.72, 0.0),
    makeBall(1, 0.7, 0.0),
  ],
  tightPocket: [
    makeBall(CUE_ID, -0.85, 0.34),
    makeBall(1, 0.62, 0.3),
    makeBall(11, 0.35, 0.36),
  ],
  // Ball 11 is not a legal target and sits directly on the line to the ones
  // that are, so candidates aimed past it strike it first: the board that
  // makes `illegal-first-contact` a real rejection rather than a reachable
  // branch nothing exercises.
  blockedLine: [
    makeBall(CUE_ID, -0.86, 0.0),
    makeBall(11, -0.45, 0.0),
    makeBall(1, 0.32, 0.0),
    makeBall(3, 0.5, 0.22),
  ],
};

interface Run {
  turn: PlannedTurn;
  /** The board as it stood BEFORE the shot, deep-copied before planning. */
  pre: Ball[];
}

const planned: Record<string, Run> = {};

describe("the selected shot, from decision to felt", () => {
  beforeAll(async () => {
    const wasm = readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm"));
    await initPhysics(wasm);
    for (const [name, balls] of Object.entries(BOARDS)) {
      const pre = balls.map(cloneBall);
      const turn = await planTurnTraced({
        state: asState(balls.map(cloneBall)),
        table,
        player: 1,
        // No model in this realm: the point of these fixtures is the route, and
        // the neural path's own failure modes are covered in `fallback.test.ts`.
        useNeural: false,
        wasmSource: wasm,
      });
      planned[name] = { turn, pre };
    }
  }, 240_000);

  const shots = (): [string, PlayedTurn, Ball[]][] =>
    Object.entries(planned)
      .filter(([, r]) => r.turn.kind === "shot")
      .map(([n, r]) => [n, r.turn as PlayedTurn, r.pre]);

  it("every board produced a real shot to examine", () => {
    expect(shots().length).toBe(Object.keys(BOARDS).length);
  });

  it("the published route is derived from the exact report that is played", () => {
    for (const [name, turn] of shots()) {
      const published = turn.trace.selected?.executed ?? null;
      expect(published, `${name}: no executed motion published`).not.toBeNull();
      // Re-derive from the report the game commits. Deep equality, so a
      // published route that came from any other simulation — a second run, a
      // verification run, a cached one — fails here.
      const fromReport = extractExecutedMotion(turn.report.sim);
      expect(fromReport, `${name}: report has no extractable motion`).not.toBeNull();
      expect(published).toEqual(fromReport);
    }
  });

  it("the route the overlay actually draws is the measured one, not the plan", () => {
    for (const [name, turn, pre] of shots()) {
      const cue = pre.find((b) => b.id === CUE_ID) ?? null;
      const geom = {
        cuePos: cue ? { x: cue.pos.x, y: cue.pos.y } : turn.trace.turn.cueBall,
      };
      const schedule = buildSchedule({ trace: turn.trace, sentenceWords: 8 });
      for (const at of ["SELECTED", "READY", "STROKE"] as const) {
        const seg = schedule.segments.find((s) => s.state === at);
        if (!seg) continue;
        const frame = frameAt(turn.trace, schedule, seg.startMs + seg.durationMs / 2, geom);
        const sel = frame.routes.filter((r) => r.role === "selected");
        expect(sel, `${name}/${at}: no selected route`).toHaveLength(1);
        expect(sel[0].source, `${name}/${at}: still drawing the plan`).toBe("simulated");
        expect(sel[0].measured).not.toBeNull();

        // And what it draws is the published motion, channel by channel.
        const expected = measuredRoute(turn.trace.selected!.executed!);
        expect(sel[0].objectLeg).toEqual(expected.object?.points ?? []);
        expect(sel[0].cueLeg).toEqual(expected.cue ? expected.cue.points : null);
      }
    }
  });

  it("candidates that are not the shot are still drawn as plans, never as measurements", () => {
    for (const [name, turn] of shots()) {
      const schedule = buildSchedule({ trace: turn.trace, sentenceWords: 8 });
      const geom = { cuePos: turn.trace.turn.cueBall };
      const seg = schedule.segments.find((s) => s.state === "SELECTED")!;
      const frame = frameAt(turn.trace, schedule, seg.startMs + 5, geom);
      for (const r of frame.routes) {
        if (r.role === "selected") continue;
        expect(r.source, `${name}: a losing candidate claims measured geometry`).toBe("plan");
        expect(r.measured).toBeNull();
      }
    }
  });

  it("contact ordering matches the simulator's event log exactly", () => {
    for (const [name, turn] of shots()) {
      const executed = turn.trace.selected!.executed!;
      const evs = turn.report.sim.events.filter((e) =>
        ["ball-ball", "ball-cushion", "pocket"].includes(e.kind),
      );
      expect(executed.contactSequence.length).toBe(evs.length);
      executed.contactSequence.forEach((c, i) => {
        expect(c.kind, `${name}: contact ${i} kind`).toBe(evs[i].kind);
        expect(c.balls, `${name}: contact ${i} balls`).toEqual(evs[i].balls);
        expect(c.timeSec).toBe(evs[i].time);
      });
      // The marks the app draws are that sequence, in that order, truncated.
      const marks = contactMarksFromExecuted(executed, 8);
      for (let i = 1; i < marks.length; i++) {
        expect(marks[i].timeSec).toBeGreaterThanOrEqual(marks[i - 1].timeSec);
        expect(marks[i].order).toBe(marks[i - 1].order + 1);
      }
    }
  });

  it("every contact mark sits on a vertex of the route drawn beside it", () => {
    for (const [name, turn] of shots()) {
      const executed = turn.trace.selected!.executed!;
      const marks = contactMarksFromExecuted(executed, 8);
      expect(marks.length, `${name}: no marks`).toBeGreaterThan(0);
      for (const m of marks) {
        const traj = executed.trajectories.find((t) => t.ballId === m.ballId)!;
        const onAVertex = traj.points.some(
          (p) => px(Math.hypot(p.x - m.at.x, p.y - m.at.y)) <= ROUTE_TOLERANCE_PX,
        );
        expect(onAVertex, `${name}: mark at ${m.timeSec} is not on ball ${m.ballId}'s route`).toBe(
          true,
        );
      }
    }
  });

  it("where a route ends is where that ball ends up in the committed state", () => {
    for (const [name, turn, pre] of shots()) {
      const executed = turn.trace.selected!.executed!;
      for (const traj of executed.trajectories) {
        const after = turn.report.next.balls.find((b) => b.id === traj.ballId)!;
        expect(after.pocketed, `${name}: ball ${traj.ballId} pocketed flag`).toBe(traj.pocketed);
        if (after.pocketed) continue;
        const end = traj.points[traj.points.length - 1];
        const off = Math.hypot(end.x - after.pos.x, end.y - after.pos.y);
        // The committed state runs through the game layer's own de-overlap
        // pass, which can nudge a resting ball by a fraction of a millimetre;
        // a ball radius is a generous bound on that and a tight one on
        // "the line ends where the ball is".
        expect(off, `${name}: ball ${traj.ballId} route ends ${off.toFixed(4)} m from rest`).toBeLessThan(
          BALL_RADIUS,
        );
      }
      // Balls with no route did not move.
      const drawn = new Set(executed.trajectories.map((t) => t.ballId));
      for (const before of pre) {
        if (drawn.has(before.id)) continue;
        const after = turn.report.next.balls.find((b) => b.id === before.id)!;
        if (after.pocketed) continue;
        expect(Math.hypot(after.pos.x - before.pos.x, after.pos.y - before.pos.y)).toBeLessThan(
          0.01,
        );
      }
    }
  });

  it("no direct shot is ever the selection, on any of these boards", () => {
    for (const [name, turn] of shots()) {
      expect(turn.trace.policy).toBe("trick-only");
      const sel = turn.trace.selected!;
      expect(sel.kind, `${name} selected a direct`).not.toBe("direct");
      for (const c of turn.trace.candidates) {
        if (c.kind !== "direct") continue;
        expect(c.eligible).toBe(false);
        expect(c.rejection).toBe("direct-excluded-by-policy");
        expect(c.index).not.toBe(sel.candidateIndex);
      }
    }
  });

  it("a candidate that scratched or fouled in simulation says so, and is not the shot", () => {
    let scratches = 0;
    let illegal = 0;
    for (const [name, turn] of shots()) {
      const sel = turn.trace.selected!;
      for (const c of turn.trace.candidates) {
        if (!c.physics || c.kind === "direct") continue;
        if (c.physics.scratched) {
          scratches++;
          expect(c.rejection, `${name}/${c.index}`).toBe("scratched-in-simulation");
          expect(c.index).not.toBe(sel.candidateIndex);
        } else if (!c.physics.legalFirstContact) {
          illegal++;
          expect(c.rejection, `${name}/${c.index}`).toBe("illegal-first-contact");
          expect(c.index).not.toBe(sel.candidateIndex);
        }
      }
      // The shot that IS played did neither.
      if (sel.candidateIndex !== null) {
        const chosen = turn.trace.candidates[sel.candidateIndex];
        if (chosen.physics) {
          expect(chosen.physics.scratched).toBe(false);
          expect(chosen.physics.legalFirstContact).toBe(true);
        }
      }
    }
    // The scratch branch must actually fire on these boards, or the assertion
    // above is vacuous and this test is decoration.
    expect(scratches, "no scratch was produced by any fixture").toBeGreaterThan(0);

    // `illegal` is recorded but NOT required to be non-zero, and that is a
    // finding rather than a shrug. `generateCandidates` will not emit a
    // candidate whose cue path is blocked: `isPathClear` rejects any ball
    // whose centre is within 2R + 4 mm of the line. The physics collides at
    // 2R. So the geometric filter is strictly stricter than the simulator, and
    // an illegal first contact can only arise from the two disagreeing — which
    // no board in this fixture set produced. It is a guard against that
    // disagreement, not an ordinary outcome. The branch itself is exercised
    // directly below.
    // Pinned at zero so that if the two ever DO disagree, this fails and the
    // note above gets revisited instead of quietly becoming false.
    expect(illegal).toBe(0);
  });

  it("the illegal-first-contact rejection is the one that fires when physics says so", () => {
    // The branch the fixtures cannot reach, exercised where it lives. This
    // drives the real `buildDecisionTrace` with a real candidate from a real
    // board; only the verification is constructed, because constructing one is
    // the only way to express "the simulator disagreed with the clearance
    // check" without a board that makes it happen.
    const balls = BOARDS.openSpread.map(cloneBall);
    const targets = [1, 3, 9];
    const cands = generateCandidates(balls, table, targets);
    const trick = cands.findIndex((c) => c.kind !== "direct");
    expect(trick).toBeGreaterThanOrEqual(0);

    const stats: CandidateStat[] = cands.map((candidate) => ({
      candidate,
      visits: 1,
      value: 0.1,
      strength: 0.1,
      rails: candidate.banks,
      potsTarget: false,
      styleScore: candidate.banks,
      verified: true,
    }));
    const verifications = cands.map((c, i) =>
      i === trick
        ? {
            index: i,
            // The cue struck something that is not the target.
            firstContact: c.target === 1 ? 3 : 1,
            legalFirstContact: false,
            scratched: false,
            legalPot: false,
            pocketed: [],
            railsBeforePot: 0,
            events: [],
          }
        : null,
    );

    const built = buildDecisionTrace({
      outcome: {
        stats,
        allStats: stats,
        verifications,
        simulations: 1,
        trace: undefined,
      } as unknown as SearchOutcome,
      decision: {
        shot: null,
        rung: null,
        qualifyingTricks: 0,
        utility: null,
        excludedIndices: [],
        safetySimsSpent: 0,
        safety: null,
        safetyQuality: null,
      } as unknown as TrickOnlyDecision,
      state: asState(balls),
      player: 1,
      targets,
      physicsUnitsAllowed: 60,
      model: null,
      fallback: null,
      timing: { totalMs: 1, neuralEncodeMs: null, neuralRunMs: null, physicsMs: 1, selectionMs: 0 },
    });

    expect(built.candidates[trick].rejection).toBe("illegal-first-contact");
    expect(built.candidates[trick].physics!.legalFirstContact).toBe(false);
    // And a scratch outranks it: the first reason that applied is the one
    // reported, which is the ordering `build.ts` documents.
    const scratched = buildDecisionTrace({
      outcome: {
        stats,
        allStats: stats,
        verifications: verifications.map((v) =>
          v === null ? null : { ...v, scratched: true },
        ),
        simulations: 1,
        trace: undefined,
      } as unknown as SearchOutcome,
      decision: {
        shot: null,
        rung: null,
        qualifyingTricks: 0,
        utility: null,
        excludedIndices: [],
        safetySimsSpent: 0,
        safety: null,
        safetyQuality: null,
      } as unknown as TrickOnlyDecision,
      state: asState(balls),
      player: 1,
      targets,
      physicsUnitsAllowed: 60,
      model: null,
      fallback: null,
      timing: { totalMs: 1, neuralEncodeMs: null, neuralRunMs: null, physicsMs: 1, selectionMs: 0 },
    });
    expect(scratched.candidates[trick].rejection).toBe("scratched-in-simulation");
  });

  it("the trace crosses the worker boundary as data, executed motion included", () => {
    // The route is only useful if it survives `structuredClone`. JSON is the
    // stricter test of the two and the one `contract.test.ts` already applies
    // to the rest of the trace.
    for (const [name, turn] of shots()) {
      const round = JSON.parse(JSON.stringify(turn.trace));
      expect(round, `${name} did not survive a JSON round trip`).toEqual(turn.trace);
    }
  });
});

// ---------------------------------------------------------------------------
// Turn scoping.
//
// No progress is streamed — see the report — so the only messages that cross
// the channel are the model-status ones and the finished turn. The rule that
// matters is the same either way: a message from a turn that is over cannot
// touch a turn that is running.
// ---------------------------------------------------------------------------

describe("a message from a finished turn cannot change a later one", () => {
  const stubChannel = () => {
    const listeners: ((e: MessageEvent<PlanResponse>) => void)[] = [];
    let terminated = false;
    const sent: PlanRequest[] = [];
    const channel: PlanChannel = {
      addEventListener: ((type: string, fn: EventListener) => {
        if (type === "message") {
          listeners.push(fn as unknown as (e: MessageEvent<PlanResponse>) => void);
        }
      }) as PlanChannel["addEventListener"],
      removeEventListener: ((_type: string, fn: EventListener) => {
        const i = listeners.indexOf(fn as unknown as (e: MessageEvent<PlanResponse>) => void);
        if (i >= 0) listeners.splice(i, 1);
      }) as PlanChannel["removeEventListener"],
      postMessage: (m: PlanRequest) => sent.push(m),
      terminate: () => {
        terminated = true;
      },
    };
    const deliver = (msg: PlanResponse) => {
      for (const fn of [...listeners]) fn({ data: msg } as MessageEvent<PlanResponse>);
    };
    return { channel, deliver, sent, listenerCount: () => listeners.length, wasTerminated: () => terminated };
  };

  const fakeTurn = (tag: string): PlannedTurn =>
    ({ kind: "shot", trace: { version: tag } } as unknown as PlannedTurn);

  it("a late `done` for turn 1 does not resolve turn 2", async () => {
    const ch = stubChannel();
    const state = asState(BOARDS.openSpread);
    const p2 = planViaWorker({
      worker: ch.channel,
      id: 2,
      state,
      table,
      player: 1,
      useNeural: false,
      modelDir: "http://x/model",
      silenceMs: 60_000,
    });

    // Turn 1's answer, arriving after turn 2 has started.
    ch.deliver({ type: "done", id: 1, planned: fakeTurn("stale") });
    // Nothing settled: the listener is still attached and nothing was posted
    // back. Then turn 2's own answer arrives and wins.
    expect(ch.listenerCount()).toBeGreaterThan(0);
    ch.deliver({ type: "done", id: 2, planned: fakeTurn("live") });

    const got = (await p2) as unknown as { trace: { version: string } };
    expect(got.trace.version).toBe("live");
  });

  it("a late model-status for turn 1 is not reported against turn 2", async () => {
    const ch = stubChannel();
    const seen: string[] = [];
    const p2 = planViaWorker({
      worker: ch.channel,
      id: 2,
      state: asState(BOARDS.openSpread),
      table,
      player: 1,
      useNeural: true,
      modelDir: "http://x/model",
      silenceMs: 60_000,
      onModelLoading: () => seen.push("loading"),
      onModelStatus: (s) => seen.push(`status:${s.status}`),
    });

    ch.deliver({ type: "model-loading", id: 1 });
    ch.deliver({ type: "model-status", id: 1, ok: false, reason: "stale", hashVerified: false });
    expect(seen).toEqual([]);

    ch.deliver({ type: "model-status", id: 2, ok: true, reason: null, hashVerified: true });
    expect(seen).toEqual(["status:ready"]);

    ch.deliver({ type: "done", id: 2, planned: fakeTurn("live") });
    await p2;
  });

  it("a late `error` for turn 1 does not reject turn 2", async () => {
    const ch = stubChannel();
    const p2 = planViaWorker({
      worker: ch.channel,
      id: 2,
      state: asState(BOARDS.openSpread),
      table,
      player: 1,
      useNeural: false,
      modelDir: "http://x/model",
      silenceMs: 60_000,
    });
    ch.deliver({ type: "error", id: 1, message: "turn one blew up" });
    ch.deliver({ type: "done", id: 2, planned: fakeTurn("live") });
    await expect(p2).resolves.toBeTruthy();
  });
});
