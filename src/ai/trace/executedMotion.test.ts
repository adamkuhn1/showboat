// The executed-motion extraction layer, against real simulations.
//
// Every fixture here is a REAL shot: a board, an action, and the WASM
// simulator's own output. Nothing is hand-written, so an assertion that passes
// is a statement about the shipped extractor and not about two pieces of test
// data agreeing with each other.
//
// The fixtures are found deterministically rather than hand-tuned. For each
// motion topology the suite walks the candidate generator's own output, in
// generation order, simulates each candidate once, and takes the FIRST whose
// event log has the shape it is looking for. Candidate generation and the
// simulator are both deterministic, so the fixture set is fixed; and if a
// topology stops being producible the `beforeAll` fails loudly instead of the
// suite quietly testing four cases where it claims six.
//
// This is also the answer to a finding from a previous sprint: `combo` and
// `rail-combo` were generated and eligible in every trace but never naturally
// SELECTED across 76 sampled decisions. Waiting for the policy to pick one is
// not coverage. These fixtures exercise the geometry directly.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../../physics/table";
import { makeBall, cloneBall, type Ball } from "../../physics/ball";
import { CUE_ID } from "../../game/rack";
import type { CueAction } from "../../physics/cue";
import type { ShotEvent, SimResult } from "../../physics/engine";
import { initPhysics, simulateShotWasm } from "../../physics/wasm-bridge";
import { generateCandidates } from "../candidates";
import { computeView } from "../../render/renderer";
import { measuredRoute } from "../../render/presentation";
import { BALL_RADIUS } from "../../physics/constants";
import {
  extractExecutedMotion,
  withExecutedMotion,
  MIN_TRAVEL_M,
  SIMPLIFY_TOLERANCE_M,
} from "./executed";
import type { DecisionTraceV1, ExecutedMotion, Vec2Trace } from "./contract";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../../..");
const table = makeTable();

// ---------------------------------------------------------------------------
// The declared tolerance.
//
// The app draws the table into a fixed 900x500 logical space, so there is one
// pixels-per-metre number and it is knowable here rather than guessable. Every
// pixel claim in this file is expressed through it.
// ---------------------------------------------------------------------------
const VIEW = computeView(900, 500, table);
/** Logical pixels. Half a pixel: under the antialiasing of a 2 px stroke. */
export const ROUTE_TOLERANCE_PX = 0.5;
const px = (metres: number): number => metres * VIEW.scale;

const MARKED = new Set(["ball-ball", "ball-cushion", "pocket"]);
const marked = (sim: SimResult): ShotEvent[] => sim.events.filter((e) => MARKED.has(e.kind));

const dist = (a: Vec2Trace, b: Vec2Trace) => Math.hypot(a.x - b.x, a.y - b.y);

/** Perpendicular distance from `p` to segment `a`-`b`. */
function perp(p: Vec2Trace, a: Vec2Trace, b: Vec2Trace): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const l2 = dx * dx + dy * dy;
  if (l2 <= 0) return dist(p, a);
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / l2));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

/** Shortest distance from `p` to a polyline. */
const toPolyline = (p: Vec2Trace, pts: Vec2Trace[]): number => {
  let best = Infinity;
  for (let i = 1; i < pts.length; i++) best = Math.min(best, perp(p, pts[i - 1], pts[i]));
  return best;
};

// ---------------------------------------------------------------------------
// Topologies, classified from the event log and nothing else
// ---------------------------------------------------------------------------

type Topology =
  | "single-bank"
  | "double-bank"
  | "multi-cushion"
  | "combo"
  | "rail-combo"
  | "safety-kick";

interface Shape {
  /** Cushion contacts made by the ball that was potted, before it dropped. */
  potBallCushions: number;
  /** Cue struck a cushion before it struck any ball. */
  kickFirst: boolean;
  /** The potted ball was set moving by an object ball, not by the cue. */
  combo: boolean;
  potted: number | null;
}

function shapeOf(sim: SimResult): Shape {
  const evs = marked(sim);
  const pot = evs.find((e) => e.kind === "pocket" && e.balls[0] !== CUE_ID);
  const potted = pot ? pot.balls[0] : null;

  const firstCueBall = evs.findIndex((e) => e.kind === "ball-ball" && e.balls.includes(CUE_ID));
  const firstCueCushion = evs.findIndex(
    (e) => e.kind === "ball-cushion" && e.balls[0] === CUE_ID,
  );
  const kickFirst =
    firstCueCushion >= 0 && (firstCueBall < 0 || firstCueCushion < firstCueBall);

  let potBallCushions = 0;
  let combo = false;
  if (potted !== null) {
    const potIdx = evs.indexOf(pot!);
    for (let i = 0; i < potIdx; i++) {
      const e = evs[i];
      if (e.kind === "ball-cushion" && e.balls[0] === potted) potBallCushions++;
    }
    const setMoving = evs.find((e) => e.kind === "ball-ball" && e.balls.includes(potted));
    combo = !!setMoving && !setMoving.balls.includes(CUE_ID);
  }
  return { potBallCushions, kickFirst, combo, potted };
}

function matches(topology: Topology, s: Shape): boolean {
  switch (topology) {
    case "single-bank":
      return s.potted !== null && !s.combo && s.potBallCushions === 1;
    case "double-bank":
      return s.potted !== null && !s.combo && s.potBallCushions === 2;
    case "multi-cushion":
      return s.potted !== null && s.potBallCushions >= 3;
    case "combo":
      // A PURE combination: cue -> ball -> ball -> pocket, no cushion in the
      // potted ball's route. Split from `rail-combo` deliberately, because a
      // rail-combo also satisfies "is a combo" and the two would otherwise
      // resolve to the same shot and claim two categories of coverage for one.
      return s.potted !== null && s.combo && s.potBallCushions === 0;
    case "rail-combo":
      return s.potted !== null && s.combo && s.potBallCushions >= 1;
    case "safety-kick":
      return s.kickFirst;
  }
}

// ---------------------------------------------------------------------------
// Boards. Chosen for variety of geometry, not for any particular outcome.
// ---------------------------------------------------------------------------

const BOARDS: { name: string; balls: Ball[]; targets: number[] }[] = [
  {
    name: "openSpread",
    balls: [
      makeBall(CUE_ID, -0.35, 0.05),
      makeBall(1, 0.25, 0.15),
      makeBall(3, -0.05, -0.28),
      makeBall(9, 0.5, -0.2),
    ],
    targets: [1, 3, 9],
  },
  {
    name: "railHeavy",
    balls: [
      makeBall(CUE_ID, 0.0, 0.0),
      makeBall(1, -0.7, 0.28),
      makeBall(3, 0.72, -0.29),
      makeBall(6, -0.4, -0.3),
    ],
    targets: [1, 3, 6],
  },
  {
    name: "clustered",
    balls: [
      makeBall(CUE_ID, -0.75, -0.05),
      makeBall(2, 0.1, 0.02),
      makeBall(4, 0.22, 0.06),
      makeBall(5, 0.45, -0.24),
      makeBall(7, -0.2, 0.3),
    ],
    targets: [2, 4, 5, 7],
  },
  {
    name: "longRail",
    balls: [
      makeBall(CUE_ID, -0.85, 0.34),
      makeBall(1, 0.62, 0.3),
      makeBall(8, 0.0, -0.36),
      makeBall(11, 0.35, 0.36),
    ],
    targets: [1, 11],
  },
];

interface Fixture {
  topology: Topology;
  board: string;
  pre: Ball[];
  action: CueAction;
  sim: SimResult;
  motion: ExecutedMotion;
}

const fixtures = new Map<Topology, Fixture>();

/** Simulate `action` on a fresh copy of `balls`. The pre-shot state is returned
 *  alongside, because `simulateShotWasm` mutates what it is given. */
function run(balls: Ball[], action: CueAction): { pre: Ball[]; sim: SimResult } {
  const working = balls.map(cloneBall);
  const sim = simulateShotWasm(working, action);
  return { pre: balls.map(cloneBall), sim };
}

const TOPOLOGIES: Topology[] = [
  "single-bank",
  "double-bank",
  "multi-cushion",
  "combo",
  "rail-combo",
  "safety-kick",
];

describe("executed motion: extraction from real simulations", () => {
  beforeAll(async () => {
    await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));

    // One pass over every candidate of every board. Deterministic in both
    // directions: `generateCandidates` is a pure function of the board and the
    // simulator is deterministic for a given action.
    for (const board of BOARDS) {
      const cands = generateCandidates(board.balls, table, board.targets);
      for (const c of cands) {
        // A few powers per candidate: the generator suggests one, and a bank
        // that falls short at 0.55 finds the pocket at 0.85. Still fully
        // determined — the list is fixed and walked in order.
        for (const power of [c.action.power, 0.7, 0.9]) {
          const action = { ...c.action, power };
          const { pre, sim } = run(board.balls, action);
          const shape = shapeOf(sim);
          for (const topology of TOPOLOGIES) {
            if (fixtures.has(topology) || !matches(topology, shape)) continue;
            const motion = extractExecutedMotion(sim);
            if (!motion) continue;
            fixtures.set(topology, { topology, board: board.name, pre, action, sim, motion });
          }
        }
      }
    }
    // A deliberate kick, for the safety topology, in case no candidate produced
    // one: the cue is driven into a rail with nothing in front of it.
    if (!fixtures.has("safety-kick")) {
      const balls = [makeBall(CUE_ID, 0.0, 0.0), makeBall(1, -0.6, 0.2), makeBall(3, 0.5, 0.3)];
      const { pre, sim } = run(balls, { phi: -Math.PI / 2.2, power: 0.75, sideSpin: 0, topSpin: 0 });
      const motion = extractExecutedMotion(sim);
      if (motion && shapeOf(sim).kickFirst) {
        fixtures.set("safety-kick", { topology: "safety-kick", board: "kick", pre, action: { phi: -Math.PI / 2.2, power: 0.75, sideSpin: 0, topSpin: 0 }, sim, motion });
      }
    }
  }, 180_000);

  it("found a real shot for every topology the suite claims to cover", () => {
    const missing = TOPOLOGIES.filter((t) => !fixtures.has(t));
    expect(missing, `no real simulation produced: ${missing.join(", ")}`).toEqual([]);
  });

  describe.each(TOPOLOGIES)("%s", (topology) => {
    const f = () => {
      const fx = fixtures.get(topology);
      if (!fx) throw new Error(`fixture ${topology} missing`);
      return fx;
    };

    it("every point is a position the simulation recorded — none interpolated, none invented", () => {
      const { sim, motion } = f();
      for (const traj of motion.trajectories) {
        const recorded = sim
          .waypoints!.map((w) => w.balls.find((b) => b.id === traj.ballId))
          .filter((b): b is Ball => !!b && !b.pocketed)
          .map((b) => ({ x: b.pos.x, y: b.pos.y }));
        // Every point but one: a potted ball's terminal point is the analytic
        // position at the capture instant, flagged by `endsAtCapture`. That
        // exemption is exactly one point, on exactly the trajectories that
        // claim it, and the case below pins where it may land.
        const last = traj.points.length - 1;
        for (let i = 0; i < traj.points.length; i++) {
          if (traj.endsAtCapture && i === last) continue;
          const p = traj.points[i];
          const exact = recorded.some((r) => r.x === p.x && r.y === p.y);
          expect(exact, `${topology}: ball ${traj.ballId} point ${JSON.stringify(p)} is not a recorded position`).toBe(true);
        }
        if (traj.endsAtCapture) expect(traj.pocketed).toBe(true);
      }
    });

    it("point times are the waypoint times, strictly increasing, and match the array length", () => {
      const { motion } = f();
      for (const traj of motion.trajectories) {
        expect(traj.timesSec.length).toBe(traj.points.length);
        for (let i = 1; i < traj.timesSec.length; i++) {
          expect(traj.timesSec[i]).toBeGreaterThan(traj.timesSec[i - 1]);
        }
        expect(traj.startSec).toBe(traj.timesSec[0]);
        expect(traj.endSec).toBe(traj.timesSec[traj.timesSec.length - 1]);
      }
    });

    it("the drawn line never departs from the true path by more than the declared tolerance", () => {
      // Independent of what the extractor reports: this walks every waypoint
      // the simplification DROPPED and measures its distance to the polyline
      // that is actually stroked.
      const { sim, motion } = f();
      let worst = 0;
      for (const traj of motion.trajectories) {
        for (const wp of sim.waypoints!) {
          if (wp.time < traj.startSec || wp.time > traj.endSec) continue;
          const b = wp.balls.find((x) => x.id === traj.ballId);
          if (!b || b.pocketed) continue;
          worst = Math.max(worst, toPolyline({ x: b.pos.x, y: b.pos.y }, traj.points));
        }
      }
      expect(px(worst)).toBeLessThanOrEqual(ROUTE_TOLERANCE_PX);
      // And the number the extractor publishes is not an optimistic one.
      expect(motion.maxDeviationM).toBeLessThanOrEqual(motion.simplifyToleranceM);
      expect(motion.simplifyToleranceM).toBe(SIMPLIFY_TOLERANCE_M);
    });

    it("simplification is doing real work — the route is shorter than the raw record", () => {
      // Without this the tolerance test above would pass trivially on a
      // polyline that kept every waypoint.
      const { sim, motion } = f();
      const rawFor = (ballId: number) =>
        sim.waypoints!.filter((w) => {
          const b = w.balls.find((x) => x.id === ballId);
          return !!b && !b.pocketed;
        }).length;
      const totalKept = motion.trajectories.reduce((n, t) => n + t.points.length, 0);
      const totalRaw = motion.trajectories.reduce((n, t) => n + rawFor(t.ballId), 0);
      expect(totalKept).toBeLessThan(totalRaw);
    });

    it("breaks are the simulator's own events, in the same order, with the same balls", () => {
      const { sim, motion } = f();
      const evs = marked(sim);
      // Every break maps onto exactly one marked event for that ball.
      for (const traj of motion.trajectories) {
        for (const brk of traj.breaks) {
          const hit = evs.find(
            (e) =>
              e.time === brk.timeSec &&
              e.balls.includes(traj.ballId) &&
              ((e.kind === "ball-ball" && brk.kind === "ball-contact") ||
                (e.kind === "ball-cushion" && brk.kind === "cushion") ||
                (e.kind === "pocket" && brk.kind === "pocket")),
          );
          expect(hit, `${topology}: break at ${brk.timeSec} has no matching event`).toBeTruthy();
          if (brk.kind === "cushion") expect(brk.cushion).toBe(hit!.cushion);
          if (brk.kind === "pocket") expect(brk.pocket).toBe(hit!.pocket);
          if (brk.kind === "ball-contact") {
            expect(hit!.balls).toContain(brk.withBall);
            expect(brk.withBall).not.toBe(traj.ballId);
          }
        }
        // In time order along the route.
        for (let i = 1; i < traj.breaks.length; i++) {
          expect(traj.breaks[i].timeSec).toBeGreaterThanOrEqual(traj.breaks[i - 1].timeSec);
          expect(traj.breaks[i].at).toBeGreaterThanOrEqual(traj.breaks[i - 1].at);
        }
      }
      // And the published contact log is the event log, in emission order.
      expect(motion.contactSequence.map((e) => [e.kind, ...e.balls].join(":"))).toEqual(
        evs.map((e) => [e.kind, ...e.balls].join(":")),
      );
    });

    it("a break vertex sits on the ball's recorded centre at that contact", () => {
      // The claim the overlay depends on: the contact marker and the corner of
      // the line are the same place. Measured in pixels against the declared
      // tolerance; in practice it is the identical float.
      const { sim, motion } = f();
      let checked = 0;
      for (const traj of motion.trajectories) {
        for (const brk of traj.breaks) {
          const vertex = traj.points[brk.at];
          expect(vertex).toBeTruthy();
          // Pocket breaks are excluded here on purpose, not for convenience:
          // there IS no recorded centre at a capture, which is the whole reason
          // `endsAtCapture` exists. Their position is pinned physically instead
          // — on the pocket circle — by "a potted ball's route ends at the
          // pocket it dropped into".
          if (brk.kind === "pocket") continue;
          checked++;
          const wp = sim.waypoints!.reduce((best, w) =>
            Math.abs(w.time - brk.timeSec) < Math.abs(best.time - brk.timeSec) ? w : best,
          );
          const ball = wp.balls.find((b) => b.id === traj.ballId)!;
          expect(px(dist(vertex, { x: ball.pos.x, y: ball.pos.y }))).toBeLessThanOrEqual(
            ROUTE_TOLERANCE_PX,
          );
        }
      }
      // The exclusion above must not empty the test.
      expect(checked, `${topology}: no non-pocket break to check`).toBeGreaterThan(0);
    });

    it("roles come from the event log, and the cue ball's route starts where the cue ball was", () => {
      const { pre, sim, motion } = f();
      const cue = motion.trajectories.find((t) => t.roles.includes("cue"));
      expect(cue, `${topology}: no cue trajectory`).toBeTruthy();
      expect(cue!.ballId).toBe(CUE_ID);
      const cueBefore = pre.find((b) => b.id === CUE_ID)!;
      expect(px(dist(cue!.points[0], { x: cueBefore.pos.x, y: cueBefore.pos.y }))).toBeLessThanOrEqual(
        ROUTE_TOLERANCE_PX,
      );

      for (const traj of motion.trajectories) {
        if (traj.roles.includes("first-contact")) expect(sim.firstContact).toBe(traj.ballId);
        expect(traj.pocketed).toBe(sim.pocketed.includes(traj.ballId));
        expect(traj.roles.includes("potted")).toBe(sim.pocketed.includes(traj.ballId));
        if (traj.roles.includes("combination")) {
          const setMoving = sim.events.find(
            (e) => e.kind === "ball-ball" && e.balls.includes(traj.ballId),
          )!;
          expect(setMoving.balls).not.toContain(CUE_ID);
        }
      }
    });

    it("a potted ball's route ends at the pocket it dropped into", () => {
      const { motion } = f();
      for (const traj of motion.trajectories) {
        if (!traj.pocketed) continue;
        const drop = traj.breaks.filter((b) => b.kind === "pocket").pop();
        expect(drop, `ball ${traj.ballId} is potted but has no pocket break`).toBeTruthy();
        const pocket = table.pockets.find((p) => p.id === drop!.pocket)!;
        const end = traj.points[traj.points.length - 1];
        // The simulator captures a ball when its CENTRE reaches the pocket
        // radius, so the capture-instant position sits on that circle. A ball
        // radius of slack covers the difference between the Rust core's event
        // solve and the TS `advanceBall` that reproduces it.
        expect(dist(end, { x: pocket.center.x, y: pocket.center.y })).toBeLessThan(
          pocket.radius + BALL_RADIUS,
        );
        // And the pocket break is that terminal point, not an earlier one.
        expect(drop!.at).toBe(traj.points.length - 1);
      }
    });

    it("only balls that really travelled get a route, and each has at least two points", () => {
      const { sim, motion } = f();
      const ids = new Set(motion.trajectories.map((t) => t.ballId));
      for (const traj of motion.trajectories) {
        expect(traj.points.length).toBeGreaterThanOrEqual(2);
        let travel = 0;
        for (let i = 1; i < traj.points.length; i++) travel += dist(traj.points[i], traj.points[i - 1]);
        expect(travel).toBeGreaterThanOrEqual(MIN_TRAVEL_M);
      }
      // A ball that finished exactly where it started is not drawn.
      for (const wp0 of [sim.waypoints![0]]) {
        for (const b of wp0.balls) {
          const final = sim.balls.find((x) => x.id === b.id)!;
          if (final.pocketed) continue;
          const moved = dist({ x: b.pos.x, y: b.pos.y }, { x: final.pos.x, y: final.pos.y });
          if (moved === 0 && !ids.has(b.id)) continue;
          if (moved === 0) {
            // It can still be drawn if it moved and came back, but not if it
            // never moved at all — check the whole record.
            const anyMotion = sim.waypoints!.some((w) => {
              const x = w.balls.find((y) => y.id === b.id);
              return !!x && !x.pocketed && dist({ x: x.pos.x, y: x.pos.y }, { x: b.pos.x, y: b.pos.y }) > MIN_TRAVEL_M;
            });
            expect(anyMotion, `ball ${b.id} is drawn but never moved`).toBe(true);
          }
        }
      }
      // Ordering: by when each ball started moving, cue first.
      const orders = motion.trajectories.map((t) => t.order);
      expect(orders).toEqual(motion.trajectories.map((_, i) => i));
      for (let i = 1; i < motion.trajectories.length; i++) {
        expect(motion.trajectories[i].startSec).toBeGreaterThanOrEqual(
          motion.trajectories[i - 1].startSec,
        );
      }
    });

    it("survives a JSON round trip with no undefined and no non-finite number", () => {
      const { motion } = f();
      expect(JSON.parse(JSON.stringify(motion))).toEqual(motion);
      const scan = (v: unknown, path: string): void => {
        expect(v, `undefined at ${path}`).not.toBeUndefined();
        if (typeof v === "number") expect(Number.isFinite(v), `${path} = ${v}`).toBe(true);
        else if (Array.isArray(v)) v.forEach((x, i) => scan(x, `${path}[${i}]`));
        else if (v && typeof v === "object") {
          for (const [k, x] of Object.entries(v as Record<string, unknown>)) scan(x, `${path}.${k}`);
        }
      };
      scan(motion, topology);
    });
  });

  describe("it refuses to invent a route", () => {
    it("returns null when the simulation captured no waypoints", () => {
      const { sim } = fixtures.get("single-bank")!;
      expect(extractExecutedMotion({ ...sim, waypoints: undefined })).toBeNull();
      expect(extractExecutedMotion({ ...sim, waypoints: [] })).toBeNull();
      expect(extractExecutedMotion({ ...sim, waypoints: [sim.waypoints![0]] })).toBeNull();
    });

    it("returns null when nothing moved far enough to have a route", () => {
      const still = fixtures.get("single-bank")!.sim;
      // Every waypoint identical to the first: a record of a table at rest.
      const frozen = still.waypoints!.map((_w, i) => ({
        time: i * 0.05,
        balls: still.waypoints![0].balls.map(cloneBall),
      }));
      expect(extractExecutedMotion({ ...still, waypoints: frozen, events: [] })).toBeNull();
    });

    it("`withExecutedMotion` cannot mint a trace and leaves the original alone", () => {
      const { motion } = fixtures.get("single-bank")!;
      const bare = { selected: null } as unknown as DecisionTraceV1;
      expect(withExecutedMotion(bare, motion)).toBe(bare);

      const trace = {
        version: "showboat-decision-trace/2",
        selected: { executed: null, kind: "bank" },
      } as unknown as DecisionTraceV1;
      const out = withExecutedMotion(trace, motion);
      expect(out).not.toBe(trace);
      expect(trace.selected!.executed).toBeNull();
      expect(out.selected!.executed).toBe(motion);
      // The version is carried, never issued.
      expect(out.version).toBe(trace.version);
    });
  });

  describe("what the overlay picks out of the motion", () => {
    it("draws the cue, the ball that drops, and the balls in between — not bystanders", () => {
      for (const topology of TOPOLOGIES) {
        const { motion } = fixtures.get(topology)!;
        const r = measuredRoute(motion);
        if (r.cue) expect(r.cue.ballId).toBe(CUE_ID);

        if (r.object) {
          const traj = motion.trajectories.find((t) => t.ballId === r.object!.ballId)!;
          const potted = motion.trajectories.filter(
            (t) => t.roles.includes("potted") && t.ballId !== CUE_ID,
          );
          // The ball that dropped, when one did; otherwise the one the cue hit.
          if (potted.length > 0) expect(traj.roles).toContain("potted");
          else expect(traj.roles.length === 0 || traj.roles.includes("first-contact")).toBe(true);
        }

        // Every drawn extra leg is a ball the EVENT LOG gave a role to.
        for (const leg of r.others) {
          const traj = motion.trajectories.find((t) => t.ballId === leg.ballId)!;
          expect(traj.roles.length, `ball ${leg.ballId} drawn with no role`).toBeGreaterThan(0);
          expect(leg.ballId).not.toBe(CUE_ID);
          expect(leg.ballId).not.toBe(r.object?.ballId);
        }
      }
    });

    it("the bystander filter is not vacuous — the combo fixture has one to drop", () => {
      // Without a case that actually exercises it, the rule above would pass on
      // a fixture set where every moving ball happened to have a role.
      const { motion } = fixtures.get("combo")!;
      const roleless = motion.trajectories.filter(
        (t) => t.roles.length === 0 && t.ballId !== CUE_ID,
      );
      expect(roleless.length, "no role-less mover in the combo fixture").toBeGreaterThan(0);
      const drawn = new Set(measuredRoute(motion).others.map((l) => l.ballId));
      for (const t of roleless) expect(drawn.has(t.ballId)).toBe(false);
      // And it is still in the published data — omitted from the drawing, not
      // dropped from the record.
      expect(motion.trajectories.some((t) => t.ballId === roleless[0].ballId)).toBe(true);
    });
  });

  describe("the declared tolerance is the one the canvas actually has", () => {
    it("1.208e-3 m is half a logical pixel at the shipped view", () => {
      // If the canvas size or the table changes, this fails rather than the
      // comment quietly becoming false.
      expect(px(SIMPLIFY_TOLERANCE_M)).toBeCloseTo(ROUTE_TOLERANCE_PX, 3);
      expect(VIEW.scale).toBeCloseTo(413.89, 2);
    });
  });
});
