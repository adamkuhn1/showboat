// The floor of the trick-only ladder: what happens when no trick is makeable.
//
// This is where the old policy handed control to a direct shot — first through
// `selectBestWithReason`'s fallback, and then, when the search returned nothing
// at all, through `App.tsx`'s nearest-ball aim. Both are gone. The replacement
// has to satisfy three things, all asserted here against the REAL ruleset and
// the REAL simulator:
//
//   * it is never a direct shot,
//   * it is legal (or, where the position makes that impossible, it is honestly
//     labelled `forced-legal-contact` and still returns a shot),
//   * it cannot hang the turn — bounded simulations, bounded wall clock,
//     entirely synchronous.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../../physics/table";
import { makeBall, type Ball } from "../../physics/ball";
import { BALL_RADIUS } from "../../physics/constants";
import { CUE_ID } from "../../game/rack";
import { type GameState } from "../../game/state";
import { takeShot, type Simulator } from "../../game/game";
import { initPhysics, simulateShotWasm } from "../../physics/wasm-bridge";
import { generateCandidates } from "../candidates";
import { defaultConfig, searchCandidates } from "../shotSearch";
import { selectTrickOnly } from "./trickOnly";
import { generateSafetyKicks, pickSafety, MAX_GENERATED, SAFETY_SIM_BUDGET } from "./safety";
import { classicalTrickOnlyBrain } from "../brain";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../../..");
const table = makeTable();

beforeAll(async () => {
  await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
}, 60_000);

/**
 * A REALISTIC mid-game state: groups are assigned, so `applyShotRules` really
 * enforces first-contact legality. An open-table state would make these tests
 * weaker than the game they claim to model — on an open table the rules skip
 * the first-contact check entirely, so a kick that struck the 8 first would
 * pass as "legal" here while being a foul in any actual mid-game position.
 */
const asState = (balls: Ball[]): GameState => ({
  balls,
  turn: 0,
  groups: { 0: "solids", 1: "stripes" },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 1,
});

const cfg = {
  ...defaultConfig,
  seed: 20260805,
  seedTimeoutMs: Infinity,
  searchTimeoutMs: Infinity,
};

/** Full live-path decision on a board, through the real search and policy. */
const decide = (balls: Ball[], targets: number[], config = cfg, simulate: Simulator = simulateShotWasm) => {
  const outcome = searchCandidates(generateCandidates(balls, table, targets), balls, targets, config);
  return {
    outcome,
    decision: selectTrickOnly(outcome.allStats, outcome.verifications, {
      state: asState(balls),
      table,
      targets,
      simulate,
    }),
  };
};

// Positions in which the candidate generator produces NOTHING — which is
// exactly the `result.best === null` condition that used to trigger App.tsx's
// nearest-ball direct aim. Groups are assigned in `asState`, so
// `applyShotRules` really enforces first-contact legality on every assertion
// below; an open-table state would let a kick that struck the 8 first pass as
// "legal" here while being a foul in any real mid-game position.
//
// Rung-4 territory: the cue is screened off its only legal target, but a
// single-rail route exists.
const BLOCKED_BY_EIGHT: Ball[] = [
  makeBall(CUE_ID, -0.6, 0.05),
  makeBall(8, -0.3, 0.05),
  makeBall(1, 0.55, 0.05),
];
const BEHIND_A_STRIPE: Ball[] = [
  makeBall(CUE_ID, 0.0, -0.25),
  makeBall(12, 0.2, -0.25),
  makeBall(14, 0.2, -0.16),
  makeBall(1, 0.7, -0.25),
];
// A genuine snooker: the cue is trapped against the end rail with the 8 six
// centimetres in front of it. Every single-rail route out clips the 8. There is
// no legal shot here — a human would take the intentional foul — so this is the
// rung-5 fixture, and the point is that the policy says so honestly rather than
// inventing a direct.
const TRAPPED_BEHIND_EIGHT: Ball[] = [
  makeBall(CUE_ID, -0.8, 0.0),
  makeBall(8, -0.74, 0.0),
  makeBall(1, 0.7, 0.0),
];
const FROZEN_ON_RAIL: Ball[] = [
  makeBall(CUE_ID, 0.0, table.width / 2 - BALL_RADIUS),
  makeBall(1, 0.35, table.width / 2 - BALL_RADIUS),
];

describe("B. no makeable trick exists", () => {
  it("B1: screened off the only legal target — a legal, non-direct kick is played", () => {
    const targets = [1];
    // The premise: no candidate at all, so there is no trick to play AND no
    // direct to fall back to.
    expect(generateCandidates(BLOCKED_BY_EIGHT, table, targets).length).toBe(0);

    const { decision } = decide(BLOCKED_BY_EIGHT, targets);
    expect(decision.shot).not.toBeNull();
    expect(decision.shot!.kind).toBe("safety-kick");
    expect(decision.rung).toBe("non-direct-safety");

    // Legality is decided by the same `applyShotRules` the game uses.
    const report = takeShot(asState(BLOCKED_BY_EIGHT), table, decision.shot!.action, simulateShotWasm);
    expect(report.outcome.foul).toBe(false);
    // The cue really struck a legal target — not the 8, not nothing.
    expect(targets).toContain(report.sim.firstContact);
    // And it got there off a cushion, which is what makes it a kick rather
    // than the direct shot this rung exists to avoid.
    const railFirst = report.sim.events.findIndex((e) => e.kind === "ball-cushion");
    const contact = report.sim.events.findIndex((e) => e.kind === "ball-ball");
    expect(railFirst).toBeGreaterThanOrEqual(0);
    expect(railFirst).toBeLessThan(contact);
  }, 30_000);

  it("B1b: screened behind two of the opponent's balls — still legal, still not a direct", () => {
    const targets = [1];
    const { decision } = decide(BEHIND_A_STRIPE, targets);
    expect(decision.shot!.kind).toBe("safety-kick");
    expect(decision.rung).toBe("non-direct-safety");
    const report = takeShot(asState(BEHIND_A_STRIPE), table, decision.shot!.action, simulateShotWasm);
    expect(report.outcome.foul).toBe(false);
  }, 30_000);

  it("B2: a genuine snooker — a shot is still returned, fast, and labelled honestly", () => {
    const targets = [1];
    const t0 = performance.now();
    const { decision } = decide(TRAPPED_BEHIND_EIGHT, targets);
    const elapsed = performance.now() - t0;

    // Never null, never a hang, never a direct.
    expect(decision.shot).not.toBeNull();
    expect(decision.shot!.kind).toBe("safety-kick");
    expect(decision.rung).toBe("forced-legal-contact");
    expect(elapsed).toBeLessThan(500);
    expect(decision.safetySimsSpent).toBeLessThanOrEqual(SAFETY_SIM_BUDGET);
    expect(decision.shot!.action.power).toBeCloseTo(0.3, 5);

    // The resulting foul is recorded truthfully rather than dressed up as a
    // success. It is a wrong-ball contact, not a whiff: the cue did get there.
    const report = takeShot(asState(TRAPPED_BEHIND_EIGHT), table, decision.shot!.action, simulateShotWasm);
    expect(report.outcome.foul).toBe(true);
    expect(report.outcome.foulReason).not.toBe("no contact");
    expect(report.sim.firstContact).not.toBeNull();
  }, 30_000);

  it("B2b: an object ball frozen on a cushion in line with the cue does not hang", () => {
    const targets = [1];
    const t0 = performance.now();
    const { decision } = decide(FROZEN_ON_RAIL, targets);
    expect(performance.now() - t0).toBeLessThan(5000);
    expect(decision.shot).not.toBeNull();
    expect(decision.shot!.kind).not.toBe("direct");
    expect(decision.safetySimsSpent).toBeLessThanOrEqual(SAFETY_SIM_BUDGET);
  }, 30_000);

  it("B3: zero legal targets — no shot, no throw", async () => {
    // Group cleared and the 8 already down: `legalTargets` is empty.
    const balls: Ball[] = [makeBall(CUE_ID, -0.5, 0.0)];
    const state: GameState = { ...asState(balls), groups: { 0: "solids", 1: "stripes" } };
    const result = await classicalTrickOnlyBrain().plan(state, table, 0, cfg);
    expect(result.shot).toBeNull();
    expect(result.best).toBeNull();
    expect(result.decision.selected).toBeNull();
  }, 30_000);

  it("B4: a 3-unit physics budget still yields a non-direct shot", () => {
    const balls: Ball[] = [
      makeBall(CUE_ID, -0.6, -0.1),
      makeBall(1, 0.3, 0.02),
      makeBall(2, 0.38, 0.1),
      makeBall(4, 0.1, -0.25),
    ];
    const targets = [1, 2, 4];
    const { outcome, decision } = decide(balls, targets, { ...cfg, simulations: 3 });
    expect(outcome.simulations).toBeLessThanOrEqual(3);
    expect(decision.shot).not.toBeNull();
    expect(decision.shot!.kind).not.toBe("direct");
    expect(decision.safetySimsSpent).toBeLessThanOrEqual(SAFETY_SIM_BUDGET);
  }, 30_000);

  it("B5: the safety budget is bounded by simulator invocation count, not by timing", () => {
    let calls = 0;
    const counting: Simulator = (balls, action) => {
      calls++;
      return simulateShotWasm(balls, action);
    };
    // Force the safety rung: an empty trick set with a legal target present.
    const targets = [1];
    const d = selectTrickOnly([], [], {
      state: asState(BEHIND_A_STRIPE),
      table,
      targets,
      simulate: counting,
    });
    expect(d.shot).not.toBeNull();
    expect(calls).toBeLessThanOrEqual(SAFETY_SIM_BUDGET);
    expect(d.safetySimsSpent).toBe(calls);
  }, 30_000);
});

describe("safety kick generation is bounded and rail-first by construction", () => {
  it("never generates more than the declared ceiling", () => {
    const crowded: Ball[] = [
      makeBall(CUE_ID, 0.0, 0.0),
      makeBall(1, 0.3, 0.1),
      makeBall(2, -0.3, -0.1),
      makeBall(3, 0.5, -0.2),
      makeBall(4, -0.5, 0.2),
      makeBall(5, 0.7, 0.05),
      makeBall(6, -0.7, -0.05),
    ];
    const kicks = generateSafetyKicks(crowded, table, [1, 2, 3, 4, 5, 6], "cue-to-rail-only");
    expect(kicks.length).toBeLessThanOrEqual(MAX_GENERATED);
    for (const k of kicks) {
      // Rail-first: the cue's first waypoint is a cushion, not the object ball.
      expect(k.cuePath.length).toBe(3);
      const onRail =
        Math.abs(Math.abs(k.railPoint.x) - (table.length / 2 - BALL_RADIUS)) < 1e-6 ||
        Math.abs(Math.abs(k.railPoint.y) - (table.width / 2 - BALL_RADIUS)) < 1e-6;
      expect(onRail, `rail point ${JSON.stringify(k.railPoint)} is not on a cushion`).toBe(true);
      expect(k.action.power).toBeLessThanOrEqual(0.7);
      expect(k.action.power).toBeGreaterThanOrEqual(0.3);
    }
  });

  it("returns `none` only when no kick can be constructed at all", () => {
    const res = pickSafety(asState([makeBall(CUE_ID, 0, 0)]), table, [], simulateShotWasm);
    expect(res.kick).toBeNull();
    expect(res.quality).toBe("none");
    expect(res.simsSpent).toBe(0);
  });

  // B6. The module header says, without qualification, that a kick sends the
  // CUE ball into a cushion FIRST. That was false for a measurable fraction of
  // what `pickSafety` returned: the top-up generation pass dropped the
  // obstruction check entirely, so a route with the target sitting between the
  // cue and the rail point survived, simulated foul-free, and shipped labelled
  // `safety-kick` and described to the visitor as "a safety off the cushion"
  // while the cue in fact went straight at the ball — and potted it in most of
  // those cases.
  //
  // Judged on what the SIMULATOR did, not on what the generator intended: the
  // shot is executed through the real ruleset and its event log is read for a
  // cue-cushion contact preceding the cue's first ball contact. That is the
  // same test the cold review used to find the defect.
  it("B6: no selected safety kick ever strikes a ball before a cushion", () => {
    let seed = 20260806 >>> 0;
    const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    const between = (a: number, b: number) => a + rnd() * (b - a);
    const hx = table.length / 2 - BALL_RADIUS;
    const hy = table.width / 2 - BALL_RADIUS;

    // Three families. The two rail-hug ones are adversarial on purpose: they
    // put the target within a ball diameter of the cue's rail point, which is
    // the geometry the un-extended obstruction check could not see.
    const families = [
      (): Ball[] => [
        makeBall(CUE_ID, between(-hx, hx), between(-hy, hy)),
        makeBall(1, between(-hx, hx), between(-hy, hy)),
        makeBall(8, between(-hx, hx), between(-hy, hy)),
      ],
      (): Ball[] => {
        const y = (rnd() < 0.5 ? 1 : -1) * (hy - between(0, 0.02));
        const cx = between(-hx, hx);
        return [
          makeBall(CUE_ID, cx, y),
          makeBall(1, cx + (rnd() < 0.5 ? 1 : -1) * between(0.06, 0.5), y + between(-0.01, 0.01)),
          makeBall(8, between(-hx, hx), between(-hy, hy)),
        ];
      },
      (): Ball[] => {
        const x = (rnd() < 0.5 ? 1 : -1) * (hx - between(0, 0.02));
        const cy = between(-hy, hy);
        return [
          makeBall(CUE_ID, x, cy),
          makeBall(1, x + between(-0.01, 0.01), cy + (rnd() < 0.5 ? 1 : -1) * between(0.06, 0.4)),
          makeBall(8, between(-hx, hx), between(-hy, hy)),
        ];
      },
    ];
    const overlapping = (b: Ball[]) => {
      for (let i = 0; i < b.length; i++) {
        for (let j = i + 1; j < b.length; j++) {
          if (Math.hypot(b[i].pos.x - b[j].pos.x, b[i].pos.y - b[j].pos.y) < 2 * BALL_RADIUS + 1e-4) {
            return true;
          }
        }
      }
      return false;
    };

    let boards = 0;
    let kicks = 0;
    const offenders: string[] = [];
    while (boards < 300) {
      const balls = families[boards % 3]();
      if (overlapping(balls)) continue;
      boards++;
      const state = asState(balls);
      // Forced onto the safety rung: an empty trick set with a legal target.
      const d = selectTrickOnly([], [], { state, table, targets: [1], simulate: simulateShotWasm });
      if (!d.shot) continue;
      kicks++;
      const report = takeShot(state, table, d.shot.action, simulateShotWasm);
      const ev = report.sim.events;
      const cushion = ev.findIndex((e) => e.kind === "ball-cushion" && e.balls.includes(CUE_ID));
      const contact = ev.findIndex((e) => e.kind === "ball-ball" && e.balls.includes(CUE_ID));
      const ballFirst = contact >= 0 && (cushion < 0 || contact < cushion);
      if (ballFirst) {
        offenders.push(
          `${d.rung} cue=(${balls[0].pos.x.toFixed(3)},${balls[0].pos.y.toFixed(3)}) ` +
            `target=(${balls[1].pos.x.toFixed(3)},${balls[1].pos.y.toFixed(3)})`,
        );
      }
    }

    // The rung has to still exist for the guarantee to be worth anything —
    // "no bad kicks" is trivially satisfiable by generating none.
    expect(kicks, "the safety rung must still produce shots").toBeGreaterThan(250);
    expect(offenders, `${offenders.length} of ${kicks} kicks hit a ball before a cushion`).toEqual(
      [],
    );
  }, 120_000);
});
