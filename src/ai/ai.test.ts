import { describe, expect, it } from "vitest";
import { makeBall } from "../physics/ball";
import { makeTable } from "../physics/table";
import { applyCue } from "../physics/cue";
import { simulateShot, type ShotEvent } from "../physics/engine";
import { type GameState } from "../game/state";
import { generateCandidates } from "./candidates";
import { classifyShot } from "./classify";
import { featuresOf, FEATURE_DIM } from "./features";
import { neuralScore, validateModel } from "./ranker";

// Assertions here use independent oracles: hand-written event logs with a
// classification worked out by hand, hand-computed forward-pass arithmetic,
// and physical invariants (a bank must touch a cushion) — never a value
// derived by re-running the code under test.

const midGameState = (): GameState => ({
  balls: [
    makeBall(0, -0.5, 0), // cue
    makeBall(1, 0.3, 0.1), // solid
    makeBall(2, 0.1, -0.25), // solid
    makeBall(8, 0.7, 0.3),
    makeBall(9, -0.2, 0.35), // stripe
  ],
  turn: 0,
  groups: { 0: "solids", 1: "stripes" },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 5,
});

describe("candidate generation", () => {
  it("targets only the shooter's group (never opponent balls, never the 8 early)", () => {
    const g = midGameState();
    const cands = generateCandidates(g, makeTable());
    expect(cands.length).toBeGreaterThan(0);
    for (const c of cands) {
      expect([1, 2]).toContain(c.targetBall);
      if (c.comboBall !== undefined) expect([1, 2]).toContain(c.comboBall);
    }
  });

  it("targets the 8-ball once the group is cleared", () => {
    const g = midGameState();
    g.balls = g.balls.filter((b) => b.id !== 1 && b.id !== 2);
    const cands = generateCandidates(g, makeTable());
    expect(cands.length).toBeGreaterThan(0);
    for (const c of cands) expect(c.targetBall).toBe(8);
  });

  it("actions stay inside the jump/masse-free action space", () => {
    const g = midGameState();
    for (const c of generateCandidates(g, makeTable())) {
      // The CueAction type has no elevation axis at all; runtime bounds too.
      expect(Object.keys(c.action).sort()).toEqual(
        ["phi", "power", "sideSpin", "topSpin"].sort(),
      );
      expect(c.action.power).toBeGreaterThan(0);
      expect(c.action.power).toBeLessThanOrEqual(1);
      expect(Number.isFinite(c.action.phi)).toBe(true);
    }
  });

  it("a generated 1-rail bank really banks: the object ball touches the planned cushion", () => {
    // Cue behind the object ball, object near the middle: bank shots off the
    // long rails exist. Simulate the candidate's actual action and check the
    // event log — the physical invariant, not the generator's own geometry.
    const table = makeTable();
    const g: GameState = {
      ...midGameState(),
      balls: [makeBall(0, -0.6, 0), makeBall(1, 0, 0), makeBall(9, 0.8, -0.4)],
    };
    const banks = generateCandidates(g, table).filter(
      (c) => c.kind === "bank" && c.railsPlanned === 1,
    );
    expect(banks.length).toBeGreaterThan(0);
    let touched = 0;
    for (const bank of banks) {
      const balls = g.balls.map((b) => ({ ...b, pos: { ...b.pos }, vel: { ...b.vel }, roll: { ...b.roll } }));
      applyCue(balls[0], bank.action);
      const sim = simulateShot(balls, table);
      const objectHitCushion = sim.events.some(
        (e) => e.kind === "ball-cushion" && e.balls.includes(1),
      );
      if (objectHitCushion) touched++;
    }
    // Mirror geometry vs real restitution/friction differ, so not every bank
    // connects — but the family must be physically real, not decorative.
    expect(touched).toBeGreaterThan(banks.length * 0.5);
  });
});

describe("measured shot classification", () => {
  const simOf = (events: ShotEvent[]) => ({
    balls: [],
    events,
    pocketed: events.filter((e) => e.kind === "pocket").flatMap((e) => e.balls),
    firstContact: null,
    duration: 2,
  });

  it("labels a bank from the event log: object ball hits a rail before dropping", () => {
    const m = classifyShot(
      simOf([
        { time: 0.2, kind: "ball-ball", balls: [0, 3] },
        { time: 0.6, kind: "ball-cushion", balls: [3], cushion: "top" },
        { time: 1.1, kind: "pocket", balls: [3], pocket: "bl" },
        { time: 2, kind: "stop", balls: [] },
      ]),
      3,
    );
    expect(m.kind).toBe("bank");
    expect(m.objectRails).toBe(1);
    expect(m.targetPotted).toBe(true);
    expect(m.pocketId).toBe("bl");
  });

  it("labels a kick when the CUE takes the rail before contact", () => {
    const m = classifyShot(
      simOf([
        { time: 0.1, kind: "ball-cushion", balls: [0], cushion: "left" },
        { time: 0.4, kind: "ball-ball", balls: [0, 5] },
        { time: 0.9, kind: "pocket", balls: [5], pocket: "tr" },
        { time: 2, kind: "stop", balls: [] },
      ]),
      5,
    );
    expect(m.kind).toBe("kick");
    expect(m.cueRailsBeforeContact).toBe(1);
  });

  it("labels a combo from the chain, not from any planner intent", () => {
    const m = classifyShot(
      simOf([
        { time: 0.2, kind: "ball-ball", balls: [0, 2] },
        { time: 0.5, kind: "ball-ball", balls: [2, 4] },
        { time: 1.0, kind: "pocket", balls: [4], pocket: "st" },
        { time: 2, kind: "stop", balls: [] },
      ]),
      4,
    );
    expect(m.kind).toBe("combo");
    expect(m.chainLength).toBe(2);
  });

  it("a shot that pots nothing is no-pot even if it was meant as a bank", () => {
    const m = classifyShot(
      simOf([
        { time: 0.2, kind: "ball-ball", balls: [0, 3] },
        { time: 0.6, kind: "ball-cushion", balls: [3], cushion: "top" },
        { time: 2, kind: "stop", balls: [] },
      ]),
      3,
    );
    expect(m.kind).toBe("no-pot");
    expect(m.targetPotted).toBe(false);
  });
});

describe("ranker model loading", () => {
  it("rejects the malformed / placeholder weights file", () => {
    expect(validateModel({ placeholder: "x", layers: [] })).toBeNull();
    expect(validateModel(null)).toBeNull();
    expect(validateModel({ layers: [{ W: [[1, 2]], b: [0] }] })).toBeNull();
  });

  it("computes the documented forward pass (hand-checked arithmetic)", () => {
    // 13 -> 1 (tanh) -> 1 (sigmoid). First layer all zeros => tanh(0) = 0.
    // Output layer: sigmoid(2*0 + 0.5) = 1 / (1 + e^-0.5) ≈ 0.62246.
    const model = validateModel({
      meta: {
        trainedAt: "t",
        datasetRows: 0,
        featureNames: [],
        normalization: {
          mean: new Array(FEATURE_DIM).fill(0),
          std: new Array(FEATURE_DIM).fill(1),
        },
        heldOut: {},
      },
      layers: [
        { W: [new Array(FEATURE_DIM).fill(0)], b: [0] },
        { W: [[2]], b: [0.5] },
      ],
    });
    expect(model).not.toBeNull();
    const p = neuralScore(model!, new Array(FEATURE_DIM).fill(0.3));
    expect(p).toBeCloseTo(0.6224593, 5);
  });
});

describe("feature extraction", () => {
  it("has the declared dimensionality and stays in sane ranges", () => {
    const g = midGameState();
    const table = makeTable();
    for (const c of generateCandidates(g, table).slice(0, 50)) {
      const f = featuresOf(g, table, c);
      expect(f).toHaveLength(FEATURE_DIM);
      for (const v of f) {
        expect(Number.isFinite(v)).toBe(true);
        expect(v).toBeGreaterThanOrEqual(-1);
        expect(v).toBeLessThanOrEqual(1.5);
      }
    }
  });
});
