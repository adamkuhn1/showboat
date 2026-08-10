// A candidate may be played as a trick only when its own rollout measurably
// executed one.
//
// The defect this file pins: `potsTarget` says a legal pot happened, and says
// nothing about the route. A candidate generated as a `bank` whose mirror point
// lands beside the pocket sends the object ball straight in, `potsTarget` is
// true, and the previous ladder played it and called it a bank. Twelve of the
// fixtures below are that shot in different disguises.
//
// The fixtures are hand-built event logs paired with real candidate stats, plus
// one end-to-end board that runs the real search and the real physics.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../../physics/table";
import { makeBall, type Ball } from "../../physics/ball";
import { type ShotEvent } from "../../physics/engine";
import { CUE_ID } from "../../game/rack";
import { type GameState } from "../../game/state";
import { initPhysics, simulateShotWasm } from "../../physics/wasm-bridge";
import { generateCandidates, type Candidate, type CandidateKind } from "../candidates";
import {
  defaultConfig,
  searchCandidates,
  type CandidateStat,
  type CandidateVerification,
} from "../shotSearch";
import { buildDecisionTrace } from "../trace/build";
import { selectTrickOnly, type TrickOnlyContext } from "./trickOnly";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../../..");
const table = makeTable();

beforeAll(async () => {
  await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
}, 60_000);

const hit = (time: number, a: number, b: number): ShotEvent => ({
  time,
  kind: "ball-ball",
  balls: [a, b],
});
const rail = (time: number, b: number, cushion = "top"): ShotEvent => ({
  time,
  kind: "ball-cushion",
  balls: [b],
  cushion,
});
const pot = (time: number, b: number, pocket = "tr"): ShotEvent => ({
  time,
  kind: "pocket",
  balls: [b],
  pocket,
});

let nextPhi = 1;
const candidate = (kind: CandidateKind, potId = 1): Candidate => ({
  kind,
  target: 1,
  potId,
  pocket: "tr",
  aimPoint: { x: 0.1, y: 0.1 },
  action: { phi: nextPhi++ / 1000, power: 0.5, sideSpin: 0, topSpin: 0 },
  path: [{ x: 0, y: 0 }, { x: 0.2, y: 0.1 }],
  banks: kind === "double-bank" ? 2 : kind === "bank" || kind === "rail-combo" ? 1 : 0,
});

const statFor = (c: Candidate, strength: number, potsTarget: boolean): CandidateStat => ({
  candidate: c,
  visits: 1,
  value: strength,
  strength,
  rails: c.banks,
  potsTarget,
  styleScore: c.banks,
  verified: true,
});

const verified = (index: number, events: ShotEvent[], potId: number): CandidateVerification => ({
  index,
  firstContact: 1,
  legalFirstContact: true,
  scratched: false,
  legalPot: true,
  pocketed: [potId],
  railsBeforePot: events.filter((e) => e.kind === "ball-cushion").length,
  events,
});

const SAFETY_BOARD: Ball[] = [makeBall(CUE_ID, -0.5, 0.05), makeBall(1, 0.35, -0.1)];

const asState = (balls: Ball[]): GameState => ({
  balls,
  turn: 0,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 1,
});

const ctx = (targets: number[] = [1, 2]): TrickOnlyContext => ({
  state: asState(SAFETY_BOARD),
  table,
  targets,
  simulate: simulateShotWasm,
});

describe("a nominal trick that measurably is not one cannot be played", () => {
  it("a 0.99 'bank' whose object ball never touched a cushion is refused", () => {
    const c = candidate("bank");
    const d = selectTrickOnly(
      [statFor(c, 0.99, true)],
      [verified(0, [hit(0.1, CUE_ID, 1), pot(0.6, 1)], 1)],
      ctx(),
    );
    expect(d.shot!.kind).toBe("safety-kick");
    expect(d.verifiedTricks).toBe(0);
    expect(d.unmeasuredTrickIndices).toEqual([0]);
  });

  it("a 'combo' the cue potted on its own, with no second ball in the chain, is refused", () => {
    const c = candidate("combo", 2);
    const d = selectTrickOnly(
      [statFor(c, 0.95, true)],
      // Ball 2 drops, but the cue put it there directly — no chain through 1.
      [
        {
          index: 0,
          firstContact: 1,
          legalFirstContact: true,
          scratched: false,
          legalPot: true,
          pocketed: [2],
          railsBeforePot: 0,
          events: [hit(0.1, CUE_ID, 1), hit(0.2, CUE_ID, 2), pot(0.6, 2)],
        },
      ],
      ctx(),
    );
    expect(d.shot!.kind).toBe("safety-kick");
  });

  it("a bank whose only cushion came AFTER the pot is refused", () => {
    const c = candidate("bank");
    const d = selectTrickOnly(
      [statFor(c, 0.99, true)],
      [verified(0, [hit(0.1, CUE_ID, 1), pot(0.5, 1), rail(0.9, CUE_ID, "left")], 1)],
      ctx(),
    );
    expect(d.shot!.kind).toBe("safety-kick");
  });

  it("a measured trick beats an unmeasured one even at a third of its strength", () => {
    const fake = candidate("bank");
    const real = candidate("bank");
    const d = selectTrickOnly(
      [statFor(fake, 0.99, true), statFor(real, 0.33, true)],
      [
        verified(0, [hit(0.1, CUE_ID, 1), pot(0.6, 1)], 1),
        verified(1, [hit(0.1, CUE_ID, 1), rail(0.3, 1), pot(0.7, 1)], 1),
      ],
      ctx(),
    );
    expect(d.shot!.candidateIndex).toBe(1);
    expect(d.shot!.kind).toBe("bank");
    expect(d.verifiedTricks).toBe(1);
    expect(d.unmeasuredTrickIndices).toEqual([0]);
  });
});

describe("the shot is named by what it did, not by what it was generated as", () => {
  it("a candidate generated as a bank that ran two cushions is played as a two-rail bank", () => {
    const c = candidate("bank");
    const d = selectTrickOnly(
      [statFor(c, 0.8, true)],
      [
        verified(
          0,
          [hit(0.1, CUE_ID, 1), rail(0.3, 1, "top"), rail(0.45, 1, "right"), pot(0.7, 1)],
          1,
        ),
      ],
      ctx(),
    );
    expect(d.shot!.plannedKind).toBe("bank");
    expect(d.shot!.kind).toBe("double-bank");
    expect(d.shot!.measured!.rails).toBe(2);
  });

  it("a candidate generated as a rail-combo that took no cushion is played as a combination", () => {
    const c = candidate("rail-combo", 2);
    const d = selectTrickOnly(
      [statFor(c, 0.8, true)],
      [verified(0, [hit(0.1, CUE_ID, 1), hit(0.25, 1, 2), pot(0.6, 2)], 2)],
      ctx(),
    );
    expect(d.shot!.plannedKind).toBe("rail-combo");
    expect(d.shot!.kind).toBe("combo");
    expect(d.shot!.measured!.classification).toBe("combination");
  });
});

describe("the trace explains the refusal in its own vocabulary", () => {
  it("a refused nominal trick is labelled planned-trick-not-measured, not did-not-pot", () => {
    const c = candidate("bank");
    const stats = [statFor(c, 0.99, true)];
    const verifications = [verified(0, [hit(0.1, CUE_ID, 1), pot(0.6, 1)], 1)];
    const decision = selectTrickOnly(stats, verifications, ctx());
    const trace = buildDecisionTrace({
      outcome: { stats, allStats: stats, verifications, simulations: 3 },
      decision,
      state: asState(SAFETY_BOARD),
      player: 0,
      targets: [1, 2],
      physicsUnitsAllowed: 60,
      model: null,
      fallback: null,
      timing: { totalMs: 1, neuralEncodeMs: null, neuralRunMs: null, physicsMs: 1, selectionMs: 0 },
    });
    expect(trace.candidates[0].rejection).toBe("planned-trick-not-measured");
    expect(trace.candidates[0].measured!.classification).toBe("direct");
    expect(trace.candidates[0].measured!.trickVerified).toBe(false);
  });
});

describe("on a real board, every trick that gets played measurably happened", () => {
  it("the selected route's own event log executes the structure it is named with", async () => {
    const balls: Ball[] = [
      makeBall(CUE_ID, -0.62, -0.08),
      makeBall(1, 0.28, 0.04),
      makeBall(2, 0.36, 0.12),
      makeBall(4, 0.08, -0.24),
      makeBall(5, -0.18, 0.3),
      makeBall(6, 0.5, -0.3),
    ];
    const targets = [1, 2, 4, 5, 6];
    const candidates = generateCandidates(balls, table, targets);
    const outcome = searchCandidates(candidates, balls, targets, {
      ...defaultConfig,
      seed: 3319,
      seedTimeoutMs: Infinity,
      searchTimeoutMs: Infinity,
    });
    const d = selectTrickOnly(outcome.allStats, outcome.verifications, {
      state: asState(balls),
      table,
      targets,
      simulate: simulateShotWasm,
    });
    expect(d.shot).not.toBeNull();
    if (d.shot!.kind === "safety-kick") {
      expect(d.shot!.measured).toBeNull();
      return;
    }
    const m = d.shot!.measured!;
    expect(m.trickVerified).toBe(true);
    expect(m.classification).not.toBe("direct");
    // The named structure and the counted cushions agree.
    if (d.shot!.kind === "bank") expect(m.rails).toBe(1);
    if (d.shot!.kind === "double-bank") expect(m.rails).toBeGreaterThanOrEqual(2);
    if (d.shot!.kind === "combo") {
      expect(m.contactChain.length).toBeGreaterThan(2);
      expect(m.rails).toBe(0);
    }
    if (d.shot!.kind === "rail-combo") {
      expect(m.contactChain.length).toBeGreaterThan(2);
      expect(m.rails).toBeGreaterThanOrEqual(1);
    }
  }, 60_000);
});
