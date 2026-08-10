// The trace contract, pinned.
//
// Three properties, in order of how expensive they are to lose:
//
//  1. **Anti-fabrication.** Every candidate carrying a non-null `physics` block
//     must be an index the search really ran a simulation on. This is the check
//     that makes "the overlay never shows a number the physics didn't produce"
//     mechanical instead of aspirational — a builder that invented a plausible
//     verification for a pruned candidate would fail here.
//  2. **JSON-serializable, `null` never `undefined`.** Asserted by a
//     `JSON.parse(JSON.stringify(x))` deep-equality round trip on a REAL trace,
//     which fails the moment an optional property creeps in.
//  3. **Shape.** An exhaustive literal that must typecheck, so a field being
//     removed or renamed breaks a compile rather than a renderer.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../../physics/table";
import { makeBall, type Ball } from "../../physics/ball";
import { CUE_ID } from "../../game/rack";
import { type GameState } from "../../game/state";
import { initPhysics } from "../../physics/wasm-bridge";
import { defaultConfig } from "../shotSearch";
import { classicalTrickOnlyBrain, type AiDecision } from "../brain";
import { DECISION_TRACE_VERSION, TRICK_KINDS, type DecisionTraceV1 } from "./contract";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../../..");
const table = makeTable();

const BOARDS: Record<string, { balls: Ball[]; targets: number[] }> = {
  openSpread: {
    balls: [
      makeBall(CUE_ID, -0.35, 0.05),
      makeBall(1, 0.25, 0.15),
      makeBall(3, -0.05, -0.28),
      makeBall(9, 0.5, -0.2),
    ],
    targets: [1, 3, 9],
  },
  railHeavy: {
    balls: [
      makeBall(CUE_ID, 0.0, 0.0),
      makeBall(1, -0.7, 0.28),
      makeBall(3, 0.72, -0.29),
      makeBall(6, -0.4, -0.3),
    ],
    targets: [1, 3, 6],
  },
  snookered: {
    balls: [makeBall(CUE_ID, -0.8, 0.0), makeBall(8, -0.74, 0.0), makeBall(1, 0.7, 0.0)],
    targets: [1],
  },
};

const asState = (balls: Ball[]): GameState => ({
  balls,
  turn: 0,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 1,
});

const decisions: Record<string, AiDecision> = {};

beforeAll(async () => {
  await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
  const brain = classicalTrickOnlyBrain();
  for (const [name, { balls }] of Object.entries(BOARDS)) {
    decisions[name] = await brain.plan(asState(balls), table, 0, {
      ...defaultConfig,
      seed: 20260805,
      seedTimeoutMs: Infinity,
      searchTimeoutMs: Infinity,
    });
  }
}, 120_000);

describe("anti-fabrication: a physics block means physics really ran", () => {
  it("every non-null `physics` index is one the search verified", () => {
    for (const [name, d] of Object.entries(decisions)) {
      const verified = new Set(d.trace!.verifiedIndices);
      const claimed = d.decision.candidates.filter((c) => c.physics !== null).map((c) => c.index);
      for (const i of claimed) {
        expect(verified.has(i), `${name}: trace claims physics on unverified candidate ${i}`).toBe(true);
      }
      // And the converse: nothing that WAS verified is silently dropped.
      expect(new Set(claimed)).toEqual(verified);
    }
  });

  it("a physics block's contact sequence is the simulator's own event log", () => {
    for (const d of Object.values(decisions)) {
      for (const c of d.decision.candidates) {
        if (!c.physics) continue;
        // Rails-before-pot must be consistent with the sequence it came from.
        const beforePot: string[] = [];
        for (const e of c.physics.contactSequence) {
          if (e.kind === "pocket") break;
          if (e.kind === "ball-cushion") beforePot.push(e.cushion!);
        }
        const potted = c.physics.contactSequence.some((e) => e.kind === "pocket");
        expect(c.physics.railsBeforePot).toBe(potted ? beforePot.length : 0);
        // Every rail contact is a ball-cushion event of the same log.
        expect(c.physics.railContacts.length).toBe(
          c.physics.contactSequence.filter((e) => e.kind === "ball-cushion").length,
        );
      }
    }
  });

  it("candidates that never reached physics carry `physics: null` and a real reason", () => {
    for (const d of Object.values(decisions)) {
      const verified = new Set(d.trace!.verifiedIndices);
      for (const c of d.decision.candidates) {
        if (verified.has(c.index)) continue;
        expect(c.physics).toBeNull();
        expect(c.rejection).not.toBeNull();
      }
    }
  });
});

describe("the trace is JSON, with null and never undefined", () => {
  it("round-trips to deep equality on every fixture", () => {
    for (const [name, d] of Object.entries(decisions)) {
      const round = JSON.parse(JSON.stringify(d.decision));
      expect(round, `${name} did not survive a JSON round trip`).toEqual(d.decision);
    }
  });

  it("contains no `undefined` anywhere, at any depth", () => {
    const scan = (v: unknown, path: string): void => {
      expect(v, `undefined at ${path}`).not.toBeUndefined();
      if (Array.isArray(v)) v.forEach((x, i) => scan(x, `${path}[${i}]`));
      else if (v && typeof v === "object") {
        for (const [k, x] of Object.entries(v as Record<string, unknown>)) scan(x, `${path}.${k}`);
      }
    };
    for (const [name, d] of Object.entries(decisions)) scan(d.decision, name);
  });

  it("contains no NaN or Infinity, which JSON turns into null", () => {
    const scan = (v: unknown, path: string): void => {
      if (typeof v === "number") expect(Number.isFinite(v), `${path} = ${v}`).toBe(true);
      else if (Array.isArray(v)) v.forEach((x, i) => scan(x, `${path}[${i}]`));
      else if (v && typeof v === "object") {
        for (const [k, x] of Object.entries(v as Record<string, unknown>)) scan(x, `${path}.${k}`);
      }
    };
    for (const [name, d] of Object.entries(decisions)) scan(d.decision, name);
  });
});

describe("the guarantees a renderer is allowed to rely on", () => {
  it("candidates are in generation order, directs included, indexed by position", () => {
    for (const d of Object.values(decisions)) {
      d.decision.candidates.forEach((c, i) => expect(c.index).toBe(i));
      expect(d.decision.candidates.length).toBe(d.allStats.length);
      d.decision.candidates.forEach((c, i) => expect(c.kind).toBe(d.allStats[i].candidate.kind));
    }
  });

  it("the shot being played is addressable by generation index", () => {
    for (const [name, d] of Object.entries(decisions)) {
      const sel = d.decision.selected;
      expect(sel, `${name} produced no selection`).not.toBeNull();
      if (sel!.candidateIndex === null) {
        // The only shot with no candidate is a generated safety kick.
        expect(sel!.kind).toBe("safety-kick");
        continue;
      }
      const c = d.decision.candidates[sel!.candidateIndex];
      expect(c.index).toBe(sel!.candidateIndex);
      expect(c.kind).toBe(sel!.kind);
      // Exactly one candidate has a null rejection, and it is that one.
      const unrejected = d.decision.candidates.filter((x) => x.rejection === null);
      expect(unrejected.map((x) => x.index)).toEqual([sel!.candidateIndex]);
    }
  });

  it("every candidate carries kind, eligible, neural, physics and a rejection", () => {
    for (const d of Object.values(decisions)) {
      for (const c of d.decision.candidates) {
        expect(typeof c.kind).toBe("string");
        expect(typeof c.eligible).toBe("boolean");
        expect(c.eligible).toBe(c.kind !== "direct");
        expect(c.neural === null || typeof c.neural.score === "number").toBe(true);
        expect(c.physics === null || typeof c.physics.strength === "number").toBe(true);
        expect(c.rejection === null || typeof c.rejection === "string").toBe(true);
      }
    }
  });

  it("every path can be drawn from the cue ball", () => {
    for (const d of Object.values(decisions)) {
      const cue = d.decision.turn.cueBall;
      for (const c of d.decision.candidates) {
        expect(c.cuePath.length).toBeGreaterThanOrEqual(2);
        expect(c.cuePath[0]).toEqual(cue);
      }
      const sel = d.decision.selected;
      if (sel) {
        expect(sel.cuePath.length).toBeGreaterThanOrEqual(2);
        expect(sel.cuePath[0]).toEqual(cue);
      }
    }
  });

  it("labels the decision truthfully: policy, mode, model, fallback", () => {
    for (const d of Object.values(decisions)) {
      expect(d.decision.version).toBe(DECISION_TRACE_VERSION);
      expect(d.decision.policy).toBe("trick-only");
      // These fixtures run the classical brain, so: no model, no fallback
      // (nothing was asked of a model in the first place).
      expect(d.decision.mode).toBe("classical-trick-only");
      expect(d.decision.model).toBeNull();
      expect(d.decision.fallback).toBeNull();
      // The selected kind is never a direct, in any mode.
      if (d.decision.selected) {
        const k = d.decision.selected.kind;
        expect(k === "safety-kick" || (TRICK_KINDS as readonly string[]).includes(k)).toBe(true);
      }
    }
  });

  it("a safety kick is labelled a safety, and only on rungs 4 and 5", () => {
    for (const d of Object.values(decisions)) {
      const sel = d.decision.selected;
      if (!sel || sel.kind !== "safety-kick") continue;
      expect(["non-direct-safety", "forced-legal-contact"]).toContain(sel.rung);
      expect(sel.utility).toBeNull();
      expect(sel.path).toEqual([]);
    }
  });

  it("budget figures agree with the search that produced them", () => {
    for (const d of Object.values(decisions)) {
      const b = d.decision.budget;
      expect(b.physicsUnitsSpent).toBe(d.simulations);
      expect(b.physicsUnitsSpent).toBeLessThanOrEqual(b.physicsUnitsAllowed);
      expect(b.candidatesGenerated).toBe(d.allStats.length);
      expect(b.physicsVerified).toBe(d.trace!.verifiedIndices.length);
      expect(b.safetySimsSpent).toBeLessThanOrEqual(6);
      // Safety sims are reported separately and never netted into the search
      // budget, so evaluation budget parity stays measurable.
      expect(b.physicsUnitsSpent + b.safetySimsSpent).toBeGreaterThanOrEqual(b.physicsUnitsSpent);
    }
  });
});

describe("the shape is pinned by an exhaustive literal", () => {
  it("typechecks", () => {
    const trace: DecisionTraceV1 = {
      version: DECISION_TRACE_VERSION,
      policy: "trick-only",
      mode: "neural-hybrid",
      turn: { player: 1, shotIndex: 4, legalTargets: [1, 2], cueBall: { x: -0.5, y: 0 } },
      model: { artifact: "m.onnx", sha256: "a".repeat(64), schema: "showboat-ranker-v2", hashVerified: true },
      budget: {
        physicsUnitsAllowed: 60,
        physicsUnitsSpent: 48,
        safetySimsSpent: 0,
        candidatesGenerated: 30,
        candidatesEligible: 18,
        candidatesConsidered: 16,
        physicsVerified: 16,
        prunedByPrior: 2,
        reservePromotions: 0,
        seedTimedOut: false,
      },
      timing: { totalMs: 41.2, neuralEncodeMs: 0.9, neuralRunMs: 1.6, physicsMs: 37.1, selectionMs: 0.2 },
      candidates: [
        {
          index: 0,
          kind: "direct",
          eligible: false,
          target: 1,
          potId: 1,
          pocket: "tr",
          aimPoint: { x: 0.1, y: 0.2 },
          cuePath: [{ x: -0.5, y: 0 }, { x: 0.1, y: 0.2 }],
          path: [{ x: 0.12, y: 0.22 }, { x: 0.88, y: 0.44 }],
          action: { phi: 0.3, power: 0.5, sideSpin: 0, topSpin: 0 },
          neural: { score: 0.81, logit: 1.4, rank: 1 },
          physics: null,
          measured: null,
          rejection: "direct-excluded-by-policy",
        },
        {
          index: 1,
          kind: "bank",
          eligible: true,
          target: 1,
          potId: 1,
          pocket: "bl",
          aimPoint: { x: 0.05, y: 0.15 },
          cuePath: [{ x: -0.5, y: 0 }, { x: 0.05, y: 0.15 }],
          path: [{ x: 0.06, y: 0.16 }, { x: 0.4, y: -0.32 }, { x: -0.88, y: -0.44 }],
          action: { phi: 0.1, power: 0.6, sideSpin: 0, topSpin: 0 },
          neural: { score: 0.62, logit: 0.5, rank: 3 },
          physics: {
            firstContact: 1,
            legalFirstContact: true,
            scratched: false,
            legalPot: true,
            pocketed: [1],
            railContacts: [{ ballId: 1, cushion: "bottom", timeSec: 0.31 }],
            railsBeforePot: 1,
            contactSequence: [
              { kind: "ball-ball", timeSec: 0.12, balls: [0, 1], cushion: null, pocket: null },
              { kind: "ball-cushion", timeSec: 0.31, balls: [1], cushion: "bottom", pocket: null },
              { kind: "pocket", timeSec: 0.62, balls: [1], cushion: null, pocket: "bl" },
            ],
            strength: 0.74,
            value: 1.35,
            visits: 3,
            styleScore: 1,
          },
          measured: {
            classification: "one-rail-bank",
            rails: 1,
            railCushions: ["bottom"],
            contactChain: [0, 1],
            pottedBall: 1,
            pocket: "bl",
            firstContact: 1,
            firstContactLegal: true,
            scratched: false,
            trickVerified: true,
          },
          rejection: null,
        },
      ],
      selected: {
        candidateIndex: 1,
        kind: "bank",
        plannedKind: "bank",
        measured: {
          classification: "one-rail-bank",
          rails: 1,
          railCushions: ["bottom"],
          contactChain: [0, 1],
          pottedBall: 1,
          pocket: "bl",
          firstContact: 1,
          firstContactLegal: true,
          scratched: false,
          trickVerified: true,
        },
        rung: "trick-qualified",
        action: { phi: 0.1, power: 0.6, sideSpin: 0, topSpin: 0 },
        cuePath: [{ x: -0.5, y: 0 }, { x: 0.05, y: 0.15 }],
        path: [{ x: 0.06, y: 0.16 }, { x: 0.4, y: -0.32 }, { x: -0.88, y: -0.44 }],
        utility: 0.86,
        passedOverDirectIndex: 0,
        reliabilityThreshold: 0.5,
        qualifyingTricks: 2,
        safetyQuality: null,
        // The /2 addition: the measured motion of the run that is played, as
        // distinct from the `cuePath`/`path` above, which are intentions.
        executed: {
          durationSec: 0.9,
          trajectories: [
            {
              ballId: 0,
              roles: ["cue"],
              order: 0,
              points: [{ x: -0.5, y: 0 }, { x: 0.04, y: 0.14 }],
              timesSec: [0, 0.12],
              breaks: [
                {
                  at: 1,
                  kind: "ball-contact",
                  timeSec: 0.12,
                  withBall: 1,
                  cushion: null,
                  pocket: null,
                },
              ],
              startSec: 0,
              endSec: 0.12,
              pocketed: false,
              endsAtCapture: false,
            },
            {
              ballId: 1,
              roles: ["first-contact", "potted"],
              order: 1,
              points: [{ x: 0.06, y: 0.16 }, { x: 0.41, y: -0.31 }, { x: -0.86, y: -0.43 }],
              timesSec: [0.12, 0.31, 0.62],
              breaks: [
                { at: 1, kind: "cushion", timeSec: 0.31, withBall: null, cushion: "bottom", pocket: null },
                { at: 2, kind: "pocket", timeSec: 0.62, withBall: null, cushion: null, pocket: "bl" },
              ],
              startSec: 0.12,
              endSec: 0.62,
              pocketed: true,
              endsAtCapture: true,
            },
          ],
          contactSequence: [
            { kind: "ball-ball", timeSec: 0.12, balls: [0, 1], cushion: null, pocket: null },
            { kind: "ball-cushion", timeSec: 0.31, balls: [1], cushion: "bottom", pocket: null },
            { kind: "pocket", timeSec: 0.62, balls: [1], cushion: null, pocket: "bl" },
          ],
          simplifyToleranceM: 1.208e-3,
          maxDeviationM: 4.1e-4,
        },
      },
      fallback: null,
    };
    expect(JSON.parse(JSON.stringify(trace))).toEqual(trace);
  });
});
