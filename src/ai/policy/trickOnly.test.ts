// Adversarial proof that the live policy cannot select a direct shot.
//
// The product decision is that Showboat wins with trick shots and never
// chooses a direct, *even when a direct would be strategically better*. So the
// interesting fixtures are the ones where choosing the direct is obviously the
// right pool decision — a 0.99-strength direct pot against a 0.10-strength
// bank — and the assertion is that the policy still doesn't.
//
// Layers 1 and 2 (the type, and the partition) are proven by A1-A5 and A8;
// layer 3 (the runtime assertion) by A10; layer 4 (the call sites) by
// trickOnlySourceGuard.test.ts.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../../physics/table";
import { makeBall, type Ball } from "../../physics/ball";
import { type ShotEvent, type SimResult } from "../../physics/engine";
import { CUE_ID } from "../../game/rack";
import { type GameState } from "../../game/state";
import { takeShot } from "../../game/game";
import { initPhysics, simulateShotWasm } from "../../physics/wasm-bridge";
import { generateCandidates, type Candidate, type CandidateKind } from "../candidates";
import {
  type CandidateStat,
  type CandidateVerification,
  TRICK_RELIABILITY_THRESHOLD,
  defaultConfig,
  isLegalPot,
  searchCandidates,
  searchWithLegacySelection,
} from "../shotSearch";
import {
  assertNotDirect,
  isTrickCandidate,
  selectTrickOnly,
  TrickOnlyInvariantError,
  type PlayableShot,
  type TrickOnlyContext,
  type TrickStat,
} from "./trickOnly";
import { classicalTrickOnlyBrain } from "../brain";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../../..");
const table = makeTable();

beforeAll(async () => {
  await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
}, 60_000);

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

let nextId = 500;
const makeCandidate = (kind: CandidateKind, overrides: Partial<Candidate> = {}): Candidate => {
  const id = nextId++;
  return {
    kind,
    target: 1,
    potId: kind === "combo" || kind === "rail-combo" ? 2 : 1,
    pocket: "tr",
    aimPoint: { x: 0.1, y: 0.1 },
    action: { phi: id / 1000, power: 0.5, sideSpin: 0, topSpin: 0 },
    path: [{ x: 0, y: 0 }, { x: 0.2, y: 0.1 }],
    banks: kind === "double-bank" ? 2 : kind === "bank" || kind === "rail-combo" ? 1 : 0,
    ...overrides,
  };
};

const stat = (
  kind: CandidateKind,
  o: { strength: number; potsTarget: boolean; styleScore?: number; verified?: boolean },
): CandidateStat => ({
  candidate: makeCandidate(kind),
  visits: 1,
  value: o.strength,
  strength: o.strength,
  rails: 0,
  potsTarget: o.potsTarget,
  styleScore: o.styleScore ?? 0,
  verified: o.verified ?? true,
});

/** A verification record consistent with a stat that did NOT pot. */
const verification = (
  index: number,
  o: Partial<CandidateVerification> = {},
): CandidateVerification => ({
  index,
  firstContact: 1,
  legalFirstContact: true,
  scratched: false,
  legalPot: false,
  pocketed: [],
  railsBeforePot: 0,
  events: [],
  ...o,
});

/**
 * The event log a rollout of `kind` produces when the trick comes off.
 *
 * A fixture that says `potsTarget: true` for a bank is claiming the physics
 * potted the ball off a cushion, and the policy now reads that claim from the
 * event log rather than from the boolean. Supplying the log is what makes the
 * fixture self-consistent; a fixture that wants to say "it potted, but not as
 * the trick it was generated as" supplies its own log instead (see D).
 */
const trickEvents = (kind: CandidateKind): ShotEvent[] => {
  const hit = (t: number, a: number, b: number): ShotEvent => ({
    time: t,
    kind: "ball-ball",
    balls: [a, b],
  });
  const rail = (t: number, b: number, cushion: string): ShotEvent => ({
    time: t,
    kind: "ball-cushion",
    balls: [b],
    cushion,
  });
  const pot = (t: number, b: number): ShotEvent => ({
    time: t,
    kind: "pocket",
    balls: [b],
    pocket: "tr",
  });
  switch (kind) {
    case "bank":
      return [hit(0.1, CUE_ID, 1), rail(0.3, 1, "top"), pot(0.6, 1)];
    case "double-bank":
      return [hit(0.1, CUE_ID, 1), rail(0.3, 1, "top"), rail(0.45, 1, "right"), pot(0.7, 1)];
    case "combo":
      return [hit(0.1, CUE_ID, 1), hit(0.25, 1, 2), pot(0.6, 2)];
    case "rail-combo":
      return [hit(0.1, CUE_ID, 1), hit(0.25, 1, 2), rail(0.4, 2, "left"), pot(0.7, 2)];
    default:
      return [hit(0.1, CUE_ID, 1), pot(0.5, 1)];
  }
};

/** A verification whose event log really executes `kind` and pots. */
const potting = (index: number, kind: CandidateKind, potId: number): CandidateVerification => ({
  index,
  firstContact: 1,
  legalFirstContact: true,
  scratched: false,
  legalPot: true,
  pocketed: [potId],
  railsBeforePot: kind === "double-bank" ? 2 : kind === "bank" || kind === "rail-combo" ? 1 : 0,
  events: trickEvents(kind),
});

/** A board with a real legal target, so the safety rung has something to aim at. */
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

const ctxFor = (balls: Ball[], targets: number[], simulate = simulateShotWasm): TrickOnlyContext => ({
  state: asState(balls),
  table,
  targets,
  simulate,
});

/** A simulator that must never be called. Proves a rung returned before physics. */
const forbiddenSimulator = (): SimResult => {
  throw new Error("the safety rung must not be reached on this fixture");
};

/**
 * Select over `stats`. Any trick stat marked `potsTarget` that the caller did
 * not give a verification for gets the canonical `potting` log for its kind, so
 * "this fixture's bank potted off a cushion" and "the event log says so" are
 * the same statement. Callers testing the measured gate supply their own.
 */
const pick = (
  stats: CandidateStat[],
  verifications: (CandidateVerification | null | undefined)[] = [],
) =>
  selectTrickOnly(
    stats,
    stats.map((s, i) => {
      if (verifications[i] !== undefined && verifications[i] !== null) return verifications[i]!;
      if (verifications[i] === null) return null;
      if (!isTrickCandidate(s.candidate) || !s.potsTarget) return null;
      return potting(i, s.candidate.kind, s.candidate.potId);
    }),
    ctxFor(SAFETY_BOARD, [1], forbiddenSimulator),
  );

// ---------------------------------------------------------------------------
// A. A high-quality direct is available and must still never be selected
// ---------------------------------------------------------------------------

describe("A. a direct is available and better, and still cannot be selected", () => {
  it("A1: a 0.99 potting direct loses to a barely-qualifying bank (rung 1)", () => {
    const direct = stat("direct", { strength: 0.99, potsTarget: true });
    const bank = stat("bank", { strength: TRICK_RELIABILITY_THRESHOLD + 0.01, potsTarget: true });
    const d = pick([direct, bank]);
    expect(d.shot?.kind).toBe("bank");
    expect(d.rung).toBe("trick-qualified");
    expect(d.shot?.candidateIndex).toBe(1);
    expect(d.excludedIndices).toEqual([0]);
  });

  it("A2: a 0.99 potting direct loses to a BAD 0.10 bank (rung 2)", () => {
    // This is the case the previous policy resolved the other way: no trick
    // cleared the reliability bar, so it played the direct. A sub-threshold
    // trick that still pots is now preferred to a certain direct pot.
    const direct = stat("direct", { strength: 0.99, potsTarget: true });
    const bank = stat("bank", { strength: 0.1, potsTarget: true });
    const d = pick([direct, bank]);
    expect(d.shot?.kind).toBe("bank");
    expect(d.rung).toBe("trick-below-threshold");
    expect(d.qualifyingTricks).toBe(0);
  });

  it("A3: a 0.99 potting direct loses to a safety when the only bank does NOT pot", () => {
    // There is no rung for "a trick that misses". A trick that does not pot in
    // simulation cannot be shown to be a trick at all, so the ladder drops to
    // the safety rung rather than playing the miss and calling it a bank. The
    // direct still loses, which is the property this fixture is here for.
    const direct = stat("direct", { strength: 0.99, potsTarget: true });
    const bank = stat("bank", { strength: 0.9, potsTarget: false });
    const d = selectTrickOnly(
      [direct, bank],
      [null, verification(1)],
      ctxFor(SAFETY_BOARD, [1]),
    );
    expect(d.shot?.kind).toBe("safety-kick");
    expect(d.rung === "non-direct-safety" || d.rung === "forced-legal-contact").toBe(true);
  });

  it("A4: directs only, all potting at 0.99 — a safety kick is played instead", () => {
    const stats = [
      stat("direct", { strength: 0.99, potsTarget: true }),
      stat("direct", { strength: 0.99, potsTarget: true }),
      stat("direct", { strength: 0.99, potsTarget: true }),
    ];
    const d = selectTrickOnly(stats, [null, null, null], ctxFor(SAFETY_BOARD, [1]));
    expect(d.shot).not.toBeNull();
    expect(d.shot!.kind).toBe("safety-kick");
    expect(d.rung === "non-direct-safety" || d.rung === "forced-legal-contact").toBe(true);
    expect(d.excludedIndices).toEqual([0, 1, 2]);
    // The shot played is not any of the three directs.
    expect(d.shot!.candidateIndex).toBeNull();
  });

  it("A5: fuzz — 10,000 random stat lists never select a direct and never throw", () => {
    // A cheap deterministic PRNG; the point is coverage of the partition over
    // arbitrary kind/strength/potsTarget/verified combinations, not physics.
    // Not the pre-registered evaluation seed (61903477); that value stays
    // reserved for the one evaluation run so it cannot be mistaken for a
    // fixture this suite has already seen.
    let s = 7788991 >>> 0;
    const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    const kinds: CandidateKind[] = ["direct", "bank", "double-bank", "combo", "rail-combo"];
    // A stub simulator so the safety rung is exercised without 10,000 real
    // shots. Its result fouls, which drives rung 5 — also worth covering.
    const stub = (): SimResult => ({
      balls: [],
      events: [],
      pocketed: [],
      firstContact: null,
      duration: 0,
    });
    const ctx = ctxFor(SAFETY_BOARD, [1], stub);

    for (let i = 0; i < 10_000; i++) {
      const n = 1 + Math.floor(rand() * 8);
      const stats: CandidateStat[] = [];
      const vs: (CandidateVerification | null)[] = [];
      for (let k = 0; k < n; k++) {
        const kind = kinds[Math.floor(rand() * kinds.length)];
        stats.push(stat(kind, { strength: rand(), potsTarget: rand() < 0.4 }));
        vs.push(rand() < 0.5 ? verification(k, { scratched: rand() < 0.3 }) : null);
      }
      const d = selectTrickOnly(stats, vs, ctx);
      // The only assertion that matters, on every one of them.
      expect(d.shot?.kind).not.toBe("direct");
      // And a shot is always produced: there is a legal target on this board.
      expect(d.shot).not.toBeNull();
      if (d.shot!.candidateIndex !== null) {
        expect(isTrickCandidate(stats[d.shot!.candidateIndex].candidate)).toBe(true);
      }
    }
  });

  it("A6: real physics — a genuinely makeable direct pot exists and is never chosen", async () => {
    // Boards where the real simulator confirms at least one `direct` candidate
    // legally pots. "Clearly available" is asserted against the oracle, not
    // assumed from the geometry.
    const boards: Record<string, { balls: Ball[]; targets: number[] }> = {
      openMid: { balls: [makeBall(CUE_ID, -0.2, 0.24), makeBall(1, 0.4, 0.26)], targets: [1] },
      cornerCut: { balls: [makeBall(CUE_ID, 0.0, 0.0), makeBall(1, 0.55, 0.22)], targets: [1] },
      lowRail: { balls: [makeBall(CUE_ID, 0.3, -0.25), makeBall(2, 0.62, -0.28)], targets: [2] },
    };

    let boardsWithPottingDirect = 0;
    for (const [name, { balls, targets } ] of Object.entries(boards)) {
      const candidates = generateCandidates(balls, table, targets);
      const directs = candidates.filter((c) => c.kind === "direct");
      expect(directs.length, `${name} must generate a direct`).toBeGreaterThan(0);

      // Oracle: run the REAL simulator on each direct, unbudgeted.
      const potting = directs.filter((c) => {
        const sim = simulateShotWasm(balls.map((b) => ({ ...b, pos: { ...b.pos }, vel: { ...b.vel } })), c.action);
        return !sim.pocketed.includes(CUE_ID) && isLegalPot(sim, c);
      });
      if (potting.length === 0) continue;
      boardsWithPottingDirect++;

      const result = await classicalTrickOnlyBrain().plan(asState(balls), table, 0, {
        ...defaultConfig,
        seed: 20260805,
        seedTimeoutMs: Infinity,
        searchTimeoutMs: Infinity,
      });
      expect(result.shot, `${name} must produce a shot`).not.toBeNull();
      expect(result.shot!.kind, `${name} chose a direct`).not.toBe("direct");

      // The direct is present in the published trace, labelled truthfully.
      const traced = result.decision.candidates.filter((c) => c.kind === "direct");
      expect(traced.length).toBe(directs.length);
      for (const c of traced) {
        expect(c.eligible).toBe(false);
        expect(c.rejection).toBe("direct-excluded-by-policy");
      }
      // And the executed shot is a real, rules-resolved outcome.
      const report = takeShot(asState(balls), table, result.shot!.action, simulateShotWasm);
      expect(typeof report.outcome.foul).toBe("boolean");
    }
    expect(
      boardsWithPottingDirect,
      "the fixture set must actually contain a makeable direct pot",
    ).toBeGreaterThan(0);
  }, 60_000);

  it("A6b: the previous opponent DOES play the direct here — and trick-only pays for refusing", () => {
    // Without this converse, A6 could pass simply because no fixture ever
    // offered a competitive direct.
    //
    // This board is not hand-drawn. It was found by scanning the evaluation's
    // own fixture distribution (`makeFixtures`, seed 909090 — deliberately not
    // the pre-registered evaluation seed) for a state where the legacy mixed
    // policy's `no-trick-qualified` branch fires. It is rare: 1 in 136
    // fixtures, consistent with the corrected gate's measured ~3.25%
    // direct-selection rate.
    //
    // On this exact board the previous opponent plays a physics-VERIFIED direct
    // pot at strength 0.63. Trick-only, given the identical candidate list,
    // plays a bank that did not pot in simulation (rung 3). That is the cost of
    // the product decision, stated as an assertion rather than a footnote.
    const balls: Ball[] = [
      makeBall(CUE_ID, -0.517415, 0.039596),
      makeBall(2, -0.637608, -0.409759),
      makeBall(3, -0.425598, -0.098914),
      makeBall(5, -0.785301, -0.080932),
      makeBall(6, -0.118187, -0.083722),
      makeBall(7, -0.362688, 0.416287),
      makeBall(8, 0.780843, 0.140419),
      makeBall(9, 0.409653, 0.298613),
      makeBall(10, -0.666774, -0.241562),
      makeBall(11, -0.486990, -0.282612),
      makeBall(12, -0.292130, 0.421391),
      makeBall(14, 0.099614, 0.379636),
      makeBall(15, 0.620042, 0.362930),
    ];
    const targets = [2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15];
    const cfg = {
      ...defaultConfig,
      seed: 20260101,
      seedTimeoutMs: Infinity,
      searchTimeoutMs: Infinity,
    };
    const candidates = generateCandidates(balls, table, targets);

    const legacy = searchWithLegacySelection(candidates, balls, targets, cfg);
    expect(legacy.trace!.selectionReason).toBe("no-trick-qualified");
    expect(legacy.best!.candidate.kind).toBe("direct");
    expect(legacy.best!.potsTarget, "the direct must be a real, verified pot").toBe(true);
    expect(legacy.best!.strength).toBeGreaterThan(0.5);

    const outcome = searchCandidates(candidates, balls, targets, cfg);
    const d = selectTrickOnly(outcome.allStats, outcome.verifications, ctxFor(balls, targets));
    expect(d.shot).not.toBeNull();
    expect(d.shot!.kind).not.toBe("direct");
    // The refusal is not free, and the shot it leaves is still legal.
    const report = takeShot(asState(balls), table, d.shot!.action, simulateShotWasm);
    expect(report.outcome.foul).toBe(false);
  }, 120_000);

  it("A7: an adversarial prior scoring every direct 0.99, WITH the eligibility filter off", () => {
    // The `eligible` predicate is a budget optimisation, not the guarantee.
    // Here it is deliberately disabled, so directs dominate the physics-verified
    // set and are ranked first — and the selection filter alone still holds.
    const balls: Ball[] = [
      makeBall(CUE_ID, -0.6, -0.1),
      makeBall(1, 0.3, 0.02),
      makeBall(2, 0.38, 0.1),
      makeBall(4, 0.1, -0.25),
      makeBall(5, -0.2, 0.3),
    ];
    const targets = [1, 2, 4, 5];
    const candidates = generateCandidates(balls, table, targets);
    const scores = candidates.map((c) => (c.kind === "direct" ? 0.99 : 0.01));

    const outcome = searchCandidates(candidates, balls, targets, {
      ...defaultConfig,
      seed: 20260805,
      seedTimeoutMs: Infinity,
      searchTimeoutMs: Infinity,
      // NOTE: no `eligible` — directs get physics and are visited first.
      prior: { scores, keepTop: 16, source: "adversarial-directs-first" },
    });
    const verifiedDirects = outcome.trace!.verifiedIndices.filter(
      (i) => candidates[i].kind === "direct",
    );
    expect(verifiedDirects.length, "the prior must really put directs into physics").toBeGreaterThan(0);

    const d = selectTrickOnly(outcome.allStats, outcome.verifications, ctxFor(balls, targets));
    expect(d.shot).not.toBeNull();
    expect(d.shot!.kind).not.toBe("direct");
  }, 60_000);

  it("A8: the type system rejects a direct and forbids forging a PlayableShot", () => {
    const directStat: CandidateStat & { candidate: Candidate & { kind: "direct" } } = {
      ...stat("direct", { strength: 0.99, potsTarget: true }),
      candidate: makeCandidate("direct") as Candidate & { kind: "direct" },
    };
    // @ts-expect-error a direct CandidateStat is not assignable to TrickStat
    const notATrick: TrickStat = directStat;
    expect(notATrick).toBeDefined();

    // @ts-expect-error PlayableShot's brand is an unexported unique symbol
    const forged: PlayableShot = {
      action: { phi: 0, power: 0.5, sideSpin: 0, topSpin: 0 },
      kind: "bank",
      candidateIndex: 0,
      rung: "trick-qualified",
      cuePath: [],
      path: [],
    };
    expect(forged).toBeDefined();
  });

  // A9 did not exist. A1-A8 and A10 were written and the number was skipped, so
  // the docs' "A1-A10" pointed at nine tests and one gap. This is the gap
  // filled with the property the other nine assume rather than check: the
  // ladder is STRICTLY ORDERED. A1-A3 each pit one rung against a direct; none
  // of them pits a rung against a weaker rung, which is what makes "four rungs,
  // strictly ordered" a claim about the code rather than about the comment
  // above it.
  it("A9: a lower rung never fires while a higher one has a member", () => {
    // Both trick rungs are populated at once, and deliberately inverted: the
    // weakest shot sits on the highest rung. A ladder that ranked by quality
    // instead of by rung would pick the 0.95 miss every time.
    const qualified = stat("bank", { strength: TRICK_RELIABILITY_THRESHOLD + 0.01, potsTarget: true });
    const belowBar = stat("combo", { strength: TRICK_RELIABILITY_THRESHOLD - 0.3, potsTarget: true });
    const miss = stat("double-bank", { strength: 0.95, potsTarget: false });

    const all = pick([qualified, belowBar, miss], [undefined, undefined, verification(2)]);
    expect(all.rung).toBe("trick-qualified");
    expect(all.shot!.candidateIndex).toBe(0);

    // Remove the top rung; the next one down must take over, not the strongest
    // remaining shot overall.
    const noQualified = pick(
      [stat("combo", { strength: 0.2, potsTarget: true }), stat("double-bank", { strength: 0.95, potsTarget: false })],
      [undefined, verification(1)],
    );
    expect(noQualified.rung).toBe("trick-below-threshold");
    expect(noQualified.shot!.candidateIndex, "a 0.20 POT outranks a 0.95 miss").toBe(0);

    // Remove rung 2 — nothing pots — and the ladder falls to the safety rung.
    // A trick that misses is not a rung, because a miss cannot be shown to be
    // a trick. `pick` supplies a simulator that throws if reached, so these
    // two run the real one.
    const onlyMisses = selectTrickOnly(
      [stat("bank", { strength: 0.3, potsTarget: false }), stat("double-bank", { strength: 0.7, potsTarget: false })],
      [verification(0), verification(1)],
      ctxFor(SAFETY_BOARD, [1]),
    );
    expect(onlyMisses.rung === "non-direct-safety" || onlyMisses.rung === "forced-legal-contact").toBe(true);
    expect(onlyMisses.shot!.kind).toBe("safety-kick");

    const scratchedOnly = selectTrickOnly(
      [stat("bank", { strength: 0.9, potsTarget: false })],
      [verification(0, { scratched: true })],
      ctxFor(SAFETY_BOARD, [1]),
    );
    expect(scratchedOnly.rung === "non-direct-safety" || scratchedOnly.rung === "forced-legal-contact").toBe(true);
    expect(scratchedOnly.shot!.kind).toBe("safety-kick");

    // And the rung the decision reports is the rung stamped on the shot — the
    // panel reads one and the felt reads the other.
    for (const d of [all, noQualified, onlyMisses, scratchedOnly]) {
      expect(d.shot!.rung).toBe(d.rung);
    }
  }, 30_000);

  it("A10: the runtime invariant throws, and the degrade path is a safety, not a direct", () => {
    const forgedDirect = {
      action: { phi: 0, power: 0.4, sideSpin: 0, topSpin: 0 },
      kind: "direct",
      candidateIndex: 0,
      rung: "trick-qualified",
      cuePath: [],
      path: [],
    } as unknown as PlayableShot;
    expect(() => assertNotDirect(forgedDirect)).toThrow(TrickOnlyInvariantError);

    // The brain's catch clause calls `selectTrickOnly` with an empty candidate
    // list, which is the safety rung. That degrade must produce a legal,
    // non-direct shot rather than nothing.
    const degraded = selectTrickOnly([], [], ctxFor(SAFETY_BOARD, [1]));
    expect(degraded.shot).not.toBeNull();
    expect(degraded.shot!.kind).toBe("safety-kick");
  }, 30_000);
});
