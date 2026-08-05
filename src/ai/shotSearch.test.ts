import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { makeBall } from "../physics/ball";
import { makeTable } from "../physics/table";
import { CUE_ID } from "../game/rack";
import { initPhysics } from "../physics/wasm-bridge";
import { type Candidate, type CandidateKind, generateCandidates } from "./candidates";
import {
  type CandidateStat,
  TRICK_RELIABILITY_THRESHOLD,
  defaultConfig,
  isLegalPot,
  searchBaseline,
  selectBest,
} from "./shotSearch";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Deterministic fixtures for the constrained selection rule (Phase 2B):
// a trick shot (anything but "direct") is preferred whenever at least one
// qualifies (pots its target AND clears TRICK_RELIABILITY_THRESHOLD); a
// direct shot is only chosen when no trick qualifies. These are unit-level
// fixtures over hand-built CandidateStat objects — deterministic and fast,
// exactly targeting the selection rule's contract rather than depending on
// real physics to organically produce each geometry. Real-physics coverage
// of the underlying candidate generator lives in candidates.test.ts and
// search-quality.test.ts; the scratch/legality fixtures further below in
// this file exercise the real search+physics pipeline directly.
// ---------------------------------------------------------------------------

let nextId = 100;

function makeCandidate(kind: CandidateKind, overrides: Partial<Candidate> = {}): Candidate {
  const id = nextId++;
  return {
    kind,
    target: 1,
    potId: kind === "combo" || kind === "rail-combo" ? 2 : 1,
    pocket: "tr",
    aimPoint: { x: 0, y: 0 },
    action: { phi: 0, power: 0.5, sideSpin: 0, topSpin: 0 },
    path: [{ x: 0, y: 0 }, { x: id, y: 0 }],
    banks: kind === "double-bank" ? 2 : kind === "bank" || kind === "rail-combo" ? 1 : 0,
    ...overrides,
  };
}

function stat(kind: CandidateKind, opts: { strength: number; potsTarget: boolean; styleScore?: number }): CandidateStat {
  return {
    candidate: makeCandidate(kind),
    visits: 1,
    value: opts.strength, // not exercised by selectBest directly; kept consistent
    strength: opts.strength,
    rails: 0,
    potsTarget: opts.potsTarget,
    styleScore: opts.styleScore ?? 0,
    verified: true,
  };
}

describe("selectBest: constrained trick-preference selection rule", () => {
  it("fixture: direct pot available plus a viable (qualifying) trick — trick wins", () => {
    const direct = stat("direct", { strength: 0.95, potsTarget: true }); // easier, higher raw strength
    const bank = stat("bank", { strength: TRICK_RELIABILITY_THRESHOLD + 0.05, potsTarget: true }); // qualifies, lower strength
    const best = selectBest([direct, bank]);
    // The trick wins even though the direct pot has materially higher raw
    // strength — this is the whole point of the rule, not a coincidence.
    expect(best).toBe(bank);
  });

  it("fixture: single-cushion bank qualifies and is chosen over a non-qualifying direct pot", () => {
    const direct = stat("direct", { strength: 0.99, potsTarget: true });
    const bank = stat("bank", { strength: TRICK_RELIABILITY_THRESHOLD, potsTarget: true });
    expect(selectBest([direct, bank])).toBe(bank);
  });

  it("fixture: double bank qualifies and is chosen over direct", () => {
    const direct = stat("direct", { strength: 0.9, potsTarget: true });
    const doubleBank = stat("double-bank", { strength: TRICK_RELIABILITY_THRESHOLD + 0.1, potsTarget: true });
    expect(selectBest([direct, doubleBank])).toBe(doubleBank);
  });

  it("fixture: combo qualifies and is chosen over direct", () => {
    const direct = stat("direct", { strength: 0.9, potsTarget: true });
    const combo = stat("combo", { strength: TRICK_RELIABILITY_THRESHOLD + 0.1, potsTarget: true });
    expect(selectBest([direct, combo])).toBe(combo);
  });

  it("fixture: rail-plus-combination qualifies and is chosen over direct", () => {
    const direct = stat("direct", { strength: 0.9, potsTarget: true });
    const railCombo = stat("rail-combo", { strength: TRICK_RELIABILITY_THRESHOLD + 0.1, potsTarget: true });
    expect(selectBest([direct, railCombo])).toBe(railCombo);
  });

  it("fixture: no viable trick shot — every trick candidate is below threshold or doesn't pot, so direct wins", () => {
    const direct = stat("direct", { strength: 0.8, potsTarget: true });
    const weakBank = stat("bank", { strength: TRICK_RELIABILITY_THRESHOLD - 0.01, potsTarget: true }); // just under the bar
    const missedCombo = stat("combo", { strength: 0.9, potsTarget: false }); // high strength but doesn't actually pot
    const best = selectBest([direct, weakBank, missedCombo]);
    expect(best).toBe(direct);
  });

  it("among multiple qualifying tricks, maximizes strength + style utility, not just strength", () => {
    const higherStrengthLowerStyle = stat("bank", { strength: 0.8, potsTarget: true, styleScore: 0 });
    const lowerStrengthHigherStyle = stat("combo", { strength: 0.7, potsTarget: true, styleScore: 3 });
    // utility = strength + 0.12*styleScore: 0.8+0=0.8 vs 0.7+0.36=1.06 — the
    // styled candidate should win despite lower raw strength.
    expect(selectBest([higherStrengthLowerStyle, lowerStrengthHigherStyle])).toBe(lowerStrengthHigherStyle);
  });

  it("rules make every trick candidate impossible (empty trick set) — falls back to any potting candidate", () => {
    const direct = stat("direct", { strength: 0.6, potsTarget: true });
    expect(selectBest([direct])).toBe(direct);
  });

  it("returns null for an empty candidate list", () => {
    expect(selectBest([])).toBeNull();
  });

  it("falls back to the first candidate when nothing pots (e.g. all fouled/missed)", () => {
    const a = stat("direct", { strength: 0.4, potsTarget: false });
    const b = stat("bank", { strength: 0.3, potsTarget: false });
    expect(selectBest([a, b])).toBe(a);
  });
});

describe("isLegalPot: first-contact and pot-target legality", () => {
  it("fixture: illegal first contact — cue struck the wrong ball, even though the intended pot ball dropped", () => {
    const candidate = makeCandidate("direct", { target: 1, potId: 1 });
    // Simulated result shows the cue's first contact was ball 5, not the
    // candidate's intended ball 1 — an obstruction/deflection the candidate
    // generator's geometry check didn't catch. Ball 1 happening to still
    // drop does not make this a legal pot.
    const sim = { firstContact: 5, pocketed: [1], events: [] };
    expect(isLegalPot(sim, candidate)).toBe(false);
  });

  it("legal direct pot: first contact matches target, target is pocketed", () => {
    const candidate = makeCandidate("direct", { target: 1, potId: 1 });
    const sim = { firstContact: 1, pocketed: [1], events: [] };
    expect(isLegalPot(sim, candidate)).toBe(true);
  });

  it("legal combo pot: first contact is the struck ball (target), the POCKETED ball is the driven intermediate (potId), and events confirm target actually struck potId", () => {
    // Regression fixture for the real bug this phase fixed: candidates.ts's
    // combo/rail-combo candidates set `target` to the first-contact ball and
    // `potId` to the ball that's actually driven into the pocket — checking
    // `pocketed.includes(target)` here would wrongly return false even
    // though the combo worked exactly as intended.
    const candidate = makeCandidate("combo", { target: 1, potId: 7 });
    const sim = {
      firstContact: 1,
      pocketed: [7],
      events: [
        { time: 0.1, kind: "ball-ball" as const, balls: [0, 1] }, // cue hits target
        { time: 0.2, kind: "ball-ball" as const, balls: [1, 7] }, // target drives potId
        { time: 0.4, kind: "pocket" as const, balls: [7], pocket: "tr" },
      ],
    };
    expect(isLegalPot(sim, candidate)).toBe(true);
  });

  it("illegal: combo's first-struck ball happens to fall too, but the intended potId ball never dropped", () => {
    const candidate = makeCandidate("combo", { target: 1, potId: 7 });
    const sim = { firstContact: 1, pocketed: [1], events: [] }; // target fell, not potId
    expect(isLegalPot(sim, candidate)).toBe(false);
  });

  it("illegal (endpoint-only labeling gap, fixed): first contact and final pocketed set match a combo's endpoint, but target never actually struck potId — potId was pocketed by an unrelated contact chain in the same shot", () => {
    // Same (firstContact, pocketed) endpoint as the legal case above, but the
    // event trace shows potId was pocketed WITHOUT ever colliding with
    // target — e.g. it was drifting toward a pocket already and something
    // else (or nothing) nudged it in during the same shot. Endpoint-only
    // checking (the pre-Stage-A-closure behavior) could not tell this apart
    // from a genuine combo; this is exactly the case that regressed it.
    const candidate = makeCandidate("combo", { target: 1, potId: 7 });
    const sim = {
      firstContact: 1,
      pocketed: [7],
      events: [
        { time: 0.1, kind: "ball-ball" as const, balls: [0, 1] }, // cue hits target
        { time: 0.15, kind: "ball-cushion" as const, balls: [1], cushion: "top" }, // target bounces away, never reaches potId
        { time: 0.3, kind: "pocket" as const, balls: [7], pocket: "tr" }, // potId drops on its own / via something else
      ],
    };
    expect(isLegalPot(sim, candidate)).toBe(false);
  });

  it("direct/bank/double-bank never need the contact-chain check (single object ball throughout, potId === target) — legal regardless of intermediate cushion count", () => {
    const candidate = makeCandidate("double-bank", { target: 3, potId: 3 });
    const sim = {
      firstContact: 3,
      pocketed: [3],
      events: [
        { time: 0.1, kind: "ball-ball" as const, balls: [0, 3] },
        { time: 0.2, kind: "ball-cushion" as const, balls: [3], cushion: "left" },
        { time: 0.3, kind: "ball-cushion" as const, balls: [3], cushion: "top" },
        { time: 0.5, kind: "pocket" as const, balls: [3], pocket: "tl" },
      ],
    };
    expect(isLegalPot(sim, candidate)).toBe(true);
  });

  it("illegal: nothing pocketed at all", () => {
    const candidate = makeCandidate("direct", { target: 1, potId: 1 });
    const sim = { firstContact: 1, pocketed: [], events: [] };
    expect(isLegalPot(sim, candidate)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Real search + real physics fixtures. These require the WASM physics
// engine initialized (Node-compatible path, same artifact the browser ships
// — see physics/wasm-bridge.ts).
// ---------------------------------------------------------------------------

const table = makeTable();

beforeAll(async () => {
  const wasmPath = join(__dirname, "../wasm/showboat_physics_bg.wasm");
  await initPhysics(readFileSync(wasmPath));
});

describe("fixture: scratch-prone candidate is rejected", () => {
  it("a candidate whose real simulated shot scratches the cue ball never appears in results", () => {
    // Empirically found geometry (corner "tr", cue trailing the object ball
    // by 0.08 at a 0.03 offset): the cue follows the object ball straight
    // into the pocket at every power level from 0.35 to 0.95 — a genuine,
    // reliable scratch produced by the real physics engine, not simulated
    // by hand.
    const pk = table.pockets.find((p) => p.id === "tr")!;
    const obj = makeBall(1, pk.center.x - 0.03, pk.center.y - 0.03);
    const cue = makeBall(CUE_ID, obj.pos.x - 0.08, obj.pos.y - 0.08);
    const balls = [cue, obj];
    const candidates = generateCandidates(balls, table, [1]);
    const direct = candidates.find((c) => c.kind === "direct");
    expect(direct).toBeDefined();

    const result = searchBaseline(balls, table, [1], defaultConfig);
    // The scratching candidate must not survive seeding (visits stay 0 and
    // it's filtered out of `stats`), and must never be chosen as best.
    const scratcher = result.stats.find(
      (s) => s.candidate.kind === "direct" && s.candidate.pocket === "tr",
    );
    expect(scratcher).toBeUndefined();
    if (result.best) {
      expect(result.best.candidate.pocket === "tr" && result.best.candidate.kind === "direct").toBe(false);
    }
  });
});

describe("fixture: strict simulation budget is never exceeded", () => {
  const pk = table.pockets.find((p) => p.id === "tr")!;
  const balls = [
    makeBall(CUE_ID, -0.3, 0.05),
    makeBall(1, 0.25, 0.15),
    makeBall(2, pk.center.x - 0.2, pk.center.y - 0.15),
    makeBall(3, -0.05, -0.28),
    makeBall(9, 0.5, -0.2),
  ];
  const targets = [1, 2, 3, 9];

  it.each([0, 1, 2, 8, 30, 60, 200])(
    "simulations budget %i is never exceeded regardless of seeding/refinement path",
    (simulations) => {
      const result = searchBaseline(balls, table, targets, { ...defaultConfig, simulations });
      expect(result.simulations).toBeLessThanOrEqual(simulations);
    },
  );

  it("net-seeded path (per-candidate scores) also respects the budget", () => {
    const candidates = generateCandidates(balls, table, targets);
    const netSeedScores = candidates.map((_, i) => (i % 2 === 0 ? 0.8 : 0.2));
    const result = searchBaseline(balls, table, targets, {
      ...defaultConfig,
      simulations: 10,
      netSeedScores,
    });
    expect(result.simulations).toBeLessThanOrEqual(10);
  });

  it("no-model fallback still spends close to the full budget when candidates are plentiful", () => {
    const result = searchBaseline(balls, table, targets, { ...defaultConfig, simulations: 60 });
    expect(result.simulations).toBeGreaterThan(0);
    expect(result.simulations).toBeLessThanOrEqual(60);
  });
});
