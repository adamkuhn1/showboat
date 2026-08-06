// The guarantee, checked where the game actually reaches it.
//
// `policy/trickOnly.test.ts` A1-A10 prove that `selectTrickOnly` cannot return
// a direct. That is a proof about a function. What the *game* runs is
// `planTurnTraced` — the single function the planning worker and the inline
// fallback both call — and between the policy and the felt there used to be two
// ways round it:
//
//   1. `App.tsx:379-416`, a nearest-legal-ball aim at power 0.3 fired whenever
//      the search returned nothing. Deleted by T2.
//   2. `lastResortAim` in this directory, the same function reproduced verbatim
//      while the two teams worked in parallel. Deleted by this integration.
//
// So these tests run the live path end to end on T2's hardest board — the A6b
// fixture, where the previous mixed policy plays a physics-verified direct pot
// — and assert on what comes back out of it. If a third escape hatch is ever
// added anywhere between `brain.plan()` and `takeShot`, this fails.
//
// The companion source assertions are in `ai/policy/trickOnlySourceGuard.test.ts`.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../../physics/table";
import { makeBall, type Ball } from "../../physics/ball";
import { CUE_ID, SOLIDS, STRIPES } from "../../game/rack";
import type { GameState } from "../../game/state";
import { initPhysics, simulateShotWasm } from "../../physics/wasm-bridge";
import { generateCandidates } from "../../ai/candidates";
import { defaultConfig, isLegalPot, searchWithLegacySelection } from "../../ai/shotSearch";
import { TRICK_KINDS } from "../../ai/trace/contract";
import { planTurnTraced } from "./plan";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../../..");
const WASM = readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm"));
const TURN_SRC = readFileSync(join(__dirname, "../useAiTurn.ts"), "utf8");
const table = makeTable();

const asState = (balls: Ball[], over: Partial<GameState> = {}): GameState => ({
  balls,
  turn: 1,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 4,
  ...over,
});

/**
 * A6b's board, verbatim from `ai/policy/trickOnly.test.ts`. Found by scanning
 * the evaluation's own fixture distribution for a state where the legacy mixed
 * policy's `no-trick-qualified` branch fires: on this exact board the previous
 * opponent plays a physics-verified direct pot at strength 0.63.
 */
const A6B_BALLS: Ball[] = [
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
const A6B_TARGETS = [2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15];

const PLAYABLE_KINDS: string[] = [...TRICK_KINDS, "safety-kick"];

beforeAll(async () => {
  await initPhysics(WASM);
}, 60_000);

describe("the live planning path refuses a direct that is available and better", () => {
  it("A6b through `planTurnTraced`: the previous policy pots the direct, this one will not", async () => {
    // Premise, re-established here rather than cited: the legacy mixed policy
    // — still the evaluation baseline and the self-play generator — selects a
    // real, physics-verified direct pot on this board.
    const candidates = generateCandidates(A6B_BALLS, table, A6B_TARGETS);
    const legacy = searchWithLegacySelection(candidates, A6B_BALLS, A6B_TARGETS, {
      ...defaultConfig,
      seed: 20260101,
      seedTimeoutMs: Infinity,
      searchTimeoutMs: Infinity,
    });
    expect(legacy.best!.candidate.kind, "the premise requires a direct here").toBe("direct");
    expect(legacy.best!.potsTarget, "and it must be a real, verified pot").toBe(true);
    expect(legacy.best!.strength).toBeGreaterThan(0.5);

    // The live path. No config argument, no injected policy: exactly what the
    // worker calls, with production's own search budget and seed timeout.
    const state = asState(A6B_BALLS);
    const planned = await planTurnTraced({
      state,
      table,
      player: 1,
      useNeural: false,
      wasmSource: WASM,
    });

    expect(planned.kind).toBe("shot");
    if (planned.kind !== "shot") return;

    // What reached the table.
    const selected = planned.trace.selected!;
    expect(selected.kind).not.toBe("direct");
    expect(PLAYABLE_KINDS).toContain(selected.kind);
    // The executed action IS the selected shot's action. This is what makes the
    // trace evidence about the shot rather than commentary beside it: there is
    // no second action anywhere in the path for the report to have come from.
    expect(planned.action).toEqual(selected.action);
    expect(planned.report.sim.duration).toBeGreaterThan(0);

    // The directs are still there, as truthfully-labelled rejected comparisons.
    const directs = planned.trace.candidates.filter((c) => c.kind === "direct");
    expect(directs.length).toBeGreaterThan(0);
    for (const d of directs) {
      expect(d.eligible).toBe(false);
      expect(d.rejection).toBe("direct-excluded-by-policy");
    }
    expect(planned.trace.policy).toBe("trick-only");
  }, 180_000);

  it("boards with a genuinely makeable direct pot still come back non-direct", async () => {
    // Three open boards, each with a direct the REAL simulator confirms pots.
    // Checked against the oracle so the test cannot pass by never being offered
    // one.
    const boards: Record<string, { balls: Ball[]; targets: number[] }> = {
      openMid: { balls: [makeBall(CUE_ID, -0.2, 0.24), makeBall(1, 0.4, 0.26)], targets: [1] },
      cornerCut: { balls: [makeBall(CUE_ID, 0.0, 0.0), makeBall(1, 0.55, 0.22)], targets: [1] },
      lowRail: { balls: [makeBall(CUE_ID, 0.3, -0.25), makeBall(2, 0.62, -0.28)], targets: [2] },
    };

    let withPottingDirect = 0;
    for (const [name, { balls, targets }] of Object.entries(boards)) {
      const potting = generateCandidates(balls, table, targets)
        .filter((c) => c.kind === "direct")
        .filter((c) => {
          const sim = simulateShotWasm(
            balls.map((b) => ({ ...b, pos: { ...b.pos }, vel: { ...b.vel } })),
            c.action,
          );
          return !sim.pocketed.includes(CUE_ID) && isLegalPot(sim, c);
        });
      if (potting.length === 0) continue;
      withPottingDirect++;

      const planned = await planTurnTraced({
        state: asState(balls, { groups: { 0: null, 1: null } }),
        table,
        player: 1,
        useNeural: false,
        wasmSource: WASM,
      });
      expect(planned.kind, `${name} produced no shot`).toBe("shot");
      if (planned.kind !== "shot") continue;
      expect(planned.trace.selected!.kind, `${name} played a direct`).not.toBe("direct");
      expect(PLAYABLE_KINDS).toContain(planned.trace.selected!.kind);
      expect(planned.action).toEqual(planned.trace.selected!.action);
    }
    expect(withPottingDirect, "the fixture set must contain a makeable direct").toBeGreaterThan(0);
  }, 180_000);
});

describe("no legal target rests the turn instead of wedging it", () => {
  it("returns `no-legal-shot` with a real trace and no report at all", async () => {
    // Player 1 is on solids and every solid is off the table, with the 8 gone
    // too — `legalTargets` returns []. This is the one condition under which
    // the ladder produces no shot, and the only one left now that rungs 4 and 5
    // cover every board where a legal target exists.
    const balls = [makeBall(CUE_ID, -0.5, 0), ...STRIPES.map((id, i) => makeBall(id, 0.2 * i - 0.5, 0.3))];
    const state = asState(balls, { groups: { 0: "stripes", 1: "solids" } });

    const planned = await planTurnTraced({
      state,
      table,
      player: 1,
      useNeural: false,
      wasmSource: WASM,
    });

    expect(planned.kind).toBe("no-legal-shot");
    // Not a nullable field the host can forget: there is no `report` here to
    // read, and the union makes reading one a compile error.
    expect("report" in planned).toBe(false);
    expect(planned.trace.selected).toBeNull();
    expect(planned.trace.policy).toBe("trick-only");
    expect(planned.trace.turn.legalTargets).toEqual([]);
    expect(SOLIDS.every((id) => !balls.some((b) => b.id === id))).toBe(true);
  }, 60_000);

  it("the hook's resting branch clears every flag that could hold `searching…`", () => {
    // There is no DOM in this package (see statusMessageVisibility.test.ts), so
    // the branch's *shape* is asserted the way opponentClaims.test.ts asserts
    // the effect's dependencies. TypeScript already forces the branch to exist:
    // `PlannedTurn` is a discriminated union, so reading `planned.report`
    // without narrowing does not compile. What it cannot force is that the
    // branch puts the UI back.
    const start = TURN_SRC.indexOf('if (planned.kind === "no-legal-shot") {');
    expect(start, "the resting branch is missing from useAiTurn").toBeGreaterThan(0);
    const branch = TURN_SRC.slice(start, TURN_SRC.indexOf("\n      }", start));
    expect(branch).toContain("setPlanning(false)");
    expect(branch).toContain("setModelLoading(false)");
    expect(branch).toContain("setBusy(false)");
    expect(branch).toContain('setPhase("aiming")');
    expect(branch).toContain("onNoLegalShot()");
    expect(branch).toContain("return;");
  });
});
