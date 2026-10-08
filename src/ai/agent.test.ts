import { describe, expect, it } from "vitest";
import { makeBall } from "../physics/ball";
import { makeTable } from "../physics/table";
import { BALL_RADIUS } from "../physics/constants";
import { type GameState } from "../game/state";
import { takeShot } from "../game/game";
import { STRIPES } from "../game/rack";
import { safetyAction, type Candidate } from "./candidates";
import { type MeasuredShot } from "./classify";
import { makeRanker } from "./ranker";
import { makeRng } from "./rollout";
import {
  aiTakeTurn,
  chooseSafety,
  ROBUSTNESS_N,
  selectShot,
  type EvaluatedCandidate,
} from "./agent";

// Selection tests use hand-built verified candidates with hand-chosen
// measurements and jitter counts, so the expected winner is worked out from
// the rule itself, not by re-running the selector. Safety tests use physical
// layouts whose legality is decided by the rules engine (a blocked nearest
// ball; a ball fully caged by opponent balls).

const measured = (
  kind: MeasuredShot["kind"],
  label: string,
  objectRails = 0,
): MeasuredShot => ({
  cueRailsBeforeContact: kind === "kick" ? 1 : 0,
  objectRails,
  chainLength: kind === "combo" ? 2 : 1,
  targetPotted: kind !== "no-pot",
  pocketId: "tr",
  railsBeforePot: objectRails,
  kind,
  label,
});

let nextId = 0;
const verified = (
  m: MeasuredShot,
  successes: number,
  prior: number,
  n = ROBUSTNESS_N,
): EvaluatedCandidate => {
  const cand: Candidate = {
    id: nextId++,
    kind: m.kind === "no-pot" ? "direct" : m.kind,
    targetBall: 3,
    pocketId: "tr",
    railsPlanned: m.objectRails,
    action: { phi: 0, power: 0.5, sideSpin: 0, topSpin: 0 },
    cuePath: [],
    objectPath: [],
  };
  return {
    cand,
    features: [],
    prior,
    orderScore: prior,
    measured: m,
    success: true,
    robustness: { successes, n },
  };
};

const direct = () => measured("direct", "direct shot");
const twoRailBank = () => measured("bank", "2-rail bank", 2);

describe("shot selection (robustness gates the trick-shot preference)", () => {
  it("a fragile trick shot loses to a robust direct shot", () => {
    const bank = verified(twoRailBank(), 1, 0.9); // 1/3: below the bar
    const pot = verified(direct(), 3, 0.4); // 3/3
    const pick = selectShot([bank, pot])!;
    expect(pick.chosen).toBe(pot);
    expect(pick.rule).toBe("robust");
    expect(pick.reason).toContain(
      `passed over a 2-rail bank (1/${ROBUSTNESS_N} under jitter) for a ${ROBUSTNESS_N}/${ROBUSTNESS_N} direct pot`,
    );
    expect(pick.reason).not.toContain("preferred as a trick shot");
  });

  it("a 0/3 trick shot never beats a direct pot, whatever its prior", () => {
    const bank = verified(twoRailBank(), 0, 0.99);
    const pot = verified(direct(), 2, 0.05);
    expect(selectShot([bank, pot])!.chosen).toBe(pot);
  });

  it("a robust trick shot beats a more robust direct shot, and says so", () => {
    const bank = verified(twoRailBank(), 2, 0.3); // 2/3: clears the bar
    const pot = verified(direct(), 3, 0.8);
    const pick = selectShot([pot, bank])!;
    expect(pick.chosen).toBe(bank);
    expect(pick.rule).toBe("trick-preferred");
    expect(pick.reason).toContain(
      `preferred as a trick shot over a ${ROBUSTNESS_N}/${ROBUSTNESS_N} direct shot`,
    );
  });

  it("does not claim the trick preference decided it when the trick was best anyway", () => {
    const bank = verified(twoRailBank(), 3, 0.5);
    const pot = verified(direct(), 2, 0.9);
    const pick = selectShot([pot, bank])!;
    expect(pick.chosen).toBe(bank);
    expect(pick.rule).toBe("robust");
    expect(pick.reason).not.toContain("preferred as a trick shot");
  });

  it("falls back to the most robust successful shot when nothing clears the bar", () => {
    const fragileBank = verified(twoRailBank(), 0, 0.9);
    const shakyPot = verified(direct(), 1, 0.2);
    const pick = selectShot([fragileBank, shakyPot])!;
    expect(pick.chosen).toBe(shakyPot);
    expect(pick.rule).toBe("most-robust-below-bar");
    expect(pick.reason).toMatch(/^No verified shot pots in 2\/3 under jitter/);
  });

  it("uses the candidate's real n for the bar, not a hard-coded 3", () => {
    // 3/5 = 0.6 < 2/3 fails; 4/5 = 0.8 passes.
    const bank35 = verified(twoRailBank(), 3, 0.9, 5);
    const pot45 = verified(direct(), 4, 0.1, 5);
    expect(selectShot([bank35, pot45])!.chosen).toBe(pot45);
  });

  it("returns null (-> safety) only when no candidate succeeded", () => {
    const miss = { ...verified(direct(), 0, 0.9), success: false };
    expect(selectShot([miss])).toBeNull();
    expect(selectShot([])).toBeNull();
  });
});

// AI is player 1 on stripes. Real 7-ft table coordinates (x in ±0.99, y in
// ±0.495).
const stripesState = (balls: GameState["balls"]): GameState => ({
  balls,
  turn: 1,
  groups: { 0: "solids", 1: "stripes" },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 12,
});

describe("safety play is simulated, not assumed legal", () => {
  it("avoids the old roll-up when it would hit an opponent ball first", () => {
    const table = makeTable();
    const g = stripesState([
      makeBall(0, -0.6, 0), // cue
      makeBall(1, -0.48, 0), // opponent solid, dead in line with...
      makeBall(9, -0.35, 0), // ...the AI's nearest stripe
      makeBall(10, -0.6, 0.4), // another stripe, clear path
      makeBall(8, 0.6, -0.3),
    ]);
    // Precondition: the old single heuristic roll-up fouls here.
    const old = takeShot(g, table, safetyAction(g, table)!);
    expect(old.outcome.foul).toBe(true);
    expect(old.outcome.foulReason).toBe("contacted opponent's ball first");

    const safe = chooseSafety(g, table, makeRanker("classical"));
    expect(safe.considered).toBeGreaterThan(1);
    expect(safe.legal).toBeGreaterThan(0);
    expect(safe.report.outcome.foul).toBe(false);
    expect(STRIPES).toContain(safe.report.sim.firstContact);
    expect(safe.report.sim.pocketed).not.toContain(0);
  });

  it("falls back to the old roll-up and says so when no safety is legal", async () => {
    const table = makeTable();
    // The AI's only stripe is caged by six solids: any path to it touches a
    // solid first, so every shot and every safety is a foul.
    const cx = 0.3;
    const cy = 0;
    const ring = Array.from({ length: 6 }, (_, k) => {
      const a = (k * Math.PI) / 3;
      const r = BALL_RADIUS * 2.1;
      return makeBall(1 + k, cx + r * Math.cos(a), cy + r * Math.sin(a));
    });
    const g = stripesState([
      makeBall(0, -0.6, 0),
      makeBall(9, cx, cy),
      ...ring,
      makeBall(8, 0.75, 0.35),
    ]);
    const move = (await aiTakeTurn(g, table, makeRanker("classical"), {
      rng: makeRng(1),
    }))!;
    expect(move.decision.selected).toBeNull();
    expect(move.decision.safety?.legal).toBe(0);
    expect(move.decision.reason).toContain("No legal safety found");
    // The fallback is exactly the old heuristic roll-up.
    expect(move.action).toEqual(safetyAction(g, table));
  });
});
