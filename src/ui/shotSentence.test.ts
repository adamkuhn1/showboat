// The two sentences the panel writes about a decision, and the two defects a
// review found in them.
//
//  1. "the trick cleared the 0.50 reliability bar" printed CHARACTER-IDENTICAL
//     under all eight plans observed in a live session, because the only number
//     in it was `TRICK_RELIABILITY_THRESHOLD` — a constant. A line that cannot
//     differ between two shots is not telling you anything about either.
//  2. The portfolio's copy promises "the opponent turns down the straight pot
//     to take a bank instead", and no sentence ever said so. Separately, the
//     comparison clause twice named the same ball as the shot itself
//     ("a two-rail bank on the 7 … over a bank on the 7").
//
// Every fixture is a decision made by the real classical brain on a real board
// with real WASM physics, so an assertion about what the sentence says is an
// assertion about what the opponent really did.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../physics/table";
import { makeBall, type Ball } from "../physics/ball";
import { CUE_ID } from "../game/rack";
import type { GameState } from "../game/state";
import { initPhysics } from "../physics/wasm-bridge";
import { defaultConfig } from "../ai/shotSearch";
import { classicalTrickOnlyBrain } from "../ai/brain";
import { isPathClear } from "../ai/candidates";
import type { DecisionTraceV1 } from "../ai/trace/contract";
import { passedOverDirect, runnerUp, rungText, shotSentence } from "./shotSentence";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../..");
const table = makeTable();

const asState = (balls: Ball[]): GameState => ({
  balls,
  turn: 1,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 3,
});

// Eight boards, because eight is the number of consecutive plans the review
// watched print the same rung line.
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
  tightPocket: [makeBall(CUE_ID, -0.85, 0.34), makeBall(1, 0.62, 0.3), makeBall(11, 0.35, 0.36)],
  lone: [makeBall(CUE_ID, -0.2, 0.0), makeBall(1, 0.6, 0.25)],
  pack: [
    makeBall(CUE_ID, -0.62, 0.02),
    makeBall(1, 0.28, 0.0),
    makeBall(2, 0.34, 0.05),
    makeBall(3, 0.34, -0.05),
    makeBall(4, 0.4, 0.1),
    makeBall(5, 0.4, 0.0),
  ],
  spread: [
    makeBall(CUE_ID, 0.1, -0.3),
    makeBall(2, -0.5, 0.2),
    makeBall(4, 0.55, 0.05),
    makeBall(13, -0.1, 0.31),
  ],
  nearRail: [makeBall(CUE_ID, -0.5, -0.34), makeBall(3, 0.2, -0.36), makeBall(5, 0.65, 0.2)],
  // Found by search, and the reason it is here: on this board the strongest
  // losing candidate is a DIFFERENT KIND OF SHOT ON THE SAME BALL as the one
  // being played — a bank on the 1, lost to a two-rail bank on the 1. That is
  // the exact collision the review saw twice, and the previous "not the
  // identical short form" rule let it straight through.
  sameBallLoser: [
    makeBall(CUE_ID, -0.751, 0.084),
    makeBall(1, -0.837, 0.364),
    makeBall(2, -0.878, 0.279),
    makeBall(3, 0.616, -0.365),
    makeBall(4, -0.111, -0.316),
  ],
  // Cue screened off its only legal target: no trick, no direct, a safety.
  screened: [makeBall(CUE_ID, -0.8, 0.0), makeBall(8, -0.72, 0.0), makeBall(1, 0.7, 0.0)],
};

const traces = new Map<string, DecisionTraceV1>();

beforeAll(async () => {
  await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
  const brain = classicalTrickOnlyBrain();
  for (const [name, balls] of Object.entries(BOARDS)) {
    // Both clocks pinned, for the reason `overlayTruthfulness.test.tsx`
    // documents: a wall-clock guard makes the fixture depend on machine load.
    const decision = await brain.plan(asState(balls), table, 1, {
      ...defaultConfig,
      seed: 20260808,
      seedTimeoutMs: Number.POSITIVE_INFINITY,
      searchTimeoutMs: Number.POSITIVE_INFINITY,
    });
    traces.set(name, decision.decision);
  }
}, 180_000);

describe("the rung line carries this shot's data, not a constant", () => {
  it("does not print the same string under every plan", () => {
    const lines = [...traces].map(([name, t]) => [name, rungText(t)] as const);
    for (const [name, text] of lines) expect(text, `${name} produced no rung line`).toBeTruthy();
    const barLines = lines.filter(([, text]) => text!.includes("reliability bar"));
    expect(barLines.length, "no board reached a reliability-bar rung").toBeGreaterThan(1);
    // The defect, stated exactly: every bar line identical.
    expect(new Set(barLines.map(([, text]) => text)).size).toBeGreaterThan(1);
  });

  it("quotes the chosen candidate's own measured strength", () => {
    for (const [name, t] of traces) {
      const text = rungText(t)!;
      if (!text.includes("reliability bar")) continue;
      const sel = t.selected!;
      const chosen = t.candidates.find((c) => c.index === sel.candidateIndex)!;
      expect(chosen.physics, `${name}: a bar rung with no physics`).not.toBeNull();
      expect(text, name).toContain(chosen.physics!.strength.toFixed(2));
      expect(text, name).toContain(sel.reliabilityThreshold.toFixed(2));
    }
  });

  it("states how many tricks cleared the bar, matching the trace", () => {
    for (const [name, t] of traces) {
      const text = rungText(t)!;
      if (t.selected!.rung !== "trick-qualified") continue;
      const n = t.selected!.qualifyingTricks;
      expect(n, name).toBeGreaterThan(0);
      expect(text, `${name}: ${n} qualifying -> "${text}"`).toContain(
        n > 1 ? `strongest of ${n} tricks` : "the only trick",
      );
    }
  });

  it("never renders the strength as a percentage", () => {
    // It is `1 - exp(-value)`, not a probability. The contract says so and the
    // panel is not allowed to imply otherwise.
    for (const [name, t] of traces) expect(rungText(t), name).not.toContain("%");
  });
});

describe("the straight-pot refusal is spoken, once, from real candidate data", () => {
  it("says it on every board where a pot was genuinely on", () => {
    let spoken = 0;
    for (const [name, t] of traces) {
      const refused = passedOverDirect(t);
      const text = shotSentence(t)!.text;
      if (refused === null) {
        expect(text, `${name} named a straight pot with no index recorded`).not.toContain(
          "straight pot",
        );
        continue;
      }
      spoken++;
      expect(text, name).toContain(`turning down the straight pot on the ${refused.target}`);
      // Once. Not once per clause.
      expect(text.match(/straight pot/g)!.length, name).toBe(1);
    }
    expect(spoken, "no board exercised the refusal at all").toBeGreaterThan(0);
  });

  it("only ever names a direct the policy really excluded", () => {
    for (const [name, t] of traces) {
      const refused = passedOverDirect(t);
      if (refused === null) continue;
      expect(refused.kind, name).toBe("direct");
      expect(refused.eligible, name).toBe(false);
      expect(refused.rejection, name).toBe("direct-excluded-by-policy");
      expect(refused.index, name).not.toBe(t.selected!.candidateIndex);
    }
  });

  it("only names a pot whose path to the pocket is actually clear", () => {
    // The claim the sentence makes, re-derived here from the board the decision
    // was made on rather than from the field under test. A direct whose object
    // ball is screened off the pocket was enumerated but was never on.
    for (const [name, balls] of Object.entries(BOARDS)) {
      const t = traces.get(name)!;
      const refused = passedOverDirect(t);
      if (refused === null) continue;
      const live = balls.filter((b) => !b.pocketed);
      const from = refused.path[0];
      const to = refused.path[refused.path.length - 1];
      expect(isPathClear(from, to, live, new Set([CUE_ID, refused.target])), name).toBe(true);
    }
  });

  it("records nothing to refuse when no direct was generated at all", () => {
    const screened = traces.get("screened")!;
    expect(screened.candidates.some((c) => c.kind === "direct")).toBe(false);
    expect(screened.selected!.passedOverDirectIndex).toBeNull();
    expect(shotSentence(screened)!.text).not.toContain("straight pot");
  });
});

describe("the comparison clause never names the ball the shot is on", () => {
  it("holds on every fixture", () => {
    for (const [name, t] of traces) {
      const text = shotSentence(t)!.text;
      const m = text.match(/, over a [a-z- ]+ on the (\d+)\./);
      if (!m) continue;
      const chosen = t.candidates.find((c) => c.index === t.selected!.candidateIndex)!;
      expect(Number(m[1]), `${name}: "${text}"`).not.toBe(chosen.target);
    }
  });

  it("refuses a same-ball loser on the board that reproduces the defect", () => {
    const t = traces.get("sameBallLoser")!;
    const chosen = t.candidates.find((c) => c.index === t.selected!.candidateIndex)!;
    const losers = t.candidates.filter(
      (c) => c.rejection === "lower-utility-than-selected" && c.physics !== null,
    );
    const strongest = losers.reduce((a, b) =>
      b.physics!.strength > a.physics!.strength ? b : a,
    );

    // The fixture really does exhibit the collision, or the assertion after it
    // proves nothing: same ball, different kind, and it is the one the old rule
    // would have picked.
    expect(strongest.target, "fixture drift: no same-ball loser").toBe(chosen.target);
    expect(strongest.kind, "fixture drift: the loser is the same kind too").not.toBe(chosen.kind);

    const over = runnerUp(t, chosen.target);
    expect(over === null || over.target !== chosen.target).toBe(true);
  });

  it("`runnerUp` excludes the chosen ball rather than only the identical wording", () => {
    // The precise hole in the previous rule: a loser of a DIFFERENT kind on the
    // SAME ball had a different short form, so it passed, and the sentence read
    // "a two-rail bank on the 7 … over a bank on the 7".
    for (const [name, t] of traces) {
      const chosen = t.candidates.find((c) => c.index === t.selected!.candidateIndex);
      if (!chosen) continue;
      const over = runnerUp(t, chosen.target);
      if (over === null) continue;
      expect(over.target, name).not.toBe(chosen.target);
      expect(over.rejection, name).toBe("lower-utility-than-selected");
    }
  });

  it("still names a loser when one exists on another ball", () => {
    // The clause must not have been quietly deleted: dropping it whenever it is
    // awkward would also pass the assertion above.
    const named = [...traces].filter(([, t]) => / over a /.test(shotSentence(t)!.text));
    const eligible = [...traces].filter(([, t]) => {
      const chosen = t.candidates.find((c) => c.index === t.selected!.candidateIndex);
      return (
        chosen !== undefined &&
        passedOverDirect(t) === null &&
        runnerUp(t, chosen.target) !== null
      );
    });
    expect(named.length).toBe(eligible.length);
  });
});
