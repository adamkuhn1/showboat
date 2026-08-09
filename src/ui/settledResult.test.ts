// The settled-result beat: the outcome has to survive long enough to be read.
//
// The failure this guards is specific and was invisible in a unit suite. When
// the opponent pots and keeps the table, `commit` is what lets the turn effect
// run again, and the first thing the next turn does is `setTrace(null)` and
// `setOutcome(null)`. Committing the instant the balls stopped therefore
// destroyed the outcome in the same React batch that created it — on exactly
// the turns where the opponent was doing best, which is most of them.
//
// So the ordering below is the behaviour, not an implementation detail: the
// outcome is set, then held, and only then is the turn committed.

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { TIMING, holdScaleForTurn, settleMs } from "../render/presentation";

const __dirname = dirname(fileURLToPath(import.meta.url));
const TURN_SRC = readFileSync(join(__dirname, "useAiTurn.ts"), "utf8");
const PANEL_SRC = readFileSync(join(__dirname, "OverlayPanel.tsx"), "utf8");

describe("the hold is long enough to read the sentence it exists for", () => {
  it("scales with the sentence and never drops below the floor", () => {
    expect(settleMs(0)).toBe(TIMING.SETTLE_MIN_MS);
    expect(settleMs(40)).toBe(TIMING.SETTLE_MAX_MS);
    expect(settleMs(12)).toBeGreaterThan(settleMs(4));
  });

  it("decays with the session, but far less than the process states do", () => {
    // A ten-word outcome on the tenth opponent turn — the worst case the decay
    // can produce — must still be readable.
    const late = settleMs(10, holdScaleForTurn(10));
    expect(holdScaleForTurn(10)).toBeLessThan(TIMING.SETTLE_MIN_HOLD_SCALE);
    expect(late).toBeGreaterThanOrEqual(TIMING.SETTLE_MIN_MS * TIMING.SETTLE_MIN_HOLD_SCALE);
    expect(late).toBeGreaterThan(850);
    // And it really is shorter than the first turn's, or the decay is a no-op.
    expect(late).toBeLessThan(settleMs(10, holdScaleForTurn(1)));
  });
});

describe("the next turn cannot erase the outcome before it is readable", () => {
  const body = TURN_SRC.slice(TURN_SRC.indexOf("await runShot(planState, planned, marks)"));

  it("sets the outcome, holds, and only then commits — in that order", () => {
    const set = body.indexOf("setOutcome(settled)");
    const hold = body.indexOf("await runHold(");
    const commit = body.indexOf("commit(planned.report)");
    expect(set, "the outcome is never set after the shot").toBeGreaterThan(0);
    expect(hold, "there is no settled-result hold").toBeGreaterThan(set);
    expect(commit, "the turn is never committed").toBeGreaterThan(hold);
  });

  it("the outcome is built from the executed trace and the rules verdict", () => {
    expect(body).toContain("shotOutcomeLine(planned.trace, planned.report.outcome)");
  });

  it("the hold is as long as the sentence needs, and decays with the session", () => {
    expect(body).toContain("settleMs(settled?.words ?? 0, holdScaleForTurn(turnIndexRef.current))");
  });

  it("only the start of the next opponent turn clears it", () => {
    expect((TURN_SRC.match(/setOutcome\(null\)/g) ?? []).length).toBe(1);
    const clear = TURN_SRC.indexOf("setOutcome(null)");
    expect(clear).toBeGreaterThan(0);
    // In the same block that clears the trace, and before the search it is
    // making room for. (Anchored on `setTrace(null)`, which occurs once, rather
    // than on `setPhase("searching")`, which the file also quotes in a comment.)
    const clearTrace = TURN_SRC.indexOf("setTrace(null)");
    expect(clear).toBeGreaterThan(clearTrace);
    expect(clear - clearTrace).toBeLessThan(400);
    expect(clear).toBeLessThan(TURN_SRC.indexOf("await planner.plan("));
  });
});

describe("the hold behaves like every other beat of the turn", () => {
  const hold = TURN_SRC.slice(TURN_SRC.indexOf("const runHold = useCallback"));
  const fn = hold.slice(0, hold.indexOf("[],\n  );"));

  it("is skippable", () => {
    expect(fn).toContain("skipRef.current");
  });

  it("clears the skip flag on entry, so skipping the roll does not skip the result", () => {
    // Each beat is skipped on its own press. Inheriting the shot's skip would
    // mean a visitor who pressed space to hurry the balls along never saw what
    // the balls did.
    expect(fn).toContain("skipRef.current = false");
  });

  it("can be cancelled by a new rack, and settles the await when it is", () => {
    // The bug class `settleLoopRef` exists for: cancelling the frame a loop is
    // waiting on leaves the turn parked on an await that never resolves, with
    // every control disabled and no error anywhere.
    expect(fn).toContain("settleLoopRef.current = resolve");
    expect(fn).toContain("cancelRef.current");
  });

  it("paints nothing, so the frame the shot ended on is what is held", () => {
    expect(fn).not.toContain("paintRef");
  });
});

describe("the panel shows the outcome and does not confuse it with the plan", () => {
  it("renders it as its own line, tagged with its tone", () => {
    expect(PANEL_SRC).toContain("outcome.text");
    expect(PANEL_SRC).toContain('data-tone={outcome.tone}');
  });

  it("prints it after the plan and the reason, not instead of them", () => {
    const plan = PANEL_SRC.indexOf("chosen-why");
    const reason = PANEL_SRC.indexOf("overlay-reason");
    const outcome = PANEL_SRC.indexOf("overlay-outcome");
    expect(plan).toBeGreaterThan(0);
    expect(reason).toBeGreaterThan(plan);
    expect(outcome).toBeGreaterThan(reason);
  });
});
