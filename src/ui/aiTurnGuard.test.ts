// Re-entrancy of the opponent-turn effect.
//
// The effect re-triggers on every `state` change, including the ones it causes
// itself, and getting this wrong hangs the turn rather than failing it — which
// is worse, because a hang has no error to log and no way out but a new rack.
//
// The bug this pins, found in Chrome against a production build: on any turn
// where the opponent had ball in hand, the effect placed the cue AND started
// planning in one pass. The placement's `setState` and the plan's
// `setPhase("searching")` landed in the same React batch, so the re-render's
// run hit the `phase !== "aiming"` guard and bailed — while the first run had
// already been torn down by that same re-render. Nothing was left to finish
// the turn, and the panel sat on "searching…" indefinitely.

import { describe, it, expect } from "vitest";
import { nextTurnAction } from "./useAiTurn";

const base = {
  active: true,
  turn: 1 as const,
  aiPlayer: 1 as const,
  winner: null,
  phase: "aiming" as const,
  ballInHand: false as const,
};

describe("the opponent-turn effect decides one thing per render", () => {
  it("plans when it is the opponent's move and the cue is on the table", () => {
    expect(nextTurnAction(base)).toBe("plan");
  });

  it("places the cue and does NOT also plan in the same pass", () => {
    expect(nextTurnAction({ ...base, ballInHand: "anywhere" })).toBe("place-cue");
    expect(nextTurnAction({ ...base, ballInHand: "behind-head" })).toBe("place-cue");
  });

  it("plans on the very next render, once the placement has committed", () => {
    // The re-run sees the placed state: ball-in-hand cleared, phase untouched
    // because nothing in the placement pass sets it.
    expect(nextTurnAction({ ...base, ballInHand: false })).toBe("plan");
  });

  it("never starts a second plan while one is in flight", () => {
    expect(nextTurnAction({ ...base, phase: "searching" })).toBe("idle");
    expect(nextTurnAction({ ...base, phase: "animating" })).toBe("idle");
  });

  it("does nothing on the human's move, after a win, or before the engine is up", () => {
    expect(nextTurnAction({ ...base, turn: 0 })).toBe("idle");
    expect(nextTurnAction({ ...base, winner: 0 })).toBe("idle");
    expect(nextTurnAction({ ...base, winner: 1 })).toBe("idle");
    expect(nextTurnAction({ ...base, active: false })).toBe("idle");
  });

  it("the placement pass is idle-safe if it somehow repeats", () => {
    // Two placements in a row would loop forever if `placeCueBall` did not
    // clear `ballInHand`; the guard's shape makes that a single fixed point.
    const first = nextTurnAction({ ...base, ballInHand: "anywhere" });
    expect(first).toBe("place-cue");
    expect(nextTurnAction({ ...base, ballInHand: false })).not.toBe("place-cue");
  });
});
