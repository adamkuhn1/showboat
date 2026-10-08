import { describe, expect, it } from "vitest";
import { type ShotReport } from "./game";
import { type ShotOutcome } from "./rules";
import { type GameState, type PlayerId } from "./state";
import { summariseOutcome } from "./summary";

// Hand-built outcomes with the expected sentence written out by hand.

const HUMAN: PlayerId = 0;
const AI: PlayerId = 1;

const report = (
  outcome: Partial<ShotOutcome>,
  nextTurn: PlayerId,
  groups: GameState["groups"] = { 0: null, 1: null },
): ShotReport => ({
  next: {
    balls: [],
    turn: nextTurn,
    groups,
    ballInHand: false,
    winner: null,
    broken: true,
    shotCount: 3,
  },
  sim: { balls: [], events: [], pocketed: [], firstContact: null, duration: 1 },
  outcome: {
    foul: false,
    foulReason: null,
    pocketedThisShot: [],
    turnPasses: false,
    ballInHandForNext: false,
    assignedGroups: false,
    gameOver: false,
    winner: null,
    ...outcome,
  },
});

describe("status-line outcome summary", () => {
  it("names the potted ball and whose turn it is", () => {
    expect(summariseOutcome(report({ pocketedThisShot: [6] }, HUMAN), HUMAN, HUMAN)).toBe(
      "You potted the 6. Your turn.",
    );
  });

  it("says plainly when nothing went down", () => {
    expect(summariseOutcome(report({ turnPasses: true }, AI), HUMAN, HUMAN)).toBe(
      "No ball potted. AI to play.",
    );
  });

  it("lists several balls and a group claim, ignoring the cue", () => {
    const r = report(
      { pocketedThisShot: [3, 11, 5], assignedGroups: true },
      AI,
      { 0: "stripes", 1: "solids" },
    );
    expect(summariseOutcome(r, AI, HUMAN)).toBe(
      "The AI potted the 3, the 11 and the 5. The AI is on solids. AI to play.",
    );
  });

  it("capitalises the AI in a foul message", () => {
    const r = report(
      { foul: true, foulReason: "contacted opponent's ball first", turnPasses: true },
      HUMAN,
    );
    expect(summariseOutcome(r, AI, HUMAN)).toBe(
      "Foul by the AI: contacted opponent's ball first. You have ball in hand.",
    );
  });

  it("never contains a collision chain", () => {
    const r = report({ pocketedThisShot: [1, 2, 9, 10] }, HUMAN);
    expect(summariseOutcome(r, HUMAN, HUMAN)).not.toMatch(/→|cue/);
  });
});
