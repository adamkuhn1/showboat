// The opponent used to name its plan precisely and then say nothing about
// whether it worked. This suite is the guard on the half that was missing.
//
// Two layers, and the split is deliberate:
//
//  1. **End to end.** Real boards, planned through `planTurnTraced` — the same
//     function the worker and the main-thread fallback call — with real WASM
//     physics. The sentence is checked against `report.outcome` and against the
//     executed simulation's own event log, so a sentence that says a ball
//     dropped when the rules recorded no pot is a failure here.
//  2. **Branch by branch.** The rules verdicts a sparse test board will not
//     produce on demand (scratch, wrong first contact, no rail, an early 8) are
//     supplied as the `ShotOutcome` values `applyShotRules` returns for them —
//     that function has its own suite in `game/rules.test.ts`, and this one is
//     about the mapping from a verdict to a sentence.
//
// What the suite is really protecting is the direction of the dependency: the
// sentence is a function of the executed simulation and the rules, and of
// nothing else. A version that reached for the plan, the neural score or the
// search utility would pass a "reads nicely" review and fail here.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../physics/table";
import { makeBall, type Ball } from "../physics/ball";
import { CUE_ID, EIGHT_ID } from "../game/rack";
import type { GameState } from "../game/state";
import type { ShotOutcome } from "../game/rules";
import { initPhysics } from "../physics/wasm-bridge";
import { planTurnTraced, type PlayedTurn } from "./planner/plan";
import { shotOutcomeLine } from "./shotOutcome";
import { POCKET_NAME } from "./shotSentence";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../..");
const table = makeTable();
const wasm = readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm"));

const asState = (balls: Ball[]): GameState => ({
  balls,
  turn: 1,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 3,
});

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
  // Cue screened off its only legal target: the ladder drops to the safety
  // rung, which pots nothing and must still be reported as something.
  screened: [makeBall(CUE_ID, -0.8, 0.0), makeBall(8, -0.72, 0.0), makeBall(1, 0.7, 0.0)],
  pack: [
    makeBall(CUE_ID, -0.62, 0.02),
    makeBall(1, 0.28, 0.0),
    makeBall(2, 0.34, 0.05),
    makeBall(3, 0.34, -0.05),
    makeBall(4, 0.4, 0.1),
    makeBall(5, 0.4, 0.0),
  ],
};

const played = new Map<string, PlayedTurn>();

beforeAll(async () => {
  await initPhysics(wasm);
  for (const [name, balls] of Object.entries(BOARDS)) {
    const turn = await planTurnTraced({
      state: asState(balls),
      table,
      player: 1,
      useNeural: false,
      wasmSource: wasm,
    });
    if (turn.kind !== "shot") throw new Error(`${name} produced no shot to report on`);
    played.set(name, turn);
  }
}, 180_000);

describe("every opponent shot is reported, and the report matches the simulation", () => {
  it("produces a sentence for every board", () => {
    for (const [name, turn] of played) {
      const line = shotOutcomeLine(turn.trace, turn.report.outcome);
      expect(line, `${name} produced no outcome sentence`).not.toBeNull();
      expect(line!.text.length, `${name}`).toBeGreaterThan(0);
      // One or two short sentences. The panel is 288px wide and the line has
      // to be readable inside a hold measured in a second and a half.
      expect(line!.words, `${name}: "${line!.text}"`).toBeLessThanOrEqual(14);
      expect(line!.text.endsWith(".")).toBe(true);
    }
  });

  it("claims a pot only when the rules recorded one, and names the pocket the log did", () => {
    for (const [name, turn] of played) {
      const line = shotOutcomeLine(turn.trace, turn.report.outcome)!;
      const potted = turn.report.outcome.pocketedThisShot.filter((id) => id !== CUE_ID);
      const claimsDrop = /dropped/.test(line.text);
      expect(claimsDrop, `${name}: "${line.text}" vs pocketed ${potted}`).toBe(potted.length > 0);

      const named = line.text.match(/[Tt]he (\d+) dropped/);
      if (named) {
        const id = Number(named[1]);
        expect(potted, `${name} named a ball the rules did not pocket`).toContain(id);
        // And the pocket it names is the one the executed event log recorded —
        // the full phrase, not merely "a pocket event exists". Dropping the
        // pocket name entirely, or taking it from the candidate's plan instead
        // of the log, both have to fail here.
        const event = turn.trace.selected!.executed!.contactSequence.find(
          (e) => e.kind === "pocket" && e.balls.includes(id),
        );
        expect(event, `${name}: no pocket event for the ball named`).toBeTruthy();
        expect(line.text, `${name}: "${line.text}"`).toMatch(
          new RegExp(`[Tt]he ${id} dropped in the ${POCKET_NAME[event!.pocket!]} pocket`),
        );
        // Named, not printed as the raw pocket id the engine uses.
        expect(line.text, name).not.toMatch(/\b(bl|br|tl|tr|sb|st)\b/);
      }
    }
  });

  it("never contradicts the foul verdict", () => {
    for (const [name, turn] of played) {
      const line = shotOutcomeLine(turn.trace, turn.report.outcome)!;
      expect(/[Ff]oul|Scratch/.test(line.text), `${name}: "${line.text}"`).toBe(
        turn.report.outcome.foul,
      );
      expect(line.tone === "foul", `${name} tone`).toBe(turn.report.outcome.foul);
    }
  });

  it("a safety that pots nothing says so rather than going quiet", () => {
    const turn = played.get("screened")!;
    expect(turn.trace.selected!.kind).toBe("safety-kick");
    const line = shotOutcomeLine(turn.trace, turn.report.outcome)!;
    expect(line.text).toMatch(/Safety|nothing fell/i);
  });

  it("names the shot from the route the ball took, not from the route it was chosen for", () => {
    // A live run caught the old behaviour out: a candidate generated as a bank
    // whose mirror point sat almost on the pocket ran straight in touching no
    // cushion, and the panel still said "Bank made." over an event log with no
    // cushion in it. The word has to come from the measurement.
    for (const [name, turn] of played) {
      const line = shotOutcomeLine(turn.trace, turn.report.outcome)!;
      const m = line.text.match(/^(Bank|Two-rail bank|Multi-rail bank|Combination|Rail combination) made\./);
      if (!m) continue;
      const chosen = turn.trace.candidates.find(
        (c) => c.index === turn.trace.selected!.candidateIndex,
      )!;
      const trajectory = turn.trace.selected!.executed!.trajectories.find(
        (t) => t.ballId === chosen.potId,
      )!;
      const cushions = trajectory.breaks.filter((b) => b.kind === "cushion").length;
      const viaCombination = trajectory.roles.includes("combination");
      const expected = viaCombination
        ? cushions > 0
          ? "Rail combination"
          : "Combination"
        : cushions >= 3
          ? "Multi-rail bank"
          : cushions === 2
            ? "Two-rail bank"
            : "Bank";
      expect(cushions + (viaCombination ? 1 : 0), `${name}: nothing to name`).toBeGreaterThan(0);
      expect(m[1], `${name}: ${cushions} cushions -> "${line.text}"`).toBe(expected);
    }
  });

  it("a pot with no cushion and no combination is not dressed up as a trick", () => {
    // Synthesised from a real trace by emptying the potted ball's cushion
    // breaks: the ball still drops, at the same pocket, off nothing.
    const turn = played.get("openSpread")!;
    const chosen = turn.trace.candidates.find(
      (c) => c.index === turn.trace.selected!.candidateIndex,
    )!;
    const executed = turn.trace.selected!.executed!;
    const flattened = {
      ...turn.trace,
      selected: {
        ...turn.trace.selected!,
        executed: {
          ...executed,
          trajectories: executed.trajectories.map((t) =>
            t.ballId === chosen.potId
              ? { ...t, roles: t.roles.filter((r) => r !== "combination"), breaks: t.breaks.filter((b) => b.kind !== "cushion") }
              : t,
          ),
        },
      },
    };
    const line = shotOutcomeLine(flattened, {
      foul: false,
      foulReason: null,
      pocketedThisShot: [chosen.potId],
      turnPasses: false,
      ballInHandForNext: false,
      assignedGroups: false,
      gameOver: false,
      winner: null,
    })!;
    expect(line.text).not.toMatch(/made\./);
    expect(line.text).toContain(`without the ${turn.trace.selected!.kind === "double-bank" ? "two-rail bank" : "bank"}`);
  });

  it("claims nothing about a route the extractor never measured", () => {
    // Seen in a live turn: a rail combination's middle ball was struck a few
    // millimetres from the pocket, travelled less than `MIN_TRAVEL_M`, and so
    // had no trajectory extracted. Reporting that as a pot "without the rail
    // combination" is a claim about a route nothing measured — it has to read
    // as a plain drop instead.
    const turn = played.get("openSpread")!;
    const chosen = turn.trace.candidates.find(
      (c) => c.index === turn.trace.selected!.candidateIndex,
    )!;
    const executed = turn.trace.selected!.executed!;
    const unmeasured = {
      ...turn.trace,
      selected: {
        ...turn.trace.selected!,
        executed: {
          ...executed,
          trajectories: executed.trajectories.filter((t) => t.ballId !== chosen.potId),
        },
      },
    };
    const line = shotOutcomeLine(unmeasured, {
      foul: false,
      foulReason: null,
      pocketedThisShot: [chosen.potId],
      turnPasses: false,
      ballInHandForNext: false,
      assignedGroups: false,
      gameOver: false,
      winner: null,
    })!;
    expect(line.text).toBe(`The ${chosen.potId} dropped in the ${POCKET_NAME[
      executed.contactSequence.find((e) => e.kind === "pocket" && e.balls.includes(chosen.potId))!
        .pocket!
    ]} pocket.`);
    expect(line.text).not.toContain("without the");
    expect(line.text).not.toMatch(/made\./);
  });

  it("says so when the ball drops somewhere other than the pocket the plan named", () => {
    const turn = played.get("openSpread")!;
    const chosen = turn.trace.candidates.find(
      (c) => c.index === turn.trace.selected!.candidateIndex,
    )!;
    const executed = turn.trace.selected!.executed!;
    const potEvent = executed.contactSequence.find(
      (e) => e.kind === "pocket" && e.balls.includes(chosen.potId),
    );
    if (!potEvent) return; // this board's shot did not pot; nothing to compare
    const elsewhere = potEvent.pocket === chosen.pocket ? "tl" : potEvent.pocket;
    const moved = {
      ...turn.trace,
      selected: {
        ...turn.trace.selected!,
        executed: {
          ...executed,
          contactSequence: executed.contactSequence.map((e) =>
            e === potEvent ? { ...e, pocket: elsewhere } : e,
          ),
        },
      },
    };
    const line = shotOutcomeLine(moved, turn.report.outcome)!;
    expect(line.text).toContain("not the planned one");
  });
});

describe("the verdicts a sparse board will not hand you", () => {
  // A real trace with real executed motion; only the rules verdict varies.
  // These are the shapes `applyShotRules` returns — see `game/rules.ts`.
  const traceOf = () => played.get("openSpread")!.trace;

  const verdict = (over: Partial<ShotOutcome>): ShotOutcome => ({
    foul: false,
    foulReason: null,
    pocketedThisShot: [],
    turnPasses: true,
    ballInHandForNext: false,
    assignedGroups: false,
    gameOver: false,
    winner: null,
    ...over,
  });

  it("a scratch says scratch, and says what it costs", () => {
    const line = shotOutcomeLine(
      traceOf(),
      verdict({
        foul: true,
        foulReason: "scratch",
        pocketedThisShot: [CUE_ID],
        ballInHandForNext: true,
      }),
    )!;
    expect(line.text).toBe("Scratch. Ball in hand.");
    expect(line.tone).toBe("foul");
  });

  it("an illegal first contact is named as one", () => {
    const line = shotOutcomeLine(
      traceOf(),
      verdict({
        foul: true,
        foulReason: "hit opponent's ball first",
        ballInHandForNext: true,
      }),
    )!;
    expect(line.text).toMatch(/^Wrong first contact/);
    expect(line.text).toContain("Ball in hand.");
    expect(line.tone).toBe("foul");
  });

  it("a no-rail foul is distinguished from a miss", () => {
    const line = shotOutcomeLine(
      traceOf(),
      verdict({ foul: true, foulReason: "no rail", ballInHandForNext: true }),
    )!;
    expect(line.text).toContain("cushion");
    expect(line.text).toContain("Foul.");
  });

  it("no contact at all is not reported as a legal miss", () => {
    const line = shotOutcomeLine(
      traceOf(),
      verdict({ foul: true, foulReason: "no contact", ballInHandForNext: true }),
    )!;
    expect(line.text).toMatch(/reached nothing/);
    expect(line.tone).toBe("foul");
  });

  it("a legal shot that pots nothing says where it ran out, from the measured route", () => {
    const trace = traceOf();
    const line = shotOutcomeLine(trace, verdict({}))!;
    expect(line.tone).toBe("missed");
    // The chosen shot on this board is a bank, so its object ball really does
    // take a cushion, and the sentence counts the cushions it took rather than
    // the cushions the plan wanted.
    const chosen = trace.candidates.find((c) => c.index === trace.selected!.candidateIndex)!;
    const cushions = trace
      .selected!.executed!.trajectories.find((t) => t.ballId === chosen.potId)!
      .breaks.filter((b) => b.kind === "cushion").length;
    expect(line.text).toBe(
      cushions > 0
        ? `Missed after the ${["", "first", "second", "third"][cushions] ?? `${cushions}th`} cushion.`
        : "Legal contact, but nothing fell.",
    );
  });

  it("winning on the 8 and losing on the 8 do not read the same", () => {
    const won = shotOutcomeLine(
      traceOf(),
      verdict({ gameOver: true, winner: 1, pocketedThisShot: [EIGHT_ID], turnPasses: false }),
    )!;
    expect(won.text).toContain("Game over.");
    expect(won.tone).toBe("made");

    const lost = shotOutcomeLine(
      traceOf(),
      verdict({
        foul: true,
        foulReason: "8 early",
        gameOver: true,
        winner: 0,
        pocketedThisShot: [EIGHT_ID],
      }),
    )!;
    expect(lost.text).toBe("The 8 went down early. Game over.");
    expect(lost.tone).toBe("foul");
    expect(lost.text).not.toEqual(won.text);
  });

  it("a pot that hands the table over is not weighted as a success", () => {
    const trace = traceOf();
    const chosen = trace.candidates.find((c) => c.index === trace.selected!.candidateIndex)!;
    // Some other ball fell — the shooter's opponent's — so the turn passes.
    const other = trace.candidates.find((c) => c.potId !== chosen.potId)!.potId;
    const line = shotOutcomeLine(
      trace,
      verdict({ pocketedThisShot: [other], turnPasses: true }),
    )!;
    expect(line.text).toMatch(/^Not the planned ball/);
    expect(line.tone).toBe("missed");
  });

  it("returns nothing when there was no shot to report on", () => {
    const trace = { ...traceOf(), selected: null };
    expect(shotOutcomeLine(trace, verdict({}))).toBeNull();
  });
});
