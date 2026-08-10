// What happened, after the balls stop.
//
// The panel names the shot precisely and then used to go quiet: a visitor read
// "Playing a two-rail bank on the 7, into the bottom-side pocket" and never
// learned whether the 7 went in. This is the missing half.
//
// It reads two records of the SAME shot and nothing else:
//
//   - `selected.executed` — the authoritative simulation's own event log and
//     measured trajectories, attached by `ui/planner/plan.ts` immediately after
//     `executeAiShot`. This is where a pocket id and a cushion count come from.
//   - `ShotOutcome` — `applyShotRules`'s verdict on that same simulation. This
//     is where foul, scratch, ball-in-hand and game over come from.
//
// Deliberately NOT read: the neural score, the search utility, the reliability
// strength, and — with one exception — the plan. A shot has to read as what
// happened, not as what it was for, so the word naming the shot is derived from
// the potted ball's own measured route (`measuredShape`) and the miss branches
// count the cushions the ball really took. The single exception is naming the
// shape that did NOT come off, which is the one place the plan is the subject.

import { CUE_ID, EIGHT_ID } from "../game/rack";
import type { ShotOutcome } from "../game/rules";
import type { DecisionTraceV1, ExecutedMotion } from "../ai/trace/contract";
import { POCKET_NAME, wordsIn } from "./shotSentence";
import { railWord } from "../ai/measure/classify";

/** Enough to colour the line and to decide nothing else. */
export type OutcomeTone = "made" | "missed" | "foul";

export interface ShotOutcomeLine {
  text: string;
  tone: OutcomeTone;
  /** For the settled-result hold, which must be long enough to read this. */
  words: number;
}

/**
 * What the shot was FOR, lower-cased for use inside a sentence. Used only to
 * name the thing that did not happen — never to describe what did.
 */
const PLANNED_SHAPE: Record<string, string> = {
  bank: "bank",
  "double-bank": "two-rail bank",
  combo: "combination",
  "rail-combo": "rail combination",
  "safety-kick": "safety",
};

type Trajectory = ExecutedMotion["trajectories"][number];

const trajectoryOf = (executed: ExecutedMotion | null, ballId: number): Trajectory | null =>
  executed?.trajectories.find((t) => t.ballId === ballId) ?? null;

/**
 * What the shot WAS, read off the potted ball's measured route.
 *
 * The label used to be `selected.kind` — the plan's word — and a live run
 * caught it out: a candidate generated as a bank whose mirror point sat almost
 * on the pocket was played, the object ball ran straight in touching no
 * cushion at all, and the panel said "Bank made." over an event log with no
 * cushion in it. The plan is not evidence about the shot. This is.
 *
 * Null means the ball reached the pocket off nothing — no cushion, and not set
 * moving by another object ball. There is no trick word for that, and the
 * caller says so rather than reaching for the plan's.
 *
 * Takes a trajectory rather than a ball id so that "no route was recorded" and
 * "the recorded route had no cushion in it" cannot collapse into one answer.
 * They did, briefly, and a live turn caught that out too: a combination's
 * middle ball was struck a few millimetres from the pocket, travelled less than
 * `MIN_TRAVEL_M` and so had no trajectory extracted at all, and the panel
 * reported it as a pot "without the rail combination" — a claim about a route
 * nothing had measured.
 */
function measuredShape(trajectory: Trajectory): string | null {
  const cushions = trajectory.breaks.filter((b) => b.kind === "cushion").length;
  const capitalise = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
  if (trajectory.roles.includes("combination")) {
    if (cushions === 0) return "Combination";
    return cushions === 1 ? "Rail combination" : capitalise(`${railWord(cushions)}-rail combination`);
  }
  if (cushions >= 2) return capitalise(`${railWord(cushions)}-rail bank`);
  if (cushions === 1) return "Bank";
  return null;
}

const ORDINALS = ["", "first", "second", "third", "fourth", "fifth", "sixth"];
const ordinal = (n: number): string => ORDINALS[n] ?? `${n}th`;

/** The pocket a ball really went into, named, or null if the log did not say. */
const pocketOf = (executed: ExecutedMotion | null, ballId: number): string | null => {
  const event = executed?.contactSequence.find(
    (e) => e.kind === "pocket" && e.balls.includes(ballId),
  );
  const id = event?.pocket ?? null;
  return id === null ? null : (POCKET_NAME[id] ?? id);
};

/** Cushions this ball really took, counted off its measured trajectory. */
const cushionsTaken = (executed: ExecutedMotion | null, ballId: number): number => {
  const trajectory = executed?.trajectories.find((t) => t.ballId === ballId);
  return trajectory ? trajectory.breaks.filter((b) => b.kind === "cushion").length : 0;
};

/** The ball the cue really struck first, per the executed roles. */
const firstContactBall = (executed: ExecutedMotion | null): number | null =>
  executed?.trajectories.find((t) => t.roles.includes("first-contact"))?.ballId ?? null;

const line = (text: string, tone: OutcomeTone): ShotOutcomeLine => ({
  text,
  tone,
  words: wordsIn(text),
});

/**
 * One sentence about the shot that just finished, or null when there was no
 * shot to report on (no legal target, so nothing was played).
 */
export function shotOutcomeLine(
  trace: DecisionTraceV1,
  outcome: ShotOutcome,
): ShotOutcomeLine | null {
  const sel = trace.selected;
  if (sel === null) return null;

  const executed = sel.executed;
  const chosen =
    sel.candidateIndex === null
      ? null
      : (trace.candidates.find((c) => c.index === sel.candidateIndex) ?? null);
  // The ball the shot was for. Differs from `target` on a combination, which is
  // exactly why it is read from `potId` and not from the ball the cue strikes.
  const plannedPot = chosen?.potId ?? null;
  const plannedShape = PLANNED_SHAPE[sel.kind] ?? "shot";
  const scratched = outcome.pocketedThisShot.includes(CUE_ID);

  const dropped = (ballId: number): string => {
    const pocket = pocketOf(executed, ballId);
    return pocket === null
      ? `The ${ballId} dropped.`
      : `The ${ballId} dropped in the ${pocket} pocket.`;
  };

  // --- Game over ---------------------------------------------------------
  // `applyShotRules` returns `foul: false` on exactly one gameOver path: the 8
  // legally potted after the group was cleared. Every other way the 8 goes down
  // is a loss and is returned as a foul.
  if (outcome.gameOver) {
    if (!outcome.foul) {
      const eight = trajectoryOf(executed, EIGHT_ID);
      const shape = eight === null ? null : measuredShape(eight);
      return line(
        shape === null
          ? `${dropped(EIGHT_ID)} Game over.`
          : `${shape} made. ${dropped(EIGHT_ID)} Game over.`,
        "made",
      );
    }
    return line(
      scratched ? "Scratched on the 8. Game over." : "The 8 went down early. Game over.",
      "foul",
    );
  }

  // --- Fouls -------------------------------------------------------------
  if (outcome.foul) {
    const tail = outcome.ballInHandForNext ? " Ball in hand." : "";
    if (scratched) return line(`Scratch.${tail}`, "foul");
    switch (outcome.foulReason) {
      case "no contact":
        return line(`The cue ball reached nothing. Foul.${tail}`, "foul");
      case "no rail":
        return line(`Legal contact, but no ball reached a cushion. Foul.${tail}`, "foul");
      default: {
        // The three first-contact reasons ("hit opponent's ball first",
        // "can't hit the 8 yet", "hit the 8 first") say the same thing to a
        // visitor and differ only in which ball was wrong, so the ball is what
        // is named — from the event log, not from the rule that fired.
        const hit = firstContactBall(executed);
        const which = hit === null ? "" : ` — the ${hit}`;
        return line(`Wrong first contact${which}. Foul.${tail}`, "foul");
      }
    }
  }

  // --- Legal shots -------------------------------------------------------
  const objectPotted = outcome.pocketedThisShot.filter((id) => id !== CUE_ID);
  if (plannedPot !== null && objectPotted.includes(plannedPot)) {
    const trajectory = trajectoryOf(executed, plannedPot);
    if (trajectory === null) {
      // The ball dropped, and nothing was measured about how it got there —
      // it travelled less than the extractor's floor. Report the drop and make
      // no claim in either direction about the route.
      return line(dropped(plannedPot), "made");
    }
    const shape = measuredShape(trajectory);
    if (shape === null) {
      // It went in, but not as the shot it was chosen to be.
      return line(`${dropped(plannedPot).slice(0, -1)}, but without the ${plannedShape}.`, "made");
    }
    // Which pocket, and whether it is the one the panel said the ball would go
    // into. That claim comes from the rollout's measured route, not from the
    // generator's pocket — the plan line already reconciles those two — so this
    // clause fires only when the executed run diverged from the run the panel
    // showed, which is the difference worth reporting after the fact.
    const predicted = sel.measured?.pocket ?? chosen?.pocket ?? null;
    const aimedAt = predicted === null ? null : (POCKET_NAME[predicted] ?? predicted);
    const actual = pocketOf(executed, plannedPot);
    const elsewhere = actual !== null && aimedAt !== null && actual !== aimedAt;
    return line(
      elsewhere
        ? `${shape} made. ${dropped(plannedPot).slice(0, -1)}, not the planned one.`
        : `${shape} made. ${dropped(plannedPot)}`,
      "made",
    );
  }
  if (objectPotted.length > 0) {
    const id = objectPotted[0];
    const pocket = pocketOf(executed, id);
    return line(
      pocket === null
        ? `Not the planned ball — the ${id} dropped.`
        : `Not the planned ball — the ${id} dropped in the ${pocket} pocket.`,
      // A ball went down and the table still changed hands, which is what
      // happens when the ball that fell belonged to the other player. That is
      // not a shot that worked, so it is not weighted like one.
      outcome.turnPasses ? "missed" : "made",
    );
  }

  // Nothing fell and nothing fouled. Where the shot ran out is read off the
  // ball's own measured route, so "after the second cushion" is a count of
  // cushions it really took and not of cushions the plan wanted.
  const cushions = plannedPot === null ? 0 : cushionsTaken(executed, plannedPot);
  if (cushions > 0) return line(`Missed after the ${ordinal(cushions)} cushion.`, "missed");
  if (sel.kind === "safety-kick") return line("Safety played. Nothing fell.", "missed");
  return line("Legal contact, but nothing fell.", "missed");
}
