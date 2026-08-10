// THE measured-shot classifier: what a simulated shot actually did.
//
// It reads one input — the ordered event log of a full-fidelity physics
// rollout — plus the shot's *intent* (which ball had to be struck first, which
// ball was meant to drop, which balls are legal to strike this turn). It never
// reads the candidate's planned `kind`, and there is no code path here that can
// return a structure because a generator said so.
//
// WHY IT EXISTS
//
// `generateCandidates` labels a route from mirror geometry, before any physics
// runs. That label is a plan. A route planned off one cushion can execute with
// none (the mirror point lands within a ball's width of the pocket and the ball
// runs straight in) or with four (a thin cut sends it round the table). Anything
// that describes the shot from the plan is describing something that may not
// have happened, and selecting a "trick" from the plan can play a straight pot
// under a bank's name.
//
// TWO RULES THAT DECIDE MOST CASES
//
//  1. **Nothing after the decisive pot counts.** A ball that drops and a cue
//     ball that then rattles round three cushions is a one-rail bank, not a
//     four-rail bank. The event log is truncated at the decisive pot and every
//     structural question is asked of the prefix.
//  2. **Causality, not co-occurrence.** Cushion and ball contacts are credited
//     only to balls on the chain that carried the shot, walked backwards from
//     the potted ball to the cue ball through the contacts that actually set
//     each ball moving. A cushion contact by an uninvolved ball elsewhere on the
//     table is not a rail in this shot's route.
//
// WHICH POT IS "DECISIVE"
//
// The first pocket event for the ball the shot intended to drop, when that ball
// drops at all; otherwise the first pocket event for any non-cue ball. So a shot
// that pots its intended ball is classified on that ball's route even when
// another ball fell first, and a shot that pots something else is classified on
// what it did instead. Only the intended ball's *identity* is consulted — never
// the planned shape of its route.

import { CUE_ID } from "../../game/rack";
import type { ShotEvent, SimResult } from "../../physics/engine";
import type { MeasuredClass, MeasuredRoute, TrickKind } from "../trace/contract";

/**
 * What the shot was for. The two ball ids are identities, not shapes: they say
 * which ball had to be hit and which had to drop, and nothing about how.
 */
export interface ShotIntent {
  /** Ball the cue was required to strike first. Null for a shot planning none. */
  target: number | null;
  /** Ball the shot was meant to drop. Null for a shot planning no pot. */
  potId: number | null;
  /** Ball ids legal to strike first this turn, from the ruleset. */
  legalTargets: readonly number[];
}

/** One ball-to-ball contact from the log, in emission order. */
export interface BallContact {
  a: number;
  b: number;
  timeSec: number;
}

/**
 * The full measurement. `MeasuredRoute` is the published subset; the extra
 * fields here are what the unit tests and the evaluation harness read.
 */
export interface MeasuredShot extends MeasuredRoute {
  /** Index of the decisive pot in the event log, or -1 when nothing dropped. */
  potEventIndex: number;
  /** Every ball-ball contact before the decisive pot, in order. */
  ballContacts: BallContact[];
  /** Cushion contacts by the CUE ball before the decisive pot. Never a bank. */
  cueRails: number;
  /** Set when the shot is a foul, in the same vocabulary the ruleset uses. */
  foulReason: "scratch" | "no contact" | "illegal first contact" | "no rail" | null;
}

/** The four measured structures Showboat is allowed to present as tricks. */
const SUPPORTED: ReadonlySet<MeasuredClass> = new Set<MeasuredClass>([
  "one-rail-bank",
  "multi-rail-bank",
  "combination",
  "rail-combination",
]);

export const isSupportedTrickClass = (c: MeasuredClass): boolean => SUPPORTED.has(c);

/**
 * Classify one rollout.
 *
 * `sim` needs only its event log: the classifier deliberately does not read
 * `firstContact` or `pocketed`, which the engine derives, so that the answer
 * comes from the ordered log and can be checked against those two fields rather
 * than inheriting them.
 */
export function classifyMeasuredShot(
  sim: Pick<SimResult, "events">,
  intent: ShotIntent,
): MeasuredShot {
  const events = sim.events;

  const scratched = events.some((e) => e.kind === "pocket" && e.balls[0] === CUE_ID);

  // --- the decisive pot ---------------------------------------------------
  let potEventIndex = -1;
  if (intent.potId !== null && intent.potId !== CUE_ID) {
    potEventIndex = events.findIndex((e) => e.kind === "pocket" && e.balls[0] === intent.potId);
  }
  if (potEventIndex < 0) {
    potEventIndex = events.findIndex((e) => e.kind === "pocket" && e.balls[0] !== CUE_ID);
  }
  const potEvent: ShotEvent | null = potEventIndex >= 0 ? events[potEventIndex] : null;
  const pottedBall = potEvent === null ? null : potEvent.balls[0];
  const pocket = potEvent?.pocket ?? null;

  // Rule 1: everything the shot did after the decisive pot is not part of the
  // route that produced it.
  const pre = potEventIndex >= 0 ? events.slice(0, potEventIndex) : events;

  // --- first contact ------------------------------------------------------
  // Read from the whole log, not the prefix: it decides legality, and a legal
  // first contact necessarily precedes any pot the cue caused anyway.
  const firstHit = events.find((e) => e.kind === "ball-ball" && e.balls.includes(CUE_ID));
  const firstContact = firstHit ? (firstHit.balls.find((id) => id !== CUE_ID) ?? null) : null;
  const firstContactLegal = firstContact !== null && intent.legalTargets.includes(firstContact);

  // --- rule 2: who set whom moving ---------------------------------------
  const moving = new Set<number>([CUE_ID]);
  const driver = new Map<number, number>();
  const startedAt = new Map<number, number>();
  const ballContacts: BallContact[] = [];
  for (const e of pre) {
    if (e.kind !== "ball-ball" || e.balls.length < 2) continue;
    const [a, b] = e.balls;
    ballContacts.push({ a, b, timeSec: e.time });
    const aMoving = moving.has(a);
    const bMoving = moving.has(b);
    if (aMoving && !bMoving) {
      driver.set(b, a);
      startedAt.set(b, e.time);
      moving.add(b);
    } else if (bMoving && !aMoving) {
      driver.set(a, b);
      startedAt.set(a, e.time);
      moving.add(a);
    }
  }

  // Walk back from the potted ball to the cue ball. An empty chain means the
  // log does not attribute the pot to the cue ball at all, which is reported as
  // unattributable rather than guessed at.
  const contactChain: number[] = [];
  if (pottedBall !== null) {
    const seen = new Set<number>();
    let cur: number | null = pottedBall;
    while (cur !== null && !seen.has(cur)) {
      seen.add(cur);
      contactChain.unshift(cur);
      if (cur === CUE_ID) break;
      cur = driver.get(cur) ?? null;
    }
    if (contactChain[0] !== CUE_ID) contactChain.length = 0;
  }

  // --- relevant cushion contacts -----------------------------------------
  // A rail counts when the ball taking it is on the chain, is not the cue ball,
  // and had already been set moving. The cue ball's own cushions are counted
  // separately: a cue ball off a rail is a kick, never a bank.
  const chainObjects = new Set(contactChain.filter((id) => id !== CUE_ID));
  const railCushions: string[] = [];
  let cueRails = 0;
  let anyCushion = false;
  for (const e of pre) {
    if (e.kind !== "ball-cushion") continue;
    anyCushion = true;
    const id = e.balls[0];
    if (id === CUE_ID) {
      cueRails++;
      continue;
    }
    if (!chainObjects.has(id)) continue;
    const t0 = startedAt.get(id);
    if (t0 === undefined || e.time < t0) continue;
    railCushions.push(e.cushion ?? "");
  }
  // A cushion after the decisive pot still satisfies the ruleset's "no rail"
  // clause, which is about the whole shot rather than about the route.
  if (!anyCushion) anyCushion = events.some((e) => e.kind === "ball-cushion");
  const rails = railCushions.length;

  // --- the verdict --------------------------------------------------------
  let foulReason: MeasuredShot["foulReason"] = null;
  if (scratched) foulReason = "scratch";
  else if (firstContact === null) foulReason = "no contact";
  else if (!firstContactLegal) foulReason = "illegal first contact";
  else if (pottedBall === null && !anyCushion) foulReason = "no rail";

  let classification: MeasuredClass;
  if (foulReason !== null) {
    classification = "foul";
  } else if (pottedBall !== null) {
    if (contactChain.length < 2) {
      classification = "miss";
    } else if (contactChain.length > 2) {
      classification = rails >= 1 ? "rail-combination" : "combination";
    } else {
      classification = rails === 0 ? "direct" : rails === 1 ? "one-rail-bank" : "multi-rail-bank";
    }
  } else {
    // Nothing dropped and nothing fouled. A shot that planned no pot did what
    // it set out to do; a shot that planned one did not. This is the only place
    // intent participates in a classification, and it participates as "was a
    // pot intended at all", never as the planned shape of one.
    classification = intent.potId === null ? "safety" : "miss";
  }

  const trickVerified =
    SUPPORTED.has(classification) &&
    pottedBall !== null &&
    intent.potId !== null &&
    pottedBall === intent.potId &&
    intent.target !== null &&
    firstContact === intent.target;

  return {
    classification,
    rails,
    railCushions,
    contactChain,
    pottedBall,
    pocket,
    firstContact,
    firstContactLegal,
    scratched,
    trickVerified,
    potEventIndex,
    ballContacts,
    cueRails,
    foulReason,
  };
}

/** The publishable subset, with no measurement-only fields on it. */
export const toMeasuredRoute = (m: MeasuredShot): MeasuredRoute => ({
  classification: m.classification,
  rails: m.rails,
  railCushions: [...m.railCushions],
  contactChain: [...m.contactChain],
  pottedBall: m.pottedBall,
  pocket: m.pocket,
  firstContact: m.firstContact,
  firstContactLegal: m.firstContactLegal,
  scratched: m.scratched,
  trickVerified: m.trickVerified,
});

/**
 * The trick kind the EVENTS support, or null when they support none. This is
 * what the displayed type is derived from — a route planned as a bank that
 * measurably ran two cushions is a `double-bank` here, and one that measurably
 * touched no cushion is null and is not a trick at all.
 */
export function measuredTrickKind(m: MeasuredRoute): TrickKind | null {
  if (!m.trickVerified) return null;
  switch (m.classification) {
    case "one-rail-bank":
      return "bank";
    case "multi-rail-bank":
      return "double-bank";
    case "combination":
      return "combo";
    case "rail-combination":
      return "rail-combo";
    default:
      return null;
  }
}

const RAIL_WORD = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight"];

/**
 * How many rails, in words. Exported so the post-shot line and the plan line
 * count cushions in the same vocabulary — a route the plan line called a
 * three-rail bank must not be reported afterwards as a "multi-rail bank".
 */
export const railWord = (n: number): string => RAIL_WORD[n] ?? String(n);

/**
 * How a measured route is named in the interface. Derived from the measured
 * class AND the measured cushion count, so a three-rail route is never called a
 * two-rail bank because `double-bank` was the nearest label in the plan's
 * vocabulary.
 */
export function measuredRouteLabel(m: MeasuredRoute): string {
  switch (m.classification) {
    case "direct":
      return "direct pot";
    case "one-rail-bank":
      return "bank";
    case "multi-rail-bank":
      return `${railWord(m.rails)}-rail bank`;
    case "combination":
      return "combination";
    case "rail-combination":
      return m.rails <= 1 ? "rail combination" : `${railWord(m.rails)}-rail combination`;
    case "safety":
      return "safety";
    case "foul":
      return "foul";
    default:
      return "miss";
  }
}
