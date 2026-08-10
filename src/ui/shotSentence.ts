// The one sentence the panel exists to write: what the opponent is playing,
// and what it played it over.
//
// Every clause is a restatement of something the trace says. The straight-pot
// clause names the candidate `selected.passedOverDirectIndex` addresses — a
// direct the policy excluded and the builder verified was genuinely on. The
// "over …" clause names a candidate the trace itself labelled
// `lower-utility-than-selected` — i.e. one the search recorded as having lost
// to the chosen shot. Nothing here compares, ranks or scores anything on its
// own.
//
// FOUR DESCRIPTIONS, KEPT APART
//
// The shot being played is named from `selected.measured` — the classification
// of the rollout the playback follows — and never from `selected.plannedKind`,
// which is the generator's proposal. When the two differ, `plannedVsMeasured`
// says so in one line rather than the panel silently showing whichever is
// handier. What the shot then DID is `ui/shotOutcome.ts`, after the balls stop.

import type { DecisionTraceV1, MeasuredRoute, TracedCandidate } from "../ai/trace/contract";
import { measuredRouteLabel } from "../ai/measure/classify";

const KIND_LABEL: Record<string, string> = {
  direct: "direct pot",
  bank: "bank",
  "double-bank": "two-rail bank",
  combo: "combination",
  "rail-combo": "rail combination",
  "safety-kick": "safety off the cushion",
};

export const POCKET_NAME: Record<string, string> = {
  bl: "bottom-left",
  tl: "top-left",
  br: "bottom-right",
  tr: "top-right",
  sb: "bottom-side",
  st: "top-side",
};

const pocketName = (id: string | null): string | null =>
  id === null ? null : (POCKET_NAME[id] ?? id);

/**
 * The shot being played, named from the MEASUREMENT: the structure the rollout
 * executed, the ball the cue strikes, and the pocket the simulation drops the
 * ball into. `measuredRouteLabel` reads the measured cushion count, so a route
 * that runs three cushions is a "three-rail bank" and not the nearest word in
 * the generator's four-kind vocabulary.
 */
function describeMeasured(m: MeasuredRoute, target: number): string {
  const label = measuredRouteLabel(m);
  const pocket = pocketName(m.pocket);
  return pocket === null
    ? `a ${label} on the ${target}`
    : `a ${label} on the ${target}, into the ${pocket} pocket`;
}

/**
 * One line reconciling the generator's proposal with the rollout, or null when
 * they agree. This is the only place the panel prints the planned route, and it
 * always prints it beside the measured one.
 */
export function plannedVsMeasured(trace: DecisionTraceV1): string | null {
  const sel = trace.selected;
  if (!sel || sel.measured === null) return null;
  const chosen =
    sel.candidateIndex === null
      ? null
      : (trace.candidates.find((c) => c.index === sel.candidateIndex) ?? null);

  const planned = KIND_LABEL[sel.plannedKind] ?? sel.plannedKind;
  const measured = measuredRouteLabel(sel.measured);
  const plannedPocket = chosen ? pocketName(chosen.pocket) : null;
  const measuredPocket = pocketName(sel.measured.pocket);

  const kindDiffers = planned !== measured;
  const pocketDiffers =
    plannedPocket !== null && measuredPocket !== null && plannedPocket !== measuredPocket;
  if (!kindDiffers && !pocketDiffers) return null;
  if (kindDiffers && pocketDiffers) {
    return `Planned as a ${planned} into the ${plannedPocket} pocket; the simulation found a ${measured} into the ${measuredPocket}.`;
  }
  if (kindDiffers) return `Planned as a ${planned}; the simulation found a ${measured}.`;
  return `Planned into the ${plannedPocket} pocket; the simulation finds the ${measuredPocket}.`;
}

/**
 * "a bank on the 11" — short form, for the comparison clause.
 *
 * Named from the loser's OWN measurement when it has one. Every candidate this
 * clause can reach carries a rollout (`runnerUp` requires `physics`), and a
 * route that reached `lower-utility-than-selected` also cleared the measured
 * check, so the label is the structure its events executed rather than the one
 * the generator proposed for it.
 */
const shortForm = (c: TracedCandidate): string => {
  const label = c.measured?.trickVerified
    ? measuredRouteLabel(c.measured)
    : (KIND_LABEL[c.kind] ?? c.kind);
  return `a ${label} on the ${c.target}`;
};

/**
 * The strongest candidate the trace recorded as losing to the selected one.
 * `lower-utility-than-selected` is a label the search applied; "strongest" only
 * chooses which of the losers to name, and the claim made about it — that the
 * chosen shot was taken over it — is true of every member of that set.
 *
 * `chosenTarget` is the ball the shot being described is played on, and losers
 * on that same ball are excluded. The bar used to be the weaker "not the
 * identical short form", which let through
 * "a two-rail bank on the 7, into the bottom-side pocket, over a bank on the 7"
 * — observed twice. The two routes really are different, but a reader takes
 * "over a bank on the 7" as a second shot on a second ball, so the clause was
 * costing more than it was carrying. The comparison stays as specific as it
 * was: a loser on a *different* ball is named when one exists, and the clause
 * is dropped when none does. The sentence is never made vaguer to hide the
 * collision.
 */
export function runnerUp(
  trace: DecisionTraceV1,
  chosenTarget: number | null = null,
): TracedCandidate | null {
  const losers = trace.candidates.filter(
    (c) =>
      c.rejection === "lower-utility-than-selected" &&
      c.physics !== null &&
      c.target !== chosenTarget,
  );
  if (losers.length === 0) return null;
  return losers.reduce((a, b) =>
    (b.physics?.strength ?? 0) > (a.physics?.strength ?? 0) ? b : a,
  );
}

/**
 * The straight pot this decision turned down, or null when none was on.
 *
 * Purely a lookup: whether a pot was available at all is settled by
 * `trace/build.ts`, against the real board, before the trace is published.
 */
export function passedOverDirect(trace: DecisionTraceV1): TracedCandidate | null {
  const index = trace.selected?.passedOverDirectIndex ?? null;
  if (index === null) return null;
  return trace.candidates.find((c) => c.index === index) ?? null;
}

export interface ShotSentence {
  text: string;
  words: number;
}

export function shotSentence(trace: DecisionTraceV1): ShotSentence | null {
  const sel = trace.selected;
  if (!sel) return null;
  const chosen =
    sel.candidateIndex === null
      ? null
      : (trace.candidates.find((c) => c.index === sel.candidateIndex) ?? null);

  const head =
    sel.measured !== null && chosen
      ? describeMeasured(sel.measured, chosen.target)
      : // A generated safety has no candidate row and no trick classification;
        // it still has a real kind, verified against the ruleset.
        `a ${KIND_LABEL[sel.kind] ?? sel.kind}`;

  // The refusal outranks the comparison, and only one of the two is ever said.
  //
  // Turning down a pot that was on is the single most specific thing this
  // opponent does, and it is what the portfolio's copy promises a visitor will
  // see; naming which trick came second is the same claim's weaker cousin. Both
  // in one sentence pushed it past the width of the panel, so the stronger one
  // wins when there is one and the comparison covers the boards where no
  // straight pot existed to refuse.
  const refused = passedOverDirect(trace);
  if (refused) {
    const text = `Playing ${head} — turning down the straight pot on the ${refused.target}.`;
    return { text, words: wordsIn(text) };
  }

  const over = runnerUp(trace, chosen?.target ?? null);
  const text =
    over && over.index !== sel.candidateIndex
      ? `Playing ${head}, over ${shortForm(over)}.`
      : `Playing ${head}.`;

  return { text, words: wordsIn(text) };
}

export const wordsIn = (text: string): number => text.split(/\s+/).filter(Boolean).length;

/**
 * Why a safety rung was reached, when part of the answer is that nominal tricks
 * potted their ball by a route that was not the trick. Counting the trace's own
 * `planned-trick-not-measured` rows keeps the clause a restatement of a
 * rejection the policy recorded, rather than a second judgement of the same
 * candidates.
 */
function withRejectedTricks(trace: DecisionTraceV1, base: string): string {
  const n = trace.candidates.filter((c) => c.rejection === "planned-trick-not-measured").length;
  if (n === 0) return base;
  return n === 1
    ? `${base} — one route potted without the cushion it was planned around, so it is not a trick`
    : `${base} — ${n} routes potted without the cushion they were planned around, so none is a trick`;
}

/**
 * Why that shot, in the selection ladder's own terms. One string per rung;
 * the renderer never composes a reason of its own.
 *
 * The two rungs decided by the reliability bar quote the measurement that
 * cleared or missed it. They used to quote only the bar, which is a constant
 * (`TRICK_RELIABILITY_THRESHOLD`), so "the trick cleared the 0.50 reliability
 * bar" printed a character-identical line under eight consecutive plans and
 * carried no information about any of them. The two numbers now shown are
 * per-shot: `physics.strength` of the chosen candidate — the same quantity the
 * ladder compared against the bar — and how many tricks cleared it. Both are
 * measured; `strength` is a bounded transform of the rollout value and never a
 * probability, which is why it is set against the bar on the bar's own scale
 * and never rendered with a percent sign.
 */
export function rungText(trace: DecisionTraceV1): string | null {
  const sel = trace.selected;
  if (!sel) return null;
  const bar = sel.reliabilityThreshold.toFixed(2);
  const chosen =
    sel.candidateIndex === null
      ? null
      : (trace.candidates.find((c) => c.index === sel.candidateIndex) ?? null);
  // Null only if the ladder ever selects a candidate no simulation ran on,
  // which rungs 1 and 2 cannot: both require a verified pot. The clause is
  // dropped rather than guessed at.
  const measured = chosen?.physics ? chosen.physics.strength.toFixed(2) : null;
  const at = measured === null ? "" : `, at ${measured}`;
  switch (sel.rung) {
    case "trick-qualified":
      return sel.qualifyingTricks > 1
        ? `the strongest of ${sel.qualifyingTricks} tricks over the ${bar} reliability bar${at}`
        : `the only trick over the ${bar} reliability bar${at}`;
    case "trick-below-threshold":
      return measured === null
        ? `nothing cleared the ${bar} reliability bar, and this one still pots in simulation`
        : `nothing cleared the ${bar} reliability bar; at ${measured} this one still pots in simulation`;
    case "non-direct-safety":
      return withRejectedTricks(
        trace,
        "no trick was makeable here, so this is a safety off the cushion",
      );
    case "forced-legal-contact":
      // Rung 5 is reached by two different states of knowledge, and saying
      // "the shortest legal contact" for both asserted a legality nothing had
      // established — on a shot that scratched in 94 % of an adversarial
      // sample. `safetyQuality` is `pickSafety`'s own verdict and the sentence
      // now follows it.
      return sel.safetyQuality === "legal-contact-only"
        ? "the target is snookered; this kick fouls, but it does reach a legal ball"
        : "the target is snookered; nothing came back legal, so this is the shortest kick off the cushion";
    default:
      return null;
  }
}
