// The one sentence the panel exists to write: what the opponent is playing,
// and what it played it over.
//
// Every clause is a restatement of something the trace says. The "over …"
// clause names a candidate the trace itself labelled
// `lower-utility-than-selected` — i.e. one the search recorded as having lost
// to the chosen shot. Nothing here compares, ranks or scores anything on its
// own.

import type { DecisionTraceV1, TracedCandidate } from "../ai/trace/contract";

const KIND_LABEL: Record<string, string> = {
  direct: "direct pot",
  bank: "bank",
  "double-bank": "two-rail bank",
  combo: "combination",
  "rail-combo": "rail combination",
  "safety-kick": "safety off the cushion",
};

const POCKET_NAME: Record<string, string> = {
  bl: "bottom-left",
  tl: "top-left",
  br: "bottom-right",
  tr: "top-right",
  sb: "bottom-side",
  st: "top-side",
};

/** "a two-rail bank on the 3, into the top-side pocket" */
export function describeCandidate(c: {
  kind: string;
  target: number;
  pocket: string;
}): string {
  const kind = KIND_LABEL[c.kind] ?? c.kind;
  const pocket = POCKET_NAME[c.pocket] ?? c.pocket;
  return `a ${kind} on the ${c.target}, into the ${pocket} pocket`;
}

/** "a bank on the 11" — short form, for the comparison clause. */
const shortForm = (c: TracedCandidate): string =>
  `a ${KIND_LABEL[c.kind] ?? c.kind} on the ${c.target}`;

/**
 * The strongest candidate the trace recorded as losing to the selected one.
 * `lower-utility-than-selected` is a label the search applied; "strongest" only
 * chooses which of the losers to name, and the claim made about it — that the
 * chosen shot was taken over it — is true of every member of that set.
 */
export function runnerUp(trace: DecisionTraceV1): TracedCandidate | null {
  const losers = trace.candidates.filter(
    (c) => c.rejection === "lower-utility-than-selected" && c.physics !== null,
  );
  if (losers.length === 0) return null;
  return losers.reduce((a, b) => ((b.physics?.strength ?? 0) > (a.physics?.strength ?? 0) ? b : a));
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

  const head = chosen
    ? describeCandidate(chosen)
    : // A generated safety has no candidate row; it still has a real kind.
      `a ${KIND_LABEL[sel.kind] ?? sel.kind}`;

  const over = runnerUp(trace);
  const text =
    over && over.index !== sel.candidateIndex
      ? `Playing ${head}, over ${shortForm(over)}.`
      : `Playing ${head}.`;

  return { text, words: text.split(/\s+/).filter(Boolean).length };
}

/**
 * Why that shot, in the selection ladder's own terms. One string per rung;
 * the renderer never composes a reason of its own.
 */
export function rungText(trace: DecisionTraceV1): string | null {
  const sel = trace.selected;
  if (!sel) return null;
  const bar = sel.reliabilityThreshold.toFixed(2);
  switch (sel.rung) {
    case "trick-qualified":
      return `the trick cleared the ${bar} reliability bar`;
    case "trick-below-threshold":
      return "no trick cleared the reliability bar, and this one still pots in simulation";
    case "trick-attempt-no-verified-pot":
      return "nothing potted in simulation, so this is the best legal attempt";
    case "non-direct-safety":
      return "no trick was makeable here, so this is a safety off the cushion";
    case "forced-legal-contact":
      return "the target is snookered; this is the shortest legal contact";
    default:
      return null;
  }
}
