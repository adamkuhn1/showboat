import {
  type CandidateStat,
  type SearchResult,
  TRICK_RELIABILITY_THRESHOLD,
} from "../ai/shotSearch";

// ---------------------------------------------------------------------------
// The reasoning overlay.
//
// Rule for this file, enforced by review and by `overlayTruthfulness.test.ts`:
// everything rendered here is read straight off the `SearchResult` the AI
// actually decided with. There is no derived-for-display quantity, no
// animation that implies work happening after the search already finished, and
// no vocabulary the algorithm doesn't earn:
//
//   - not "MCTS": the search is a flat UCB bandit over candidates, so it says
//     "physics search".
//   - not "win probability" and not "AI confidence": nothing here predicts
//     game outcomes, so nothing here claims to.
//   - no fabricated natural-language reasoning. The last line is
//     `trace.selectionReason`, emitted by the selection function itself.
//
// It is also, deliberately, three sentences long. It used to be a ranked table
// of eight candidates carrying a model-internal enumeration index, two
// unlabelled decimals set flush together, per-row bars several of which
// rendered empty, five saturated hues, and raw pocket enum keys. Every number
// in it was true and none of it was legible. What survives is the part a
// player can actually use: that there were alternatives, which shot is being
// played, and why that one.
// ---------------------------------------------------------------------------

const KIND_LABEL: Record<string, string> = {
  direct: "direct pot",
  bank: "bank",
  "double-bank": "two-rail bank",
  combo: "combination",
  "rail-combo": "rail combination",
};

const POCKET_NAME: Record<string, string> = {
  bl: "bottom-left",
  tl: "top-left",
  br: "bottom-right",
  tr: "top-right",
  sb: "bottom-side",
  st: "top-side",
};

export const SELECTION_TEXT: Record<string, string> = {
  "trick-qualified": `the trick cleared the ${TRICK_RELIABILITY_THRESHOLD.toFixed(2)} reliability bar`,
  "no-trick-qualified": "no trick cleared the reliability bar, so this is the best verified pot",
  "no-verified-pot": "nothing potted in simulation, so this is the best available",
  none: "no candidate survived physics verification",
};

export interface ModelBadge {
  /** What is configured to run. The rendered mode always comes from the trace. */
  mode: "classical" | "neural-hybrid";
  /** Set when the neural path was requested but the artifact could not load. */
  fallbackReason?: string;
  /** Whether the artifact's sha256 was verified against the manifest. */
  hashVerified?: boolean;
}

/** "a bank on the 3, into the top-side pocket" — no key required to read it. */
function describeCandidate(c: CandidateStat["candidate"]): string {
  const kind = KIND_LABEL[c.kind] ?? c.kind;
  const pocket = POCKET_NAME[c.pocket] ?? c.pocket;
  return `a ${kind} on the ${c.target}, into the ${pocket} pocket`;
}

export function OverlayPanel({
  result,
  searching,
  stale = false,
  badge,
}: {
  result: SearchResult | null;
  /** True while the physics search is actually running for this turn. */
  searching: boolean;
  stale?: boolean;
  badge: ModelBadge;
}) {
  const trace = result?.trace;
  // Title reflects the mode of the decision actually on screen; the configured
  // mode is only used before there is a decision to describe.
  const mode = trace?.mode ?? badge.mode;
  const usedNeural = mode === "neural-hybrid" && !trace?.fallbackReason;
  const title = usedNeural ? "Neural evaluator and physics search" : "Physics search";

  if (!result || result.stats.length === 0) {
    // Reachable in real play only when a search actually ran and found
    // nothing to evaluate (e.g. no legal target) — the app doesn't mount this
    // panel at all before the opponent's first turn (see App.tsx), so there
    // is no "nothing has happened yet" placeholder to write here.
    return (
      <aside className="overlay">
        <p className="overlay-title">{title}</p>
        <p className="overlay-empty">
          {searching ? "searching…" : trace?.fallbackReason ?? "no shot to evaluate this turn"}
        </p>
        {badge.fallbackReason && (
          <p className="overlay-warn">classical fallback: {badge.fallbackReason}</p>
        )}
      </aside>
    );
  }

  const best = result.best;
  const considered = result.stats.length;

  return (
    <aside className="overlay" style={stale ? { opacity: 0.45 } : undefined}>
      <p className="overlay-title">
        {title}
        {stale && <span className="overlay-meta">last turn</span>}
      </p>

      {trace?.fallbackReason && !usedNeural && (
        <p className="overlay-warn">classical fallback: {trace.fallbackReason}</p>
      )}

      {/* What the panel says, and all it says: that alternatives existed, which
          one is being played, and why that one. The ranked table that used to
          sit here printed a model-internal enumeration index beside an already
          sorted list and two unlabelled decimals flush together; it was a
          scoreboard nobody could read. The alternatives themselves are already
          drawn on the felt as dashed lines, which is the legible version of
          the same fact. */}
      <p className="overlay-line">
        {considered === 1
          ? "One shot survived physics verification this turn."
          : `${considered} shots survived physics verification this turn.`}
      </p>

      {best && <p className="overlay-line chosen-why">Playing {describeCandidate(best.candidate)}.</p>}

      {trace?.selectionReason && (
        <p className="overlay-line overlay-reason">{SELECTION_TEXT[trace.selectionReason]}</p>
      )}
    </aside>
  );
}
