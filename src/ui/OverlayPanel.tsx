// ---------------------------------------------------------------------------
// The reasoning panel.
//
// Rule for this file, enforced by `overlayTruthfulness.test.tsx`: everything
// rendered here is read straight off the `DecisionTraceV1` the opponent
// actually decided with. No derived-for-display quantity, no animation that
// implies work happening after the search finished, and no vocabulary the
// algorithm hasn't earned:
//
//   - not "MCTS": the search is a flat UCB bandit over candidates, so it says
//     "physics search".
//   - not "win probability" and not "confidence": nothing here predicts game
//     outcomes, so nothing here claims to.
//   - no fabricated natural-language reasoning. The last line is the selection
//     ladder's own rung, mapped one-to-one to a phrase.
//
// It is deliberately about five lines long, and it must stay that way: at the
// portfolio's ~1180 px embed the panel takes 288 px and the table gets the
// remaining ~74 %. That ratio is the floor. Which is also why the per-route
// elimination reasons are drawn on the felt beside the route they belong to
// rather than listed here.
// ---------------------------------------------------------------------------

import type { DecisionTraceV1 } from "../ai/trace/contract";
import { STATE_LABEL, isReasoning, type PresentationState } from "../render/presentation";
import { rungText, shotSentence } from "./shotSentence";

export interface ModelBadge {
  /** What is configured to run. The rendered mode always comes from the trace. */
  mode: "classical" | "neural-hybrid";
  /** Set when the neural path was requested but the artifact could not load. */
  fallbackReason?: string;
  /** Whether the artifact's sha256 was verified against the manifest. */
  hashVerified?: boolean;
}

export interface OverlayPanelProps {
  trace: DecisionTraceV1 | null;
  state: PresentationState;
  /** True between "the opponent is up" and the trace arriving. */
  planning: boolean;
  /** True while the onnxruntime session is being created for the first time. */
  modelLoading: boolean;
  badge: ModelBadge;
  /** Shown from the second opponent turn onward, never on the first. */
  showSkipHint: boolean;
  /** Shown on the first opponent turn of a session only. */
  showDisclosure: boolean;
  compare: { available: boolean; useNeural: boolean; onChange: (v: boolean) => void };
  /** Non-null only at SETTLED. Both replays are free — the trace is retained. */
  replay: { onDecision: () => void; onShot: () => void } | null;
}

export function OverlayPanel({
  trace,
  state,
  planning,
  modelLoading,
  badge,
  showSkipHint,
  showDisclosure,
  compare,
  replay,
}: OverlayPanelProps) {
  // The title names what decided *this* decision. Before there is one, it names
  // what is configured.
  const mode = trace?.mode ?? (badge.mode === "neural-hybrid" ? "neural-hybrid" : "classical-trick-only");
  const usedNeural = mode === "neural-hybrid" && !trace?.fallback;
  const title = usedNeural ? "Neural evaluator and physics search" : "Physics search";

  const sentence = trace ? shotSentence(trace) : null;
  const rung = trace ? rungText(trace) : null;
  const label = isReasoning(state) ? STATE_LABEL[state] : null;
  const decided =
    state === "SELECTED" ||
    state === "READY" ||
    state === "STROKE" ||
    state === "SHOOTING" ||
    state === "SETTLED";
  const fallbackDetail = trace?.fallback?.detail ?? badge.fallbackReason ?? null;

  return (
    <aside className="overlay">
      <p className="overlay-title">{title}</p>

      {/* The state slot. Only the five reasoning labels ever appear here. */}
      {label !== null && (
        <p className="overlay-state" data-state={state}>
          {label}
        </p>
      )}

      {planning && label === null && (
        <p className="overlay-state overlay-state--live">
          {modelLoading ? "loading the trained model…" : "searching…"}
        </p>
      )}

      {fallbackDetail && !usedNeural && (
        <p className="overlay-warn">classical fallback: {fallbackDetail}</p>
      )}

      {/* The plan is written when it has been made, not before. Showing it
          during ENUMERATING would give the answer away and make the SELECTED
          beat meaningless — the sequence would be narrating a conclusion the
          panel had already printed. */}
      {decided && sentence && <p className="overlay-line chosen-why">{sentence.text}</p>}
      {decided && rung && <p className="overlay-line overlay-reason">{rung}</p>}

      {showDisclosure && trace && (
        <>
          <p className="overlay-note">
            The search finishes before this plays; what you are watching is the decision it
            made, in the order it made it.
          </p>
          {/* The one distinction the felt has to get across, said once. A dashed
              route is geometry the opponent considered; the solid one is the
              route the simulation produces, so it is the shot you are about to
              see rather than a drawing of the shot it meant to take. */}
          <p className="overlay-note">
            Dashed routes are shots it weighed. The solid one is the path its simulation
            produces — the same run the score comes from.
          </p>
        </>
      )}

      {showSkipHint && isReasoning(state) && (
        <p className="overlay-note">press space to skip</p>
      )}

      {replay && (
        <p className="overlay-actions">
          <button type="button" className="linkish" onClick={replay.onDecision}>
            replay the decision
          </button>
          <button type="button" className="linkish" onClick={replay.onShot}>
            replay the shot
          </button>
        </p>
      )}

      {/* The comparison is one of the most interesting things in the project;
          it was sitting among the gameplay controls as an A/B research switch
          with a 40-word tooltip. Here it reads as what it is. The structurally
          shorter classical sequence (no "neural ranking" state at all) does the
          rest of the explaining. */}
      <label className={`overlay-compare${compare.available ? "" : " is-disabled"}`}>
        <input
          type="checkbox"
          checked={compare.useNeural && compare.available}
          disabled={!compare.available}
          onChange={(e) => compare.onChange(e.target.checked)}
        />
        rank with the trained model
      </label>
    </aside>
  );
}
