import {
  type CandidateStat,
  type SearchResult,
  TRICK_RELIABILITY_THRESHOLD,
} from "../ai/shotSearch";

// ---------------------------------------------------------------------------
// The reasoning overlay.
//
// Rule for this file, enforced by review and by `overlayTruthfulness.test.ts`:
// every number and label rendered here is read straight off the `SearchResult`
// the AI actually decided with. There is no derived-for-display quantity, no
// animation that implies work happening after the search already finished, and
// no vocabulary the algorithm doesn't earn:
//
//   - not "MCTS": the search is a flat UCB bandit over candidates, so it says
//     "physics search".
//   - not "win probability": nothing here predicts game outcomes. `strength`
//     is a bounded monotonic transform of the physics rollout value and is
//     shown as a plain decimal.
//   - not "AI confidence": the model's output IS a calibrated probability of a
//     legal pot under execution noise (ECE measured pre/post-Platt on val AND
//     test — see the manifest's calibration_evidence), so it is labelled
//     exactly that, "make est.", and nothing more.
//   - no fabricated natural-language reasoning. The one sentence at the bottom
//     is `trace.selectionReason`, emitted by the selection function itself.
// ---------------------------------------------------------------------------

const isTrickShot = (s: CandidateStat): boolean =>
  s.candidate.banks >= 2 || s.candidate.kind === "combo" || s.candidate.kind === "rail-combo";

const KIND_LABEL: Record<string, string> = {
  direct: "direct",
  bank: "bank",
  "double-bank": "2-rail",
  combo: "combo",
  "rail-combo": "rail+combo",
};

const POCKET_SHORT: Record<string, string> = {
  bl: "BL", tl: "TL", br: "BR", tr: "TR", sb: "SB", st: "ST",
};

export const SELECTION_TEXT: Record<string, string> = {
  "trick-qualified": `trick cleared the ${TRICK_RELIABILITY_THRESHOLD.toFixed(2)} reliability bar`,
  "no-trick-qualified": "no trick cleared the reliability bar — best verified pot",
  "no-verified-pot": "nothing potted in simulation — best available",
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

export function OverlayPanel({
  result,
  thinking,
  stale = false,
  badge,
}: {
  result: SearchResult | null;
  thinking: boolean;
  stale?: boolean;
  badge: ModelBadge;
}) {
  const trace = result?.trace;
  // Title reflects the mode of the decision actually on screen; the configured
  // mode is only used before there is a decision to describe.
  const mode = trace?.mode ?? badge.mode;
  const usedNeural = mode === "neural-hybrid" && !trace?.fallbackReason;
  const title = usedNeural ? "Neural evaluator + physics search" : "Physics search";

  if (!result || result.stats.length === 0) {
    return (
      <aside className="overlay">
        <div className="overlay-header">
          <span className="overlay-title">{title}</span>
        </div>
        <p className="overlay-empty">
          {thinking ? "searching…" : "candidates appear here on the opponent's turn"}
        </p>
        {badge.fallbackReason && (
          <p className="overlay-warn">classical fallback — {badge.fallbackReason}</p>
        )}
      </aside>
    );
  }

  const top = result.stats.slice(0, 8);
  const best = result.best;
  const maxVisits = Math.max(1, ...top.map((s) => s.visits));

  return (
    <aside className="overlay" style={stale ? { opacity: 0.45 } : undefined}>
      <div className="overlay-header">
        <span className="overlay-title">{title}</span>
        {stale && <span className="overlay-meta">prev. turn</span>}
      </div>

      {trace && (
        <div className="stage-list">
          <div className="stage">
            <span className="stage-name">candidates generated</span>
            <span className="stage-val">{trace.candidatesGenerated}</span>
          </div>
          {usedNeural ? (
            <>
              <div className="stage">
                <span className="stage-name">learned ranking</span>
                <span className="stage-val">
                  {trace.candidatesGenerated} scored
                  {trace.neuralInferenceMs !== undefined &&
                    ` · ${trace.neuralInferenceMs.toFixed(1)}ms`}
                </span>
              </div>
              <div className="stage">
                <span className="stage-name">pruned before physics</span>
                <span className="stage-val">{trace.prunedByPrior}</span>
              </div>
            </>
          ) : (
            trace.fallbackReason && (
              <div className="stage stage-warn">
                <span className="stage-name">classical fallback</span>
                <span className="stage-val">{trace.fallbackReason}</span>
              </div>
            )
          )}
          <div className="stage">
            <span className="stage-name">physics-verified</span>
            <span className="stage-val">
              {trace.physicsVerified} · {trace.legalPots} legal pot{trace.legalPots === 1 ? "" : "s"}
            </span>
          </div>
          <div className="stage">
            <span className="stage-name">scratched in sim</span>
            <span className="stage-val">{trace.scratched}</span>
          </div>
          <div className="stage">
            <span className="stage-name">physics calls spent</span>
            <span className="stage-val">{trace.physicsCalls}</span>
          </div>
        </div>
      )}

      <div className="cand-list">
        <div className="cand-head">
          <span>shot</span>
          <span>{usedNeural ? "make est. · strength" : "strength"}</span>
        </div>
        {top.map((s, i) => {
          const isBest = s === best;
          const strengthPct = Math.round(s.strength * 100);
          const kind = KIND_LABEL[s.candidate.kind] ?? s.candidate.kind;
          const pocket = POCKET_SHORT[s.candidate.pocket] ?? s.candidate.pocket;
          const isTrick = isTrickShot(s);
          const visitShare = s.visits / maxVisits;

          return (
            <div key={i} className={`cand-row ${isBest ? "cand-chosen" : ""}`}>
              <div className="cand-shot">
                <span className={`cand-kind kind-${s.candidate.kind.replace("-", "")}`}>
                  {kind}·{s.candidate.target}
                </span>
                <span className="cand-pocket">{pocket}</span>
                {isTrick && <span className="cand-star">★</span>}
                {s.priorRank !== undefined && (
                  <span className="cand-rank" title="rank the learned model gave this candidate">
                    #{s.priorRank}
                  </span>
                )}
              </div>
              <div className="cand-bar-wrap">
                <div
                  className="cand-bar"
                  style={{ width: `${strengthPct}%`, opacity: 0.5 + visitShare * 0.5 }}
                />
                <span className="cand-score">
                  {s.priorScore !== undefined && (
                    <span className="cand-prior" title="model's calibrated legal-pot estimate">
                      {s.priorScore.toFixed(2)}
                    </span>
                  )}
                  {s.strength.toFixed(2)}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {best && (
        <div className="chosen-callout">
          <span className="chosen-label">playing</span>
          <span className={`chosen-kind kind-${best.candidate.kind.replace("-", "")}`}>
            {KIND_LABEL[best.candidate.kind] ?? best.candidate.kind}·{best.candidate.target}
          </span>
          <span className="chosen-arrow">→</span>
          <span className="chosen-pocket">
            {POCKET_SHORT[best.candidate.pocket] ?? best.candidate.pocket}
          </span>
          {isTrickShot(best) && <span className="cand-star">★</span>}
          <span className="chosen-score">
            strength {best.strength.toFixed(2)} · style {best.styleScore}
          </span>
        </div>
      )}

      {trace?.selectionReason && (
        <p className="chosen-why">{SELECTION_TEXT[trace.selectionReason]}</p>
      )}

      {usedNeural && trace?.modelId && (
        <p className="overlay-foot">
          {trace.modelId}
          {badge.hashVerified === false && " · hash unverified (no WebCrypto)"}
        </p>
      )}
    </aside>
  );
}
