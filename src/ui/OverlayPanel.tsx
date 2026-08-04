import type { SearchResult } from "../ai/mcts";

const KIND_LABEL: Record<string, string> = {
  direct:       "direct",
  bank:         "bank",
  "double-bank":"2-rail",
  combo:        "combo",
};

const POCKET_SHORT: Record<string, string> = {
  bl: "BL", tl: "TL", br: "BR", tr: "TR", sb: "SB", st: "ST",
};

export function OverlayPanel({
  result,
  thinking,
  stale = false,
}: {
  result: SearchResult | null;
  thinking: boolean;
  stale?: boolean;
}) {
  if (!result || result.stats.length === 0) {
    return (
      <aside className="overlay">
        <div className="overlay-header">
          <span className="overlay-title">MCTS</span>
          {thinking && <span className="thinking-dots"><span /><span /><span /></span>}
        </div>
        <p className="overlay-empty">
          {thinking ? "running rollouts…" : "candidates appear here on AI turn"}
        </p>
      </aside>
    );
  }

  const top = result.stats.slice(0, 8);
  const best = result.best;
  const maxVisits = Math.max(1, ...top.map((s) => s.visits));

  return (
    <aside className="overlay" style={stale ? { opacity: 0.45 } : undefined}>
      <div className="overlay-header">
        <span className="overlay-title">MCTS</span>
        <span className="overlay-meta">
          {stale ? "prev · " : ""}{result.simulations} rollouts · {result.stats.length} cand
        </span>
      </div>

      <div className="cand-list">
        {top.map((s, i) => {
          const isBest = s === best;
          const winPct = Math.round(s.winProb * 100);
          const kind = KIND_LABEL[s.candidate.kind] ?? s.candidate.kind;
          const pocket = POCKET_SHORT[s.candidate.pocket] ?? s.candidate.pocket;
          const isTrick = s.candidate.banks >= 2 || s.candidate.kind === "combo";
          const visitShare = s.visits / maxVisits;

          return (
            <div key={i} className={`cand-row ${isBest ? "cand-chosen" : ""}`}>
              <div className="cand-shot">
                <span className={`cand-kind kind-${s.candidate.kind.replace("-", "")}`}>
                  {kind}·{s.candidate.target}
                </span>
                <span className="cand-pocket">{pocket}</span>
                {isTrick && <span className="cand-star">★</span>}
              </div>
              <div className="cand-bar-wrap">
                <div
                  className="cand-bar"
                  style={{ width: `${winPct}%`, opacity: 0.5 + visitShare * 0.5 }}
                />
                <span className="cand-pct">{winPct}</span>
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
          {(best.candidate.banks >= 2 || best.candidate.kind === "combo") && (
            <span className="cand-star">★</span>
          )}
          <span className="chosen-conf">{Math.round(best.winProb * 100)}%</span>
        </div>
      )}
    </aside>
  );
}
