import type { SearchResult } from "../ai/mcts";

// Quantitative reasoning panel. Every row is REAL search output: the candidate
// kind (direct / bank / combo), the pocket, per-candidate win-prob (squashed
// rollout value), MCTS visit count, and the rails-before-pot measured from the
// actual simulated event trace. This is the "numbers are primary" half of the
// overlay; the ghost lines on the canvas are the geometric half. No decoration.

const kindTag: Record<string, string> = {
  direct: "direct",
  bank: "bank",
  combo: "combo",
};

export function OverlayPanel({
  result,
  thinking,
}: {
  result: SearchResult | null;
  thinking: boolean;
}) {
  if (!result || result.stats.length === 0) {
    return (
      <aside className="overlay">
        <h2>Opponent reasoning</h2>
        <p className="overlay-empty">
          {thinking ? "Searching…" : "The opponent's real search shows here on its turn."}
        </p>
      </aside>
    );
  }

  const top = result.stats.slice(0, 7);
  const best = result.best;

  return (
    <aside className="overlay">
      <h2>Opponent reasoning</h2>
      <p className="overlay-sub">
        {result.simulations.toLocaleString()} physics rollouts · {result.stats.length}{" "}
        candidate paths
      </p>
      <table className="cand-table">
        <thead>
          <tr>
            <th>shot</th>
            <th>→</th>
            <th>win</th>
            <th>visits</th>
            <th>rails</th>
          </tr>
        </thead>
        <tbody>
          {top.map((s, i) => (
            <tr key={i} className={s === best ? "chosen" : ""}>
              <td className={`kind ${s.candidate.kind}`}>
                {kindTag[s.candidate.kind]}·{s.candidate.target}
              </td>
              <td>{s.candidate.pocket}</td>
              <td>{Math.round(s.winProb * 100)}%</td>
              <td>{s.visits}</td>
              <td>{s.rails}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {best && (
        <p className="chosen-note">
          Playing <b>{best.candidate.kind}</b> on ball {best.candidate.target} →{" "}
          {best.candidate.pocket}
          {best.rails > 0 ? ` (${best.rails}-rail route)` : ""}. Chosen as the
          most-visited line — trick routes only appear here when the search values
          them, never scripted.
        </p>
      )}
    </aside>
  );
}
