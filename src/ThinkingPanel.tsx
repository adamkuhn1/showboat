import { type Decision, type EvaluatedCandidate } from "./ai/agent";

// The AI's live thinking display. Every string here is derived from the same
// Decision / EvaluatedCandidate objects the agent actually decided with —
// this component receives that data and formats it, nothing more. It shows:
// what kind of candidate is being considered, what the physics verification
// measured, and the selected shot with its reason. Deliberately compact: no
// raw telemetry, no invented probabilities, no candidate leaderboard.

export interface ThinkingState {
  active: boolean;
  ranker: "neural" | "classical";
  phase: string; // "placing" | "generating" | "verifying" | "selecting" | "done"
  generated?: number;
  byKind?: Record<string, number>;
  current?: EvaluatedCandidate;
  index?: number;
  total?: number;
  decision?: Decision;
}

const kindWord: Record<string, string> = {
  direct: "direct shot",
  bank: "bank",
  kick: "kick (cue off the rail)",
  combo: "combo",
};

const currentLine = (t: ThinkingState): string | null => {
  const c = t.current;
  if (!c || t.index === undefined) return null;
  const consider = `${t.index + 1}/${t.total}: ${kindWord[c.cand.kind]} on the ${
    c.cand.targetBall === 8 ? "8-ball" : `${c.cand.targetBall}-ball`
  }`;
  if (!c.measured) return consider;
  if (c.success && c.robustness) {
    return `${consider} — ${c.measured.label}, pots it (${c.robustness.successes}/${c.robustness.n} under jitter)`;
  }
  return `${consider} — ${
    c.measured.targetPotted ? "pots it but fouls" : "misses in simulation"
  }`;
};

export default function ThinkingPanel({ t }: { t: ThinkingState }) {
  if (!t.active && !t.decision) return null;

  const counts =
    t.byKind &&
    Object.entries(t.byKind)
      .filter(([, n]) => n > 0)
      .sort((a, b) => b[1] - a[1])
      .map(([k, n]) => `${n} ${k}`)
      .join(" · ");

  return (
    <aside className="thinking" aria-live="polite">
      <div className="thinking-head">
        <span className="dot" data-active={t.active || undefined} />
        Opponent — {t.ranker} ranker
      </div>

      {t.phase === "placing" && <p>Ball in hand — choosing cue position…</p>}

      {t.generated !== undefined && (
        <p className="counts">
          {t.generated} candidates ({counts})
        </p>
      )}

      {t.active && t.phase === "verifying" && (
        <p className="current">{currentLine(t) ?? "Verifying…"}</p>
      )}

      {t.decision && (
        <p className="reason">
          <strong>
            {t.decision.selected
              ? "Selected"
              : t.decision.phase === "break"
                ? "Break"
                : "Safety"}
            :
          </strong>{" "}
          {t.decision.reason}
        </p>
      )}
    </aside>
  );
}
