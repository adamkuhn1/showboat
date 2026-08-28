import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { makeGame, placeCueBall, takeShot, type ShotReport } from "./game/game";
import { CUE_ID } from "./game/rack";
import type { GameState } from "./game/state";
import type { CueAction } from "./physics/cue";
import { BALL_RADIUS } from "./physics/constants";
import type { SimResult } from "./physics/engine";
import {
  computeView,
  drawAim,
  drawRoute,
  render,
  renderPoses,
  type ViewTransform,
} from "./render/renderer";
import { playShot, type PlaybackHandle } from "./render/playback";
import { describeShot } from "./ai/trace";
import { aiTakeTurn, type EvaluatedCandidate } from "./ai/agent";
import { makeRanker } from "./ai/ranker";
import ThinkingPanel, { type ThinkingState } from "./ThinkingPanel";

const CANVAS_W = 900;
const CANVAS_H = 500;
const HUMAN = 0 as const;
const AI = 1 as const;

type Phase = "aiming" | "animating";

// Route polyline of one ball extracted from the chosen simulation's frames —
// the verified route is literally the path the committed simulation takes.
const routeFromFrames = (sim: SimResult, ballId: number) => {
  const pts: { x: number; y: number }[] = [];
  for (const f of sim.frames ?? []) {
    const b = f.balls.find((x) => x.id === ballId);
    if (!b || b.pocketed) break;
    const last = pts[pts.length - 1];
    if (!last || Math.hypot(b.x - last.x, b.y - last.y) > 0.012) {
      pts.push({ x: b.x, y: b.y });
    }
  }
  return pts;
};

export default function App() {
  const game = useRef(makeGame());
  // ?ranker=classical|neural forces one or the other (useful to compare);
  // the default without the param reflects training/evaluate.ts's held-out
  // gate result (see src/ai/ranker.ts).
  const ranker = useRef(
    makeRanker(
      (new URLSearchParams(window.location.search).get("ranker") as
        | "classical"
        | "neural"
        | null) ?? undefined,
    ),
  );
  const [state, setState] = useState<GameState>(game.current.state);
  const [phase, setPhase] = useState<Phase>("aiming");
  const [aim, setAim] = useState(0);
  const [power, setPower] = useState(0.6);
  const [side, setSide] = useState(0);
  const [top, setTop] = useState(0);
  const [slow, setSlow] = useState(false);
  const [message, setMessage] = useState("You break. Aim with the mouse, then Shoot.");
  const [thinking, setThinking] = useState<ThinkingState>({
    active: false,
    ranker: ranker.current.name,
    phase: "",
  });
  // Bumped when an AI turn fully finishes so the effect re-checks whether the
  // AI still holds the table (it keeps shooting after a legal pot).
  const [aiTick, setAiTick] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const playbackRef = useRef<PlaybackHandle | null>(null);
  const aiAbortRef = useRef<{ cancelled: boolean } | null>(null);
  const aiRunningRef = useRef(false);
  const slowRef = useRef(slow);
  slowRef.current = slow;

  const table = game.current.table;
  // Stable identity matters: paint() must not be recreated on unrelated
  // renders (thinking updates), or its effect would repaint the base table
  // over the AI's candidate-route overlay.
  const view = useMemo(() => computeView(CANVAS_W, CANVAS_H, table), [table]);

  const ctx = (): CanvasRenderingContext2D | null =>
    canvasRef.current?.getContext("2d") ?? null;

  const paint = useCallback(
    (s: GameState) => {
      const c = ctx();
      if (!c) return;
      render(c, s, table, view);
      if (phase === "aiming" && s.turn === HUMAN) {
        const cue = s.balls.find((b) => b.id === CUE_ID);
        if (cue && !cue.pocketed) drawAim(c, cue, aim, power, view, table);
      }
    },
    [table, view, phase, aim, power],
  );

  useEffect(() => {
    if (phase !== "animating") paint(state);
  }, [state, aim, power, phase, paint]);

  // --- shared playback + commit ---------------------------------------------
  const playAndCommit = useCallback(
    (
      report: ShotReport,
      overlay?: (c: CanvasRenderingContext2D, v: ViewTransform) => void,
    ): Promise<void> => {
      setPhase("animating");
      const handle = playShot(report.sim, slowRef.current ? 0.45 : 1, (poses) => {
        const c = ctx();
        if (!c) return;
        renderPoses(c, poses, table, view);
        overlay?.(c, view);
      });
      playbackRef.current = handle;
      return handle.done.then(() => {
        playbackRef.current = null;
        game.current.state = report.next;
        setState(report.next);
        setPhase("aiming");
      });
    },
    [table, view],
  );

  const outcomeMessage = (report: ShotReport, shooter: number): string => {
    const o = report.outcome;
    const trace = describeShot(report.sim);
    const who = (p: number): string => (p === HUMAN ? "You" : "The AI");
    if (o.gameOver) {
      return `${who(o.winner ?? 0)} win${o.winner === AI ? "s" : ""} the rack!${
        o.foul ? ` (${o.foulReason})` : ""
      }`;
    }
    if (o.foul) {
      return `Foul by ${who(shooter).toLowerCase()}: ${o.foulReason}. ${
        report.next.turn === HUMAN ? "You have" : "AI has"
      } ball in hand.`;
    }
    const next =
      report.next.turn === shooter
        ? shooter === HUMAN
          ? "You continue."
          : "AI continues."
        : report.next.turn === HUMAN
          ? "Your turn."
          : "AI's turn.";
    return `${next} ${trace}`;
  };

  // --- human input ----------------------------------------------------------
  const humanTurn = phase === "aiming" && state.turn === HUMAN && state.winner === null;
  // Aiming is press-and-drag, not hover-follow: the aim only updates while
  // the pointer is held down on the table, so moving the mouse around to look
  // at the table (or to reach a slider) never disturbs a prepared shot.
  const draggingRef = useRef(false);

  const aimFromEvent = (e: React.PointerEvent<HTMLCanvasElement>): { x: number; y: number; inBounds: boolean } => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const inBounds = x >= 0 && x <= rect.width && y >= 0 && y <= rect.height;
    return { x, y, inBounds };
  };

  const updateAimTo = (x: number, y: number, cue: { pos: { x: number; y: number } }) => {
    const cx = view.offsetX + cue.pos.x * view.scale;
    const cy = view.offsetY - cue.pos.y * view.scale;
    setAim(Math.atan2(-(y - cy), x - cx));
  };

  // Clicking ONLY places the cue ball during ball-in-hand; shooting is the
  // explicit button. Aim and shoot stay separate, deliberate actions.
  const placeCueAt = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const wx = (e.clientX - rect.left - view.offsetX) / view.scale;
    const wy = -(e.clientY - rect.top - view.offsetY) / view.scale;
    const hx = table.length / 2 - BALL_RADIUS;
    const hy = table.width / 2 - BALL_RADIUS;
    const nx = Math.max(-hx, Math.min(hx, wx));
    const ny = Math.max(-hy, Math.min(hy, wy));
    const placed = placeCueBall(state, nx, ny);
    game.current.state = placed;
    setState(placed);
    setMessage("Cue ball placed. Take your shot.");
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!humanTurn) return;
    if (state.ballInHand !== false) {
      placeCueAt(e);
      return;
    }
    const cue = state.balls.find((b) => b.id === CUE_ID);
    if (!cue || cue.pocketed) return;
    draggingRef.current = true;
    // Pointer capture keeps move/up events coming to this canvas even if the
    // drag continues past its edges, so releasing off-table still freezes
    // the aim cleanly instead of leaving the drag "stuck" open.
    canvasRef.current?.setPointerCapture(e.pointerId);
    const { x, y } = aimFromEvent(e);
    updateAimTo(x, y, cue);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!humanTurn || !draggingRef.current) return;
    const cue = state.balls.find((b) => b.id === CUE_ID);
    if (!cue || cue.pocketed) return;
    const { x, y, inBounds } = aimFromEvent(e);
    // Leaving the table surface stops the aim from updating (it freezes at
    // its last in-bounds value) without ending the drag -- coming back while
    // still held resumes it, same as a real cue stroke you can pause.
    if (inBounds) updateAimTo(x, y, cue);
  };

  const endDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    canvasRef.current?.releasePointerCapture(e.pointerId);
  };

  const shoot = () => {
    if (!humanTurn) return;
    if (state.ballInHand !== false) {
      setMessage("Place the cue ball first (click the table).");
      return;
    }
    setThinking({ active: false, ranker: ranker.current.name, phase: "" });
    const action: CueAction = { phi: aim, power, sideSpin: side, topSpin: top };
    // ONE simulation resolves the shot; its frames are the playback.
    const report = takeShot(state, table, action, { recordFrames: true });
    void playAndCommit(report).then(() => {
      setMessage(outcomeMessage(report, HUMAN));
    });
  };

  // --- AI turn --------------------------------------------------------------
  useEffect(() => {
    if (phase !== "aiming" || state.turn !== AI || state.winner !== null) return;
    if (aiRunningRef.current) return;
    aiRunningRef.current = true;
    const signal = { cancelled: false };
    aiAbortRef.current = signal;

    const paintCandidate = (base: GameState, ec: EvaluatedCandidate): void => {
      const c = ctx();
      if (!c) return;
      render(c, base, table, view);
      // Dashed = hypothetical geometry under consideration, not a result.
      drawRoute(c, ec.cand.cuePath, view, "hypothetical", "#f7f5ee");
      drawRoute(c, ec.cand.objectPath, view, "hypothetical", "#ffd166");
    };

    const run = async (): Promise<void> => {
      let base = state;
      setMessage("AI is thinking…");
      setThinking({ active: true, ranker: ranker.current.name, phase: "starting" });
      const move = await aiTakeTurn(state, table, ranker.current, {
        delayMs: 420,
        signal,
        onPhase: (p) => setThinking((t) => ({ ...t, phase: p })),
        onPlaced: (placed) => {
          base = placed;
          paint(placed);
        },
        onGenerated: (n, byKind) =>
          setThinking((t) => ({ ...t, generated: n, byKind })),
        onCandidate: (ec, index, total) => {
          setThinking((t) => ({ ...t, current: ec, index, total }));
          paintCandidate(base, ec);
        },
      });
      if (!move || signal.cancelled) return;

      // Publish the one decision; show the VERIFIED route of the chosen
      // simulation (solid), pause so it can be read, then play that exact
      // simulation back.
      setThinking((t) => ({ ...t, active: false, current: undefined, decision: move.decision }));
      const target =
        move.decision.selected?.cand.targetBall ?? move.report.sim.firstContact;
      const overlay = (c: CanvasRenderingContext2D, v: ViewTransform): void => {
        drawRoute(c, routeFromFrames(move.report.sim, CUE_ID), v, "verified", "#f7f5ee");
        if (target !== null) {
          drawRoute(c, routeFromFrames(move.report.sim, target), v, "verified", "#ffd166");
        }
      };
      const c = ctx();
      if (c) {
        render(c, move.stateAfterPlacement, table, view);
        overlay(c, view);
      }
      await new Promise((res) => setTimeout(res, 1500));
      if (signal.cancelled) return;

      await playAndCommit(move.report, overlay);
      if (signal.cancelled) return;
      setMessage(outcomeMessage(move.report, AI));
    };

    void run().finally(() => {
      aiRunningRef.current = false;
      // Re-run this effect: if the AI legally potted, it is still at the
      // table and must take the next shot.
      setAiTick((k) => k + 1);
    });
    // Cancellation is EXPLICIT (New rack) — not tied to effect cleanup, which
    // re-fires on every phase change and would cancel mid-playback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, state, aiTick]);

  const reset = () => {
    // Cancel any AI work and any animation; playback.stop() resolves its own
    // promise, so nothing stays pending.
    if (aiAbortRef.current) aiAbortRef.current.cancelled = true;
    playbackRef.current?.stop();
    playbackRef.current = null;
    aiRunningRef.current = false;
    draggingRef.current = false;
    game.current = makeGame();
    setState(game.current.state);
    setPhase("aiming");
    setThinking({ active: false, ranker: ranker.current.name, phase: "" });
    setMessage("New rack. You break.");
  };

  const grp = state.groups[state.turn];
  const turnLabel =
    state.winner !== null
      ? state.winner === HUMAN
        ? "You won"
        : "AI won"
      : state.turn === HUMAN
        ? "Your turn"
        : "AI's turn";

  return (
    <main className="shell">
      <header className="topbar">
        <h1>Showboat</h1>
        <p className="tag">2D bar pool · you vs a trick-shot opponent</p>
      </header>

      <div className="layout">
        <div className="board">
          <canvas
            ref={canvasRef}
            width={CANVAS_W}
            height={CANVAS_H}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            className="table"
            data-human={humanTurn || undefined}
          />
          <div className="status">
            <span className={`turn p${state.turn}`}>
              {turnLabel}
              {grp ? ` · ${grp}` : " · open table"}
              {state.ballInHand !== false && state.turn === HUMAN ? " · ball in hand" : ""}
            </span>
            <span className="msg">{message}</span>
          </div>
        </div>

        <ThinkingPanel t={thinking} />
      </div>

      <div className="controls">
        <label>
          Power <span>{Math.round(power * 100)}%</span>
          <input type="range" min={0.05} max={1} step={0.01} value={power}
            onChange={(e) => setPower(Number(e.target.value))} disabled={!humanTurn} />
        </label>
        <label>
          Side (english) <span>{side.toFixed(2)}</span>
          <input type="range" min={-1} max={1} step={0.05} value={side}
            onChange={(e) => setSide(Number(e.target.value))} disabled={!humanTurn} />
        </label>
        <label>
          Draw / Follow <span>{top.toFixed(2)}</span>
          <input type="range" min={-1} max={1} step={0.05} value={top}
            onChange={(e) => setTop(Number(e.target.value))} disabled={!humanTurn} />
        </label>
        <div className="buttons">
          <button onClick={shoot} disabled={!humanTurn}>
            {phase === "animating"
              ? "Rolling…"
              : state.turn === AI && state.winner === null
                ? "AI thinking…"
                : "Shoot"}
          </button>
          <button onClick={() => setSlow((s) => !s)} className="secondary">
            {slow ? "Playback: slow" : "Playback: normal"}
          </button>
          <button onClick={reset} className="secondary">New rack</button>
        </div>
      </div>
      <p className="hint">
        Dashed lines are candidate geometry the AI is still considering; solid
        lines are the verified route of the exact simulation you then watch.
        No jump or massé shots exist — the cue action has no elevation axis by
        construction.
      </p>
    </main>
  );
}
