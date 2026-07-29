import { useRef, useState, useEffect, useCallback } from "react";
import { makeGame, takeShot, placeCueBall, cloneState } from "./game/game";
import { CUE_ID } from "./game/rack";
import type { GameState, PlayerId } from "./game/state";
import type { CueAction } from "./physics/cue";
import { applyCue } from "./physics/cue";
import { computeView, render, drawAim } from "./render/renderer";
import { drawOverlay } from "./render/overlay";
import { stepWorld } from "./render/animate";
import { describeShot } from "./ai/trace";
import { BALL_RADIUS } from "./physics/constants";
import { initPhysics, simulateShotWasm } from "./physics/wasm-bridge";
import { planTurn } from "./ai/turn";
import { getBrain, brainLabel } from "./ai/brain";
import { tryLoadModel } from "./ai/onnx";
import type { SearchResult } from "./ai/mcts";
import { OverlayPanel } from "./ui/OverlayPanel";

const CANVAS_W = 900;
const CANVAS_H = 500;

// Portfolio embed contract (apps/portfolio/src/lib/embedProtocol.ts): posting
// `ready` makes the shell crossfade its loading veil out. Sent once the engine
// outcome is known — loaded or failed, both are painted, final states (the
// failure message is the app's honest UI, better shown than veiled). No-op
// when running standalone.
function postEmbedReady() {
  if (window.parent === window) return;
  window.parent.postMessage(
    { source: "portfolio-embed", type: "ready", id: "showboat" },
    "*",
  );
}

// Player 1 (id 0) is human; Player 2 (id 1) is the AI opponent.
const AI_PLAYER: PlayerId = 1;

type Phase = "aiming" | "thinking" | "animating";

export default function App() {
  const game = useRef(makeGame());
  const [state, setState] = useState<GameState>(game.current.state);
  const [phase, setPhase] = useState<Phase>("aiming");
  const [aim, setAim] = useState(0);
  const [power, setPower] = useState(0.6);
  const [side, setSide] = useState(0);
  const [top, setTop] = useState(0);
  const [message, setMessage] = useState("Loading physics engine…");
  const [engineReady, setEngineReady] = useState(false);
  const [vsAI, setVsAI] = useState(true);
  const [search, setSearch] = useState<SearchResult | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const table = game.current.table;
  const view = computeView(CANVAS_W, CANVAS_H, table);

  useEffect(() => {
    Promise.all([initPhysics(), tryLoadModel()])
      .then(() => {
        setEngineReady(true);
        setMessage("Player 1 to break.");
      })
      .catch(() => setMessage("Failed to load the WASM physics engine."))
      .finally(postEmbedReady);
  }, []);

  const paint = useCallback(
    (s: GameState, overlay?: SearchResult | null) => {
      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx) return;
      render(ctx, s, table, view);
      if (overlay && showOverlay) drawOverlay(ctx, overlay, table, view);
      if (phase === "aiming" && s.turn !== AI_PLAYER) {
        const cue = s.balls.find((b) => b.id === CUE_ID);
        if (cue && !cue.pocketed) drawAim(ctx, cue, aim, power, view);
      }
    },
    [table, view, phase, aim, power, showOverlay],
  );

  useEffect(() => {
    paint(state, phase === "thinking" ? search : null);
  }, [state, aim, power, phase, search, paint]);

  // Animate a shot in real time, then commit the authoritative WASM outcome.
  const animateAndCommit = useCallback(
    (fromState: GameState, action: CueAction, report: ReturnType<typeof takeShot>) => {
      setPhase("animating");
      const anim = cloneState(fromState);
      const cue = anim.balls.find((b) => b.id === CUE_ID)!;
      applyCue(cue, action);
      let last = performance.now();
      const tick = (now: number) => {
        const dt = Math.min((now - last) / 1000, 0.05);
        last = now;
        const moving = stepWorld(anim.balls, table, dt * 1.4);
        setState({ ...anim });
        paint(anim, null);
        if (moving) requestAnimationFrame(tick);
        else commitShot(report);
      };
      requestAnimationFrame(tick);
    },
    [table, paint],
  );

  const commitShot = useCallback(
    (report: ReturnType<typeof takeShot>) => {
      setState(report.next);
      setSearch(null);
      setPhase("aiming");
      game.current.state = report.next;
      const o = report.outcome;
      const trace = describeShot(report.sim);
      let msg = "";
      if (o.gameOver) {
        msg = `Player ${(o.winner ?? 0) + 1} wins!${o.foul ? " (" + o.foulReason + ")" : ""}`;
      } else if (o.foul) {
        msg = `Foul: ${o.foulReason}. Player ${report.next.turn + 1} — ball in hand.`;
      } else if (o.assignedGroups) {
        msg = `Groups set. ${trace}`;
      } else if (o.turnPasses) {
        msg = `Player ${report.next.turn + 1}'s turn. ${trace}`;
      } else {
        msg = `Continue. ${trace}`;
      }
      setMessage(msg);
    },
    [],
  );

  // AI turn: plan (real search), show the overlay + a short "thinking" pause,
  // then play the chosen shot. Runs whenever it becomes the AI's move.
  useEffect(() => {
    if (!vsAI || !engineReady) return;
    if (state.turn !== AI_PLAYER || state.winner !== null || phase !== "aiming") return;

    let cancelled = false;
    setPhase("thinking");
    setMessage("Opponent is thinking…");

    // Ball-in-hand for the AI: place the cue at a simple legal spot (centre of
    // the head area) before searching.
    let planState = state;
    if (state.ballInHand !== false) {
      planState = placeCueBall(state, -table.length / 4, 0);
      setState(planState);
    }

    // Defer to next frame so the "thinking" UI paints first.
    const t = setTimeout(() => {
      if (cancelled) return;
      const brain = getBrain();
      const result = brain.plan(planState, table, planTurn);
      setSearch(result);
      paint(planState, result);

      // Hold the overlay briefly so the reasoning is visible, then shoot.
      setTimeout(() => {
        if (cancelled) return;
        if (!result.best) {
          // No makeable shot found: play a safe soft shot toward legal targets.
          const fallback: CueAction = { phi: Math.random() * Math.PI * 2, power: 0.3, sideSpin: 0, topSpin: 0 };
          const report = takeShot(planState, table, fallback, simulateShotWasm);
          animateAndCommit(planState, fallback, report);
          return;
        }
        const action = result.best.candidate.action;
        const report = takeShot(planState, table, action, simulateShotWasm);
        animateAndCommit(planState, action, report);
      }, 1100);
    }, 30);

    return () => {
      cancelled = true;
      clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.turn, state.winner, phase, vsAI, engineReady]);

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (phase !== "aiming" || (vsAI && state.turn === AI_PLAYER)) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const cue = state.balls.find((b) => b.id === CUE_ID);
    if (!cue || cue.pocketed) return;
    const cx = view.offsetX + cue.pos.x * view.scale;
    const cy = view.offsetY - cue.pos.y * view.scale;
    setAim(Math.atan2(-(my - cy), mx - cx));
  };

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (state.ballInHand === false || phase !== "aiming") return;
    if (vsAI && state.turn === AI_PLAYER) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const wx = (mx - view.offsetX) / view.scale;
    const wy = -(my - view.offsetY) / view.scale;
    const hx = table.length / 2 - BALL_RADIUS;
    const hy = table.width / 2 - BALL_RADIUS;
    const cx = Math.max(-hx, Math.min(hx, wx));
    const cy = Math.max(-hy, Math.min(hy, wy));
    setState(placeCueBall(state, cx, cy));
    setMessage("Cue ball placed. Take your shot.");
  };

  const shoot = () => {
    if (phase !== "aiming" || state.winner !== null) return;
    if (vsAI && state.turn === AI_PLAYER) return;
    if (state.ballInHand !== false) {
      setMessage("Place the cue ball first (click the table).");
      return;
    }
    const action: CueAction = { phi: aim, power, sideSpin: side, topSpin: top };
    const report = takeShot(state, table, action, simulateShotWasm);
    animateAndCommit(state, action, report);
  };

  const reset = () => {
    game.current = makeGame();
    setState(game.current.state);
    setSearch(null);
    setPhase("aiming");
    setMessage("New rack. Player 1 to break.");
  };

  const grp = state.groups[state.turn];
  const aiTurn = vsAI && state.turn === AI_PLAYER;

  return (
    <main className="shell">
      <header className="topbar">
        <h1>Showboat</h1>
        <p className="tag">
          2D bar pool · {vsAI ? `you vs ${brainLabel()}` : "human vs human"}
        </p>
      </header>

      <div className="layout">
        <div className="board">
          <canvas
            ref={canvasRef}
            width={CANVAS_W}
            height={CANVAS_H}
            onMouseMove={onMouseMove}
            onClick={onClick}
            className="table"
          />
          <div className="status">
            <span className={`turn p${state.turn}`}>
              {state.turn === AI_PLAYER && vsAI ? "Opponent" : `Player ${state.turn + 1}`}
              {grp ? ` · ${grp}` : " · open table"}
              {state.ballInHand !== false ? " · ball in hand" : ""}
            </span>
            <span className="msg">{message}</span>
          </div>
        </div>

        {vsAI && (
          <OverlayPanel result={search} thinking={phase === "thinking"} />
        )}
      </div>

      <div className="controls">
        <label>
          Power <span>{Math.round(power * 100)}%</span>
          <input type="range" min={0.05} max={1} step={0.01} value={power}
            onChange={(e) => setPower(Number(e.target.value))} disabled={phase !== "aiming" || aiTurn} />
        </label>
        <label>
          Side (english) <span>{side.toFixed(2)}</span>
          <input type="range" min={-1} max={1} step={0.05} value={side}
            onChange={(e) => setSide(Number(e.target.value))} disabled={phase !== "aiming" || aiTurn} />
        </label>
        <label>
          Draw / Follow <span>{top.toFixed(2)}</span>
          <input type="range" min={-1} max={1} step={0.05} value={top}
            onChange={(e) => setTop(Number(e.target.value))} disabled={phase !== "aiming" || aiTurn} />
        </label>
        <div className="buttons">
          <button
            onClick={shoot}
            disabled={phase !== "aiming" || state.winner !== null || !engineReady || aiTurn}
          >
            {phase === "animating" ? "Rolling…" : phase === "thinking" ? "Thinking…" : "Shoot"}
          </button>
          <button onClick={reset} className="secondary">New rack</button>
          <label className="toggle">
            <input type="checkbox" checked={vsAI} onChange={(e) => setVsAI(e.target.checked)} />
            vs AI
          </label>
          <label className="toggle">
            <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} />
            overlay
          </label>
        </div>
      </div>
      <p className="hint">
        The overlay shows the opponent's real search: candidate aiming paths (direct,
        bank, combo), per-candidate win-prob and MCTS visit counts, and the shot's
        event trace — all from actual decision data. No jump or massé shots exist; the
        cue action has no elevation axis by construction.
      </p>
    </main>
  );
}
