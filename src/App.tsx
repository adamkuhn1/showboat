import { useCallback, useEffect, useRef, useState } from "react";
import { cloneState, makeGame, placeCueBall, takeShot } from "./game/game";
import { CUE_ID } from "./game/rack";
import type { GameState } from "./game/state";
import type { CueAction } from "./physics/cue";
import { BALL_RADIUS } from "./physics/constants";
import {
  computeView,
  drawAim,
  render,
  renderPoses,
} from "./render/renderer";
import { playShot, type PlaybackHandle } from "./render/playback";
import { describeShot } from "./ai/trace";

const CANVAS_W = 900;
const CANVAS_H = 500;

type Phase = "aiming" | "animating";

export default function App() {
  const game = useRef(makeGame());
  const [state, setState] = useState<GameState>(game.current.state);
  const [phase, setPhase] = useState<Phase>("aiming");
  const [aim, setAim] = useState(0);
  const [power, setPower] = useState(0.6);
  const [side, setSide] = useState(0);
  const [top, setTop] = useState(0);
  const [slow, setSlow] = useState(false);
  const [message, setMessage] = useState("Player 1 to break.");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const playbackRef = useRef<PlaybackHandle | null>(null);

  const table = game.current.table;
  const view = computeView(CANVAS_W, CANVAS_H, table);

  const paint = useCallback(
    (s: GameState) => {
      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx) return;
      render(ctx, s, table, view);
      if (phase === "aiming") {
        const cue = s.balls.find((b) => b.id === CUE_ID);
        if (cue && !cue.pocketed) drawAim(ctx, cue, aim, power, view);
      }
    },
    [table, view, phase, aim, power],
  );

  useEffect(() => {
    if (phase === "aiming") paint(state);
  }, [state, aim, power, phase, paint]);

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (phase !== "aiming") return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const cue = state.balls.find((b) => b.id === CUE_ID);
    if (!cue || cue.pocketed) return;
    const cx = view.offsetX + cue.pos.x * view.scale;
    const cy = view.offsetY - cue.pos.y * view.scale;
    setAim(Math.atan2(-(my - cy), mx - cx));
  };

  // Clicking the table ONLY places the cue ball during ball-in-hand. Aiming is
  // mouse-move, shooting is the explicit button — deliberately separate actions.
  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (state.ballInHand === false || phase !== "aiming") return;
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

  const commitShot = useCallback((report: ReturnType<typeof takeShot>) => {
    game.current.state = report.next;
    setState(report.next);
    setPhase("aiming");
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
  }, []);

  const shoot = () => {
    if (phase !== "aiming" || state.winner !== null) return;
    if (state.ballInHand !== false) {
      setMessage("Place the cue ball first (click the table).");
      return;
    }
    const action: CueAction = { phi: aim, power, sideSpin: side, topSpin: top };
    // ONE simulation resolves the shot; its recorded frames are what we play
    // back. There is no second simulation for display.
    const report = takeShot(state, table, action, { recordFrames: true });

    setPhase("animating");
    const preState = cloneState(state);
    const handle = playShot(report.sim, slow ? 0.45 : 1, (poses) => {
      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx) return;
      renderPoses(ctx, poses, table, view);
      void preState;
    });
    playbackRef.current = handle;
    void handle.done.then(() => {
      playbackRef.current = null;
      commitShot(report);
    });
  };

  const reset = () => {
    // Stop any in-flight animation first; stop() resolves its promise itself,
    // so nothing is left pending.
    playbackRef.current?.stop();
    playbackRef.current = null;
    game.current = makeGame();
    setState(game.current.state);
    setPhase("aiming");
    setMessage("New rack. Player 1 to break.");
  };

  const grp = state.groups[state.turn];
  return (
    <main className="shell">
      <header className="topbar">
        <h1>Showboat</h1>
        <p className="tag">2D bar pool · human vs human</p>
      </header>

      <div className="board">
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          onMouseMove={onMouseMove}
          onClick={onClick}
          className="table"
        />
      </div>

      <div className="status">
        <span className={`turn p${state.turn}`}>
          Player {state.turn + 1}
          {grp ? ` · ${grp}` : " · open table"}
          {state.ballInHand !== false ? " · ball in hand" : ""}
        </span>
        <span className="msg">{message}</span>
      </div>

      <div className="controls">
        <label>
          Power <span>{Math.round(power * 100)}%</span>
          <input type="range" min={0.05} max={1} step={0.01} value={power}
            onChange={(e) => setPower(Number(e.target.value))} disabled={phase !== "aiming"} />
        </label>
        <label>
          Side (english) <span>{side.toFixed(2)}</span>
          <input type="range" min={-1} max={1} step={0.05} value={side}
            onChange={(e) => setSide(Number(e.target.value))} disabled={phase !== "aiming"} />
        </label>
        <label>
          Draw / Follow <span>{top.toFixed(2)}</span>
          <input type="range" min={-1} max={1} step={0.05} value={top}
            onChange={(e) => setTop(Number(e.target.value))} disabled={phase !== "aiming"} />
        </label>
        <div className="buttons">
          <button onClick={shoot} disabled={phase !== "aiming" || state.winner !== null}>
            {phase === "animating" ? "Rolling…" : "Shoot"}
          </button>
          <button onClick={() => setSlow((s) => !s)} className="secondary">
            {slow ? "Speed: slow" : "Speed: normal"}
          </button>
          <button onClick={reset} className="secondary">New rack</button>
        </div>
      </div>
      <p className="hint">
        Move the mouse to aim, press Shoot to strike. No jump or massé shots
        exist — the cue action has no elevation axis by construction.
      </p>
    </main>
  );
}
