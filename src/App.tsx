import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  cueSpotLimits,
  makeGame,
  placeCueBall,
  takeShot,
  type ShotReport,
} from "./game/game";
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
import { summariseOutcome } from "./game/summary";
import { type EvaluatedCandidate } from "./ai/agent";
import { AiRunner } from "./ai/aiClient";
import { makeRanker } from "./ai/ranker";
import ThinkingPanel, { type ThinkingState } from "./ThinkingPanel";

const CANVAS_W = 900;
const CANVAS_H = 500;
const DPR = Math.min(3, Math.max(1, Math.round(window.devicePixelRatio || 1)));
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

// Instruction wording follows the input device: on a touch screen the aim
// is a tap-and-drag, everywhere else a press-and-drag.
const coarsePointer = (): boolean =>
  typeof window.matchMedia === "function" &&
  window.matchMedia("(pointer: coarse)").matches;

export default function App({ embed = false }: { embed?: boolean }) {
  const game = useRef(makeGame());
  const coarse = useMemo(coarsePointer, []);
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
  const [message, setMessage] = useState(
    `You break. ${coarse ? "Tap and drag" : "Drag"} to aim, then Shoot.`,
  );
  // Full collision chain of the last committed shot — behind "details", never
  // in the status line itself.
  const [lastTrace, setLastTrace] = useState<string | null>(null);
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
  // The AI thinks in a Web Worker so the table keeps animating meanwhile.
  const aiRunner = useRef<AiRunner | null>(null);
  aiRunner.current ??= new AiRunner();
  const slowRef = useRef(slow);
  slowRef.current = slow;

  const table = game.current.table;
  // Stable identity matters: paint() must not be recreated on unrelated
  // renders (thinking updates), or its effect would repaint the base table
  // over the AI's candidate-route overlay.
  const view = useMemo(() => computeView(CANVAS_W, CANVAS_H, table), [table]);

  // Drawing happens in CANVAS_W x CANVAS_H units; the backing store is
  // scaled by the device pixel ratio so the table is sharp on retina screens.
  const ctx = (): CanvasRenderingContext2D | null => {
    const c = canvasRef.current?.getContext("2d") ?? null;
    c?.setTransform(DPR, 0, 0, DPR, 0, 0);
    return c;
  };

  const paint = useCallback(
    (s: GameState) => {
      const c = ctx();
      if (!c) return;
      render(c, s, table, view);
      if (phase === "aiming" && s.turn === HUMAN) {
        const cue = s.balls.find((b) => b.id === CUE_ID);
        if (cue && !cue.pocketed) drawAim(c, cue, aim, s.balls, view, table);
      }
    },
    [table, view, phase, aim],
  );

  useEffect(() => {
    if (phase !== "animating") paint(state);
  }, [state, aim, phase, paint]);

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

  // Status line gets a plain summary; the full event chain goes behind the
  // details toggle (and the console) instead of flooding the line.
  const showOutcome = (report: ShotReport, shooter: typeof HUMAN | typeof AI): void => {
    const trace = describeShot(report.sim, Infinity);
    console.debug(`[showboat] ${shooter === HUMAN ? "you" : "AI"}: ${trace}`);
    setLastTrace(trace);
    const summary = summariseOutcome(report, shooter, HUMAN);
    setMessage(
      report.next.ballInHand !== false && report.next.turn === HUMAN && report.next.winner === null
        ? `${summary} ${coarse ? "Tap" : "Click"} inside the shaded kitchen to put it down.`
        : summary,
    );
  };

  // --- human input ----------------------------------------------------------
  const humanTurn = phase === "aiming" && state.turn === HUMAN && state.winner === null;
  // Aiming is press-and-drag, not hover-follow: the aim only updates while
  // the pointer is held down on the table, so moving the mouse around to look
  // at the table (or to reach a slider) never disturbs a prepared shot.
  const draggingRef = useRef(false);

  // Pointer position in canvas pixels. The canvas is drawn at CANVAS_W x
  // CANVAS_H but may be displayed scaled (max-width on narrow screens, full
  // frame width in embed mode), so CSS pixels are mapped back to the drawing
  // coordinates the view transform uses.
  const canvasPoint = (e: React.PointerEvent<HTMLCanvasElement>): { x: number; y: number; inBounds: boolean } => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const cssY = e.clientY - rect.top;
    const inBounds = cssX >= 0 && cssX <= rect.width && cssY >= 0 && cssY <= rect.height;
    return {
      x: (cssX * CANVAS_W) / (rect.width || CANVAS_W),
      y: (cssY * CANVAS_H) / (rect.height || CANVAS_H),
      inBounds,
    };
  };

  const updateAimTo = (x: number, y: number, cue: { pos: { x: number; y: number } }) => {
    const cx = view.offsetX + cue.pos.x * view.scale;
    const cy = view.offsetY - cue.pos.y * view.scale;
    setAim(Math.atan2(-(y - cy), x - cx));
  };

  // Clicking ONLY places the cue ball during ball-in-hand; shooting is the
  // explicit button. Aim and shoot stay separate, deliberate actions.
  const placeCueAt = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const { x, y } = canvasPoint(e);
    const wx = (x - view.offsetX) / view.scale;
    const wy = -(y - view.offsetY) / view.scale;
    const lim = cueSpotLimits(state, table);
    const nx = Math.max(lim.minX, Math.min(lim.maxX, wx));
    const ny = Math.max(lim.minY, Math.min(lim.maxY, wy));
    const overlaps = state.balls.some(
      (b) =>
        !b.pocketed &&
        b.id !== CUE_ID &&
        Math.hypot(b.pos.x - nx, b.pos.y - ny) < BALL_RADIUS * 2,
    );
    if (overlaps) {
      setMessage("There's a ball there. Pick a clear spot.");
      return;
    }
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
    const { x, y } = canvasPoint(e);
    updateAimTo(x, y, cue);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!humanTurn || !draggingRef.current) return;
    const cue = state.balls.find((b) => b.id === CUE_ID);
    if (!cue || cue.pocketed) return;
    const { x, y, inBounds } = canvasPoint(e);
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
      setMessage(`Place the cue ball first (${coarse ? "tap" : "click"} the table).`);
      return;
    }
    setThinking({ active: false, ranker: ranker.current.name, phase: "" });
    const action: CueAction = { phi: aim, power, sideSpin: side, topSpin: top };
    // ONE simulation resolves the shot; its frames are the playback.
    const report = takeShot(state, table, action, { recordFrames: true });
    void playAndCommit(report).then(() => {
      showOutcome(report, HUMAN);
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

    // Ball in hand, shown as it happens on a real table: the cue ball comes
    // out from the head rail and rolls to the chosen spot, rather than
    // appearing there.
    const rollCueIn = (
      placed: GameState,
      spot: { x: number; y: number },
      sig: { cancelled: boolean },
    ): Promise<void> =>
      new Promise((resolve) => {
        const from = { x: -table.length / 2 + BALL_RADIUS, y: spot.y };
        const ms = 650;
        const start = performance.now();
        const frame = (now: number): void => {
          if (sig.cancelled) return resolve();
          const k = Math.min(1, (now - start) / ms);
          const e = 1 - (1 - k) ** 3; // ease-out: it slows into place
          const s = {
            ...placed,
            balls: placed.balls.map((b) =>
              b.id === CUE_ID
                ? {
                    ...b,
                    pos: {
                      x: from.x + (spot.x - from.x) * e,
                      y: from.y + (spot.y - from.y) * e,
                    },
                  }
                : b,
            ),
          };
          const c = ctx();
          if (c) render(c, s, table, view);
          if (k < 1) requestAnimationFrame(frame);
          else setTimeout(resolve, 250);
        };
        requestAnimationFrame(frame);
      });

    const run = async (): Promise<void> => {
      let base = state;
      setMessage("AI is thinking…");
      setLastTrace(null);
      setThinking({ active: true, ranker: ranker.current.name, phase: "starting" });
      const turn = aiRunner.current!.run(state, table, ranker.current.name, {
        onPhase: (p) => setThinking((t) => ({ ...t, phase: p })),
        onPlaced: async (placed, spot) => {
          base = placed;
          setMessage("AI takes ball in hand behind the line.");
          await rollCueIn(placed, spot, signal);
        },
        onGenerated: (n, byKind) =>
          setThinking((t) => ({ ...t, generated: n, byKind })),
        onCandidate: (ec, index, total) => {
          setThinking((t) => ({ ...t, current: ec, index, total }));
          paintCandidate(base, ec);
        },
      }, { delayMs: 420 });
      const move = await turn.move;
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
      showOutcome(move.report, AI);
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
    // Cancel any AI work (terminating the worker mid-search) and any
    // animation; playback.stop() resolves its own promise, so nothing stays
    // pending.
    if (aiAbortRef.current) aiAbortRef.current.cancelled = true;
    aiRunner.current?.cancel();
    playbackRef.current?.stop();
    playbackRef.current = null;
    aiRunningRef.current = false;
    draggingRef.current = false;
    game.current = makeGame();
    setState(game.current.state);
    setPhase("aiming");
    setThinking({ active: false, ranker: ranker.current.name, phase: "" });
    setLastTrace(null);
    setMessage("New rack. You break.");
  };

  const turnLabel =
    state.winner !== null
      ? state.winner === HUMAN
        ? "You won the rack"
        : "The AI won the rack"
      : state.turn === HUMAN
        ? "Your shot"
        : "AI at the table";

  const englishLabel =
    Math.abs(side) < 0.01 ? "centre" : `${Math.round(Math.abs(side) * 100)}% ${side < 0 ? "left" : "right"}`;
  const spinLabel =
    Math.abs(top) < 0.01 ? "stun" : `${Math.round(Math.abs(top) * 100)}% ${top < 0 ? "draw" : "follow"}`;

  return (
    <main className="shell">
      {!embed && (
        <header className="topbar">
          <h1>Showboat</h1>
          <p className="tag">Eight-ball on a seven-foot bar table, against an opponent who would rather bank it.</p>
        </header>
      )}

      <div className="layout">
        <div className="board">
          <div className="scores">
            <Side who="You" player={HUMAN} state={state} />
            <span className={`turn p${state.turn}`}>{turnLabel}</span>
            <Side who="AI" player={AI} state={state} />
          </div>
          <canvas
            ref={canvasRef}
            width={CANVAS_W * DPR}
            height={CANVAS_H * DPR}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            className="table"
            data-human={humanTurn || undefined}
            data-placing={(humanTurn && state.ballInHand !== false) || undefined}
          />
          <div className="status">
            <span className="msg">{message}</span>
            {lastTrace && phase === "aiming" && (
              <details className="trace">
                <summary>what happened</summary>
                <span>{lastTrace}</span>
              </details>
            )}
          </div>
          <div className="controls">
            <label className="power">
              Power <span>{Math.round(power * 100)}%</span>
              <input type="range" min={0.05} max={1} step={0.01} value={power}
                onChange={(e) => setPower(Number(e.target.value))} disabled={!humanTurn} />
            </label>
            <label>
              English <span>{englishLabel}</span>
              <input type="range" min={-1} max={1} step={0.05} value={side}
                onChange={(e) => setSide(Number(e.target.value))}
                onDoubleClick={() => setSide(0)} disabled={!humanTurn} />
            </label>
            <label>
              Draw / follow <span>{spinLabel}</span>
              <input type="range" min={-1} max={1} step={0.05} value={top}
                onChange={(e) => setTop(Number(e.target.value))}
                onDoubleClick={() => setTop(0)} disabled={!humanTurn} />
            </label>
            <div className="buttons">
              <button onClick={shoot} disabled={!humanTurn}>
                {phase === "animating"
                  ? "Rolling…"
                  : state.turn === AI && state.winner === null
                    ? "AI's shot"
                    : "Shoot"}
              </button>
              <button onClick={() => setSlow((s) => !s)} className="secondary" aria-pressed={slow}>
                Slow motion {slow ? "on" : "off"}
              </button>
              <button onClick={reset} className="secondary">Rack 'em</button>
            </div>
          </div>
          <p className="hint">
            While the AI thinks, dashed lines are shots it's still weighing. The
            solid line is the one it picked, and it's exactly what you'll see rolled.
            Double-click a spin slider to centre it.
          </p>
        </div>

        <ThinkingPanel t={thinking} placeholder />
      </div>

    </main>
  );
}

// One side of the scoreboard: who, which group, and the balls still to sink
// (the 8 alone once the group is cleared).
function Side({ who, player, state }: { who: string; player: 0 | 1; state: GameState }) {
  const group = state.groups[player];
  const left = group
    ? state.balls
        .filter((b) => !b.pocketed && (group === "solids" ? b.id >= 1 && b.id <= 7 : b.id >= 9))
        .map((b) => b.id)
        .sort((x, y) => x - y)
    : [];
  const show = group && left.length === 0 ? [8] : left;
  return (
    <span className={`side p${player}`}>
      <strong>{who}</strong>
      <em>{group ?? "open table"}</em>
      <span className="chips" aria-label={group ? `${show.length} to go` : undefined}>
        {show.map((id) => (
          <i key={id} className={`chip c${id > 8 ? id - 8 : id}${id > 8 ? " stripe" : ""}`} title={`${id}`} />
        ))}
      </span>
    </span>
  );
}
