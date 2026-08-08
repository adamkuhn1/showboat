import { useRef, useState, useEffect, useCallback } from "react";
import { makeGame, takeShot, placeCueBall, type ShotReport } from "./game/game";
import { CUE_ID, SOLIDS, STRIPES } from "./game/rack";
import type { GameState, PlayerId } from "./game/state";
import type { CueAction } from "./physics/cue";
import { aimTowards, computeView, render, drawAim, drawCueStroke } from "./render/renderer";
import { drawPresentation } from "./render/overlay";
import { initPhysics, simulateShotWasm } from "./physics/wasm-bridge";
import { neuralEvaluator } from "./ai/neural/evaluator";
import { OverlayPanel, type ModelBadge } from "./ui/OverlayPanel";
import { useAiTurn, type Phase, type Scene } from "./ui/useAiTurn";
import { interpolateBalls, type AnimTrack } from "./render/animate";
import { PLAYBACK_SPEEDS, PLAYBACK_SPEED_LABEL, usePlaybackSpeed } from "./ui/playbackSpeed";
import { simulationRate } from "./ui/pacing";

const CANVAS_W = 900;
const CANVAS_H = 500;

/** Fine-aim nudge, in radians. A fingertip covers ~26 mm of felt at embed scale. */
const NUDGE = (0.25 * Math.PI) / 180;
/** Coarse aim step for the keyboard. */
const STEP = (2 * Math.PI) / 180;
/** Keyboard power step. */
const POWER_STEP = 0.05;

// --- Cue interaction --------------------------------------------------------
//
// Aiming and striking are separate acts, and the pointer only aims. Drag
// anywhere on the felt and the cue ball points AT the pointer; power is the
// slider or the vertical arrow keys; the shot is the Shoot button or space.
//
// They are separate because a gesture that both sets a shot up and takes it has
// no way to be wrong safely. A pull-back stroke on this canvas fired a
// full-power shot from any drag of more than four pixels, anywhere on the
// table, with no way to cancel and no requirement that the press began near the
// cue ball — and it aimed the cue ball away from the pointer, so a drag toward
// the ball you meant to hit sent the cue 180 degrees the other way. Two
// controls that each do one thing are worth more here than one that does both.
//
// Every action is reachable from the keyboard: horizontal keys aim (Shift for a
// 0.25-degree step), vertical keys set power, space shoots. See the key handler.

// Convert a pointer event's CSS-pixel coordinates into the canvas's intrinsic
// pixel space. `.table { max-width: 100% }` (index.css) lets the canvas render
// smaller than its intrinsic CANVAS_W/CANVAS_H on narrow viewports —
// getBoundingClientRect() reports the CSS-rendered box, but view.offsetX/scale
// (from computeView(CANVAS_W, CANVAS_H, ...)) are in intrinsic-pixel space.
// Without this ratio, aiming/placement is measurably off on any window
// narrower than the canvas's intrinsic width.
const getCanvasPoint = (e: React.PointerEvent<HTMLCanvasElement>): { x: number; y: number } => {
  const rect = e.currentTarget.getBoundingClientRect();
  // Maps into the LOGICAL 900x500 space `computeView` works in — deliberately
  // not `canvas.width / rect.width`. The backing store is `CANVAS_W * dpr`
  // device pixels (see the resolution effect below), so that ratio would be
  // 2x off on a retina display and every aim and cue-ball placement would land
  // at half the intended offset.
  return {
    x: (e.clientX - rect.left) * (CANVAS_W / rect.width),
    y: (e.clientY - rect.top) * (CANVAS_H / rect.height),
  };
};

// Portfolio embed contract (apps/portfolio/src/lib/embedProtocol.ts): posting
// `ready` makes the shell crossfade its loading veil out. Sent once the engine
// outcome is known — loaded or failed, both are painted, final states (the
// failure message is the app's honest UI, better shown than veiled). No-op
// when running standalone.
function postToShell(type: string) {
  if (window.parent === window) return;
  window.parent.postMessage({ source: "portfolio-embed", type, id: "showboat" }, "*");
}

/**
 * True when this app is running inside the portfolio's iframe.
 *
 * Read once, at module load: whether a document is framed cannot change during
 * its lifetime, and making it state would only add a first-paint flash of the
 * header this exists to suppress. Same test `postToShell` uses.
 */
const EMBEDDED = typeof window !== "undefined" && window.parent !== window;

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const on = () => setReduced(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduced;
}

// Player 1 (id 0) is human; Player 2 (id 1) is the opponent.
const AI_PLAYER: PlayerId = 1;

export default function App() {
  const game = useRef(makeGame());
  const [state, setState] = useState<GameState>(game.current.state);
  const [phase, setPhase] = useState<Phase>("aiming");
  const [aim, setAim] = useState(0);
  const [power, setPower] = useState(0.6);
  const [side, setSide] = useState(0);
  const [top, setTop] = useState(0);
  const [message, setMessage] = useState("loading…");
  const [engineReady, setEngineReady] = useState(false);
  // Neural ranking is the DEFAULT; the physics-only comparison stays one click
  // away, in the reasoning panel. The corrected pre-registered gate that made
  // it the default (n=400, fresh seed 55508219) is written up at
  // docs/repair/visual-authorship/showboat/CORRECTED_GATE_RESULT.md; every
  // measurable criterion passed, including multi-wall-combo selection
  // +16.75pp [+12.71, +20.79]. Reproduce with
  //   npm run eval:hybrid -- --fixtures 400 --games 40 --seed 55508219
  const [useNeural, setUseNeural] = useState(true);
  const [modelBadge, setModelBadge] = useState<ModelBadge>({ mode: "classical" });
  // Preflight succeeded: the artifact is present, hash-verified and
  // schema-compatible, so the comparison can be offered. Distinct from
  // "loaded" — the onnxruntime session is only created on first use.
  const [modelAvailable, setModelAvailable] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reducedMotion = usePrefersReducedMotion();
  const [playbackSpeed, setPlaybackSpeed] = usePlaybackSpeed();

  const table = game.current.table;
  const view = computeView(CANVAS_W, CANVAS_H, table);

  // Physics gates playability, so it is awaited. The ranker is loaded in the
  // background and deliberately NOT awaited here: creating its session pulls
  // onnxruntime-web's ~27MB WASM runtime, and blocking first paint on that
  // would be a real regression for the portfolio embed.
  useEffect(() => {
    initPhysics()
      .then(() => {
        setEngineReady(true);
        setMessage("break to start");
      })
      .catch(() => setMessage("couldn't load the physics engine"))
      .finally(() => postToShell("ready"));

    // Preflight only: validates the manifest and hashes the 14 KB artifact
    // without importing onnxruntime-web, so a missing/corrupted/schema-
    // mismatched model is caught loudly at startup while the ~27 MB WASM
    // runtime is deferred until someone actually plays.
    neuralEvaluator.preflight().then((pre) => {
      if (pre.ok) {
        setModelAvailable(true);
      } else {
        console.error(
          `[showboat] ranker artifact failed preflight: ${pre.reason} — the neural mode will be ` +
            `unavailable and the opponent stays on the classical physics search.`,
        );
        setModelBadge({ mode: "classical", fallbackReason: pre.reason });
      }
    });
  }, []);

  // Last scene handed to `paintScene`, so a backing-store resize can repaint
  // exactly what was on screen. Assigning canvas.width wipes the surface.
  const lastSceneRef = useRef<Scene | null>(null);
  // Live game state for the resize handler, which is registered once and would
  // otherwise close over the state as it stood on mount.
  const stateRef = useRef(state);
  stateRef.current = state;

  /** The one place anything reaches the canvas. */
  const paintScene = useCallback(
    (scene: Scene) => {
      lastSceneRef.current = scene;
      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx) return;
      render(ctx, scene.state, table, view);
      // One branch. Ball playback used to arrive here with `frame: null` and a
      // synthetic marks-only frame was assembled on the spot; the opponent's
      // shot now carries its real SHOOTING frame, which is what draws the
      // measured route under the moving balls. The player's own shot passes no
      // frame and no marks, because the reasoning overlay is the opponent's.
      if (scene.frame) {
        // The route's head is pinned to the ball's ACTUAL position this frame —
        // the same array `render` just drew — rather than to a position
        // interpolated along the thinned route. See `legCutAt` for the drift
        // that made this necessary.
        const ballAt = (id: number) => {
          const b = scene.state.balls.find((x) => x.id === id);
          return b && !b.pocketed ? { x: b.pos.x, y: b.pos.y } : null;
        };
        drawPresentation(ctx, scene.frame, view, scene.marks, scene.simTime, ballAt);
      }
      if (scene.stroke) {
        const cue = scene.state.balls.find((b) => b.id === CUE_ID);
        if (cue && !cue.pocketed) {
          drawCueStroke(ctx, cue, scene.stroke.phi, scene.stroke.power, scene.stroke.progress, view);
        }
      }
    },
    [table, view],
  );

  // Backing-store resolution follows devicePixelRatio while every draw call
  // keeps working in the fixed 900x500 logical space `computeView` was built
  // for. Without this the table renders at half resolution on a retina display
  // — visibly soft on the one surface a visitor is asked to study closely,
  // while the rocket-lab canvas on the same page (rocket-lab/shared/canvas.ts)
  // has always been scaled correctly. Capped at 2: beyond that the fill-rate
  // cost is real and the visible gain is not.
  //
  // Re-applied on resize because moving a window between displays changes the
  // ratio at runtime. Assigning width/height resets the context state and
  // clears the surface, so the transform is re-set and the last scene repainted.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const applyResolution = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.round(CANVAS_W * dpr);
      const h = Math.round(CANVAS_H * dpr);
      if (canvas.width === w && canvas.height === h) return;
      canvas.width = w;
      canvas.height = h;
      canvas.getContext("2d")?.setTransform(dpr, 0, 0, dpr, 0, 0);
      // Repaint whatever was on screen. Before the opponent's first turn
      // `lastSceneRef` is still null — `useAiTurn` is what fills it — and an
      // early `if (…) paint` here left the table BLANK after a resize that
      // also changed the ratio (a second display, or browser zoom), until the
      // next React render happened to repaint it. Measured 88.9% inked -> 0%.
      // The resting board is always paintable, so fall back to it rather than
      // skipping the repaint.
      paintScene(
        lastSceneRef.current ?? {
          state: stateRef.current,
          frame: null,
          marks: [],
          simTime: null,
          stroke: null,
        },
      );
    };
    applyResolution();
    window.addEventListener("resize", applyResolution);
    return () => window.removeEventListener("resize", applyResolution);
  }, [paintScene]);

  const commit = useCallback((report: ShotReport) => {
    setState(report.next);
    setPhase("aiming");
    game.current.state = report.next;
    const o = report.outcome;
    const playerName = (id: number) => (id === AI_PLAYER ? "opponent" : "you");
    // "you" takes a plural verb; "the opponent" doesn't.
    const verb = (id: number, plural: string, singular: string) =>
      id === AI_PLAYER ? singular : plural;
    // The turn has already moved in `report.next` when it passes, so the player
    // who just shot is recovered from the outcome, not from a pre-shot state
    // this function doesn't hold.
    const shooter = o.turnPasses ? (report.next.turn === 0 ? 1 : 0) : report.next.turn;

    let msg = "";
    if (o.gameOver) {
      msg = `${playerName(o.winner ?? 0)} ${verb(o.winner ?? 0, "win", "wins")}!${o.foul ? ` (${o.foulReason})` : ""}`;
    } else if (o.foul) {
      msg = `foul, ${o.foulReason}. ball in hand.`;
    } else if (o.assignedGroups) {
      const group = report.next.groups[shooter];
      msg = group ? `${group} for ${playerName(shooter)}.` : `${playerName(shooter)} claimed a group.`;
    } else if (o.turnPasses) {
      msg = `${playerName(report.next.turn)} up.`;
    } else {
      msg = `${playerName(shooter)} ${verb(shooter, "stay", "stays")} at the table.`;
    }
    setMessage(msg);
  }, []);

  const ai = useAiTurn({
    state,
    table,
    aiPlayer: AI_PLAYER,
    active: engineReady,
    phase,
    useNeural,
    reducedMotion,
    playbackSpeed,
    paintScene,
    setState,
    setPhase,
    commit,
    // `shot === null` reaches here only when the opponent had no legal target
    // at all. The hook has already rested the phase; this is the sentence that
    // tells the visitor why nothing is happening.
    onNoLegalShot: () => setMessage("no legal shot for the opponent."),
    onModelStatus: (s) =>
      setModelBadge(
        s.status === "ready"
          ? { mode: "neural-hybrid", hashVerified: s.hashVerified }
          : { mode: "classical", fallbackReason: s.reason ?? "not loaded" },
      ),
  });

  // Build the ranker session in the worker as soon as the page is idle, rather
  // than on the first opponent turn. The session lives in the worker and
  // nowhere else — the ~27 MB onnxruntime-web runtime is never instantiated on
  // this thread. Deliberately inside the app, not the shell: the portfolio
  // never mounts this iframe until the visitor presses "Play a rack", so a
  // visitor who scrolls past the section downloads none of it.
  /**
   * Build the ranker session when the visitor first touches the game.
   *
   * This used to warm on `requestIdleCallback`, described as paying for the
   * download "while the human is lining up a break". Measured against the
   * production build (`qa/warm-cost.mjs`), the `warm` message was posted **81 ms
   * after navigation on a page nobody had touched** — the page is idle
   * immediately after load, so "warm on idle" was warm on launch with extra
   * words, and a visitor who opened Showboat and left transferred the whole
   * 13.4 MB runtime for nothing.
   *
   * First interaction costs that visitor nothing and costs a player nothing
   * either: the trigger is their aim or their break, and their own break shot
   * then animates for several seconds before the opponent's first turn needs
   * the model. The `preflight()` at startup is unaffected — it is a 177 KB
   * artifact and a hash check, pulls no runtime, and is what the comparison
   * toggle's availability is decided by.
   */
  const warmedRef = useRef(false);
  const warmOnFirstInteraction = useCallback(() => {
    if (warmedRef.current || !modelAvailable) return;
    warmedRef.current = true;
    ai.warmModel();
  }, [modelAvailable, ai]);

  // The opponent's turn owns the canvas while it runs; this is the idle paint.
  useEffect(() => {
    if (ai.busy) return;
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    render(ctx, state, table, view);
    if (phase === "aiming" && state.turn !== AI_PLAYER) {
      const cue = state.balls.find((b) => b.id === CUE_ID);
      if (cue && !cue.pocketed) drawAim(ctx, cue, aim, power, view, table, state.balls);
    }
  }, [state, aim, power, phase, ai.busy, table, view]);

  // ---- human input ------------------------------------------------------
  const draggingRef = useRef(false);
  /** Set by space/tap while the player's own shot is rolling. */
  const humanSkipRef = useRef(false);
  /** Identifies the rack a human roll belongs to; see the loop in `shoot`. */
  const humanShotTokenRef = useRef(0);
  // The shot loop reads the speed per frame, so changing it mid-roll takes
  // effect without the balls jumping.
  const speedRef = useRef(playbackSpeed);
  speedRef.current = playbackSpeed;
  const yourTurn = phase === "aiming" && state.turn !== AI_PLAYER && !ai.busy;

  /**
   * Point the cue ball at the pointer.
   *
   * Aim only. Power comes from the slider or the vertical arrow keys, and the
   * shot is taken by the Shoot button or space — so no drag can fire one, and
   * no drag can overwrite a power that was set deliberately.
   */
  const aimAt = (px: number, py: number) => {
    const cue = state.balls.find((b) => b.id === CUE_ID);
    if (!cue || cue.pocketed) return;
    setAim(aimTowards(cue.pos, px, py, view));
  };

  // Drag to aim, on Pointer Events: one code path for mouse, pen and touch.
  // There were no pointer or touch handlers at all before this, so on a phone
  // `setAim` was never called and the cue stayed at 0 rad forever — you could
  // shoot, but only due east. Drag-and-release is also a better gesture for a
  // mouse than the hover-then-leave-the-canvas lock it replaces, which needed
  // a line of instructions underneath the table to explain it.
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    // Any touch of the felt counts, including one that only skips a replay:
    // the visitor is engaging with the game, which is the signal.
    warmOnFirstInteraction();
    // Pointer-down anywhere on the felt skips whatever is being shown: the
    // opponent's reasoning, the opponent's shot, or your own shot rolling out.
    // None of the three can be corrupted by it — every one is a replay of
    // something already decided and already simulated.
    if (ai.busy) {
      ai.skip();
      return;
    }
    if (phase === "animating") {
      humanSkipRef.current = true;
      return;
    }
    if (!yourTurn) return;
    const { x, y } = getCanvasPoint(e);
    if (state.ballInHand !== false) {
      const wx = (x - view.offsetX) / view.scale;
      const wy = -(y - view.offsetY) / view.scale;
      // Clamping to the cushions and clearing any ball already sitting there is
      // `placeCueBall`'s job, so every caller gets the same legal spot.
      setState(placeCueBall(state, wx, wy, table));
      setMessage("cue ball placed. drag to aim.");
      return;
    }
    // Capture so a drag that leaves the canvas keeps tracking.
    e.currentTarget.setPointerCapture(e.pointerId);
    draggingRef.current = true;
    aimAt(x, y);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!draggingRef.current || !yourTurn) return;
    const { x, y } = getCanvasPoint(e);
    aimAt(x, y);
  };

  const endDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  };

  const shoot = useCallback(() => {
    // The keyboard and the Shoot button reach the game without ever touching
    // the canvas, so the warm has to hang off this too.
    warmOnFirstInteraction();
    if (phase !== "aiming" || state.winner !== null || ai.busy) return;
    if (state.turn === AI_PLAYER) return;
    if (state.ballInHand !== false) {
      setMessage("tap the table to place the cue ball");
      return;
    }
    const action: CueAction = { phi: aim, power, sideSpin: side, topSpin: top };
    const report = takeShot(state, table, action, simulateShotWasm);
    setPhase("animating");

    // The human's shot is replayed by exactly the machinery the opponent's is:
    // the same `AnimTrack`, the same `interpolateBalls`, the same presentation
    // speed, the same skip. It was not before — it snapped to the nearest
    // earlier waypoint, which quantises every ball to the ~50 ms simulation
    // event grid and shows as a visible stutter that the opponent's shots do
    // not have. There is no reason for the player's own shot to be the
    // lower-fidelity one.
    const waypoints = report.sim.waypoints ?? [];
    if (waypoints.length === 0) {
      commit(report);
      return;
    }
    const track: AnimTrack = {
      waypoints: waypoints.map((wp) => ({ simTime: wp.time, balls: wp.balls })),
      duration: report.sim.duration,
    };
    humanSkipRef.current = false;
    let simTime = 0;
    let last = performance.now();
    // The roll is owned by the rack it was taken on. Without this, "New rack"
    // mid-roll left the loop running against the old report: it kept animating
    // and then called `commit(report)`, replacing the fresh rack with the
    // previous board four seconds after the visitor asked for a new one, ball
    // credited. Bumping the token on reset makes the abandoned loop stop
    // painting and, crucially, never commit.
    const token = ++humanShotTokenRef.current;
    const tick = (now: number) => {
      if (humanShotTokenRef.current !== token) return;
      // Exactly the pacing the opponent's shots use: one rate for the shot.
      const rate = simulationRate(speedRef.current);
      simTime += ((now - last) / 1000) * rate;
      last = now;
      if (humanSkipRef.current) simTime = track.duration;
      const ctx = canvasRef.current?.getContext("2d");
      if (ctx) {
        paintScene({
          state: { ...state, balls: interpolateBalls(track, simTime) },
          frame: null,
          marks: [],
          simTime: null,
          stroke: null,
        });
      }
      if (simTime < track.duration) requestAnimationFrame(tick);
      else commit(report);
    };
    requestAnimationFrame(tick);
  }, [phase, state, ai.busy, aim, power, side, top, table, paintScene, commit]);

  const shootRef = useRef(shoot);
  shootRef.current = shoot;
  const aiRef = useRef(ai);
  aiRef.current = ai;
  const phaseRef = useRef(phase);
  phaseRef.current = phase;

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        // While anything is playing back, space skips ahead rather than firing
        // a shoot that is disabled anyway.
        if (aiRef.current.busy) aiRef.current.skip();
        else if (phaseRef.current === "animating") humanSkipRef.current = true;
        else shootRef.current();
      }
      // The pointer-free way to take the same shot. Aim on the horizontal keys
      // (Shift for the 0.25-degree step the nudge buttons used to give), power
      // on the vertical ones. Held keys repeat, which is why `e.repeat` is not
      // filtered here — a fine aim wants to be draggable by key too.
      const yours = phaseRef.current === "aiming" && !aiRef.current.busy;
      if (yours && (e.code === "ArrowLeft" || e.code === "ArrowRight")) {
        e.preventDefault();
        const d = (e.shiftKey ? NUDGE : STEP) * (e.code === "ArrowLeft" ? 1 : -1);
        setAim((a) => a + d);
        return;
      }
      if (yours && (e.code === "ArrowUp" || e.code === "ArrowDown")) {
        e.preventDefault();
        const d = POWER_STEP * (e.code === "ArrowUp" ? 1 : -1);
        setPower((p) => Math.max(0.05, Math.min(1, p + d)));
        return;
      }
      if (e.code === "Escape" && !e.repeat) {
        // The shell's focused embed route tells the visitor Escape leaves.
        // Since d269d9b this frame takes the focus the shell offers (so
        // space-to-shoot works inside the embed), which means the parent's own
        // Escape listener never fires and that promise stopped being true.
        // Hand the request back up instead of dropping window.focus(), which
        // would trade one broken key for another.
        postToShell("releaseFocus");
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Keyboard shortcuts listen on this window, which inside an iframe holds no
  // focus until something in it is clicked. The shell hands control over by
  // posting `focus` (apps/portfolio/src/lib/useEmbed.ts) — take the focus it is
  // offering, so "press space to shoot" is true from that moment rather than
  // only after the visitor happens to click the felt.
  useEffect(() => {
    if (window.parent === window) return;
    const onMessage = (e: MessageEvent) => {
      const data = e.data as { source?: unknown; type?: unknown; id?: unknown } | null;
      if (!data || data.source !== "portfolio-embed") return;
      if (data.type !== "focus" || data.id !== "showboat") return;
      window.focus();
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  const reset = () => {
    // Abandon any roll still in flight so it cannot commit over the new rack,
    // and end the opponent's turn cleanly rather than leaving its loop parked
    // on an await that never settles.
    humanShotTokenRef.current += 1;
    ai.skip();
    game.current = makeGame();
    setState(game.current.state);
    setPhase("aiming");
    setMessage("new rack. break to start.");
  };

  const grp = state.groups[state.turn];
  const aiTurn = state.turn === AI_PLAYER;

  const rack = (ids: number[]) =>
    ids.map((id) => ({ id, pocketed: state.balls.find((b) => b.id === id)?.pocketed ?? false }));

  return (
    <main className="shell">
      {/* The title is printed by whoever owns the page. Standalone that is this
          app; inside the portfolio it is the portfolio, which sets "Showboat"
          in the same typeface with the same rule under it about 180 px above
          this one — the same word twice, once on cream and once on black. The
          tagline stays either way: nothing upstream says it. */}
      <header className={`topbar${EMBEDDED ? " topbar--embedded" : ""}`}>
        {!EMBEDDED && <h1>Showboat</h1>}
        <p className="tag">Eight-ball, against an opponent that goes looking for the bank shot.</p>
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
          />
          <div className="status">
            <span className={`turn p${state.turn}`}>
              {aiTurn ? "opponent" : "you"}
              {grp ? ` · ${grp}` : " · open"}
              {state.ballInHand !== false ? " · ball in hand" : ""}
            </span>
            <span className="rack" aria-label="balls remaining">
              {rack(SOLIDS).map(({ id, pocketed }) => (
                <span key={id} className={`rack-ball solid${pocketed ? " down" : ""}`}>
                  {id}
                </span>
              ))}
              <span className="rack-gap" />
              {rack(STRIPES).map(({ id, pocketed }) => (
                <span key={id} className={`rack-ball stripe${pocketed ? " down" : ""}`}>
                  {id}
                </span>
              ))}
            </span>
            <span className="msg">{message}</span>
          </div>
        </div>

        {/* Nothing renders here until the opponent has actually planned once.
            An empty "candidates appear here" box beside the table was dead
            weight that also broke the embedded layout: at the ~1180px embed
            width it pushed `.layout` over its wrap breakpoint, which reflowed
            this panel below the table and buried the Shoot button below the
            visible frame. See index.css's `.board` comment. */}
        {(ai.trace !== null || ai.planning || ai.liveCounts !== null) && (
          <OverlayPanel
            trace={ai.trace}
            state={ai.presentation}
            planning={ai.planning}
            modelLoading={ai.modelLoading}
            badge={useNeural ? modelBadge : { mode: "classical" }}
            showSkipHint={ai.turnIndex >= 2}
            showDisclosure={ai.turnIndex === 1}
            replaying={ai.replaying}
            liveCounts={ai.liveCounts}
            compare={{ available: modelAvailable, useNeural, onChange: setUseNeural }}
            replay={
              ai.presentation === "SETTLED" && !ai.busy
                ? { onDecision: ai.replayDecision, onShot: ai.replayShot }
                : null
            }
          />
        )}
      </div>

      <div className="controls">
        <div className="primary-row">
          {/* The pointer-free power control. Direct cue interaction on the felt
              is the primary way in; this mirrors it, and remains the way to set
              power without a drag gesture. */}
          <label className="power">
            Power <span>{Math.round(power * 100)}%</span>
            <input
              type="range"
              min={0.05}
              max={1}
              step={0.01}
              value={power}
              onChange={(e) => setPower(Number(e.target.value))}
              disabled={!yourTurn}
            />
          </label>
          <div className="buttons">
            <button onClick={shoot} disabled={!yourTurn || state.winner !== null || !engineReady}>
              {phase === "animating" ? "Rolling…" : ai.busy ? "Opponent" : "Shoot"}
            </button>
            <button onClick={reset} className="secondary">
              New rack
            </button>
          </div>
        </div>

        {/* Playback rate. Deliberately labelled as a property of the screen and
            not of the game: the shot is simulated once, at full physical
            fidelity, before any of it is drawn, and this only decides how fast
            that recording is played back. It sits with the controls rather than
            in the reasoning panel because it governs your own shots too, and
            the panel does not exist until the opponent has had a turn. */}
        <div className="speed" role="group" aria-label={PLAYBACK_SPEED_LABEL}>
          <span className="speed-label">{PLAYBACK_SPEED_LABEL}</span>
          {PLAYBACK_SPEEDS.map((s) => (
            <button
              key={s}
              type="button"
              className={`speed-btn${s === playbackSpeed ? " is-on" : ""}`}
              aria-pressed={s === playbackSpeed}
              onClick={() => setPlaybackSpeed(s)}
            >
              {s}&times;
            </button>
          ))}
        </div>

        {/* Spin. Fine aim moved onto the arrow keys, where a fingertip's ~26 mm
            of felt is no longer the limiting resolution — Shift gives the same
            0.25 degrees the two buttons here used to. */}
        <div className="fine">
          <label>
            English <span>{side > 0 ? `+${side.toFixed(2)}` : side.toFixed(2)}</span>
            <input
              type="range"
              min={-1}
              max={1}
              step={0.05}
              value={side}
              onChange={(e) => setSide(Number(e.target.value))}
              disabled={!yourTurn}
            />
          </label>
          <label>
            Draw · Follow <span>{top > 0 ? `+${top.toFixed(2)}` : top.toFixed(2)}</span>
            <input
              type="range"
              min={-1}
              max={1}
              step={0.05}
              value={top}
              onChange={(e) => setTop(Number(e.target.value))}
              disabled={!yourTurn}
            />
          </label>
        </div>
      </div>

      {state.winner !== null && (
        <div className="win-screen">
          <div className="win-card">
            <p className="win-eyebrow">game over</p>
            <h2 className="win-headline">
              {state.winner === AI_PLAYER ? "opponent wins" : "you win"}
            </h2>
            <p className="win-sub">
              {state.winner === 0 ? "nice run." : "the search didn't miss."}
            </p>
            <button onClick={reset} className="win-btn">
              play again
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
