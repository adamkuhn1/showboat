import { useRef, useState, useEffect, useCallback } from "react";
import { makeGame, takeShot, placeCueBall, cloneState } from "./game/game";
import { CUE_ID, SOLIDS, STRIPES } from "./game/rack";
import type { GameState, PlayerId } from "./game/state";
import type { CueAction } from "./physics/cue";
import { applyCue } from "./physics/cue";
import { computeView, render, drawAim } from "./render/renderer";
import { drawOverlay } from "./render/overlay";
import { buildAnimTrack, interpolateBalls, type AnimTrack } from "./render/animate";
import { BALL_RADIUS } from "./physics/constants";
import { initPhysics, simulateShotWasm } from "./physics/wasm-bridge";
import { legalTargets } from "./ai/turn";
import { getBrain } from "./ai/brain";
import { neuralEvaluator } from "./ai/neural/evaluator";
import type { SearchResult } from "./ai/shotSearch";
import { OverlayPanel, type ModelBadge } from "./ui/OverlayPanel";

const CANVAS_W = 900;
const CANVAS_H = 500;

// Bounds for the post-search overlay READ hold (see the AI-turn effect below).
// This is a rendering concession, not simulated deliberation: the search has fully
// completed before this timer starts, and the panel shows no progress
// animation. Without a hold the overlay would be painted and replaced by the
// shot animation in the same few frames, so nothing would be legible.
const OVERLAY_READ_HOLD_MIN_MS = 350;
const OVERLAY_READ_HOLD_MAX_MS = 1100;

// Convert a mouse event's CSS-pixel coordinates into the canvas's intrinsic
// pixel space. `.table { max-width: 100% }` (index.css) lets the canvas
// render smaller than its intrinsic CANVAS_W/CANVAS_H on narrow viewports —
// getBoundingClientRect() reports the CSS-rendered box, but view.offsetX/
// scale (from computeView(CANVAS_W, CANVAS_H, ...)) are in intrinsic-pixel
// space. Without this ratio, aiming/placement is measurably off on any
// window narrower than the canvas's intrinsic width. Reads canvas.width/
// height directly off the element rather than the CANVAS_W/CANVAS_H
// constants so this stays correct even if those constants ever change.
const getCanvasPoint = (e: React.MouseEvent<HTMLCanvasElement>): { x: number; y: number } => {
  const canvas = e.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
};

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

/** Run `fn` when the browser is next idle, or soon, in engines without it. */
function idle(fn: () => void): void {
  const ric = (window as unknown as { requestIdleCallback?: (cb: () => void) => number })
    .requestIdleCallback;
  if (typeof ric === "function") ric(fn);
  else window.setTimeout(fn, 1200);
}

// Player 1 (id 0) is human; Player 2 (id 1) is the AI opponent.
const AI_PLAYER: PlayerId = 1;

type Phase = "aiming" | "searching" | "animating";

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
  const [vsAI, setVsAI] = useState(true);
  const vsAIRef = useRef(vsAI);
  const [search, setSearch] = useState<SearchResult | null>(null);
  const searchRef = useRef<SearchResult | null>(null);
  const [lastSearch, setLastSearch] = useState<SearchResult | null>(null);
  // Neural ranking is the DEFAULT. Classical physics search remains fully
  // available as an explicitly selectable comparison mode via the toggle.
  //
  // History: the release-candidate gate (docs/repair/release-candidate/
  // showboat/DECISION_GATE.md), pooled over 210 held-out fixtures, failed 2
  // of 7 measurable criteria (C: foul/scratch ceilings, D: trick-shot
  // benefit) — but both failures were mis-specifications, not real
  // regressions, disclosed by the team that ran it and independently
  // confirmed arithmetically by a cold review (docs/repair/release-candidate/
  // reviews/ml-truth.md). C's threshold was narrower than n=210 could ever
  // resolve, even under a true null; D's baseline (classical already attempts
  // a "trick" on 96.2% of turns under an overbroad definition that counted a
  // plain single-rail bank shot) capped the maximum possible improvement at
  // +3.8pp against a +5pp bar, and measured the wrong construct besides.
  //
  // A corrected gate (docs/repair/visual-authorship/showboat/
  // CORRECTED_GATE.md) was frozen BEFORE running — fixing C's power (same
  // +0.02/+0.01 thresholds, n increased to 400) and D's construct (restricted
  // to {double-bank, combo, rail-combo}, the "multi-wall combos" the product
  // constraint actually names, same +5pp/CI-excluding-zero bar) — then run
  // exactly once on a genuinely fresh held-out seed (55508219, verified
  // unused anywhere in prior sweeps). Result: ALL measurable criteria pass:
  //
  //   - PASS  legal-pot -0.25pp [-2.27pp, +1.77pp], regret +0.25pp
  //           [-1.77pp, +2.27pp] — non-inferior
  //   - PASS  C': foul -0.50pp [-2.58pp, +1.58pp], scratch 0.0% both arms
  //   - PASS  D': multi-wall-combo selection +16.75pp [+12.71pp, +20.79pp]
  //           (25.5% vs 8.8% of 400 decisions), completion +16.50pp
  //           [+12.54pp, +20.46pp] — CI excludes zero on both
  //   - PASS  equal budget (0 over-budget), latency, direct-fallback
  //           preservation (95.0% vs 100.0%, within the -5pp allowance)
  //
  // Reproduce: `npm run eval:hybrid -- --fixtures 400 --games 40 --seed
  // 55508219 --out corrected_55508219.json` then `npx tsx
  // eval/correctedGateReport.ts eval/results/corrected_55508219.json`. Full
  // writeup: docs/repair/visual-authorship/showboat/CORRECTED_GATE_RESULT.md.
  const [useNeural, setUseNeural] = useState(true);
  const useNeuralRef = useRef(useNeural);
  const [modelBadge, setModelBadge] = useState<ModelBadge>({ mode: "classical" });
  // Preflight succeeded: the artifact is present, hash-verified and
  // schema-compatible, so the toggle can be offered. Distinct from "loaded" —
  // the onnxruntime session is only created on first use.
  const [modelAvailable, setModelAvailable] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const aimLockedRef = useRef(false);
  const shootRef = useRef<() => void>(() => {});

  const table = game.current.table;
  const view = computeView(CANVAS_W, CANVAS_H, table);

  vsAIRef.current = vsAI;
  searchRef.current = search;
  useNeuralRef.current = useNeural;

  // Physics gates playability, so it is awaited. The ranker is loaded in the
  // background and deliberately NOT awaited here: creating its session pulls
  // onnxruntime-web's ~27MB WASM runtime, and blocking first paint on that
  // would be a real regression for the portfolio embed. The AI turn awaits the
  // same (idempotent) promise before it plans, so a slow model load delays one
  // opponent turn rather than the whole app.
  useEffect(() => {
    initPhysics()
      .then(() => {
        setEngineReady(true);
        setMessage("break to start");
      })
      .catch(() => setMessage("couldn't load the physics engine"))
      .finally(postEmbedReady);

    // Preflight only: validates the manifest and hashes the 14 KB artifact
    // without importing onnxruntime-web, so a missing/corrupted/schema-
    // mismatched model is caught loudly at startup while the ~27 MB WASM
    // runtime is deferred until someone actually turns the model on.
    neuralEvaluator.preflight().then((pre) => {
      if (pre.ok) {
        setModelAvailable(true);
        // Neural ranking is the default, so the ~27MB onnxruntime-web WASM
        // runtime is now on the path of an ordinary game rather than of a
        // visitor who ticked a box. Start it as soon as the app is idle
        // instead of on the first opponent turn: nothing awaits it here, the
        // turn still falls back to classical while it is in flight
        // (getBrain gates on evaluator.isReady()), and by the time the human
        // has taken a break shot it is normally already resident.
        //
        // Deliberately inside the app, not the shell: the portfolio never
        // mounts this iframe until the visitor presses "Play a rack", so a
        // visitor who scrolls past the section downloads none of it.
        idle(() => {
          void neuralEvaluator.load();
        });
      } else {
        console.error(
          `[showboat] ranker artifact failed preflight: ${pre.reason} — the neural mode will be ` +
            `unavailable and the opponent stays on the classical physics search.`,
        );
        setModelBadge({ mode: "classical", fallbackReason: pre.reason });
      }
    });
  }, []);

  const paint = useCallback(
    (s: GameState, overlay?: SearchResult | null) => {
      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx) return;
      render(ctx, s, table, view);
      if (overlay) drawOverlay(ctx, overlay, table, view);
      if (phase === "aiming" && s.turn !== AI_PLAYER) {
        const cue = s.balls.find((b) => b.id === CUE_ID);
        if (cue && !cue.pocketed) drawAim(ctx, cue, aim, power, view, table, s.balls);
      }
    },
    [table, view, phase, aim, power],
  );

  useEffect(() => {
    paint(state, phase === "searching" ? search : null);
  }, [state, aim, power, phase, search, paint]);

  // Animate a shot by replaying the WASM engine's own trajectory.
  //
  // This used to run a SECOND, independently-implemented TS simulation
  // (buildAnimTrack) purely for the animation preview, then snap or glide to
  // the WASM engine's authoritative final state once it finished. Over a long
  // collision cascade — a break chains 20+ ball-ball/cushion events — two
  // separately-coded event-driven simulations are numerically chaotic against
  // each other: measured on a real break, several balls diverged by tens of
  // centimetres and the two runs even disagreed on which balls were pocketed.
  // A glide correction only softened the resulting visible "rearrange" right
  // before commit; it didn't remove it, because a large divergence eased
  // over a few hundred ms still reads as balls sliding to new spots.
  //
  // The fix is architectural: simulateShotWasm (physics/wasm-bridge.ts) now
  // captures a full waypoint snapshot after every resolved step of the SAME
  // run that produces the authoritative outcome (physics-core's
  // simulate_shot, capture_waypoints=true). Replaying those waypoints here
  // means the animation IS the authoritative simulation — there is no second
  // run to diverge from, so commit is a true no-op frame, not a correction.
  const animateAndCommit = useCallback(
    (fromState: GameState, action: CueAction, report: ReturnType<typeof takeShot>) => {
      setPhase("animating");

      const track: AnimTrack =
        report.sim.waypoints && report.sim.waypoints.length > 0
          ? {
              waypoints: report.sim.waypoints.map((wp) => ({
                simTime: wp.time,
                balls: wp.balls,
              })),
              duration: report.sim.duration,
            }
          : (() => {
              // Defensive fallback — only reachable if a non-WASM simulator
              // is ever wired in without waypoint capture. Real play always
              // takes the branch above.
              const anim = cloneState(fromState);
              const cue = anim.balls.find((b) => b.id === CUE_ID)!;
              applyCue(cue, action);
              return buildAnimTrack(anim.balls, table);
            })();

      // Real-time playback. This used to run at 2x, which made every shot —
      // especially the break, where a dozen balls are moving at once — read
      // as a blur rather than something you could watch the AI's reasoning
      // play out in.
      const ANIM_SPEED = 1.0;
      const start = performance.now();

      const tick = (now: number) => {
        const simTime = ((now - start) / 1000) * ANIM_SPEED;
        const balls = interpolateBalls(track, simTime);
        const displayState = { ...fromState, balls };
        setState(displayState);
        paint(displayState, null);

        if (simTime < track.duration) {
          requestAnimationFrame(tick);
        } else {
          commitShot(report);
        }
      };
      requestAnimationFrame(tick);
    },
    [table, paint],
  );

  const commitShot = useCallback(
    (report: ReturnType<typeof takeShot>) => {
      setState(report.next);
      if (searchRef.current !== null) setLastSearch(searchRef.current);
      setSearch(null);
      setPhase("aiming");
      game.current.state = report.next;
      const o = report.outcome;
      const ai = vsAIRef.current;
      const playerName = (id: number) =>
        ai ? (id === AI_PLAYER ? "opponent" : "you") : `player ${id + 1}`;
      // "you" takes a plural verb; "the opponent" and "player 2" don't.
      const isPlural = (id: number) => ai && id !== AI_PLAYER;
      const verb = (id: number, plural: string, singular: string) =>
        isPlural(id) ? plural : singular;
      // The turn has already moved in `report.next` when it passes, so the
      // player who just shot is recovered from the outcome rather than from a
      // pre-shot state this function doesn't hold.
      const shooter = o.turnPasses ? (report.next.turn === 0 ? 1 : 0) : report.next.turn;

      let msg = "";
      if (o.gameOver) {
        msg = `${playerName(o.winner ?? 0)} ${verb(o.winner ?? 0, "win", "wins")}!${o.foul ? ` (${o.foulReason})` : ""}`;
      } else if (o.foul) {
        msg = `foul, ${o.foulReason}. ball in hand.`;
      } else if (o.assignedGroups) {
        // The table was open and this shot claimed a group. That is the only
        // thing worth saying about it.
        const group = report.next.groups[shooter];
        msg = group ? `${group} for ${playerName(shooter)}.` : `${playerName(shooter)} claimed a group.`;
      } else if (o.turnPasses) {
        msg = `${playerName(report.next.turn)} up.`;
      } else {
        msg = `${playerName(shooter)} ${verb(shooter, "stay", "stays")} at the table.`;
      }
      setMessage(msg);
    },
    [],
  );

  // AI turn: plan (real search), show the overlay for a bounded read hold,
  // then play the chosen shot. Runs whenever it becomes the AI's move.
  useEffect(() => {
    if (!vsAI || !engineReady) return;
    if (state.turn !== AI_PLAYER || state.winner !== null || phase !== "aiming") return;

    let cancelled = false;

    // Ball-in-hand for the AI: place the cue at a simple legal spot (centre of
    // the head area) before searching.  Calling setState here re-triggers this
    // effect (because `state` is in deps), but the new run sees ballInHand===false
    // and skips this block.  The old timeout (registered below) is cancelled by
    // that re-run's cleanup — which is correct; the re-run registers a fresh
    // timeout that executes cleanly.
    let planState = state;
    if (state.ballInHand !== false) {
      planState = placeCueBall(state, -table.length / 4, 0);
      setState(planState);
    }

    // Defer to next frame.  setPhase("searching") is intentionally inside the
    // callback: calling it in the effect body would change `phase`, which used
    // to be in the deps list, causing the cleanup to fire and cancel this
    // timeout before it ran.  Moving it here avoids that self-cancellation.
    const t = setTimeout(async () => {
      if (cancelled) return;
      setPhase("searching");
      // The model load is normally already done (see the idle preload in the
      // startup effect), but if this turn arrives first the wait is stated
      // rather than hidden behind a generic "searching…". It is the honest
      // reason the opponent's own name in the header can change mid-session.
      setMessage(
        useNeuralRef.current && !neuralEvaluator.isReady()
          ? "loading the trained model…"
          : "searching…",
      );
      // First neural turn pays for the onnxruntime-web session (idempotent
      // afterwards). A failure here is reported, never silently downgraded.
      if (useNeuralRef.current) {
        const loaded = await neuralEvaluator.load();
        if (loaded.status === "ready") {
          setModelBadge({ mode: "neural-hybrid", hashVerified: loaded.hashVerified });
        } else {
          console.error(`[showboat] neural ranker failed to load: ${loaded.reason}`);
          setModelBadge({ mode: "classical", fallbackReason: loaded.reason });
        }
      }
      if (cancelled) return;
      setMessage("searching…");
      const brain = getBrain(useNeuralRef.current);
      const result = await brain.plan(planState, table, AI_PLAYER);
      if (cancelled) return; // check again after the (possibly async) net eval
      setSearch(result);
      paint(planState, result);

      // Bounded, content-adaptive hold so the (already-computed) decision is
      // readable before the balls move. Scales with how much there is to read.
      // Nothing is computed during it and nothing on screen pretends otherwise.
      const holdMs = Math.min(
        OVERLAY_READ_HOLD_MAX_MS,
        OVERLAY_READ_HOLD_MIN_MS + result.stats.length * 15,
      );
      setTimeout(() => {
        if (cancelled) return;
        if (!result.best) {
          // All search candidates were filtered. Aim at the nearest legal target
          // with a clear cue path to avoid hitting the 8-ball or opponent balls first.
          const cueBall = planState.balls.find((b) => b.id === CUE_ID)!;
          const targets = legalTargets(planState, AI_PLAYER);
          const live = planState.balls.filter((b) => !b.pocketed);
          const byDist = live
            .filter((b) => targets.includes(b.id))
            .sort((a, b) =>
              Math.hypot(a.pos.x - cueBall.pos.x, a.pos.y - cueBall.pos.y) -
              Math.hypot(b.pos.x - cueBall.pos.x, b.pos.y - cueBall.pos.y)
            );
          // Prefer a target whose direct cue path clears all other balls.
          const pathClearTo = (t: typeof byDist[0]) => {
            const dx = t.pos.x - cueBall.pos.x;
            const dy = t.pos.y - cueBall.pos.y;
            const len = Math.hypot(dx, dy);
            if (len < 1e-9) return true;
            const nx = dx / len; const ny = dy / len;
            for (const b of live) {
              if (b.id === CUE_ID || b.id === t.id) continue;
              const vx = b.pos.x - cueBall.pos.x;
              const vy = b.pos.y - cueBall.pos.y;
              const proj = vx * nx + vy * ny;
              if (proj <= 0 || proj >= len) continue;
              const perp2 = (vx - proj * nx) ** 2 + (vy - proj * ny) ** 2;
              if (perp2 < (2 * BALL_RADIUS) ** 2) return false;
            }
            return true;
          };
          const nearest = byDist.find(pathClearTo) ?? byDist[0];
          const phi = nearest
            ? Math.atan2(nearest.pos.y - cueBall.pos.y, nearest.pos.x - cueBall.pos.x)
            : Math.random() * Math.PI * 2;
          const fallback: CueAction = { phi, power: 0.3, sideSpin: 0, topSpin: 0 };
          const report = takeShot(planState, table, fallback, simulateShotWasm);
          animateAndCommit(planState, fallback, report);
          return;
        }
        const action = result.best.candidate.action;
        const report = takeShot(planState, table, action, simulateShotWasm);
        animateAndCommit(planState, action, report);
      }, holdMs);
    }, 30);

    return () => {
      cancelled = true;
      clearTimeout(t);
    };
    // `phase` is intentionally excluded: including it caused the effect cleanup
    // to cancel the search timeout the moment setPhase("searching") was called.
    // Full `state` (not just state.turn) lets the effect re-trigger when the AI
    // pockets a ball and continues its turn with the same turn index.
    // `useNeural` is intentionally excluded as well, and this is a bug fix,
    // not an oversight. It used to be in this list, so flipping the toggle
    // during the opponent's turn ran the cleanup (cancelling the in-flight
    // search) and then re-entered the effect, which bailed immediately on the
    // `phase !== "aiming"` guard — leaving the turn wedged at "searching…"
    // forever with no way out but a new rack. The effect body already reads
    // the live value through `useNeuralRef`, so the dependency bought nothing
    // even when it worked. A toggle mid-turn now lets the turn in flight
    // finish under the mode it started in, and takes effect from the next
    // one. See searchToggle.test.ts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, vsAI, engineReady]);

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (aimLockedRef.current) return;
    if (phase !== "aiming" || (vsAI && state.turn === AI_PLAYER)) return;
    const { x: mx, y: my } = getCanvasPoint(e);
    const cue = state.balls.find((b) => b.id === CUE_ID);
    if (!cue || cue.pocketed) return;
    const cx = view.offsetX + cue.pos.x * view.scale;
    const cy = view.offsetY - cue.pos.y * view.scale;
    setAim(Math.atan2(-(my - cy), mx - cx));
  };

  const onMouseLeave = () => { aimLockedRef.current = true; };
  const onMouseEnter = () => { aimLockedRef.current = false; };

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (state.ballInHand === false || phase !== "aiming") return;
    if (vsAI && state.turn === AI_PLAYER) return;
    const { x: mx, y: my } = getCanvasPoint(e);
    const wx = (mx - view.offsetX) / view.scale;
    const wy = -(my - view.offsetY) / view.scale;
    const hx = table.length / 2 - BALL_RADIUS;
    const hy = table.width / 2 - BALL_RADIUS;
    const cx = Math.max(-hx, Math.min(hx, wx));
    const cy = Math.max(-hy, Math.min(hy, wy));
    setState(placeCueBall(state, cx, cy));
    setMessage("cue ball placed. shoot when ready.");
  };

  const shoot = () => {
    if (phase !== "aiming" || state.winner !== null) return;
    if (vsAI && state.turn === AI_PLAYER) return;
    if (state.ballInHand !== false) {
      setMessage("click the table to place the cue ball");
      return;
    }
    const action: CueAction = { phi: aim, power, sideSpin: side, topSpin: top };
    const report = takeShot(state, table, action, simulateShotWasm);
    animateAndCommit(state, action, report);
  };

  shootRef.current = shoot;

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        shootRef.current();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Keyboard shortcuts listen on this window, which inside an iframe holds no
  // focus until something in it is clicked. The shell hands control over by
  // posting `focus` (see apps/portfolio/src/lib/useEmbed.ts) — take the focus
  // it is offering, so "press space to shoot" is true from that moment rather
  // than only after the visitor happens to click the felt. Standalone, the
  // window already has focus and this never fires.
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
    game.current = makeGame();
    setState(game.current.state);
    setSearch(null);
    setLastSearch(null);
    setPhase("aiming");
    setMessage("new game. break to start.");
  };

  const grp = state.groups[state.turn];
  const aiTurn = vsAI && state.turn === AI_PLAYER;

  // Ball-count dot colors (index = ball id)
  const BALL_COLOR: Record<number, string> = {
    1: "#f4c724", 2: "#1f4fd8", 3: "#e23c2e", 4: "#6b2fb3",
    5: "#e8792b", 6: "#1f8a4c", 7: "#8c2f2a",
    9: "#f4c724", 10: "#1f4fd8", 11: "#e23c2e", 12: "#6b2fb3",
    13: "#e8792b", 14: "#1f8a4c", 15: "#8c2f2a",
  };

  const solidDots = SOLIDS.map((id) => ({
    id,
    pocketed: state.balls.find((b) => b.id === id)?.pocketed ?? false,
  }));

  const stripeDots = STRIPES.map((id) => ({
    id,
    pocketed: state.balls.find((b) => b.id === id)?.pocketed ?? false,
  }));

  return (
    <main className="shell">
      <header className="topbar">
        <h1>Showboat</h1>
        {/* Introduces the opponent by what it does, not what it's built from —
            "physics-search opponent" / "neural + physics opponent" named an
            implementation. Which mode is actually running is still stated
            accurately, just not here: the "neural ranking" toggle below and
            the reasoning panel's own title (OverlayPanel.tsx, derived from
            the search's real trace) are the honest, mode-specific readouts. */}
        {/* Describes what the opponent does, not which implementation is
            deciding this instant. `brainLabel()` (ai/brain.ts) still derives
            that from validated model state and is still the honest answer to
            "what is running" — but it flips from "the physics-search
            opponent" to "the neural + physics opponent" when the runtime
            finishes downloading, which as a page header read as copy that
            changed between loads. The mode is stated where it is actionable
            and stable instead: on the "neural ranking" toggle, and on the
            reasoning panel's own title, both of which are read off the real
            decision. */}
        <p className="tag">
          {vsAI
            ? "Eight-ball, against an opponent that goes looking for the bank shot."
            : "Eight-ball, two players."}
        </p>
      </header>

      <div className="layout">
        <div className="board">
          <canvas
            ref={canvasRef}
            width={CANVAS_W}
            height={CANVAS_H}
            onMouseMove={onMouseMove}
            onMouseLeave={onMouseLeave}
            onMouseEnter={onMouseEnter}
            onClick={onClick}
            className="table"
          />
          <div className="status">
            <span className={`turn p${state.turn}`}>
              {state.turn === AI_PLAYER && vsAI ? "opponent" : "you"}
              {grp ? ` · ${grp}` : " · open"}
              {state.ballInHand !== false ? " · ball in hand" : ""}
            </span>
            <span className="ball-dots">
              {solidDots.map(({ id, pocketed }) => (
                <span
                  key={id}
                  className={`ball-dot${pocketed ? " pocketed" : ""}`}
                  style={{ background: BALL_COLOR[id] }}
                  title={`ball ${id}`}
                />
              ))}
            </span>
            <span className="ball-dots" style={{ opacity: 0.65 }}>
              {stripeDots.map(({ id, pocketed }) => (
                <span
                  key={id}
                  className={`ball-dot${pocketed ? " pocketed" : ""}`}
                  style={{ background: BALL_COLOR[id] }}
                  title={`ball ${id}`}
                />
              ))}
            </span>
            <span className="msg">{message}</span>
          </div>
        </div>

        {/* Nothing renders here at all until the opponent has actually
            searched once. An empty "candidates appear here" box beside the
            table — before there is anything to show — was dead weight that
            also broke the embedded layout: at the portfolio's ~1178px embed
            width it pushed `.layout` (index.css) over its wrap breakpoint,
            which reflowed this panel below the table and buried the Shoot
            button below the visible frame. See index.css's `.board` comment
            for the layout half of that fix. */}
        {vsAI && (search !== null || lastSearch !== null) && (
          <OverlayPanel
            result={search ?? lastSearch}
            searching={phase === "searching"}
            stale={search === null && lastSearch !== null}
            badge={useNeural ? modelBadge : { mode: "classical" }}
          />
        )}
      </div>

      <div className="controls">
        <label>
          Power <span>{Math.round(power * 100)}%</span>
          <input type="range" min={0.05} max={1} step={0.01} value={power}
            onChange={(e) => setPower(Number(e.target.value))} disabled={phase !== "aiming" || aiTurn} />
        </label>
        <label>
          English <span>{side > 0 ? `+${side.toFixed(2)}` : side.toFixed(2)}</span>
          <input type="range" min={-1} max={1} step={0.05} value={side}
            onChange={(e) => setSide(Number(e.target.value))} disabled={phase !== "aiming" || aiTurn} />
        </label>
        <label>
          Draw · Follow <span>{top > 0 ? `+${top.toFixed(2)}` : top.toFixed(2)}</span>
          <input type="range" min={-1} max={1} step={0.05} value={top}
            onChange={(e) => setTop(Number(e.target.value))} disabled={phase !== "aiming" || aiTurn} />
        </label>
        <div className="buttons">
          <button
            onClick={shoot}
            disabled={phase !== "aiming" || state.winner !== null || !engineReady || aiTurn}
          >
            {phase === "animating" ? "Rolling…" : phase === "searching" ? "Searching…" : "Shoot"}
          </button>
          <button onClick={reset} className="secondary">New rack</button>
          <label className="toggle">
            <input type="checkbox" checked={vsAI} onChange={(e) => setVsAI(e.target.checked)} />
            vs AI
          </label>
          <label
            className="toggle"
            title={
              modelAvailable
                ? "Ranks candidate shots with a trained model, then verifies them with the same physics search either way; turn it off to compare against physics search alone."
                : `model unavailable: ${modelBadge.fallbackReason ?? "not loaded"}`
            }
          >
            <input
              type="checkbox"
              checked={useNeural && modelAvailable}
              disabled={!modelAvailable}
              onChange={(e) => setUseNeural(e.target.checked)}
            />
            neural ranking
          </label>
        </div>
      </div>
      <p className="hint">
        Hover to aim, leave the canvas to lock the angle, then press space to shoot.
      </p>

      {state.winner !== null && (
        <div className="win-screen">
          <div className="win-card">
            <p className="win-eyebrow">game over</p>
            <h2 className="win-headline">
              {state.winner === AI_PLAYER && vsAI ? "opponent wins" : "you win"}
            </h2>
            <p className="win-sub">
              {state.winner === 0 || !vsAI ? "nice run." : "the search didn't miss."}
            </p>
            <button onClick={reset} className="win-btn">play again</button>
          </div>
        </div>
      )}
    </main>
  );
}
