// The opponent's turn, end to end: plan it off the main thread, present the
// decision, stroke the cue, replay the shot, commit.
//
// This used to be ~110 lines of orchestration inline in a 700-line `App.tsx`,
// tangled with human input and layout. It is here so the sequence can be read
// in one sitting, and so `App.tsx` is composition again.
//
// The canvas belongs to this hook while `busy` is true. It paints every frame
// directly and does NOT push a React state update per frame — the old loop
// called `setState` sixty times a second to move balls that no other component
// reads, which is React churn in the middle of the one animation that has to
// be smooth. The single authoritative state update happens once, at commit.

import { useCallback, useEffect, useRef, useState } from "react";
import type { GameState, PlayerId } from "../game/state";
import type { Table } from "../physics/table";
import type { ShotReport } from "../game/game";
import { placeCueBall } from "../game/game";
import { CUE_ID } from "../game/rack";
import type { DecisionTraceV1 } from "../ai/trace/contract";
import { interpolateBalls, type AnimTrack } from "../render/animate";
import { contactMarks, contactMarksFromExecuted, type ContactMark } from "../render/annotate";
import { type PlaybackSpeed } from "./playbackSpeed";
import { simulationRate } from "./pacing";
import {
  buildSchedule,
  frameAt,
  holdScaleForTurn,
  stateAt,
  withSkip,
  type PresentationFrame,
  type PresentationSchedule,
  type PresentationState,
} from "../render/presentation";
import { shotSentence } from "./shotSentence";
import { usePlanner } from "./planner/usePlanner";
import type { ModelStatus, PlannedTurn, PlayedTurn } from "./planner/plan";
import {
  applyProgress,
  createLiveSearch,
  liveFrame,
  liveState,
  type LiveSearch,
} from "../render/liveSearch";

export type Phase = "aiming" | "searching" | "animating";

/** Everything the canvas needs for one frame. The hook decides what; the host decides how. */
export interface Scene {
  state: GameState;
  frame: PresentationFrame | null;
  marks: ContactMark[];
  /** Simulation seconds elapsed during playback, or null when nothing is moving. */
  simTime: number | null;
  stroke: { phi: number; power: number; progress: number } | null;
}

export interface UseAiTurnArgs {
  state: GameState;
  table: Table;
  aiPlayer: PlayerId;
  /** vs-AI mode is on and the physics engine is ready. */
  active: boolean;
  phase: Phase;
  useNeural: boolean;
  reducedMotion: boolean;
  /** Presentation only: the multiplier from wall-clock onto simulation time. */
  playbackSpeed: PlaybackSpeed;
  paintScene: (scene: Scene) => void;
  setState: (s: GameState) => void;
  setPhase: (p: Phase) => void;
  commit: (report: ShotReport) => void;
  onModelStatus: (s: ModelStatus) => void;
  /**
   * The policy returned no shot, which now means only "no legal target". The
   * host says so; this hook has already put the phase back to a resting state.
   */
  onNoLegalShot: () => void;
}

export interface AiTurnView {
  presentation: PresentationState;
  trace: DecisionTraceV1 | null;
  /** The search is running and nothing has arrived yet. */
  planning: boolean;
  modelLoading: boolean;
  /** 1-based count of opponent turns this session; drives the presentation decay. */
  turnIndex: number;
  /** True while this hook owns the canvas. */
  busy: boolean;
  /**
   * True while a REPLAY of a finished decision is on screen, false while a
   * search is being watched live. The panel must say which, because the two
   * look similar and mean different things.
   */
  replaying: boolean;
  /**
   * Counts of events the search has published so far, or null when no search
   * is being watched. Held as React state — `LiveSearch` itself is mutated in
   * place on the frame path, so a component reading its fields directly would
   * never re-render.
   */
  liveCounts: LiveSearch["counts"] | null;
  /**
   * The retained event stream for the turn on screen. Mutated in place, so it
   * is a snapshot to read on demand (a replay, a debugging surface), not
   * something to render from directly.
   */
  liveStream: () => LiveSearch | null;
  /**
   * Skip ahead. During the reasoning sequence it lands in READY; during ball
   * playback it runs the remaining simulation time out at once.
   *
   * Neither can corrupt anything: the decision, the shot and the resulting game
   * state are all fixed before the first frame is painted, and `commit` runs off
   * `planned.report` either way. Skipping changes how much of an already-decided
   * turn you watch, and nothing else.
   */
  skip: () => void;
  replayDecision: () => void;
  replayShot: () => void;
  /** Build the ranker session in the worker before the first turn needs it. */
  warmModel: () => void;
}

/**
 * What the opponent-turn effect should do this render.
 *
 * Pulled out as a pure function because the interesting part is not the
 * planning, it is the re-entrancy: the effect re-triggers on every `state`
 * change, including the ones it causes itself, and getting that wrong hangs
 * the turn rather than failing it. Testable without a DOM.
 */
export type TurnAction = "idle" | "place-cue" | "plan";

export function nextTurnAction(a: {
  active: boolean;
  turn: PlayerId;
  aiPlayer: PlayerId;
  winner: PlayerId | null;
  phase: Phase;
  ballInHand: GameState["ballInHand"];
}): TurnAction {
  if (!a.active) return "idle";
  if (a.turn !== a.aiPlayer || a.winner !== null) return "idle";
  // `phase !== "aiming"` is the re-entrancy guard: a turn already in flight
  // has set the phase, and a second run must not start a second plan.
  if (a.phase !== "aiming") return "idle";
  // Placement is its own render. See the comment at the call site for what
  // happens when it is not.
  if (a.ballInHand !== false) return "place-cue";
  return "plan";
}

export function useAiTurn(args: UseAiTurnArgs): AiTurnView {
  const {
    state,
    table,
    aiPlayer,
    active,
    useNeural,
    reducedMotion,
    playbackSpeed,
    paintScene,
    setState,
    setPhase,
    commit,
    onModelStatus,
    onNoLegalShot,
  } = args;

  const planner = usePlanner();

  const [presentation, setPresentation] = useState<PresentationState>("IDLE");
  const [trace, setTrace] = useState<DecisionTraceV1 | null>(null);
  const [planning, setPlanning] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [turnIndex, setTurnIndex] = useState(0);
  const [busy, setBusy] = useState(false);
  const [replaying, setReplaying] = useState(false);
  const [liveCounts, setLiveCounts] = useState<LiveSearch["counts"] | null>(null);

  // Live values the turn body reads without becoming a dependency of it.
  const useNeuralRef = useRef(useNeural);
  useNeuralRef.current = useNeural;
  const reducedMotionRef = useRef(reducedMotion);
  reducedMotionRef.current = reducedMotion;
  // Read per frame, so changing the speed mid-shot takes effect immediately
  // without the ball jumping: `runShot` integrates dt * speed rather than
  // recomputing elapsed * speed from the start.
  const speedRef = useRef(playbackSpeed);
  speedRef.current = playbackSpeed;
  const paintRef = useRef(paintScene);
  paintRef.current = paintScene;
  const turnIndexRef = useRef(0);

  const skipRef = useRef(false);
  const rafRef = useRef(0);
  /**
   * The search currently being watched, and its own paint loop.
   *
   * Events arrive as individual worker messages, which on a busy search can be
   * several within one display frame. Painting on each would repaint the felt
   * more often than the screen can show it, so an event marks the observation
   * dirty and one rAF paints whatever has accumulated. That coalescing changes
   * nothing about WHAT is drawn — `liveFrame` renders the state the events have
   * left behind, so a frame that folds three events in shows all three.
   */
  const liveRef = useRef<LiveSearch | null>(null);
  const liveRafRef = useRef(0);
  const cancelRef = useRef(false);
  const runIdRef = useRef(0);
  /** Retained so both replays are free: no search re-runs, ever. */
  const lastTurnRef = useRef<{ pre: GameState; planned: PlayedTurn; marks: ContactMark[] } | null>(
    null,
  );

  /**
   * Settles whichever rAF-driven loop is currently awaited.
   *
   * Both loops below resolve themselves on `cancelRef`, but only from inside
   * `tick` — so cancelling the frame they are waiting on means `tick` never
   * runs again and the promise never settles. The turn body then parks
   * forever on its `await`, `setBusy(false)` is never reached, and every
   * control stays disabled with no error anywhere: pressing "New rack" during
   * the opponent's turn wedged the page until reload.
   *
   * Holding the resolver here is what lets `stopLoop` end the wait rather than
   * merely stopping the animation.
   */
  const settleLoopRef = useRef<null | (() => void)>(null);

  const stopLoop = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = 0;
    if (liveRafRef.current) cancelAnimationFrame(liveRafRef.current);
    liveRafRef.current = 0;
    const settle = settleLoopRef.current;
    settleLoopRef.current = null;
    settle?.();
  }, []);

  // ---- the reasoning sequence ------------------------------------------
  const runSequence = useCallback(
    (
      pre: GameState,
      planned: PlayedTurn,
      marks: ContactMark[],
      thenShoot: boolean,
      /**
       * `"post-search"` on a live turn — the search has just been watched
       * happening, so only the beats that are not observations of it remain.
       * `"full"` is the replay: the whole sequence, paced from the trace.
       */
      scope: "full" | "post-search",
    ) =>
      new Promise<void>((rawResolve) => {
        // Registered so `stopLoop` can end this wait; see `settleLoopRef`.
        const resolve = () => {
          settleLoopRef.current = null;
          rawResolve();
        };
        settleLoopRef.current = resolve;
        const sentence = shotSentence(planned.trace);
        let schedule: PresentationSchedule = buildSchedule({
          trace: planned.trace,
          sentenceWords: sentence?.words ?? 0,
          decay: holdScaleForTurn(turnIndexRef.current),
          reducedMotion: reducedMotionRef.current,
          scope,
        });
        const cue = pre.balls.find((b) => b.id === CUE_ID);
        const geom = { cuePos: cue ? { x: cue.pos.x, y: cue.pos.y } : { x: 0, y: 0 } };
        // The stroke belongs to the shot, not to the decision, so a decision
        // replay stops when the plan is ready. Read off the live `schedule`,
        // which a skip replaces.
        const endOf = () => (thenShoot ? schedule.strokeEndMs : schedule.reasoningEndMs);

        skipRef.current = false;
        let skipped = false;
        let last: PresentationState | null = null;
        const start = performance.now();

        const tick = (now: number) => {
          if (cancelRef.current) return resolve();
          const t = now - start;

          if (skipRef.current && !skipped) {
            // Always safe: the trace is complete before any of this starts, so
            // skipping cannot desync the display from the decision.
            schedule = withSkip(schedule, t);
            skipped = true;
          }

          if (t >= endOf()) {
            setPresentation(thenShoot ? "SHOOTING" : "SETTLED");
            return resolve();
          }

          const place = stateAt(schedule, t);
          if (place.state !== last) {
            last = place.state;
            setPresentation(place.state);
          }
          const frame = frameAt(planned.trace, schedule, t, geom);
          paintRef.current({
            state: pre,
            frame,
            marks,
            simTime: null,
            stroke:
              frame.strokeProgress === null
                ? null
                : {
                    phi: planned.action.phi,
                    power: planned.action.power,
                    progress: frame.strokeProgress,
                  },
          });
          rafRef.current = requestAnimationFrame(tick);
        };
        rafRef.current = requestAnimationFrame(tick);
      }),
    [],
  );

  // ---- shot playback ----------------------------------------------------
  const runShot = useCallback(
    (pre: GameState, planned: PlayedTurn, marks: ContactMark[]) =>
      new Promise<void>((rawResolve) => {
        // Registered so `stopLoop` can end this wait; see `settleLoopRef`.
        const resolve = () => {
          settleLoopRef.current = null;
          rawResolve();
        };
        settleLoopRef.current = resolve;
        const sim = planned.report.sim;
        const track: AnimTrack = {
          waypoints: (sim.waypoints ?? []).map((wp) => ({ simTime: wp.time, balls: wp.balls })),
          duration: sim.duration,
        };
        if (track.waypoints.length === 0) return resolve();

        // The route drawn under the moving balls is the measured motion of THIS
        // simulation, read from the trace. `frameAt` past the end of the
        // schedule is the SHOOTING frame; building it once keeps the paint loop
        // free of decision logic.
        const cue = pre.balls.find((b) => b.id === CUE_ID);
        const geom = { cuePos: cue ? { x: cue.pos.x, y: cue.pos.y } : { x: 0, y: 0 } };
        const shootSchedule = buildSchedule({
          trace: planned.trace,
          sentenceWords: 0,
          reducedMotion: true,
        });
        const frame = frameAt(
          planned.trace,
          shootSchedule,
          shootSchedule.strokeEndMs + 1,
          geom,
        );

        skipRef.current = false;
        let simTime = 0;
        let last = performance.now();
        const tick = (now: number) => {
          if (cancelRef.current) return resolve();
          // One rate for the whole shot (`ui/pacing.ts`). Read per frame only
          // so that changing the presentation speed mid-shot takes effect at
          // once; within a single setting the value is the same every frame.
          const rate = simulationRate(speedRef.current);
          simTime += ((now - last) / 1000) * rate;
          last = now;
          // Skipping runs the remaining simulation time out in one frame. The
          // outcome is `planned.report` regardless — see `skip`.
          if (skipRef.current) simTime = track.duration;
          const balls = interpolateBalls(track, simTime);
          paintRef.current({
            state: { ...pre, balls },
            frame,
            marks,
            simTime,
            stroke: null,
          });
          if (simTime < track.duration) rafRef.current = requestAnimationFrame(tick);
          else resolve();
        };
        rafRef.current = requestAnimationFrame(tick);
      }),
    [],
  );

  // ---- the turn ---------------------------------------------------------
  useEffect(() => {
    const action = nextTurnAction({
      active,
      turn: state.turn,
      aiPlayer,
      winner: state.winner,
      phase: args.phase,
      ballInHand: state.ballInHand,
    });
    if (action === "idle") return;

    if (action === "place-cue") {
      // Ball-in-hand for the opponent: place the cue at a simple legal spot,
      // and RETURN. The `setState` re-triggers this effect, and the re-run —
      // which now sees `ballInHand === false` — does the planning.
      //
      // Doing both in one pass is what hung the turn: the effect body's
      // `setState` and the async body's `setPhase("searching")` land in the
      // same React batch, so the re-run hit the `phase !== "aiming"` guard and
      // bailed, while this run had already been torn down by the re-render.
      // Nothing was left to finish the turn and the panel sat on "searching…"
      // forever. Reproduced in Chrome against a production build: every AI
      // ball-in-hand turn, indefinitely.
      setState(placeCueBall(state, -table.length / 4, 0, table));
      return;
    }

    const planState = state;
    cancelRef.current = false;
    const runId = ++runIdRef.current;
    let disposed = false;
    /** Has a newer run taken ownership of the presentation? */
    const superseded = () => runIdRef.current !== runId;

    void (async () => {
      // Painted BEFORE the search starts, and it now actually paints: the
      // planning happens in a worker, so React gets to commit this frame. It
      // never did before — `plan()` blocked the main thread first.
      setBusy(true);
      setPlanning(true);
      setTrace(null);
      setPresentation("IDLE");
      setPhase("searching");
      const nextTurn = turnIndexRef.current + 1;
      turnIndexRef.current = nextTurn;
      setTurnIndex(nextTurn);

      // The board the search is reasoning about, and the cue-ball origin every
      // candidate's cue leg is drawn from.
      const planCue = planState.balls.find((b) => b.id === CUE_ID);
      const liveGeom = {
        cuePos: planCue ? { x: planCue.pos.x, y: planCue.pos.y } : { x: 0, y: 0 },
      };
      liveRef.current = null;

      let planned: PlannedTurn;
      try {
        planned = await planner.plan(planState, table, aiPlayer, useNeuralRef.current, {
          onModelLoading: () => setModelLoading(true),
          onModelStatus: (s) => {
            setModelLoading(false);
            onModelStatus(s);
          },
          // THE LIVE PATH. Each event is folded in as it lands and the felt is
          // repainted from what the search has actually published. Nothing here
          // advances on a timer, and nothing draws a candidate the search has
          // not sent geometry for.
          onProgress: (e) => {
            if (cancelRef.current || superseded()) return;
            const l = liveRef.current ?? createLiveSearch();
            liveRef.current = l;
            applyProgress(l, e);
            setPresentation(liveState(l));
            // A fresh object each time: the accumulator's own `counts` is
            // mutated in place, so handing that reference to React would never
            // register as a change.
            setLiveCounts({ ...l.counts });
            // `planning` is cleared by the first event: the panel's "searching…"
            // line exists for the stretch before there is anything to show, and
            // once routes are on the felt there is.
            if (l.phase !== "started") setPlanning(false);
            if (liveRafRef.current) return;
            liveRafRef.current = requestAnimationFrame(() => {
              liveRafRef.current = 0;
              if (cancelRef.current || superseded()) return;
              const current = liveRef.current;
              if (!current) return;
              paintRef.current({
                state: planState,
                frame: liveFrame(current, liveGeom),
                marks: [],
                simTime: null,
                stroke: null,
              });
            });
          },
        });
      } catch (err) {
        console.error("[showboat] the opponent's turn failed to plan:", err);
        setPlanning(false);
        setModelLoading(false);
        setBusy(false);
        setPhase("aiming");
        return;
      }
      // A torn-down run must never leave the panel stuck on "searching…". If a
      // newer run has taken over it owns those flags; if nothing has, this run
      // is responsible for putting them back.
      if (disposed || cancelRef.current) {
        if (!superseded()) {
          setPlanning(false);
          setModelLoading(false);
          setBusy(false);
        }
        return;
      }

      // No legal target — the only thing `shot === null` can mean now that the
      // trick-only ladder's rungs 4 and 5 cover every board where one exists.
      // There is no aim to invent here and nothing to fall back to, so the turn
      // rests: the phase goes back to "aiming", `planning`/`busy` are cleared,
      // and the host says why. Leaving any of those set is what wedged the
      // panel at "searching…". The effect does not re-run on a phase change, so
      // this cannot spin.
      if (planned.kind === "no-legal-shot") {
        setTrace(planned.trace);
        setPlanning(false);
        setModelLoading(false);
        setBusy(false);
        setPhase("aiming");
        onNoLegalShot();
        return;
      }

      // Capped at 8: a break chains 20+ events and the marks would bury the
      // table. The cap drops the LATEST events, so what is shown is always a
      // true prefix of what happened.
      //
      // Taken from the published trace, not from the raw `SimResult`, so each
      // mark sits on a vertex of the route drawn beside it by construction.
      // `report.sim` is the fallback for the one case the extractor refuses:
      // a simulation that captured no waypoints, where it yields nothing and
      // `contactMarks` yields nothing either.
      const executed = planned.trace.selected?.executed ?? null;
      const marks =
        executed === null
          ? contactMarks(planned.report.sim, 8)
          : contactMarksFromExecuted(executed, 8);
      lastTurnRef.current = { pre: planState, planned, marks };
      setTrace(planned.trace);
      setPlanning(false);
      setModelLoading(false);

      // The live observation is over; its pending repaint must not land on top
      // of the hold that follows.
      if (liveRafRef.current) cancelAnimationFrame(liveRafRef.current);
      liveRafRef.current = 0;

      await runSequence(planState, planned, marks, true, "post-search");
      if (disposed || cancelRef.current) return;

      setPhase("animating");
      await runShot(planState, planned, marks);
      if (disposed || cancelRef.current) return;

      setPresentation("SETTLED");
      setBusy(false);
      commit(planned.report);
    })();

    return () => {
      disposed = true;
      cancelRef.current = true;
      stopLoop();
    };
    // `args.phase` is intentionally excluded: including it made the cleanup
    // fire the moment `setPhase("searching")` was called, cancelling the turn
    // it had just started. Full `state` (not just `state.turn`) lets the effect
    // re-trigger when the opponent pockets and continues with the same turn
    // index.
    //
    // `useNeural` is intentionally excluded as well, and this is a bug fix, not
    // an oversight. It used to be a dependency, so flipping the toggle during
    // the opponent's turn ran the cleanup (cancelling the in-flight search) and
    // then re-entered the effect, which bailed immediately on the
    // `phase !== "aiming"` guard — leaving the turn wedged at "searching…"
    // forever with no way out but a new rack. The body reads the live value
    // through `useNeuralRef`, so the dependency bought nothing even when it
    // worked. See opponentClaims.test.ts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, active]);

  const skip = useCallback(() => {
    skipRef.current = true;
  }, []);

  const replayDecision = useCallback(() => {
    const t = lastTurnRef.current;
    if (!t || busy) return;
    // Free: the retained trace is replayed, no search re-runs. The board shown
    // is the pre-shot board, so the routes line up with the balls that were
    // actually there.
    cancelRef.current = false;
    setBusy(true);
    setReplaying(true);
    // `"full"` — every state, paced from the completed trace. This is the
    // replay, and the panel labels it as one.
    void runSequence(t.pre, t.planned, t.marks, false, "full").then(() => {
      setBusy(false);
      setReplaying(false);
      setPresentation("SETTLED");
    });
  }, [busy, runSequence]);

  const replayShot = useCallback(() => {
    const t = lastTurnRef.current;
    if (!t || busy) return;
    cancelRef.current = false;
    // A rewatch runs at whatever presentation speed the visitor has chosen —
    // the same one first play used. There is no separate hidden replay rate,
    // which is what the old fixed 0.35x was: slow motion nobody asked for and
    // nothing labelled.
    setBusy(true);
    setReplaying(true);
    setPresentation("SHOOTING");
    void runShot(t.pre, t.planned, t.marks).then(() => {
      setBusy(false);
      setReplaying(false);
      setPresentation("SETTLED");
    });
  }, [busy, runShot]);

  useEffect(() => stopLoop, [stopLoop]);

  return {
    presentation,
    trace,
    planning,
    modelLoading,
    turnIndex,
    busy,
    replaying,
    liveCounts,
    liveStream: () => liveRef.current,
    skip,
    replayDecision,
    replayShot,
    warmModel: planner.warm,
  };
}
