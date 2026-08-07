// The executed shot, as geometry: waypoints + events -> `ExecutedMotion`.
//
// WHY THIS EXISTS
//
// Three different things were being called "the shot":
//
//   1. the **candidate plan** — `TracedCandidate.cuePath` / `.path`, mirror
//      geometry produced by the candidate generator. An intention. It starts at
//      the object ball, it assumes a perfect rebound, and it is what the
//      overlay used to draw as the shot being played;
//   2. the **physics-verified path** — what a real simulation of that candidate
//      did. Until now the trace kept only its event log, never its geometry;
//   3. the **executed motion** — the authoritative run whose outcome is
//      committed to the game state, and whose waypoints the animation replays.
//
// This module derives (3) from the run that produces it, and publishes it on
// the trace so a renderer can draw the route the balls actually take instead of
// the route the generator hoped for. `contract.ts` grew `ExecutedMotion` for
// it, which is why the contract's version moved to /2.
//
// Note what this file does NOT do: it never mints a trace. `withExecutedMotion`
// takes one and returns a copy — the version stamp comes from the trace it was
// handed, and `trickOnlySourceGuard.test.ts` holds the line that only
// `ai/trace/build.ts` may produce one in the first place.
//
// RULES
//
//  - **Nothing is invented.** Every point is a position the simulation
//    recorded, every break is an event it emitted. A ball with no usable
//    waypoints gets no trajectory, and a simulation with no waypoints at all
//    returns `null`. There is no fallback geometry, because a fallback here
//    would be a drawing of a shot that did not happen.
//  - **A pocketed ball's route ends at its last measured position.** The
//    waypoint emitted after a pocket capture reports the ball at the x=0
//    sentinel (see `wasm-bridge.ts`'s `ballsFromFlat`), which is the middle of
//    the table. Reading it would put the end of a potting route — and the
//    pocket contact mark, which had this bug — nowhere near the pocket.
//  - **Simplification is bounded and measured.** Collinear runs collapse only
//    within `SIMPLIFY_TOLERANCE_M`, break points are never moved, and the
//    largest deviation actually introduced is published on the result rather
//    than assumed to be under the bound.

import { CUE_ID } from "../../game/rack";
import type { ShotEvent, SimResult, SimWaypoint } from "../../physics/engine";
import { ballPositionAdvancedTo, waypointIndexNearest } from "../../physics/waypoints";
import type {
  ContactEvent,
  DecisionTraceV1,
  ExecutedMotion,
  ExecutedTrajectory,
  TrajectoryBreak,
  TrajectoryRole,
  Vec2Trace,
} from "./contract";

/**
 * Straight-line simplification tolerance, in metres.
 *
 * 1.208 mm is **0.5 logical pixels** at the shipped view: the table is drawn
 * into the fixed 900x500 logical space `computeView` builds, which works out at
 * 413.89 px/m, and 0.5 / 413.89 = 1.2080e-3 m. Half a logical pixel is under
 * the width of the antialiasing on a 2 px stroke, and one device pixel at the
 * DPR-2 backing store the canvas actually uses.
 *
 * `executed.test.ts` recomputes that conversion from `computeView` itself, so
 * the number stops agreeing with the comment the moment the canvas changes.
 */
export const SIMPLIFY_TOLERANCE_M = 1.208e-3;

/**
 * A ball that travelled less than this in total did not go anywhere worth
 * drawing: 10 mm is about 4 logical pixels, roughly a third of a ball radius.
 * Below it a "route" is a jostle, and a two-point line across four pixels
 * reads as an artefact rather than as motion.
 */
export const MIN_TRAVEL_M = 0.01;

/**
 * Two recorded positions closer together than this count as the same place.
 * 0.1 mm — four hundredths of a pixel — so it only ever collapses the exactly
 * stationary head and tail of a ball's waypoint record, never real creep.
 */
const STILL_EPS_M = 1e-4;

const MARKED_KINDS: ReadonlySet<string> = new Set(["ball-ball", "ball-cushion", "pocket"]);

const dist = (a: Vec2Trace, b: Vec2Trace): number => Math.hypot(a.x - b.x, a.y - b.y);

/** Perpendicular distance from `p` to the segment `a`-`b`. */
function perpDistance(p: Vec2Trace, a: Vec2Trace, b: Vec2Trace): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  if (len2 <= 0) return dist(p, a);
  let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

/**
 * Ramer-Douglas-Peucker over `pts[lo..hi]`, keeping both endpoints. Returns the
 * indices retained, and reports the largest deviation it accepted through
 * `onDrop` so the caller can publish a measured bound rather than a claimed one.
 */
function rdp(
  pts: Vec2Trace[],
  lo: number,
  hi: number,
  eps: number,
  keep: Set<number>,
  onDrop: (deviation: number) => void,
): void {
  keep.add(lo);
  keep.add(hi);
  if (hi - lo < 2) return;
  let worst = -1;
  let worstAt = -1;
  for (let i = lo + 1; i < hi; i++) {
    const d = perpDistance(pts[i], pts[lo], pts[hi]);
    if (d > worst) {
      worst = d;
      worstAt = i;
    }
  }
  if (worst <= eps) {
    onDrop(worst < 0 ? 0 : worst);
    return;
  }
  rdp(pts, lo, worstAt, eps, keep, onDrop);
  rdp(pts, worstAt, hi, eps, keep, onDrop);
}

/** Positions of one ball across the waypoint stream; null where unusable. */
function trackOf(wps: readonly SimWaypoint[], ballId: number): (Vec2Trace | null)[] {
  return wps.map((wp) => {
    const b = wp.balls.find((x) => x.id === ballId);
    // A pocketed ball's recorded position is a sentinel, not a place.
    if (!b || b.pocketed) return null;
    return { x: b.pos.x, y: b.pos.y };
  });
}

/** Ball ids that appear in at least one waypoint. */
function ballIds(wps: readonly SimWaypoint[]): number[] {
  const ids = new Set<number>();
  for (const wp of wps) for (const b of wp.balls) ids.add(b.id);
  return [...ids].sort((a, b) => a - b);
}

const toContactEvents = (events: readonly ShotEvent[]): ContactEvent[] =>
  events
    .filter((e) => MARKED_KINDS.has(e.kind))
    .map((e) => ({
      kind: e.kind as ContactEvent["kind"],
      timeSec: e.time,
      balls: [...e.balls],
      cushion: e.cushion ?? null,
      pocket: e.pocket ?? null,
    }));

/**
 * Which roles the EVENT LOG supports for this ball. Never the candidate's
 * intent: a shot that was planned as a combination and struck the wrong ball
 * must read as what it did, not as what it meant.
 */
function rolesOf(sim: SimResult, ballId: number): TrajectoryRole[] {
  const roles: TrajectoryRole[] = [];
  if (ballId === CUE_ID) roles.push("cue");
  if (sim.firstContact === ballId && ballId !== CUE_ID) roles.push("first-contact");
  if (ballId !== CUE_ID) {
    // The first ball-ball event this ball took part in is the one that set it
    // moving. If the cue was not the other party, it was set moving by another
    // object ball — which is what a combination is.
    const firstHit = sim.events.find((e) => e.kind === "ball-ball" && e.balls.includes(ballId));
    if (firstHit && !firstHit.balls.includes(CUE_ID)) roles.push("combination");
  }
  if (sim.pocketed.includes(ballId)) roles.push("potted");
  return roles;
}

/**
 * The motion one executed simulation produced, as drawable geometry.
 *
 * Returns `null` when the simulation carries no waypoints — the search's
 * rollouts and the TS reference engine both produce results in that shape, and
 * a caller handed one has nothing to draw and must draw nothing.
 */
export function extractExecutedMotion(sim: SimResult): ExecutedMotion | null {
  const wps = sim.waypoints;
  if (!wps || wps.length < 2) return null;

  const contactSequence = toContactEvents(sim.events);
  let maxDeviationM = 0;
  const noted = (d: number) => {
    if (d > maxDeviationM) maxDeviationM = d;
  };

  const drafts: (ExecutedTrajectory & { _startSec: number })[] = [];

  for (const id of ballIds(wps)) {
    const track = trackOf(wps, id);
    // Usable range: from the first recorded position to the last one. A ball
    // that is pocketed simply has no positions after the capture.
    let first = track.findIndex((p) => p !== null);
    if (first < 0) continue;
    let last = first;
    for (let i = first; i < track.length; i++) if (track[i] !== null) last = i;
    // Any gap inside the range would mean a ball vanished and came back, which
    // the simulator cannot produce; refuse to bridge one rather than draw over it.
    for (let i = first; i <= last; i++) if (track[i] === null) return null;

    const at = (i: number): Vec2Trace => track[i] as Vec2Trace;

    // Trim the stationary head and tail: the record runs for the whole shot,
    // but the ball's ROUTE starts when it moves and ends where it comes to rest.
    let startIdx = first;
    while (startIdx < last && dist(at(startIdx + 1), at(first)) <= STILL_EPS_M) startIdx++;
    let stopIdx = last;
    while (stopIdx > startIdx && dist(at(stopIdx - 1), at(last)) <= STILL_EPS_M) stopIdx--;
    if (stopIdx <= startIdx) continue;

    let travel = 0;
    for (let i = startIdx + 1; i <= stopIdx; i++) travel += dist(at(i), at(i - 1));
    if (travel < MIN_TRAVEL_M) continue;

    const rawPts: Vec2Trace[] = [];
    const rawTimes: number[] = [];
    for (let i = startIdx; i <= stopIdx; i++) {
      rawPts.push(at(i));
      rawTimes.push(wps[i].time);
    }

    // A pot needs one derived point, and only this one. Waypoints are up to
    // 50 ms apart, so the last RECORDED position of a ball dropping at speed
    // can be 0.2 m — 80 logical pixels — short of the pocket, and a route that
    // stops there says the ball stopped short of a pocket it went into. The
    // point appended is `ballPositionAdvancedTo`: the simulator's own analytic
    // motion, from a recorded state, to the time it emitted the capture at, and
    // the identical computation the animation uses to place the ball between
    // waypoints. Read its comment before changing this.
    const capture = sim.events.find((e) => e.kind === "pocket" && e.balls[0] === id);
    let endsAtCapture = false;
    if (capture) {
      const atCapture = ballPositionAdvancedTo(sim, id, capture.time);
      if (atCapture && capture.time > rawTimes[rawTimes.length - 1]) {
        rawPts.push({ x: atCapture.x, y: atCapture.y });
        rawTimes.push(capture.time);
        endsAtCapture = true;
      }
    }

    const local = (wpIndex: number) =>
      Math.max(0, Math.min(rawPts.length - 1, wpIndex - startIdx));

    // Breaks: the simulator's own events for this ball, at the waypoint each
    // was recorded against. `waypointIndexNearest` is the same lookup
    // `render/annotate.ts` uses for contact marks, so a mark and the vertex it
    // belongs to are the same point by construction rather than by agreement.
    const rawBreaks: { at: number; ev: ShotEvent }[] = [];
    for (const e of sim.events) {
      if (!MARKED_KINDS.has(e.kind)) continue;
      if (!e.balls.includes(id)) continue;
      const isCapture = endsAtCapture && e === capture;
      rawBreaks.push({
        at: isCapture ? rawPts.length - 1 : local(waypointIndexNearest(wps, e.time)),
        ev: e,
      });
    }

    // Simplify each run between consecutive breaks on its own, so a break point
    // is never a candidate for removal and never moves.
    const anchors = [0, ...rawBreaks.map((b) => b.at), rawPts.length - 1]
      .filter((v, i, a) => a.indexOf(v) === i)
      .sort((a, b) => a - b);
    const keep = new Set<number>();
    for (let k = 0; k + 1 < anchors.length; k++) {
      rdp(rawPts, anchors[k], anchors[k + 1], SIMPLIFY_TOLERANCE_M, keep, noted);
    }
    if (anchors.length === 1) keep.add(anchors[0]);
    const kept = [...keep].sort((a, b) => a - b);
    const remap = new Map(kept.map((oldIdx, newIdx) => [oldIdx, newIdx]));

    const points = kept.map((i) => rawPts[i]);
    const timesSec = kept.map((i) => rawTimes[i]);
    if (points.length < 2) continue;

    const breaks: TrajectoryBreak[] = rawBreaks.map((b) => ({
      at: remap.get(b.at) ?? 0,
      kind:
        b.ev.kind === "ball-ball" ? "ball-contact" : b.ev.kind === "pocket" ? "pocket" : "cushion",
      timeSec: b.ev.time,
      withBall: b.ev.kind === "ball-ball" ? (b.ev.balls.find((x) => x !== id) ?? null) : null,
      cushion: b.ev.cushion ?? null,
      pocket: b.ev.pocket ?? null,
    }));

    drafts.push({
      ballId: id,
      roles: rolesOf(sim, id),
      order: 0,
      points,
      timesSec,
      breaks,
      startSec: timesSec[0],
      endSec: timesSec[timesSec.length - 1],
      pocketed: sim.pocketed.includes(id),
      endsAtCapture,
      _startSec: timesSec[0],
    });
  }

  if (drafts.length === 0) return null;

  drafts.sort((a, b) => a._startSec - b._startSec || a.ballId - b.ballId);
  const trajectories: ExecutedTrajectory[] = drafts.map((d, i) => ({
    ballId: d.ballId,
    roles: d.roles,
    order: i,
    points: d.points,
    timesSec: d.timesSec,
    breaks: d.breaks,
    startSec: d.startSec,
    endSec: d.endSec,
    pocketed: d.pocketed,
    endsAtCapture: d.endsAtCapture,
  }));

  return {
    durationSec: sim.duration,
    trajectories,
    contactSequence,
    simplifyToleranceM: SIMPLIFY_TOLERANCE_M,
    maxDeviationM,
  };
}

/**
 * Publish the executed motion on a finished trace.
 *
 * The search publishes its trace BEFORE the shot is played — that ordering is
 * why `executed` is nullable and why this is a separate step rather than a
 * field `buildDecisionTrace` could fill in. Returns a new trace; nothing is
 * mutated, so the object the search handed over stays exactly as it was.
 */
export function withExecutedMotion(
  trace: DecisionTraceV1,
  executed: ExecutedMotion | null,
): DecisionTraceV1 {
  if (trace.selected === null) return trace;
  return { ...trace, selected: { ...trace.selected, executed } };
}
