// Cushion contacts and combination order, derived from the shot that actually
// decides the outcome.
//
// These marks come from `SimResult.events` + `SimResult.waypoints` — the real
// WASM run whose waypoints the animation replays — and NOT from the candidate
// generator's planned mirror-geometry vertices. `candidate.path` is the
// search's intention; the event trace is the physics' result. When they differ,
// the physics is right, and annotating the intention would be drawing a claim
// the table is about to contradict.
//
// `ShotEvent` carries no position, so a contact's location is looked up: the
// position of ball `b` at `event.time` is `b`'s position in the waypoint
// nearest that time. Waypoints are emitted after every resolved event, so the
// nearest one is the event itself in every case that matters. That lookup lives
// in `physics/waypoints.ts` because `ai/trace/executed.ts` places route
// vertices with it too — a mark and the vertex it sits on are then the same
// point by construction, not by two files happening to agree.
//
// The exception is a POCKET. The waypoint recorded for a capture reports the
// ball at the off-table sentinel, so a pocket mark placed by the general rule
// landed in the middle of the cloth; it is placed at the ball's last measured
// position instead. See `lastMeasuredPosition`.

import type { SimResult, ShotEvent } from "../physics/engine";
import type { Vec2 } from "../physics/vec";
import { ballPositionAdvancedTo, ballPositionAt, lastMeasuredPosition } from "../physics/waypoints";
import type { ExecutedMotion } from "../ai/trace/contract";

export type ContactKind = "cushion" | "ball" | "pocket";

export interface ContactMark {
  kind: ContactKind;
  /** World coordinates of the contact. */
  at: Vec2;
  timeSec: number;
  /** 1-based order within the shot — this is the "combination order". */
  order: number;
  /** The ball whose position the mark sits on. */
  ballId: number;
  /** Cushion side for `cushion`, pocket id for `pocket`, else null. */
  detail: string | null;
}

const MARKED: ReadonlySet<string> = new Set(["ball-cushion", "ball-ball", "pocket"]);

/** Where a mark for `e` belongs, or null if the simulation cannot say. */
const markPosition = (sim: SimResult, e: ShotEvent, ballId: number): Vec2 | null =>
  e.kind === "pocket"
    ? // The pocket instant, advanced from the last recorded state — see
      // `ballPositionAdvancedTo`. `lastMeasuredPosition` is the floor for a
      // capture the advance cannot reach (a ball already classified stationary).
      ballPositionAdvancedTo(sim, ballId, e.time) ?? lastMeasuredPosition(sim, ballId)
    : ballPositionAt(sim, ballId, e.time);

/**
 * Contact marks for one executed shot, in time order.
 *
 * `maxMarks` exists so a break (20+ events) does not bury the table; it drops
 * the *latest* events, never a reordering, so the marks shown are always a
 * true prefix of what happened.
 */
export function contactMarks(sim: SimResult, maxMarks = 12): ContactMark[] {
  if (!sim.waypoints || sim.waypoints.length === 0) return [];

  const out: ContactMark[] = [];
  for (const e of sim.events as ShotEvent[]) {
    if (!MARKED.has(e.kind)) continue;
    const ballId = e.balls[0];
    if (ballId === undefined) continue;
    const at = markPosition(sim, e, ballId);
    if (!at) continue;
    out.push({
      kind: e.kind === "ball-cushion" ? "cushion" : e.kind === "pocket" ? "pocket" : "ball",
      at,
      timeSec: e.time,
      order: out.length + 1,
      ballId,
      detail: e.cushion ?? e.pocket ?? null,
    });
    if (out.length >= maxMarks) break;
  }
  return out;
}

/**
 * The same marks, from the published trace instead of from a raw `SimResult`.
 *
 * This is the path the app uses. It matters that it is: a mark taken from here
 * sits on a vertex of the route drawn beside it — `TrajectoryBreak.at` indexes
 * the very points the overlay strokes — so "the marker is on the line" is a
 * property of the data structure and not something two files have to keep
 * agreeing about. `annotate.test.ts` checks it against `contactMarks` on the
 * same shot, so the two cannot drift.
 *
 * An event whose ball has no trajectory (it never travelled far enough to be
 * given one) is skipped rather than placed by guesswork.
 */
export function contactMarksFromExecuted(m: ExecutedMotion, maxMarks = 12): ContactMark[] {
  const byBall = new Map(m.trajectories.map((t) => [t.ballId, t]));
  const out: ContactMark[] = [];
  for (const e of m.contactSequence) {
    const ballId = e.balls[0];
    if (ballId === undefined) continue;
    const traj = byBall.get(ballId);
    if (!traj) continue;
    const brk = traj.breaks.find(
      (b) =>
        b.timeSec === e.timeSec &&
        ((e.kind === "ball-ball" && b.kind === "ball-contact") ||
          (e.kind === "ball-cushion" && b.kind === "cushion") ||
          (e.kind === "pocket" && b.kind === "pocket")),
    );
    if (!brk) continue;
    const at = traj.points[brk.at];
    if (!at) continue;
    out.push({
      kind: e.kind === "ball-cushion" ? "cushion" : e.kind === "pocket" ? "pocket" : "ball",
      at: { x: at.x, y: at.y },
      timeSec: e.timeSec,
      order: out.length + 1,
      ballId,
      detail: e.cushion ?? e.pocket ?? null,
    });
    if (out.length >= maxMarks) break;
  }
  return out;
}
