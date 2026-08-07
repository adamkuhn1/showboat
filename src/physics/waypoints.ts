// Reading positions out of a simulation's waypoint stream.
//
// One rule, in one place, because two callers depend on agreeing exactly:
// `render/annotate.ts` puts a contact mark at a ball's position at an event
// time, and `ai/trace/executed.ts` puts a route vertex there. If those two used
// different lookups the mark would sit off the line it belongs to. They don't:
// both call `waypointIndexNearest`.
//
// Waypoints are emitted after every resolved event (see the WASM core and the
// TS reference engine's `buildAnimTrack`), so the waypoint nearest an event's
// time IS that event, to floating point, in every case that matters. Nothing
// here interpolates: an interpolated contact position is a guess about where a
// ball was, and the simulation already recorded where it was.

import type { SimResult, SimWaypoint } from "./engine";
import type { Vec2 } from "./vec";

/**
 * Index of the waypoint closest in time to `t`. `-1` when there are none.
 *
 * Binary-searches for the last waypoint at or before `t`, then takes whichever
 * of the two bracketing waypoints is nearer.
 */
export function waypointIndexNearest(wps: readonly SimWaypoint[], t: number): number {
  if (wps.length === 0) return -1;
  let lo = 0;
  let hi = wps.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (wps[mid].time <= t) lo = mid;
    else hi = mid - 1;
  }
  const next = Math.min(lo + 1, wps.length - 1);
  return Math.abs(wps[lo].time - t) <= Math.abs(wps[next].time - t) ? lo : next;
}

/**
 * Position of `ballId` at `t`, from the waypoint nearest `t`.
 *
 * Null when the simulation captured no waypoints, or when that waypoint does
 * not carry the ball — including the case that matters most: a pocketed ball,
 * whose recorded x is a sentinel rather than a place on the table. A caller
 * that gets null must draw nothing, never a plausible substitute.
 */
export function ballPositionAt(sim: SimResult, ballId: number, t: number): Vec2 | null {
  const wps = sim.waypoints;
  if (!wps || wps.length === 0) return null;
  const i = waypointIndexNearest(wps, t);
  const ball = wps[i].balls.find((b) => b.id === ballId);
  if (!ball || ball.pocketed) return null;
  return { x: ball.pos.x, y: ball.pos.y };
}

/**
 * The last place `ballId` was actually recorded — its resting position, or, for
 * a ball that was pocketed, the lip it was last seen at.
 *
 * This is the position a pocket contact belongs at. `ballPositionAt` cannot
 * supply it: the waypoint emitted for a pocket capture reports the ball at the
 * x=0 sentinel `wasm-bridge.ts` writes for a ball that is off the table, which
 * is the middle of the cloth. Marks drawn from it landed ~0.99 m — about 410
 * logical pixels, half the table — from the pocket they claimed to be on.
 */
export function lastMeasuredPosition(sim: SimResult, ballId: number): Vec2 | null {
  const wps = sim.waypoints;
  if (!wps || wps.length === 0) return null;
  for (let i = wps.length - 1; i >= 0; i--) {
    const b = wps[i].balls.find((x) => x.id === ballId);
    if (b && !b.pocketed) return { x: b.pos.x, y: b.pos.y };
  }
  return null;
}
