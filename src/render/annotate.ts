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
// nearest one is the event itself in every case that matters.

import type { SimResult, ShotEvent } from "../physics/engine";
import type { Vec2 } from "../physics/vec";

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

/** Position of `ballId` at `t`, from the waypoint nearest `t`. */
function positionAt(sim: SimResult, ballId: number, t: number): Vec2 | null {
  const wps = sim.waypoints;
  if (!wps || wps.length === 0) return null;

  let lo = 0;
  let hi = wps.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (wps[mid].time <= t) lo = mid;
    else hi = mid - 1;
  }
  // Prefer whichever of the bracketing waypoints is closer in time.
  const a = wps[lo];
  const b = wps[Math.min(lo + 1, wps.length - 1)];
  const wp = Math.abs(a.time - t) <= Math.abs(b.time - t) ? a : b;
  const ball = wp.balls.find((x) => x.id === ballId);
  return ball ? { x: ball.pos.x, y: ball.pos.y } : null;
}

const MARKED: ReadonlySet<string> = new Set(["ball-cushion", "ball-ball", "pocket"]);

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
    const at = positionAt(sim, ballId, e.time);
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
