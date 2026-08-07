// Where the slowness goes.
//
// THE PROBLEM THIS SOLVES, IN ADAM'S WORDS: the balls are too fast to
// understand, and the route is hard to reconcile with the motion, even at 0.6x.
//
// The previous model had one rate for the whole shot up to the last contact,
// and a capped sprint afterwards. That is the right idea applied at too coarse
// a grain. A shot is not uniformly interesting: a five-rail bank spends most of
// its time crossing open felt in a straight line, and all of its meaning in the
// half-second around each cushion. A single multiplier either makes the whole
// thing slow — which is how a 5.4 s shot became a nine-second animation nobody
// wants to watch twice — or leaves the contacts flashing past.
//
// So the rate is a function of simulation time, built from the shot's OWN event
// log: slow inside a window around every real contact, faster across the
// stretches with nothing in them, and the existing cap on the coast to rest.
//
// WHAT THIS IS NOT
//
// It is not physics. Nothing here can reach a physics constant, a candidate
// action, a simulation or the committed game state — all of those are fixed
// before a frame is painted. This reparameterises time on the way from
// wall-clock to simulation time and does nothing else, which is exactly what
// `playbackSpeed.ts` already documented for the flat multiplier.
//
// It is also not a new set of events. Every time in `contactTimes` is read off
// `SimResult.events` — the same log the contact marks and the route vertices
// come from. A shot with no contacts gets no dips, because it had none.

import { settleRate } from "./playbackSpeed";

/**
 * Simulation seconds either side of a contact that are played at the slow rate.
 *
 * 0.18 s. At the slow rate below that is ~0.7 s of screen time per contact at
 * the 0.5x default — long enough to see a ball arrive at a cushion, deform the
 * route and leave, which is the thing that was being missed. Wider starts
 * merging adjacent contacts in a combination into one long crawl.
 */
export const CONTACT_WINDOW_SEC = 0.18;

/** Multiplier on the chosen speed inside a contact window. */
export const CONTACT_RATE = 0.45;

/**
 * Multiplier on the chosen speed across open felt.
 *
 * The time spent on contacts has to come from somewhere or every shot just gets
 * longer. It comes from here.
 */
export const COAST_RATE = 1.7;

/**
 * Absolute ceiling on the coast rate, as a multiple of real time.
 *
 * Coasting may be sped up relative to the chosen presentation speed, but not
 * past reality: a ball crossing the table faster than it physically would is a
 * different kind of lie from a slow one, and it also undoes the legibility this
 * whole module exists for.
 */
export const COAST_MAX_ABSOLUTE = 1;

export interface PacingInput {
  /** Times of every real contact in this shot, in simulation seconds. */
  contactTimes: number[];
  /** Total simulation seconds of the shot. */
  durationSec: number;
  /** The visitor's chosen presentation speed. */
  speed: number;
}

export interface Pacing {
  /** Wall-clock-to-simulation rate at a given simulation time. */
  rateAt: (simTime: number) => number;
  /** Simulation time of the last contact; the coast to rest starts here. */
  lastContactSec: number;
  /** Simulation seconds of coast after the last contact. */
  tailSec: number;
}

/**
 * `1` at a contact, falling to `0` at `CONTACT_WINDOW_SEC` away from every
 * contact. Smooth (cosine), so the rate does not step — a step in playback rate
 * reads as a stutter and gets blamed on the physics.
 */
const focus = (t: number, contacts: number[]): number => {
  let nearest = Infinity;
  for (const c of contacts) {
    const d = Math.abs(t - c);
    if (d < nearest) nearest = d;
    // The list is in time order, so once a contact is beyond the window and
    // ahead of `t`, nothing later can be nearer.
    if (c > t && d > CONTACT_WINDOW_SEC) break;
  }
  if (!Number.isFinite(nearest) || nearest >= CONTACT_WINDOW_SEC) return 0;
  return 0.5 * (1 + Math.cos((nearest / CONTACT_WINDOW_SEC) * Math.PI));
};

/**
 * Build the rate schedule for one shot.
 *
 * The three regimes, in the order they apply:
 *   1. after the last contact — the coast to rest, capped so it cannot take
 *      more than `SETTLE_MAX_SEC` of screen time and never slower than the
 *      chosen speed. This is the existing rule and is unchanged.
 *   2. inside a contact window — `speed * CONTACT_RATE`.
 *   3. everywhere else — `speed * COAST_RATE`, but never faster than real time.
 */
export function buildPacing(input: PacingInput): Pacing {
  const { durationSec, speed } = input;
  const contacts = [...input.contactTimes].sort((a, b) => a - b);
  const lastContactSec = contacts.length > 0 ? contacts[contacts.length - 1] : 0;
  const tailSec = Math.max(0, durationSec - lastContactSec);
  // The existing coast-to-rest rule, called rather than re-derived: fast enough
  // that the remaining coast fits inside `SETTLE_MAX_SEC` of screen time, never
  // slower than what the visitor asked for. `playbackSpeed.test.ts` owns it.
  const settle = settleRate(tailSec, speed);
  const coast = Math.min(speed * COAST_RATE, COAST_MAX_ABSOLUTE);
  const contactRate = speed * CONTACT_RATE;

  return {
    lastContactSec,
    tailSec,
    rateAt: (simTime: number): number => {
      if (simTime >= lastContactSec) return settle;
      const f = focus(simTime, contacts);
      // Blend rather than switch, so there is no discontinuity at the window
      // edge. `f` is already smooth, so the rate is too.
      return contactRate * f + coast * (1 - f);
    },
  };
}

/**
 * Screen seconds this pacing will take, integrated numerically.
 *
 * Used by the QA harness to compare candidate defaults without watching a
 * hundred shots, and by the test that pins the shape of the curve. Numerical
 * because `rateAt` is piecewise and there is no reason to have a closed form
 * for something evaluated a few hundred times offline.
 */
export function screenSeconds(pacing: Pacing, durationSec: number, steps = 2000): number {
  const dt = durationSec / steps;
  let screen = 0;
  for (let i = 0; i < steps; i++) {
    const rate = pacing.rateAt(i * dt + dt / 2);
    screen += rate > 0 ? dt / rate : 0;
  }
  return screen;
}
