// Playback speed: a PRESENTATION control and nothing else.
//
// It multiplies wall-clock elapsed time on its way to becoming simulation time.
// That is the whole mechanism. It cannot reach a physics constant, a candidate
// action, a simulation, or the committed game state — those are all decided
// before a single frame is painted, by `ui/planner/plan.ts`, and the animation
// only reads the result. `playbackSpeed.test.ts` proves the reparameterisation
// property (same simulation time, same ball states, whatever the speed) and
// pins the physics constants so a future pacing change cannot quietly move one.
//
// WHY THE COAST TO REST IS CAPPED
//
// First playback ran at a flat 1.0x. Measured over 399 real simulations across
// four boards: a shot runs 5.39 simulation seconds at the median (p90 6.74),
// the last contact lands at 3.04 s (p90 4.82), and the remaining 1.99 s (p90
// 3.72, max 5.90) is balls coasting to a stop with nothing left to happen —
// **37% of the median shot, after everything interesting is over**.
//
// So the settle is capped at `SETTLE_MAX_SEC` of screen time — never played
// SLOWER than the chosen speed, only faster. That rule is unchanged and lives
// in `settleRate` below; `ui/pacing.ts` calls it rather than restating it.
//
// WHY 0.5x, AND HOW IT WAS CHOSEN
//
// The previous default was 0.6x against a FLAT multiplier, and Adam's report on
// it was that the balls were still too fast to understand and the route was
// hard to reconcile with the motion. Two things changed in response: the rate
// is now a curve that spends its slowness on the contacts (`ui/pacing.ts`), and
// this number was re-derived rather than inherited.
//
// The metric is how far a ball moves across the screen between two displayed
// frames, because that is what smooth pursuit of a small object can and cannot
// follow. The ball is 23.7 logical pixels across at the shipped view. Measured
// by `qa/speed-sweep.ts` over 61 real simulations, walking each shot frame by
// frame through the real pacing curve at 60 Hz:
//
//   speed   mean screen s   px/frame p90   at contacts p90   frames > 1 ball
//   0.35x           11.02            9.0               9.6              0.1%
//   0.5x             7.98           12.7              13.7              1.3%
//   0.6x             6.85           15.0              16.3              2.3%
//   1x               5.53           19.9              23.0              5.7%
//
// 0.6x moves 0.69 ball-widths per frame at the p90 contact — which is the
// complaint, in numbers. 0.35x fixes it and costs 11 seconds a shot, which is
// not a game anybody finishes a rack of. 0.5x lands at 0.58 ball-widths for
// about a second more per shot than 0.6x, and combined with the contact curve
// the moments that matter now play at 0.225x against the old default's flat
// 0.6x — 2.7x slower where the shot actually happens.
//
// 0.35x stays as the explicitly slower replay and 1.0x stays because some
// visitors want the game to just move. 0.6x is retired: the curve superseded it.

import { useCallback, useState } from "react";

/** The offered speeds, slowest first. Three is a choice, not a slider. */
export const PLAYBACK_SPEEDS = [0.35, 0.5, 1] as const;
export type PlaybackSpeed = (typeof PLAYBACK_SPEEDS)[number];

export const DEFAULT_PLAYBACK_SPEED: PlaybackSpeed = 0.5;

/**
 * Session storage, deliberately: the choice should survive a new rack and a
 * reload of the embed, and should not follow the visitor to another day as a
 * setting they no longer remember making.
 */
export const PLAYBACK_SPEED_KEY = "showboat.playbackSpeed";

const isSpeed = (v: unknown): v is PlaybackSpeed =>
  (PLAYBACK_SPEEDS as readonly number[]).includes(v as number);

export function loadPlaybackSpeed(): PlaybackSpeed {
  try {
    const raw = globalThis.sessionStorage?.getItem(PLAYBACK_SPEED_KEY);
    const n = raw === null || raw === undefined ? NaN : Number(raw);
    return isSpeed(n) ? n : DEFAULT_PLAYBACK_SPEED;
  } catch {
    // Storage can throw outright (Safari private mode, a sandboxed iframe with
    // storage blocked). A speed control is not worth failing a page over.
    return DEFAULT_PLAYBACK_SPEED;
  }
}

export function savePlaybackSpeed(speed: PlaybackSpeed): void {
  try {
    globalThis.sessionStorage?.setItem(PLAYBACK_SPEED_KEY, String(speed));
  } catch {
    /* see loadPlaybackSpeed */
  }
}

/** The label the UI must use. Never "slow motion" on its own — that reads as a
 *  claim about the simulation rather than about the screen. */
export const PLAYBACK_SPEED_LABEL = "presentation speed";

/** Screen seconds the post-contact settle is allowed to take, at most. */
export const SETTLE_MAX_SEC = 1;

/**
 * The rate for the stretch after the shot's last contact.
 *
 * Never slower than the chosen speed — asking for slow motion must not make
 * the coast to rest faster than you asked for — and fast enough that the
 * remaining `tailSimSec` fits inside `SETTLE_MAX_SEC` of screen time. Takes
 * plain numbers so this module keeps its distance from the simulation: the
 * caller reads the last contact off the shot's own event log.
 */
export const settleRate = (tailSimSec: number, speed: number): number =>
  tailSimSec <= 0 ? speed : Math.max(speed, tailSimSec / SETTLE_MAX_SEC);

export function usePlaybackSpeed(): [PlaybackSpeed, (s: PlaybackSpeed) => void] {
  const [speed, setSpeed] = useState<PlaybackSpeed>(loadPlaybackSpeed);
  const set = useCallback((s: PlaybackSpeed) => {
    setSpeed(s);
    savePlaybackSpeed(s);
  }, []);
  return [speed, set];
}
