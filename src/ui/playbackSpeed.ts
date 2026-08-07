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
// WHY 0.6x, AND WHY IT DOES NOT APPLY TO THE WHOLE SHOT
//
// First playback ran at a flat 1.0x. Measured over 399 real simulations across
// four boards: a shot runs 5.39 simulation seconds at the median (p90 6.74),
// the last contact lands at 3.04 s (p90 4.82), and the remaining 1.99 s (p90
// 3.72, max 5.90) is balls coasting to a stop with nothing left to happen —
// **37% of the median shot, after everything interesting is over**.
//
// A flat multiplier is the wrong instrument for that shape. At a flat 0.6x the
// median shot becomes a nine-second animation, and three of those nine seconds
// are watching a ball roll to rest. So the chosen speed governs the part with
// contacts in it, and the settle afterwards is capped at `SETTLE_MAX_SEC` of
// screen time — never played SLOWER than the chosen speed, only faster.
//
// At the 0.6x default that puts the median shot at 3.04/0.6 + 1.0 = 6.1 s
// against 5.4 s before: the part you have to follow is 1.7x longer, the part
// you do not is up to 3.7x shorter, and the whole thing is 13% longer rather
// than 67% longer. 0.35x stays available as explicit slow motion, and 1.0x
// stays because some visitors want the game to just move.

import { useCallback, useState } from "react";

/** The offered speeds, slowest first. Three is a choice, not a slider. */
export const PLAYBACK_SPEEDS = [0.35, 0.6, 1] as const;
export type PlaybackSpeed = (typeof PLAYBACK_SPEEDS)[number];

export const DEFAULT_PLAYBACK_SPEED: PlaybackSpeed = 0.6;

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
