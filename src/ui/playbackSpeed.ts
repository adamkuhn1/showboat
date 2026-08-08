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
// THE SPEED IS THE WHOLE MAPPING
//
// One number, applied to the entire shot. It is not blended with anything, it
// does not change at a contact, and it does not change when the balls are
// nearly stopped. `ui/pacing.ts` argues why that has to be true on a table with
// sixteen balls on it; this module is only where the number comes from.
//
// WHY 0.5x, AND HOW IT WAS CHOSEN
//
// The metric is how far a ball moves across the screen between two displayed
// frames, because that is what smooth pursuit of a small object can and cannot
// follow. The ball is 23.7 logical pixels across at the shipped view. Measured
// by `qa/time-mapping.ts` over 82 real simulations, walking each shot frame by
// frame at 60 Hz:
//
//   speed   mean screen s   px/frame p90   px/frame max   ball widths at p90
//   0.35x          14.84            5.6           20.5                 0.24
//   0.5x           10.39            8.0           29.3                 0.34
//   0.6x            8.66            9.5           35.2                 0.40
//   0.75x           6.93           11.9           44.0                 0.50
//   1x              5.20           15.8           58.6                 0.67
//
// 0.5x is a third of a ball width per frame at the p90 and never more than 1.24
// ball widths at the very fastest instant of the very fastest shot, which is
// the break. It costs about ten and a half seconds a shot; a tap on the felt or
// the space bar runs the rest out at once for anyone who does not want to watch
// it. 0.35x stays as the explicitly slower replay and 1.0x stays because some
// visitors want the game to just move.

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

export function usePlaybackSpeed(): [PlaybackSpeed, (s: PlaybackSpeed) => void] {
  const [speed, setSpeed] = useState<PlaybackSpeed>(loadPlaybackSpeed);
  const set = useCallback((s: PlaybackSpeed) => {
    setSpeed(s);
    savePlaybackSpeed(s);
  }, []);
  return [speed, set];
}
