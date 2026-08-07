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
// WHY 0.6x IS THE DEFAULT
//
// First playback ran at 1.0x. Measured on the shipped build, shots settle in
// 0.9-1.6 simulation seconds, and a two-cushion bank spends about 250 ms of
// that between the first rail and the pot — fast enough that the interesting
// part of a trick shot is over before you have found it. 0.6x puts a typical
// shot at 1.5-2.7 s of screen time, which is long enough to follow the cue ball
// through a bank and still short enough that five racks do not feel padded.
// 0.35x is kept as an explicit slow-motion option rather than as the default,
// and 1.0x is kept because some visitors will want the game to just move.

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

export function usePlaybackSpeed(): [PlaybackSpeed, (s: PlaybackSpeed) => void] {
  const [speed, setSpeed] = useState<PlaybackSpeed>(loadPlaybackSpeed);
  const set = useCallback((s: PlaybackSpeed) => {
    setSpeed(s);
    savePlaybackSpeed(s);
  }, []);
  return [speed, set];
}
