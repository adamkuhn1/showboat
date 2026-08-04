import { type SimResult, type ShotEvent } from "../physics/engine";
import { CUE_ID, EIGHT_ID } from "../game/rack";

// Build a shot caption from the physics event trace: "cue → rail → 3-ball → corner".

const ballName = (id: number): string => {
  if (id === CUE_ID) return "cue";
  if (id === EIGHT_ID) return "8-ball";
  return `${id}-ball`;
};

const pocketName = (id: string): string => {
  const map: Record<string, string> = {
    bl: "bottom-left",
    tl: "top-left",
    br: "bottom-right",
    tr: "top-right",
    sb: "bottom-side",
    st: "top-side",
  };
  return `${map[id] ?? id} pocket`;
};

const segmentFor = (e: ShotEvent): string | null => {
  switch (e.kind) {
    case "ball-cushion":
      return "rail";
    case "ball-ball": {
      const other = e.balls.find((id) => id !== CUE_ID);
      // Name the struck ball (skip pure cue mention; the caption starts at cue).
      return other !== undefined ? ballName(other) : ballName(e.balls[1]);
    }
    case "pocket":
      return e.pocket ? pocketName(e.pocket) : "pocket";
    default:
      return null;
  }
};

// Capped at 8 segments so a chaotic multi-ball bounce can't overflow the status bar.
export const describeShot = (sim: SimResult): string => {
  const MAX_SEGMENTS = 8;
  const segments: string[] = ["cue"];
  let truncated = false;
  for (const e of sim.events) {
    const seg = segmentFor(e);
    if (seg && segments[segments.length - 1] !== seg) {
      if (segments.length >= MAX_SEGMENTS) { truncated = true; break; }
      segments.push(seg);
    }
  }
  if (segments.length === 1) return "cue rolled without contact";
  return segments.join(" → ") + (truncated ? " → …" : "");
};

// Count cushion contacts before the first ball is pocketed.
// Returns 0 if nothing was pocketed — don't credit rails to a missed shot.
export const railsBeforePot = (sim: SimResult): number => {
  let rails = 0;
  for (const e of sim.events) {
    if (e.kind === "ball-cushion") rails++;
    if (e.kind === "pocket") return rails;
  }
  return 0;
};
