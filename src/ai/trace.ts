import { type SimResult, type ShotEvent } from "../physics/engine";
import { CUE_ID, EIGHT_ID } from "../game/rack";

// Turn a real physics event trace into a human-readable caption, e.g.
// "cue → rail → 3-ball → corner". This reads ONLY from the simulation's actual
// event sequence — it is a presentation layer over real decision/physics data,
// never a scripted description. Same mechanism CueTip uses to caption shots.

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

// Build the arrow-joined caption from the ordered event trace. Long shots
// (breaks especially) produce dozens of events; the caption keeps the first
// `maxSegments` and says how much it elided rather than flooding the status
// line.
export const describeShot = (sim: SimResult, maxSegments = 9): string => {
  const segments: string[] = ["cue"];
  for (const e of sim.events) {
    const seg = segmentFor(e);
    if (seg && segments[segments.length - 1] !== seg) segments.push(seg);
  }
  if (segments.length === 1) return "cue rolled without contact";
  if (segments.length > maxSegments) {
    const elided = segments.length - maxSegments;
    return `${segments.slice(0, maxSegments).join(" → ")} → … (${elided} more)`;
  }
  return segments.join(" → ");
};

// Count cushion contacts before the first ball is pocketed — used by the overlay
// to flag bank/multi-wall shots (which is what makes an emergent trick shot
// legible, not any scripting).
export const railsBeforePot = (sim: SimResult): number => {
  let rails = 0;
  for (const e of sim.events) {
    if (e.kind === "ball-cushion") rails++;
    if (e.kind === "pocket") break;
  }
  return rails;
};
