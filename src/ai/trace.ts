import { type SimResult } from "../physics/engine";

// What survives of the shot trace.
//
// `describeShot` used to live here: it turned the physics event log into
// "cue → 8-ball → 5-ball → 3-ball → rail → …" and App.tsx put that string in
// the game's one status slot after every non-foul shot, for the human's shots
// as well as the opponent's. It was a debug trace occupying the line that
// should say what happened, and what happened was already computed in
// `report.outcome` immediately above it. Deleted, along with the ball/pocket
// name tables that existed only to feed it.
//
// `railsBeforePot` is different: it is read by the shot search to tell a bank
// from a direct pot, so it is decision data, not narration.

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
