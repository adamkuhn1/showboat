// The canvas's accessible name.
//
// A `<canvas>` exposes no accessible content at all — to anything that is not
// reading pixels it is an unlabelled box, which is what Showboat's table was.
// The portfolio's network diagram solves the same problem the same way
// (`role="img"` plus a label that says what the picture shows), and this is
// that label for a picture whose content changes every shot: it is rebuilt from
// the live `GameState`, so it describes the board as it stands rather than the
// board as it was racked.
//
// Deliberately coarse. It reports whose shot it is, the group situation and how
// much is left — the things a sighted player reads off the felt in a glance.
// Ball coordinates are not in it: fifteen positions read aloud are not a
// picture, and the per-ball detail is already in the `.rack` list beside it.

import { EIGHT_ID, SOLIDS, STRIPES } from "../game/rack";
import type { GameState, PlayerId } from "../game/state";

const count = (state: GameState, ids: number[]): number =>
  ids.filter((id) => {
    const ball = state.balls.find((b) => b.id === id);
    return ball !== undefined && !ball.pocketed;
  }).length;

const plural = (n: number, word: string): string => `${n} ${word}${n === 1 ? "" : "s"}`;

export function boardLabel(state: GameState, aiPlayer: PlayerId): string {
  const opening = "Eight-ball table, seen from above.";

  if (state.winner !== null) {
    return `${opening} Game over — ${state.winner === aiPlayer ? "the opponent wins" : "you win"}.`;
  }

  const yours = state.turn !== aiPlayer;
  const group = state.groups[state.turn];
  const whose = `${yours ? "Your shot" : "The opponent's shot"}, ${
    group === null ? "the table is open" : `on ${group}`
  }.`;

  const solids = count(state, SOLIDS);
  const stripes = count(state, STRIPES);
  const eight = count(state, [EIGHT_ID]) > 0 ? ", and the 8" : "";
  const left = `${plural(solids, "solid")} and ${plural(stripes, "stripe")} still up${eight}.`;

  const inHand = state.ballInHand !== false ? " Ball in hand." : "";
  return `${opening} ${whose} ${left}${inHand}`;
}
