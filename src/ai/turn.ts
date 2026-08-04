import { type GameState, type PlayerId } from "../game/state";
import { type Table } from "../physics/table";
import { SOLIDS, STRIPES, EIGHT_ID, CUE_ID } from "../game/rack";
import { searchBaseline, type SearchResult, type SearchConfig } from "./shotSearch";

// Determine the legal target ball ids for a player: their group's remaining
// balls, or the 8-ball once the group is cleared; on an open table, everything
// except the cue and the 8. This feeds the candidate generator / search with the
// correct objective — the reward is outcomes only, never "bank because we said
// so".
export const legalTargets = (state: GameState, player: PlayerId): number[] => {
  const group = state.groups[player];
  const live = (ids: number[]) =>
    ids.filter((id) => {
      const b = state.balls.find((x) => x.id === id);
      return b && !b.pocketed;
    });

  if (group === null) {
    // Open table: any solid or stripe is a legal first target.
    return [...live(SOLIDS), ...live(STRIPES)];
  }
  const groupIds = group === "solids" ? SOLIDS : STRIPES;
  const remaining = live(groupIds);
  if (remaining.length === 0) {
    // Group cleared -> the 8-ball is the target.
    const eight = state.balls.find((b) => b.id === EIGHT_ID);
    return eight && !eight.pocketed ? [EIGHT_ID] : [];
  }
  return remaining;
};

// Run the baseline search for whichever player is to move. Returns the full
// SearchResult (best shot + all candidate stats) so the overlay can render the
// real decision data.
export const planTurn = (
  state: GameState,
  table: Table,
  config?: SearchConfig,
): SearchResult => {
  const targets = legalTargets(state, state.turn);
  const cue = state.balls.find((b) => b.id === CUE_ID);
  if (!cue || targets.length === 0) {
    return { best: null, stats: [], simulations: 0 };
  }
  return searchBaseline(state.balls, table, targets, config);
};
