import { type Ball } from "../physics/ball";
import { type Group } from "./rack";

export type PlayerId = 0 | 1;

// Whole-game state. Kept as a plain serializable object so the ML env, the UI,
// and the search all share one representation and copies are cheap.
export interface GameState {
  balls: Ball[];
  turn: PlayerId;
  // null until the table is "open" and someone legally pockets to claim a group.
  groups: Record<PlayerId, Group | null>;
  // Ball-in-hand: the incoming player may place the cue ball anywhere (after a
  // scratch/foul). "behind-head" restricts placement behind the head string
  // (only on the opening break scratch).
  ballInHand: false | "anywhere" | "behind-head";
  winner: PlayerId | null;
  // True until the first legal contact of the game has resolved (the break).
  broken: boolean;
  shotCount: number;
}

export const otherPlayer = (p: PlayerId): PlayerId => (p === 0 ? 1 : 0);

export const activeBalls = (g: GameState): Ball[] =>
  g.balls.filter((b) => !b.pocketed);

export const ballById = (g: GameState, id: number): Ball | undefined =>
  g.balls.find((b) => b.id === id);
