// The only AI-side `takeShot` wrapper.
//
// It accepts a `PlayableShot` and nothing else. Since `PlayableShot` carries an
// unexported brand that only `trickOnly.ts` can stamp, "the AI played a shot"
// and "the trick-only policy chose that shot" are the same statement — there is
// no way to hand a raw `CueAction` to this function, and no other AI code path
// reaches `takeShot`.
//
// `trickOnlySourceGuard.test.ts` asserts that mechanically against the source.

import { type Table } from "../../physics/table";
import { type GameState } from "../../game/state";
import { takeShot, type ShotReport, type Simulator } from "../../game/game";
import { actionOf, assertNotDirect, type PlayableShot } from "./trickOnly";

export const executeAiShot = (
  state: GameState,
  table: Table,
  shot: PlayableShot,
  simulate: Simulator,
): ShotReport => takeShot(state, table, actionOf(assertNotDirect(shot)), simulate);
