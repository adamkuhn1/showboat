// THE choke point. This is the only module in the codebase that can produce a
// shot the AI is allowed to play, and it cannot produce a direct.
//
// The locked product decision is: **Showboat wins using trick shots, and never
// chooses a direct shot in live gameplay — even when a direct shot would be
// strategically better.** Directs keep existing: they are generated, they are
// shown in the trace as truthfully-labelled rejected comparisons, and they are
// still selectable by the *evaluation baseline* (`selectBestWithReason`). They
// are simply not playable.
//
// FOUR LAYERS, STRONGEST FIRST
//
//  1. **Type (compile time).** The selection ladder is typed over `TrickEntry`,
//     whose `stat` is a `TrickStat` — a `CandidateStat` whose candidate's
//     `kind` is statically narrowed to the four trick kinds. Returning a direct
//     is a type error, not a bug someone has to notice in review. And
//     `PlayableShot` carries an unexported `unique symbol` brand, so no other
//     module can construct one at all: there is no object literal anywhere else
//     in the repo that TypeScript will accept as a playable shot.
//  2. **Structural (runtime).** The partition happens in the first statement of
//     `selectTrickOnly`. `excluded` is write-only — it flows into the trace and
//     is never read by any selection expression. No rung below holds a
//     reference to a direct, so there is nothing for a direct to win.
//  3. **Assertion (defence in depth).** `assertNotDirect` runs on the way out
//     and throws `TrickOnlyInvariantError`. Unreachable unless layer 1 or 2 is
//     broken; the brain catches it and degrades to the safety rung, so a future
//     editing mistake yields a *worse* shot, never a direct one and never a
//     hang.
//  4. **Call site (source test).** `trickOnlySourceGuard.test.ts` asserts that
//     the live path never imports the legacy selector and that `App.tsx` no
//     longer contains its own nearest-ball aim.

import { type Candidate } from "../candidates";
import { type Table } from "../../physics/table";
import { type CueAction } from "../../physics/cue";
import { type GameState } from "../../game/state";
import { CUE_ID } from "../../game/rack";
import { type Simulator } from "../../game/game";
import {
  type CandidateStat,
  type CandidateVerification,
  TRICK_RELIABILITY_THRESHOLD,
  trickUtility,
} from "../shotSearch";
import { TRICK_KINDS, type SelectionRung, type TrickKind } from "../trace/contract";
import { pickSafety, SAFETY_SIM_BUDGET, type SafetyKick, type SafetyResult } from "./safety";

export { TRICK_KINDS, type TrickKind };

/**
 * The single exclusion predicate in the codebase. Imported, never
 * re-implemented — a second copy is exactly how these guarantees rot.
 */
export const isTrickCandidate = (c: Candidate): c is Candidate & { kind: TrickKind } =>
  (TRICK_KINDS as readonly string[]).includes(c.kind);

/** A `CandidateStat` whose candidate is statically known to be non-direct. */
export type TrickStat = CandidateStat & { candidate: Candidate & { kind: TrickKind } };

const isTrickStat = (s: CandidateStat): s is TrickStat => isTrickCandidate(s.candidate);

/** A trick candidate with its position in generation order and its physics. */
export interface TrickEntry {
  /** Index in `generateCandidates` output order — the stable trace id. */
  index: number;
  stat: TrickStat;
  physics: CandidateVerification | null;
}

// Unexported `unique symbol`: the brand cannot be *named* outside this module,
// therefore it cannot be forged. A `PlayableShot` object literal is a compile
// error anywhere else in the repo, including in tests.
const PLAYABLE: unique symbol = Symbol("showboat.playable");

export interface PlayableShot {
  readonly [PLAYABLE]: true;
  readonly action: CueAction;
  readonly kind: TrickKind | "safety-kick";
  /** Index in generation order; null only for a generated safety kick. */
  readonly candidateIndex: number | null;
  readonly rung: SelectionRung;
  /** Cue-ball route from the cue ball's real position. At least two points. */
  readonly cuePath: { x: number; y: number }[];
  /** Object-ball route. Empty for a safety kick, which plans no pot. */
  readonly path: { x: number; y: number }[];
}

/** Thrown only if layer 1 or 2 above has been broken by a later edit. */
export class TrickOnlyInvariantError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TrickOnlyInvariantError";
  }
}

export interface TrickOnlyContext {
  /** Pre-shot state. `state.turn` must be the player about to shoot. */
  state: GameState;
  table: Table;
  targets: number[];
  /** Real simulator, injected so tests can count calls. */
  simulate: Simulator;
  /** Physics calls the safety rung may spend. Clamped to SAFETY_SIM_BUDGET. */
  safetyBudget?: number;
}

export interface TrickOnlyDecision {
  /** Null iff no shot could be constructed at all (no legal target). */
  shot: PlayableShot | null;
  rung: SelectionRung | null;
  /** Verified tricks that pot AND clear the reliability threshold. */
  qualifyingTricks: number;
  /** `strength + STYLE_WEIGHT * styleScore` of the chosen trick; null safety. */
  utility: number | null;
  /** Generation indices of every candidate excluded for being a direct. */
  excludedIndices: number[];
  /** Real simulations the safety rung spent. Reported, never netted. */
  safetySimsSpent: number;
  /** Set iff `shot.kind === "safety-kick"`. Geometry for the trace. */
  safety: SafetyKick | null;
  /**
   * `pickSafety`'s own verdict on how far verification got. Set iff `safety` is.
   *
   * The rung below collapses four qualities onto two rungs, which is the right
   * shape for the *ladder* and the wrong shape for the *description*: rung 5 is
   * reached both by a kick that made legal contact and fouled, and by a kick
   * nothing verified at all. This field is what keeps that distinction, and it
   * used to be dropped one line before the trace was built.
   */
  safetyQuality: SafetyResult["quality"] | null;
}

/** The one function that stamps the brand. Nothing else can. */
const brand = (s: Omit<PlayableShot, typeof PLAYABLE>): PlayableShot => ({
  [PLAYABLE]: true,
  ...s,
});

/** Layer 3. Unreachable unless the type or the partition has been broken. */
export const assertNotDirect = (shot: PlayableShot): PlayableShot => {
  if (shot.kind !== "safety-kick" && !(TRICK_KINDS as readonly string[]).includes(shot.kind)) {
    throw new TrickOnlyInvariantError(`trick-only policy produced kind "${shot.kind}"`);
  }
  return shot;
};

/** Read the action off a branded shot. The only way to get one out. */
export const actionOf = (shot: PlayableShot): CueAction => shot.action;

const argmax = <T,>(xs: T[], score: (x: T) => number): T =>
  xs.reduce((a, b) => (score(b) > score(a) ? b : a));

/**
 * THE selection. `allStats` and `verifications` are index-aligned with
 * `generateCandidates`'s output, so `index` here is the stable trace id.
 *
 * Five rungs, strictly ordered, none of which can return a direct:
 *
 *   1 `trick-qualified`               pots + clears the reliability bar
 *   2 `trick-below-threshold`         pots, below the bar (still a real pot)
 *   3 `trick-attempt-no-verified-pot` verified, legal contact, no scratch
 *   4 `non-direct-safety`             a rail-first kick that the real rules
 *                                     confirmed is foul-free
 *   5 `forced-legal-contact`          nothing was foul-free; play the shortest
 *                                     kick anyway, and say so
 *
 * Rung 2 is what makes the win-rate cost survivable: a sub-threshold *potting*
 * trick is still a pot, so most decisions that used to fall through to a direct
 * still put a ball down.
 */
export function selectTrickOnly(
  allStats: readonly CandidateStat[],
  verifications: readonly (CandidateVerification | null)[],
  ctx: TrickOnlyContext,
): TrickOnlyDecision {
  // ---- the partition. Everything below reads `tricks` and nothing else. ----
  const tricks: TrickEntry[] = [];
  const excludedIndices: number[] = [];
  allStats.forEach((s, index) => {
    if (isTrickStat(s)) tricks.push({ index, stat: s, physics: verifications[index] ?? null });
    else excludedIndices.push(index); // write-only: trace material, never selected from
  });

  const qualifying = tricks.filter(
    (t) => t.stat.potsTarget && t.stat.strength >= TRICK_RELIABILITY_THRESHOLD,
  );

  const play = (t: TrickEntry, rung: SelectionRung): TrickOnlyDecision => ({
    shot: assertNotDirect(
      brand({
        action: t.stat.candidate.action,
        kind: t.stat.candidate.kind,
        candidateIndex: t.index,
        rung,
        cuePath: [ctxCuePos(ctx), t.stat.candidate.aimPoint],
        path: t.stat.candidate.path,
      }),
    ),
    rung,
    qualifyingTricks: qualifying.length,
    utility: trickUtility(t.stat),
    excludedIndices,
    safetySimsSpent: 0,
    safety: null,
    safetyQuality: null,
  });

  // Rung 1 — the shot Showboat wants: a trick that pots and is reliable.
  if (qualifying.length > 0) {
    return play(argmax(qualifying, (t) => trickUtility(t.stat)), "trick-qualified");
  }

  // Rung 2 — a trick that pots but below the reliability bar. This is where
  // the old policy handed the table to a direct.
  const potting = tricks.filter((t) => t.stat.potsTarget);
  if (potting.length > 0) {
    return play(argmax(potting, (t) => t.stat.strength), "trick-below-threshold");
  }

  // Rung 3 — no trick pots, but one was verified to strike the right ball
  // without scratching. Attempting it is a legal shot with upside.
  const attempts = tricks.filter(
    (t) => t.physics !== null && t.physics.legalFirstContact && !t.physics.scratched,
  );
  if (attempts.length > 0) {
    return play(argmax(attempts, (t) => t.stat.strength), "trick-attempt-no-verified-pot");
  }

  // Rungs 4 and 5 — no trick is playable at all. A rail-first kick, verified
  // against the real ruleset. Never a direct, bounded, synchronous.
  const safety = pickSafety(
    ctx.state,
    ctx.table,
    ctx.targets,
    ctx.simulate,
    Math.min(ctx.safetyBudget ?? SAFETY_SIM_BUDGET, SAFETY_SIM_BUDGET),
  );
  if (!safety.kick) {
    return {
      shot: null,
      rung: null,
      qualifyingTricks: 0,
      utility: null,
      excludedIndices,
      safetySimsSpent: safety.simsSpent,
      safety: null,
      safetyQuality: safety.quality,
    };
  }
  const rung: SelectionRung =
    safety.quality === "foul-free" ? "non-direct-safety" : "forced-legal-contact";
  return {
    shot: assertNotDirect(
      brand({
        action: safety.kick.action,
        kind: "safety-kick",
        candidateIndex: null,
        rung,
        cuePath: safety.kick.cuePath,
        path: [],
      }),
    ),
    rung,
    qualifyingTricks: 0,
    utility: null,
    excludedIndices,
    safetySimsSpent: safety.simsSpent,
    safety: safety.kick,
    safetyQuality: safety.quality,
  };
}

const ctxCuePos = (ctx: TrickOnlyContext): { x: number; y: number } => {
  const cue = ctx.state.balls.find((b) => b.id === CUE_ID);
  return cue ? { x: cue.pos.x, y: cue.pos.y } : { x: 0, y: 0 };
};
