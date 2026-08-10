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
//
// A FIFTH REQUIREMENT, ON TOP OF ALL FOUR: the shot must measurably BE a trick.
//
// The layers above establish that a selected candidate is not labelled
// `direct`. They say nothing about what its route does, and a candidate's label
// is mirror geometry computed before any physics ran — a "bank" whose mirror
// point lands beside the pocket sends the object ball straight in, touching no
// cushion. Selecting it plays a straight pot under a bank's name.
//
// So every candidate that reaches a trick rung is classified from the ordered
// event log of its own production rollout (`ai/measure/classify.ts`), and only
// a route the log shows executing one of the four supported structures — with
// the intended ball dropping, off the intended first contact — is playable.
// The measured class also *is* the shot's kind from here on: a route planned as
// one bank and measured as two is played, and named, as a two-rail bank.
//
// There is no rung below the two trick rungs other than the safety ladder. A
// trick that does not pot in simulation cannot be shown to be a trick, so the
// rung that used to play "the best legal attempt" and describe it with the
// plan's word is gone rather than reworded.

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
import {
  TRICK_KINDS,
  type MeasuredRoute,
  type SelectionRung,
  type TrickKind,
} from "../trace/contract";
import { classifyMeasuredShot, measuredTrickKind, toMeasuredRoute } from "../measure/classify";
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
  /**
   * The classification of `physics.events`. Null exactly when no simulation was
   * run on this candidate, so a null here means "unmeasured", never "measured
   * and found to be nothing".
   */
  measured: MeasuredRoute | null;
}

// Unexported `unique symbol`: the brand cannot be *named* outside this module,
// therefore it cannot be forged. A `PlayableShot` object literal is a compile
// error anywhere else in the repo, including in tests.
const PLAYABLE: unique symbol = Symbol("showboat.playable");

export interface PlayableShot {
  readonly [PLAYABLE]: true;
  readonly action: CueAction;
  /** The MEASURED structure. Derived from the rollout's event log, not the plan. */
  readonly kind: TrickKind | "safety-kick";
  /** What the generator called it. Equal to `kind` unless the route differed. */
  readonly plannedKind: TrickKind | "safety-kick";
  /** The measurement `kind` came from. Null only for a safety kick. */
  readonly measured: MeasuredRoute | null;
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
  /**
   * Generation indices of trick candidates whose rollout potted the intended
   * ball and whose measured route is not a supported trick structure — the
   * nominal banks that used no cushion, and their relatives. Write-only, like
   * `excludedIndices`: it flows into the trace and nothing selects from it.
   */
  unmeasuredTrickIndices: number[];
  /** Trick candidates whose rollout measurably executed a supported structure. */
  verifiedTricks: number;
  /** Real simulations the safety rung spent. Reported, never netted. */
  safetySimsSpent: number;
  /** Set iff `shot.kind === "safety-kick"`. Geometry for the trace. */
  safety: SafetyKick | null;
  /**
   * `pickSafety`'s own verdict on how far verification got. Set iff `safety` is.
   *
   * The rung below collapses four qualities onto two rungs, which is the right
   * shape for the *ladder* and the wrong shape for the *description*:
   * `forced-legal-contact` is reached both by a kick that made legal contact
   * and fouled, and by a kick nothing verified at all. This field is what keeps
   * that distinction, and it used to be dropped one line before the trace was
   * built.
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
 * Four rungs, strictly ordered, none of which can return a direct and none of
 * which can return a trick the physics did not show happening:
 *
 *   1 `trick-qualified`       measured trick + clears the reliability bar
 *   2 `trick-below-threshold` measured trick, below the bar (still a real pot)
 *   3 `non-direct-safety`     a rail-first kick that the real rules confirmed
 *                             is foul-free
 *   4 `forced-legal-contact`  nothing was foul-free; play the shortest kick
 *                             anyway, and say so
 *
 * Rung 2 is what keeps the win-rate cost survivable: a sub-threshold trick that
 * measurably pots is still a pot.
 *
 * Both trick rungs draw from `verified` — candidates whose own rollout was
 * classified as one of the four supported structures. `potsTarget` alone is not
 * enough and is no longer consulted on its own: it is true of a nominal bank
 * that ran straight into the pocket, which is a straight pot.
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
    if (!isTrickStat(s)) {
      excludedIndices.push(index); // write-only: trace material, never selected from
      return;
    }
    const physics = verifications[index] ?? null;
    // The candidate's own production rollout, classified from its event log.
    // Intent is two ball identities and the turn's legal target set; the
    // candidate's KIND is not passed and cannot influence the answer.
    const measured =
      physics === null
        ? null
        : toMeasuredRoute(
            classifyMeasuredShot(
              { events: physics.events },
              {
                target: s.candidate.target,
                potId: s.candidate.potId,
                legalTargets: ctx.targets,
              },
            ),
          );
    tricks.push({ index, stat: s, physics, measured });
  });

  // A candidate is a trick here only if the events say so.
  const verified = tricks.filter((t) => t.measured !== null && t.measured.trickVerified);
  // Potted the ball it was for, but not by any supported structure. Reported so
  // the trace can give each of these its real reason instead of "did not pot".
  const unmeasuredTrickIndices = tricks
    .filter((t) => t.stat.potsTarget && !(t.measured?.trickVerified ?? false))
    .map((t) => t.index);

  const qualifying = verified.filter((t) => t.stat.strength >= TRICK_RELIABILITY_THRESHOLD);

  const play = (t: TrickEntry, rung: SelectionRung): TrickOnlyDecision => {
    const measured = t.measured!;
    // The displayed type. `measuredTrickKind` returns null only when
    // `trickVerified` is false, which `verified` has already excluded, so the
    // fallback is unreachable and is here so the expression is total.
    const kind = measuredTrickKind(measured) ?? t.stat.candidate.kind;
    return {
      shot: assertNotDirect(
        brand({
          action: t.stat.candidate.action,
          kind,
          plannedKind: t.stat.candidate.kind,
          measured,
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
      unmeasuredTrickIndices,
      verifiedTricks: verified.length,
      safetySimsSpent: 0,
      safety: null,
      safetyQuality: null,
    };
  };

  // Rung 1 — the shot Showboat wants: a measured trick that is also reliable.
  if (qualifying.length > 0) {
    return play(argmax(qualifying, (t) => trickUtility(t.stat)), "trick-qualified");
  }

  // Rung 2 — a measured trick below the reliability bar. It still pots.
  if (verified.length > 0) {
    return play(argmax(verified, (t) => t.stat.strength), "trick-below-threshold");
  }

  // Rungs 3 and 4 — no trick is playable at all. A rail-first kick, verified
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
      unmeasuredTrickIndices,
      verifiedTricks: 0,
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
        plannedKind: "safety-kick",
        measured: null,
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
    unmeasuredTrickIndices,
    verifiedTricks: 0,
    safetySimsSpent: safety.simsSpent,
    safety: safety.kick,
    safetyQuality: safety.quality,
  };
}

const ctxCuePos = (ctx: TrickOnlyContext): { x: number; y: number } => {
  const cue = ctx.state.balls.find((b) => b.id === CUE_ID);
  return cue ? { x: cue.pos.x, y: cue.pos.y } : { x: 0, y: 0 };
};
