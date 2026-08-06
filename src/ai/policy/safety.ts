// Rail-first kick safeties: the non-direct floor of the trick-only ladder.
//
// When no trick candidate can be played, the previous policy fell back to a
// direct pot and, failing that, `App.tsx` aimed the cue straight at the nearest
// legal ball. Both are direct shots. This module is what replaces them.
//
// A kick sends the CUE ball into a cushion first, and off the cushion into a
// legal target. It is not one of the four trick kinds and is never labelled as
// one — it is `safety-kick` everywhere, including in the trace and the UI.
//
// Two properties matter more than shot quality here:
//
//   * **Legality is decided by the real ruleset**, not by geometry. Each
//     verified kick is executed through `takeShot` + the real simulator and
//     judged by `applyShotRules` — the same function the game uses. A kick is
//     legal when the cue's first ball contact is a legal target and either a
//     ball is pocketed or some ball touches a rail; the cue's own pre-contact
//     cushion already satisfies the second clause (rules.ts:96-104), which is
//     why a geometrically valid kick is legal in almost every position.
//   * **Bounded by construction.** At most `MAX_GENERATED` kicks are built and
//     at most `SAFETY_SIM_BUDGET` are simulated, all synchronously. A turn
//     cannot hang here, and the cost is reported separately from the search
//     budget (`safetySimsSpent`) so budget parity in evaluation stays honest.

import { type Ball } from "../../physics/ball";
import { type Table } from "../../physics/table";
import { type Vec2 } from "../../physics/vec";
import { type CueAction } from "../../physics/cue";
import { CUE_ID } from "../../game/rack";
import { type GameState } from "../../game/state";
import { takeShot, type Simulator } from "../../game/game";
import { isPathClear, mirrorAcross, railCrossing, SIDES } from "../candidates";

/** Kicks generated before any physics. Hard ceiling: 3 targets x 4 rails. */
export const MAX_GENERATED = 12;
/** Real simulations the safety search may spend. Never exceeded. */
export const SAFETY_SIM_BUDGET = 6;

/** Low and controlled: a kick is a positional shot, not a break. */
const MIN_KICK_POWER = 0.3;
const MAX_KICK_POWER = 0.7;

export interface SafetyKick {
  target: number;
  cushion: string;
  /** Where the cue meets the rail. */
  railPoint: Vec2;
  /** Cue-ball route: [cue position, rail point, target position]. */
  cuePath: Vec2[];
  action: CueAction;
  /** Total cue travel, used to prefer the shortest, most controllable kick. */
  length: number;
}

export interface SafetyResult {
  kick: SafetyKick | null;
  /**
   * `foul-free` — a real simulation showed `applyShotRules().foul === false`.
   * `legal-contact-only` — no kick came back foul-free; this one at least made
   *   contact with a legal target in simulation.
   * `unverified` — the physics budget could not be spent, or no kick survived
   *   simulation; the shortest geometric kick is returned at minimum power.
   * `none` — no kick could be constructed at all.
   */
  quality: "foul-free" | "legal-contact-only" | "unverified" | "none";
  simsSpent: number;
}

const sub = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y });
const mag = (a: Vec2): number => Math.hypot(a.x, a.y);

const aimAt = (from: Vec2, to: Vec2, power: number): CueAction => {
  const d = sub(to, from);
  return { phi: Math.atan2(d.y, d.x), power, sideSpin: 0, topSpin: 0 };
};

const kickPower = (length: number, tableLength: number): number =>
  Math.max(MIN_KICK_POWER, Math.min(MAX_KICK_POWER, 0.3 + (length / tableLength) * 0.22));

/**
 * Rail-first kicks at the nearest legal targets.
 *
 * Geometry: mirror the target across a cushion and aim the cue at the mirror.
 * The straight line cue -> mirror crosses that cushion at exactly the point
 * where a perfectly elastic rebound sends the cue on to the real target — the
 * same reflection principle `candidates.ts` uses for bank shots, applied to the
 * cue ball rather than to the object ball.
 *
 * `clearPathsOnly=false` drops the obstruction check, which is how the caller
 * degrades gracefully in a fully snookered position rather than returning
 * nothing.
 */
export const generateSafetyKicks = (
  balls: Ball[],
  table: Table,
  targets: number[],
  clearPathsOnly = true,
): SafetyKick[] => {
  const cue = balls.find((b) => b.id === CUE_ID);
  if (!cue || cue.pocketed) return [];
  const live = balls.filter((b) => !b.pocketed);
  const cuePos = cue.pos;

  const nearest = live
    .filter((b) => targets.includes(b.id))
    .sort((a, b) => mag(sub(a.pos, cuePos)) - mag(sub(b.pos, cuePos)))
    .slice(0, 3);

  const out: SafetyKick[] = [];
  for (const t of nearest) {
    for (const side of SIDES) {
      const mirror = mirrorAcross(t.pos, table, side);
      const railPoint = railCrossing(cuePos, mirror, table, side);
      if (!railPoint) continue;
      if (clearPathsOnly) {
        const skip = new Set([CUE_ID]);
        if (!isPathClear(cuePos, railPoint, live, skip)) continue;
        skip.add(t.id);
        if (!isPathClear(railPoint, t.pos, live, skip)) continue;
      }
      const length = mag(sub(railPoint, cuePos)) + mag(sub(t.pos, railPoint));
      out.push({
        target: t.id,
        cushion: side,
        railPoint,
        cuePath: [cuePos, railPoint, t.pos],
        action: aimAt(cuePos, railPoint, kickPower(length, table.length)),
        length,
      });
    }
  }
  return out.sort((a, b) => a.length - b.length).slice(0, MAX_GENERATED);
};

/**
 * Pick a kick, spending at most `SAFETY_SIM_BUDGET` real simulations.
 *
 * Fully synchronous. The returned `quality` is the honest description of how
 * far verification got — the caller maps it onto a selection rung and the trace
 * reports that rung verbatim.
 */
export const pickSafety = (
  state: GameState,
  table: Table,
  targets: number[],
  simulate: Simulator,
  budget: number = SAFETY_SIM_BUDGET,
): SafetyResult => {
  // Clear-path kicks first, then top up with obstructed ones. The geometric
  // obstruction check is conservative — it rejects a route whose *centre line*
  // passes within a ball diameter of another ball, which the real simulator
  // often resolves as a glancing miss — so in a tight snooker the filtered set
  // can be small or empty while a genuinely legal route exists. Simulating the
  // top-ups costs nothing extra: the budget below is what bounds the work, and
  // the filtered candidates are still tried first.
  const clear = generateSafetyKicks(state.balls, table, targets, true);
  const seen = new Set(clear.map((k) => `${k.target}:${k.cushion}`));
  const topUp = generateSafetyKicks(state.balls, table, targets, false).filter(
    (k) => !seen.has(`${k.target}:${k.cushion}`),
  );
  let kicks = [...clear, ...topUp].slice(0, MAX_GENERATED);
  if (kicks.length === 0) return { kick: null, quality: "none", simsSpent: 0 };

  // Power variants for the shortest routes, if the geometry left budget spare.
  //
  // The mirror construction assumes a perfectly elastic cushion. Real cushions
  // have restitution below 1 and compress the rebound angle, so a geometrically
  // exact kick can still miss — measured on fixture `blockedMid`, where every
  // one of the four single-rail routes came back "no contact". The error grows
  // with impact speed, so re-trying the shortest route at minimum power is a
  // physically motivated second attempt rather than a random retry. It costs
  // nothing: it only fills simulation slots the generated set left unused.
  const cap = Math.min(budget, SAFETY_SIM_BUDGET);
  if (kicks.length < cap) {
    const variants = kicks
      .slice(0, cap - kicks.length)
      .map((k) => ({ ...k, action: { ...k.action, power: MIN_KICK_POWER } }))
      .filter((k) => k.action.power !== kicks[0].action.power);
    kicks = [...kicks, ...variants].slice(0, MAX_GENERATED);
  }

  const allowed = Math.max(0, Math.min(budget, SAFETY_SIM_BUDGET, kicks.length));
  let simsSpent = 0;
  let legalContact: SafetyKick | null = null;

  for (let i = 0; i < allowed; i++) {
    const k = kicks[i];
    const report = takeShot(state, table, k.action, simulate);
    simsSpent++;
    if (!report.outcome.foul) return { kick: k, quality: "foul-free", simsSpent };
    // Not foul-free, but did the cue at least strike a legal ball? That is the
    // difference between "a bad safety" and "a whiff", and it is worth keeping
    // as the fallback rather than shooting blind.
    if (legalContact === null && report.sim.firstContact !== null && targets.includes(report.sim.firstContact)) {
      legalContact = k;
    }
  }

  if (legalContact) {
    return {
      kick: { ...legalContact, action: { ...legalContact.action, power: MIN_KICK_POWER } },
      quality: "legal-contact-only",
      simsSpent,
    };
  }
  return {
    kick: { ...kicks[0], action: { ...kicks[0].action, power: MIN_KICK_POWER } },
    quality: "unverified",
    simsSpent,
  };
};
