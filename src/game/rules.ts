import { type SimResult } from "../physics/engine";
import { type GameState, type PlayerId, otherPlayer } from "./state";
import {
  CUE_ID,
  EIGHT_ID,
  SOLIDS,
  STRIPES,
  type Group,
  groupOf,
} from "./rack";

// The outcome of resolving one completed shot against the 8-ball ruleset. This
// is a pure function of (pre-shot state, sim result) so it is fully testable and
// shared identically by the UI and the ML environment.
export interface ShotOutcome {
  foul: boolean;
  foulReason: string | null;
  pocketedThisShot: number[];
  turnPasses: boolean; // does control move to the other player?
  ballInHandForNext: boolean; // opponent gets cue-ball-in-hand?
  assignedGroups: boolean; // did this shot claim groups (table was open)?
  gameOver: boolean;
  winner: PlayerId | null;
}

const ballsFor = (group: Group): number[] =>
  group === "solids" ? SOLIDS : STRIPES;

// How many of a player's group balls remain on the table (not pocketed).
const remainingInGroup = (g: GameState, group: Group): number =>
  ballsFor(group).filter((id) => {
    const b = g.balls.find((x) => x.id === id);
    return b && !b.pocketed;
  }).length;

// Apply 8-ball rules to a resolved shot. Mutates `g` (pocketed flags already set
// by the physics sim) to reflect group assignment / turn / winner, and returns a
// structured outcome describing what happened.
//
// Rules implemented (standard bar / BCA-flavored 8-ball, v1 scope):
//   - Cue must contact a legal ball first; on the break, contact any ball.
//   - Pocketing the cue ball (scratch) is a foul -> ball in hand to opponent.
//   - Table is "open" until someone legally pockets and claims a group.
//   - You continue your turn if you legally pocket one of your balls and don't
//     foul; otherwise turn passes.
//   - The 8-ball may only be legally pocketed after clearing your group.
//     Pocketing the 8 early, or scratching while pocketing the 8, loses.
export const applyShotRules = (
  g: GameState,
  pre: GameState,
  result: SimResult,
): ShotOutcome => {
  const shooter = pre.turn;
  const opp = otherPlayer(shooter);
  const pocketed = result.pocketed;
  const cueScratched = pocketed.includes(CUE_ID);
  const eightPocketed = pocketed.includes(EIGHT_ID);

  let foul = false;
  let foulReason: string | null = null;

  // --- First-contact legality --------------------------------------------
  const shooterGroup = pre.groups[shooter];
  const firstHit = result.firstContact;
  if (firstHit === null) {
    // No object ball contacted at all — always a foul (except a legal break is
    // handled by requiring at least one ball to be hit; here none was).
    foul = true;
    foulReason = "no object ball contacted";
  } else if (pre.broken && shooterGroup !== null) {
    // Groups assigned: must strike your own group first, unless only the 8
    // remains for you (then you must hit the 8 first).
    const myBallsLeft = remainingInGroup(pre, shooterGroup);
    const mustHitEight = myBallsLeft === 0;
    const firstGroup = groupOf(firstHit);
    if (mustHitEight && firstHit !== EIGHT_ID) {
      foul = true;
      foulReason = "must contact the 8-ball first";
    } else if (!mustHitEight && firstHit === EIGHT_ID) {
      foul = true;
      foulReason = "cannot contact the 8-ball first";
    } else if (
      !mustHitEight &&
      firstGroup !== "eight" &&
      firstGroup !== shooterGroup
    ) {
      foul = true;
      foulReason = "contacted opponent's ball first";
    }
  }

  // --- Cushion / pocket requirement --------------------------------------
  // After contact, a legal shot must either pocket a ball or drive some ball to
  // a rail. If nothing is pocketed and no ball hit a cushion after first
  // contact, it's a foul (a standard "no rail" rule).
  if (!foul && pocketed.length === 0) {
    const cushionAfterContact = result.events.some(
      (e) => e.kind === "ball-cushion",
    );
    if (!cushionAfterContact) {
      foul = true;
      foulReason = "no ball reached a rail";
    }
  }

  // --- Scratch ------------------------------------------------------------
  if (cueScratched) {
    foul = true;
    foulReason = foulReason ?? "scratch (cue ball pocketed)";
  }

  // --- 8-ball win / loss --------------------------------------------------
  if (eightPocketed) {
    const myBallsLeftPre = shooterGroup
      ? remainingInGroup(pre, shooterGroup)
      : 8; // open table -> group not cleared
    const clearedGroup = shooterGroup !== null && myBallsLeftPre === 0;
    if (clearedGroup && !cueScratched && !foul) {
      // Legal 8-ball pot AFTER clearing group -> shooter wins.
      g.winner = shooter;
      return {
        foul: false,
        foulReason: null,
        pocketedThisShot: pocketed,
        turnPasses: false,
        ballInHandForNext: false,
        assignedGroups: false,
        gameOver: true,
        winner: shooter,
      };
    }
    // Any other way the 8 goes down -> shooter loses immediately.
    g.winner = opp;
    return {
      foul: true,
      foulReason: foulReason ?? "8-ball pocketed illegally",
      pocketedThisShot: pocketed,
      turnPasses: true,
      ballInHandForNext: false,
      assignedGroups: false,
      gameOver: true,
      winner: opp,
    };
  }

  // --- Group assignment on an open table ---------------------------------
  let assignedGroups = false;
  const objectPocketed = pocketed.filter((id) => id !== CUE_ID && id !== EIGHT_ID);
  if (pre.broken && shooterGroup === null && !foul && objectPocketed.length > 0) {
    // The table is open; the shooter claims the group of the first ball they
    // legally pocketed. (If they made both solids and stripes on an open shot,
    // convention: claim by the first pocketed in event order.)
    const firstPocketedId = objectPocketed[0];
    const claimed: Group = SOLIDS.includes(firstPocketedId) ? "solids" : "stripes";
    g.groups[shooter] = claimed;
    g.groups[opp] = claimed === "solids" ? "stripes" : "solids";
    assignedGroups = true;
  }

  // --- Did the shooter legally pocket one of their own balls? -------------
  const grpAfter = g.groups[shooter];
  const madeOwnBall =
    !foul &&
    grpAfter !== null &&
    objectPocketed.some((id) => groupOf(id) === grpAfter);
  // On an open table (pre-assignment) any legal object-ball pot keeps the turn.
  const madeOnOpenTable = !foul && shooterGroup === null && objectPocketed.length > 0;

  const continueTurn = !foul && (madeOwnBall || madeOnOpenTable);

  if (!continueTurn) g.turn = opp;

  return {
    foul,
    foulReason,
    pocketedThisShot: pocketed,
    turnPasses: !continueTurn,
    ballInHandForNext: foul && !eightPocketed,
    assignedGroups,
    gameOver: false,
    winner: null,
  };
};
