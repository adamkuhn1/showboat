import { type ShotReport } from "./game";
import { type PlayerId } from "./state";
import { CUE_ID } from "./rack";

// One-line, human-readable result of a committed shot for the status line:
// "You potted the 6. Your turn." / "No ball potted. AI to play." The raw
// collision chain (ai/trace.ts describeShot) is deliberately NOT part of it —
// the UI keeps that behind a details toggle.

const ballWord = (id: number): string => `the ${id}`;

const listBalls = (ids: number[]): string =>
  ids.length <= 1
    ? ids.map(ballWord).join("")
    : `${ids.slice(0, -1).map(ballWord).join(", ")} and ${ballWord(ids[ids.length - 1])}`;

export const summariseOutcome = (
  report: ShotReport,
  shooter: PlayerId,
  human: PlayerId,
): string => {
  const o = report.outcome;
  const isHuman = (p: PlayerId | null): boolean => p === human;

  if (o.gameOver) {
    return `${isHuman(o.winner) ? "You win" : "The AI wins"} the rack!${
      o.foul ? ` (${o.foulReason})` : ""
    }`;
  }

  if (o.foul) {
    return `Foul by ${isHuman(shooter) ? "you" : "the AI"}: ${o.foulReason}. ${
      isHuman(report.next.turn) ? "You have" : "The AI has"
    } ball in hand.`;
  }

  const potted = o.pocketedThisShot.filter((id) => id !== CUE_ID);
  let first: string;
  if (potted.length === 0) {
    first = "No ball potted.";
  } else {
    const group = o.assignedGroups ? report.next.groups[shooter] : null;
    first =
      `${isHuman(shooter) ? "You" : "The AI"} potted ${listBalls(potted)}.` +
      (group ? ` ${isHuman(shooter) ? "You're" : "The AI is"} on ${group}.` : "");
  }
  return `${first} ${isHuman(report.next.turn) ? "Your turn." : "AI to play."}`;
};
