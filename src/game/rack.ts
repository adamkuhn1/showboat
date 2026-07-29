import { type Ball, makeBall } from "../physics/ball";
import { FOOT_SPOT, HEAD_SPOT } from "../physics/table";
import { BALL_DIAMETER } from "../physics/constants";

// Ball id conventions used across the game and the ML action space:
//   0        cue ball
//   1..7     solids (low)
//   8        eight ball
//   9..15    stripes (high)
export const CUE_ID = 0;
export const EIGHT_ID = 8;
export const SOLIDS = [1, 2, 3, 4, 5, 6, 7];
export const STRIPES = [9, 10, 11, 12, 13, 14, 15];

export type Group = "solids" | "stripes";

// Build a standard 8-ball rack: triangle with apex on the foot spot, the eight
// ball in the centre of the third row, and one solid + one stripe in the back
// corners (a legal common rack). Cue ball on the head spot.
//
// The triangle points toward the head (cue) end: apex at the foot spot, rows
// receding toward the foot rail.
export const rackEightBall = (): Ball[] => {
  const balls: Ball[] = [];
  balls.push(makeBall(CUE_ID, HEAD_SPOT.x, HEAD_SPOT.y));

  // Rack layout by row (apex row 0). Row r has r+1 balls. Standard legal-ish
  // arrangement: 8 in the middle of row 2; corners of the back row are a solid
  // and a stripe. Remaining balls fill in.
  const layout: number[][] = [
    [1], // apex
    [9, 2],
    [3, 8, 10], // 8-ball centre of 3rd row
    [11, 4, 5, 12],
    [6, 13, 14, 7, 15],
  ];

  // Geometry: apex at foot spot, rows step toward +x (foot rail); balls in a
  // row are spread along y. Spacing slightly over a diameter to avoid initial
  // overlap from floating point.
  const gap = BALL_DIAMETER * 1.001;
  const rowDx = gap * Math.sqrt(3) / 2;

  for (let r = 0; r < layout.length; r++) {
    const row = layout[r];
    const x = FOOT_SPOT.x + r * rowDx;
    const count = row.length;
    // Centre the row on y = 0.
    const y0 = -((count - 1) / 2) * gap;
    for (let i = 0; i < count; i++) {
      const id = row[i];
      const y = y0 + i * gap;
      balls.push(makeBall(id, x, y));
    }
  }

  return balls;
};

export const groupOf = (id: number): Group | "eight" | "cue" => {
  if (id === CUE_ID) return "cue";
  if (id === EIGHT_ID) return "eight";
  return SOLIDS.includes(id) ? "solids" : "stripes";
};
