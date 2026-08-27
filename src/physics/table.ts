import { type Vec2, vec } from "./vec";
import {
  TABLE_LENGTH,
  TABLE_WIDTH,
  CORNER_POCKET_RADIUS,
  SIDE_POCKET_RADIUS,
} from "./constants";

// Table geometry. The origin is the table centre; the long axis is x
// (-L/2 .. +L/2), the short axis is y (-W/2 .. +W/2). Cushions are the four
// straight rails; pockets sit at the four corners and the two long-rail
// midpoints (six-pocket table).

export type CushionSide = "left" | "right" | "top" | "bottom";

export interface Cushion {
  side: CushionSide;
  // The cushion is an axis-aligned line. `axis` says which coordinate is fixed
  // at `pos`; the ball rebounds by flipping that component of velocity.
  axis: "x" | "y";
  pos: number; // fixed coordinate value of the cushion nose line
  // Inward normal (points toward the playing area).
  normal: Vec2;
}

export interface Pocket {
  id: string;
  center: Vec2;
  radius: number;
}

export interface Table {
  length: number;
  width: number;
  cushions: Cushion[];
  pockets: Pocket[];
}

export const makeTable = (): Table => {
  const hx = TABLE_LENGTH / 2;
  const hy = TABLE_WIDTH / 2;

  const cushions: Cushion[] = [
    { side: "left", axis: "x", pos: -hx, normal: vec(1, 0) },
    { side: "right", axis: "x", pos: hx, normal: vec(-1, 0) },
    { side: "bottom", axis: "y", pos: -hy, normal: vec(0, 1) },
    { side: "top", axis: "y", pos: hy, normal: vec(0, -1) },
  ];

  const pockets: Pocket[] = [
    { id: "bl", center: vec(-hx, -hy), radius: CORNER_POCKET_RADIUS },
    { id: "tl", center: vec(-hx, hy), radius: CORNER_POCKET_RADIUS },
    { id: "br", center: vec(hx, -hy), radius: CORNER_POCKET_RADIUS },
    { id: "tr", center: vec(hx, hy), radius: CORNER_POCKET_RADIUS },
    { id: "sb", center: vec(0, -hy), radius: SIDE_POCKET_RADIUS },
    { id: "st", center: vec(0, hy), radius: SIDE_POCKET_RADIUS },
  ];

  return { length: TABLE_LENGTH, width: TABLE_WIDTH, cushions, pockets };
};

// Standard 8-ball spot positions on this coordinate system.
// The head string is at x = -L/4; the foot spot (rack apex reference) at x=+L/4.
export const FOOT_SPOT: Vec2 = vec(TABLE_LENGTH / 4, 0);
export const HEAD_SPOT: Vec2 = vec(-TABLE_LENGTH / 4, 0);
export const HEAD_STRING_X = -TABLE_LENGTH / 4;
