// Single source of truth for the position-level train/validation/test split,
// shared by train.ts and evaluate.ts so they can never disagree about which
// positions belong to which set.
//
// Split by whole table position, never by candidate row -- candidates from
// the same position share geometry, so a row-level split would leak. Test
// positions are held out completely: train.ts never reads them, not even for
// early stopping. Validation positions are for early stopping / any model
// judgment calls. Test positions are read exactly once, by evaluate.ts, for
// the final gate.
export type Split = "train" | "val" | "test";

export const splitOf = (pos: number): Split => {
  const m = ((pos % 10) + 10) % 10;
  if (m === 0) return "test";
  if (m === 1) return "val";
  return "train";
};
