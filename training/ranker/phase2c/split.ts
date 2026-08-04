// Deterministic, group-aware (family-based) split assignment + automated
// leakage checks. See DATASET_DESIGN.md's "Splitting and leakage prevention"
// — fixes the Phase 2A/2B gap (docs/repair/showboat-ml/phase-2c/
// 03-split-quality-audit.md §1/§4/§7-8): splitting must be by family_id
// (rack/game/trajectory), never by state_id or example_id, or near-identical
// in-game snapshots can straddle train/test.

import { createHash } from "node:crypto";

export type Split = "train" | "val" | "test";

const TRAIN_PCT = 70;
const VAL_PCT = 15;
// test = remaining 15%

/**
 * Deterministic bucket assignment from a hash of (familyId, seed) — every
 * state/candidate/perturbation sharing a family_id gets the identical split,
 * with no separate global pass required over the whole dataset. Golden
 * fixtures are never passed through this function; callers must special-case
 * state_origin === "golden" to stay out of all splits entirely.
 */
export function splitForFamily(familyId: string, seed: number): Split {
  const h = createHash("sha256").update(`${seed}:${familyId}`).digest();
  // Use the first 4 bytes as a uint32 bucket in [0, 100).
  const bucket = h.readUInt32BE(0) % 100;
  if (bucket < TRAIN_PCT) return "train";
  if (bucket < TRAIN_PCT + VAL_PCT) return "val";
  return "test";
}

export interface LeakageReport {
  ok: boolean;
  issues: string[];
  familyCountBySplit: Record<Split, number>;
  stateCountBySplit: Record<Split, number>;
}

export interface RowForLeakageCheck {
  family_id: string;
  state_id: string;
  split: Split;
  state_origin: "trajectory" | "controlled" | "golden";
}

/**
 * Automated leakage checks required by the Phase 2C acceptance criteria:
 * no family_id spans more than one split, no state_id is claimed by more
 * than one family_id, and no golden-origin row is assigned any split.
 */
export function checkLeakage(rows: RowForLeakageCheck[]): LeakageReport {
  const issues: string[] = [];
  const familySplit = new Map<string, Split>();
  const stateFamily = new Map<string, string>();
  const familyCountBySplit: Record<Split, number> = { train: 0, val: 0, test: 0 };
  const stateCountBySplit: Record<Split, number> = { train: 0, val: 0, test: 0 };
  const seenStatesBySplit = new Set<string>();

  for (const r of rows) {
    if (r.state_origin === "golden") {
      issues.push(`golden-origin state ${r.state_id} was assigned split ${r.split}`);
      continue;
    }
    const priorSplit = familySplit.get(r.family_id);
    if (priorSplit === undefined) {
      familySplit.set(r.family_id, r.split);
    } else if (priorSplit !== r.split) {
      issues.push(`family_id ${r.family_id} spans multiple splits: ${priorSplit} and ${r.split}`);
    }

    const priorFamily = stateFamily.get(r.state_id);
    if (priorFamily === undefined) {
      stateFamily.set(r.state_id, r.family_id);
    } else if (priorFamily !== r.family_id) {
      issues.push(`state_id ${r.state_id} claimed by multiple families: ${priorFamily} and ${r.family_id}`);
    }

    const stateSplitKey = `${r.split}:${r.state_id}`;
    if (!seenStatesBySplit.has(stateSplitKey)) {
      seenStatesBySplit.add(stateSplitKey);
      stateCountBySplit[r.split] += 1;
    }
  }

  const familyCounted = new Set<string>();
  for (const [fam, split] of familySplit.entries()) {
    if (!familyCounted.has(fam)) {
      familyCounted.add(fam);
      familyCountBySplit[split] += 1;
    }
  }

  return { ok: issues.length === 0, issues, familyCountBySplit, stateCountBySplit };
}
