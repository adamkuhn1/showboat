import { describe, it, expect } from "vitest";
import { makeBall } from "../physics/ball";
import { makeTable } from "../physics/table";
import { generateCandidates } from "./candidates";
import { simulateShot } from "../physics/engine";
import { applyCue } from "../physics/cue";
import { CUE_ID } from "../game/rack";

const table = makeTable();

describe("candidate-shot generator (geometry enumeration, not scripting)", () => {
  it("produces a direct-pot candidate for a ball sitting near a pocket", () => {
    // Cue near centre, object ball just in front of the top-right corner.
    const cue = makeBall(CUE_ID, 0, 0);
    const corner = table.pockets.find((p) => p.id === "tr")!;
    const obj = makeBall(1, corner.center.x - 0.2, corner.center.y - 0.1);
    const cands = generateCandidates([cue, obj], table, [1]);
    const direct = cands.filter((c) => c.kind === "direct" && c.pocket === "tr");
    expect(direct.length).toBeGreaterThan(0);
  });

  it("a generated direct candidate actually pockets the ball in simulation", () => {
    // This proves the generator's geometry is physically sound: aim through the
    // ghost point it computed and the object ball should drop.
    const cue = makeBall(CUE_ID, -0.3, 0);
    const corner = table.pockets.find((p) => p.id === "tr")!;
    const obj = makeBall(1, corner.center.x - 0.35, corner.center.y - 0.18);
    const cands = generateCandidates([cue, obj], table, [1]);
    const direct = cands.find((c) => c.kind === "direct" && c.pocket === "tr");
    expect(direct).toBeDefined();

    const balls = [makeBall(CUE_ID, -0.3, 0), makeBall(1, obj.pos.x, obj.pos.y)];
    const c = balls.find((b) => b.id === CUE_ID)!;
    applyCue(c, { ...direct!.action, power: 0.6 });
    const res = simulateShot(balls, table);
    expect(res.pocketed).toContain(1);
  });

  it("enumerates bank candidates (single cushion) as first-class options", () => {
    const cue = makeBall(CUE_ID, 0, 0);
    const obj = makeBall(1, 0.3, 0.1);
    const cands = generateCandidates([cue, obj], table, [1]);
    const banks = cands.filter((c) => c.kind === "bank");
    // Bank routes exist alongside direct ones — availability, not scripting.
    expect(banks.length).toBeGreaterThan(0);
    for (const b of banks) expect(b.banks).toBe(1);
  });

  it("returns nothing when the cue ball is pocketed", () => {
    const cue = makeBall(CUE_ID, 0, 0);
    cue.pocketed = true;
    const obj = makeBall(1, 0.3, 0.1);
    expect(generateCandidates([cue, obj], table, [1])).toHaveLength(0);
  });
});
