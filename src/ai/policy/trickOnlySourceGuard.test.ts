// Layer 4: the call sites.
//
// The type system stops a direct being *selected*. It cannot, by itself, stop
// someone reintroducing a second path to `takeShot` that never consults the
// policy — which is exactly what `App.tsx:379-416` was: a nearest-legal-ball
// aim at power 0.3, fired whenever the search returned no candidate. It was a
// direct shot in the ordinary sense, it bypassed the policy entirely, and
// fixing `selectBestWithReason` alone left it in place.
//
// So these are source-level assertions. The precedent is
// `src/ui/opponentClaims.test.ts`, which pins App.tsx's effect dependencies the
// same way and for the same reason: the property lives in the shape of the
// code, so the code is what gets asserted.

import { describe, it, expect } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, dirname, relative } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC = join(__dirname, "../..");
const AI = join(__dirname, "..");

const walk = (dir: string): string[] =>
  readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });

const productionFiles = (root: string): string[] =>
  walk(root).filter((p) => /\.tsx?$/.test(p) && !/\.test\.tsx?$/.test(p));

describe("the AI has exactly one path to a played shot", () => {
  it("only `policy/execute.ts` calls `takeShot` from AI code", () => {
    const offenders = productionFiles(AI)
      .filter((p) => readFileSync(p, "utf8").includes("takeShot("))
      .map((p) => relative(SRC, p));
    expect(offenders).toEqual(["ai/policy/execute.ts", "ai/policy/safety.ts"]);
    // `safety.ts` is the second, and it is deliberate: verifying a kick means
    // executing it through the real rules. It returns a *candidate*, never a
    // committed shot — the committed shot still goes through `execute.ts`.
  });

  it("App.tsx plays `result.shot` and has no aim of its own", () => {
    const app = readFileSync(join(SRC, "App.tsx"), "utf8");
    expect(app).toContain("executeAiShot(planState, table, result.shot, simulateShotWasm)");
    // The deleted block's own identifiers. If any of these come back, so has
    // the second direct-shot escape hatch.
    expect(app).not.toContain("pathClearTo");
    expect(app).not.toContain("const byDist");
    expect(app).not.toContain("const nearest = byDist");
    expect(app).not.toMatch(/const fallback: CueAction/);
    // And the AI turn does not fabricate an aim when the policy declines.
    expect(app).not.toMatch(/Math\.random\(\) \* Math\.PI \* 2/);
  });

  it("the live path never imports the legacy mixed policy", () => {
    for (const name of ["ai/brain.ts", "ai/policy/trickOnly.ts", "ai/policy/execute.ts", "App.tsx"]) {
      const src = readFileSync(join(SRC, name), "utf8");
      // Doc comments legitimately *name* these; the import statements must not.
      const imports = src.match(/^import[\s\S]*?from\s+"[^"]+";$/gm) ?? [];
      const joined = imports.join("\n");
      expect(joined, `${name} imports selectBestWithReason`).not.toContain("selectBestWithReason");
      expect(joined, `${name} imports selectBest`).not.toMatch(/\bselectBest\b/);
      expect(joined, `${name} imports searchWithLegacySelection`).not.toContain("searchWithLegacySelection");
      expect(joined, `${name} imports planTurn`).not.toContain("planTurn");
      expect(joined, `${name} imports searchBaseline`).not.toContain("searchBaseline");
    }
  });

  it("`isTrickCandidate` is defined once and imported everywhere else", () => {
    const defs = productionFiles(SRC).filter((p) =>
      /(export )?const isTrickCandidate\s*=/.test(readFileSync(p, "utf8")),
    );
    expect(defs.map((p) => relative(SRC, p))).toEqual(["ai/policy/trickOnly.ts"]);
  });

  it("the trace contract has no runtime imports", () => {
    const src = readFileSync(join(SRC, "ai/trace/contract.ts"), "utf8");
    expect(src).not.toMatch(/^import /m);
    expect(src).not.toMatch(/\brequire\(/);
  });
});
