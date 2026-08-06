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

  it("the planner plays `result.shot` and has no aim of its own", () => {
    // The opponent's turn used to be inline in `App.tsx`; T3 moved it to
    // `ui/planner/plan.ts`, which the worker and the inline fallback both call,
    // so this is now where the property has to hold. Asserting the file rather
    // than the app is also why the next test exists: a *move* must not be able
    // to smuggle the escape hatch back in somewhere else.
    const plan = readFileSync(join(SRC, "ui/planner/plan.ts"), "utf8");
    expect(plan).toContain("executeAiShot(input.state, input.table, result.shot, simulateShotWasm)");
    // Not through `takeShot` directly, and not through a re-executed action:
    // `executeAiShot` is the only AI-side wrapper and it takes the brand.
    expect(plan).not.toContain("takeShot(");
    // The renderer is handed the trace the decision was made with. The interim
    // adapter that synthesised one from `SearchResult` is deleted.
    expect(plan).toContain("result.decision");
    expect(plan).not.toContain("adaptSearchResult");
  });

  it("no production file anywhere contains a nearest-legal-ball aim", () => {
    // The identifiers of BOTH deleted escape hatches: the one that lived at
    // `App.tsx:379-416` and the verbatim copy of it that landed in
    // `ui/planner/plan.ts` as `lastResortAim` during the parallel-team merge.
    // Checked across the whole tree, because the last two times this code moved
    // it moved into a file this test was not reading.
    const banned = [
      "lastResortAim",
      "pathClearTo",
      "const byDist",
      "const nearest = byDist",
      /const fallback: CueAction/,
      // Nor a fabricated aim of any other shape when the policy declines.
      /Math\.random\(\) \* Math\.PI \* 2/,
    ];
    for (const p of productionFiles(SRC)) {
      const src = readFileSync(p, "utf8");
      for (const b of banned) {
        if (typeof b === "string") expect(src, `${relative(SRC, p)} contains ${b}`).not.toContain(b);
        else expect(src, `${relative(SRC, p)} matches ${b}`).not.toMatch(b);
      }
    }
  });

  it("only `ai/trace/build.ts` can produce a decision trace", () => {
    // A second trace builder is how a *synthesised* trace gets back in front of
    // the renderer. Stamping the version is what makes an object a
    // `DecisionTraceV1`, so only the contract and the one builder may name it.
    const stampers = productionFiles(SRC)
      .filter((p) => readFileSync(p, "utf8").includes("DECISION_TRACE_VERSION"))
      .map((p) => relative(SRC, p))
      .sort();
    expect(stampers).toEqual(["ai/trace/build.ts", "ai/trace/contract.ts"]);
  });

  it("the live path never imports the legacy mixed policy", () => {
    for (const name of [
      "ai/brain.ts",
      "ai/policy/trickOnly.ts",
      "ai/policy/execute.ts",
      "ui/planner/plan.ts",
      "ui/planner/planWorker.ts",
      "ui/planner/usePlanner.ts",
      "ui/useAiTurn.ts",
      "App.tsx",
    ]) {
      const src = readFileSync(join(SRC, name), "utf8");
      // Doc comments legitimately *name* these; the import statements must not.
      const imports = src.match(/^import[\s\S]*?from\s+"[^"]+";$/gm) ?? [];
      const joined = imports.join("\n");
      expect(joined, `${name} imports selectBestWithReason`).not.toContain("selectBestWithReason");
      expect(joined, `${name} imports selectBest`).not.toMatch(/\bselectBest\b/);
      expect(joined, `${name} imports searchWithLegacySelection`).not.toContain("searchWithLegacySelection");
      // Word-bounded: the live planner's own `planTurnTraced` is a different
      // function and must not be caught by a substring match.
      expect(joined, `${name} imports planTurn`).not.toMatch(/\bplanTurn\b/);
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
