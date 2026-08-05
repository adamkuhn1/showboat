// The overlay is a claim surface. This suite checks that every claim it makes
// is backed by the decision data it was handed, and that it never uses
// vocabulary the algorithm hasn't earned.
//
// It renders the real component with `react-dom/server` against a real
// `SearchResult` produced by the real physics search on a fixed board — not a
// hand-written stub — so a number appearing in the markup that isn't in the
// search result is a test failure.

import { describe, it, expect, beforeAll } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { OverlayPanel } from "./OverlayPanel";
import { makeTable } from "../physics/table";
import { makeBall } from "../physics/ball";
import { CUE_ID } from "../game/rack";
import { initPhysics } from "../physics/wasm-bridge";
import { generateCandidates } from "../ai/candidates";
import { searchCandidates, defaultConfig, type SearchResult } from "../ai/shotSearch";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../..");
const table = makeTable();

const board = [
  makeBall(CUE_ID, -0.6, -0.1),
  makeBall(1, 0.3, 0.02),
  makeBall(2, 0.38, 0.1),
  makeBall(4, 0.1, -0.25),
  makeBall(5, -0.2, 0.3),
];
const targets = [1, 2, 4, 5];

// Words the algorithm does not earn. Checked against the rendered TEXT, and
// separately against the component source so a future edit can't sneak one in
// via a code path this fixture doesn't hit.
const BANNED = [
  /\bMCTS\b/i,
  /monte[- ]carlo/i,
  /win\s*prob/i,
  /win\s*rate/i,
  /confidence/i,
  /\bcertainty\b/i,
  /\bthinking\b/i,
  /\banalyz/i,
  /\bneural net(work)? is\b/i,
];

const textOf = (markup: string) =>
  markup
    .replace(/<[^>]*>/g, " ")
    .replace(/&[a-z]+;/g, " ")
    .replace(/\s+/g, " ")
    .trim();

let classical: SearchResult;
let hybridLike: SearchResult;

describe("reasoning overlay: only real values, only earned vocabulary", () => {
  beforeAll(async () => {
    await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
    const candidates = generateCandidates(board, table, targets);
    const config = { ...defaultConfig, simulations: 40, seed: 99 };
    classical = searchCandidates(candidates, board, targets, config);
    // A synthetic-but-well-formed prior: this test is about rendering, and the
    // real model is exercised end to end in src/ai/neural/hybrid.test.ts.
    const scores = candidates.map((_, i) => ((i * 37) % 101) / 100);
    hybridLike = searchCandidates(candidates, board, targets, {
      ...config,
      prior: { scores, keepTop: 10, source: "showboat-ranker-phase2d", inferenceMs: 1.234 },
    });
  }, 60_000);

  it("classical mode says 'Physics search' and shows no model claims", () => {
    const markup = renderToStaticMarkup(
      <OverlayPanel result={classical} searching={false} badge={{ mode: "classical" }} />,
    );
    const text = textOf(markup);
    expect(text).toContain("Physics search");
    expect(text).not.toContain("Neural");
    expect(text).not.toContain("make est.");
    for (const re of BANNED) expect(text).not.toMatch(re);
  });

  it("every stage number rendered is a number the search actually produced", () => {
    const t = classical.trace!;
    const text = textOf(
      renderToStaticMarkup(
        <OverlayPanel result={classical} searching={false} badge={{ mode: "classical" }} />,
      ),
    );
    expect(text).toContain(`candidates generated ${t.candidatesGenerated}`);
    expect(text).toContain(`scratched in sim ${t.scratched}`);
    expect(text).toContain(`physics calls spent ${t.physicsCalls}`);
    expect(text).toContain(`physics-verified ${t.physicsVerified}`);
    // And the trace's own counts are internally consistent with the stats list.
    expect(t.physicsVerified).toBe(t.verifiedIndices.length);
    // `legalPots` counts pots among ALL verified candidates, which can exceed
    // the pots visible in `stats`: a candidate can be simulated (and pot) and
    // then be dropped for `visits === 0` when the budget ran out before its
    // rollout. The trace line reads "physics-verified N · M legal pots", which
    // is exactly that quantity, so >= is the correct relationship here.
    expect(t.legalPots).toBeGreaterThanOrEqual(
      classical.stats.filter((s) => s.potsTarget).length,
    );
  });

  it("candidate scores rendered match the stats to the displayed precision", () => {
    const text = textOf(
      renderToStaticMarkup(
        <OverlayPanel result={classical} searching={false} badge={{ mode: "classical" }} />,
      ),
    );
    for (const s of classical.stats.slice(0, 8)) {
      expect(text).toContain(s.strength.toFixed(2));
    }
  });

  it("neural mode adds only fields the prior really supplied", () => {
    const text = textOf(
      renderToStaticMarkup(
        <OverlayPanel result={hybridLike} searching={false} badge={{ mode: "neural-hybrid", hashVerified: true }} />,
      ),
    );
    expect(text).toContain("Neural evaluator + physics search");
    expect(text).toContain("learned ranking");
    expect(text).toContain("make est.");
    expect(text).toContain(`pruned before physics ${hybridLike.trace!.prunedByPrior}`);
    expect(text).toContain("1.2ms"); // the measured inferenceMs, to 1dp
    expect(text).toContain("showboat-ranker-phase2d");
    for (const re of BANNED) expect(text).not.toMatch(re);
  });

  it("a fallback decision is never dressed up as a neural one", () => {
    const fallback: SearchResult = {
      ...classical,
      trace: { ...classical.trace!, mode: "neural-hybrid", fallbackReason: "model unavailable: HTTP 404" },
    };
    const text = textOf(
      renderToStaticMarkup(
        <OverlayPanel
          result={fallback}
          searching={false}
          badge={{ mode: "neural-hybrid", fallbackReason: "model unavailable: HTTP 404" }}
        />,
      ),
    );
    expect(text).toContain("Physics search");
    expect(text).not.toContain("Neural evaluator");
    expect(text).not.toContain("learned ranking");
    expect(text).toContain("classical fallback");
    expect(text).toContain("HTTP 404");
  });

  it("the selection sentence is the one the selection function emitted", () => {
    const text = textOf(
      renderToStaticMarkup(
        <OverlayPanel result={classical} searching={false} badge={{ mode: "classical" }} />,
      ),
    );
    const reason = classical.trace!.selectionReason!;
    const expected: Record<string, string> = {
      "trick-qualified": "trick cleared the 0.50 reliability bar",
      "no-trick-qualified": "no trick cleared the reliability bar",
      "no-verified-pot": "nothing potted in simulation",
      none: "no candidate survived physics verification",
    };
    expect(text).toContain(expected[reason]);
  });

  it("the component source contains no progress animation and no banned words", () => {
    const src = readFileSync(join(__dirname, "OverlayPanel.tsx"), "utf8");
    // Strip the file's own explanatory header, which necessarily *names* the
    // words it forbids.
    const body = src.slice(src.indexOf("const isTrickShot"));
    for (const re of BANNED) expect(body).not.toMatch(re);
    expect(body).not.toContain("thinking-dots");
    expect(body).not.toContain("setTimeout");
    expect(body).not.toContain("setInterval");
    const css = readFileSync(join(APP_ROOT, "src/index.css"), "utf8");
    expect(css).not.toContain("thinking-dots");
    expect(css).not.toContain("dot-pulse");
  });
});
