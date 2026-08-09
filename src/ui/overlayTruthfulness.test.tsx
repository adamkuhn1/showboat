// The overlay is a claim surface. This suite checks that every claim it makes
// is backed by the decision data it was handed, and that it never uses
// vocabulary the algorithm hasn't earned.
//
// It renders the real component with `react-dom/server` against the real
// `DecisionTraceV1` the opponent decides with — the one `brain.plan()`
// publishes after a real physics search on a fixed board, not a hand-written
// stub and no longer an adapted reconstruction — so a number appearing in the
// markup that isn't in the trace is a test failure.

import { describe, it, expect, beforeAll } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { OverlayPanel } from "./OverlayPanel";
import { makeTable } from "../physics/table";
import { makeBall } from "../physics/ball";
import { CUE_ID } from "../game/rack";
import type { GameState } from "../game/state";
import { initPhysics, simulateShotWasm } from "../physics/wasm-bridge";
import { generateCandidates } from "../ai/candidates";
import { searchCandidates, defaultConfig, type SearchConfig } from "../ai/shotSearch";
import { classicalTrickOnlyBrain } from "../ai/brain";
import { isTrickCandidate, selectTrickOnly } from "../ai/policy/trickOnly";
import { buildDecisionTrace } from "../ai/trace/build";
import { REASONING_STATES, STATE_LABEL } from "../render/presentation";
import type { DecisionTraceV1 } from "../ai/trace/contract";

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
const state: GameState = {
  balls: board,
  turn: 1,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 3,
};

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
    .replace(/&[a-z#0-9]+;/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const idleCompare = { available: true, useNeural: true, onChange: () => {} };

const panel = (trace: DecisionTraceV1 | null, over: Partial<Parameters<typeof OverlayPanel>[0]> = {}) =>
  textOf(
    renderToStaticMarkup(
      <OverlayPanel
        trace={trace}
        state="READY"
        planning={false}
        modelLoading={false}
        badge={{ mode: "classical" }}
        showSkipHint={false}
        showDisclosure={false}
        compare={idleCompare}
        replay={null}
        // Default to the LIVE reading, so every existing assertion below runs
        // against the surface a visitor actually sees during a turn. The
        // replay reading is asserted explicitly by its own case.
        replaying={false}
        liveCounts={null}
        {...over}
      />,
    ),
  );

/**
 * The fixture's search budget.
 *
 * `seedTimeoutMs: Infinity` is the fix for this file's long-standing flake
 * (~1 run in 3-5). `defaultConfig` already sets `searchTimeoutMs: Infinity` for
 * a documented reason — a wall-clock guard makes a test depend on how loaded
 * the machine is rather than on the physics budget — but it leaves
 * `seedTimeoutMs` at its 2,000 ms default, and the seeding loop is the one this
 * fixture spends its time in. Two full searches run here: `brain.plan()` in
 * `beforeAll`, and `traceFor()` inside "the fixture is the live decision". When
 * the machine was busy enough for the clock to truncate one and not the other,
 * the two traces genuinely differed — the observed failure was one candidate
 * reading `seed-timeout` where the other read `budget-exhausted`, which is a
 * correct report of two different searches, not a bad assertion.
 *
 * `contract.test.ts` already pins both clocks for the same reason. Live play is
 * unaffected: `withinDeadline` (ai/brain.ts) folds the turn's real remaining
 * time in with `Math.min`, so `Infinity` never reaches a played turn.
 */
const config: SearchConfig = {
  ...defaultConfig,
  simulations: 40,
  seed: 99,
  seedTimeoutMs: Number.POSITIVE_INFINITY,
};

/**
 * Search, select and publish — the same three calls `ai/brain.ts`'s `decide`
 * makes, in the same order, with the same `eligible` filter. It exists only
 * because the neural fixture needs a prior injected without an ONNX session;
 * the classical trace below comes straight off `brain.plan()` so the two are
 * checked against each other in "the fixture is the live decision".
 */
function traceFor(prior?: SearchConfig["prior"]): DecisionTraceV1 {
  const candidates = generateCandidates(board, table, targets);
  const outcome = searchCandidates(candidates, board, targets, {
    ...config,
    prior,
    eligible: isTrickCandidate,
  });
  const decision = selectTrickOnly(outcome.allStats, outcome.verifications, {
    state,
    table,
    targets,
    simulate: simulateShotWasm,
  });
  return buildDecisionTrace({
    outcome,
    decision,
    state,
    player: 1,
    targets,
    physicsUnitsAllowed: config.simulations,
    model: null,
    fallback: null,
    timing: {
      totalMs: 812,
      neuralEncodeMs: null,
      neuralRunMs: prior ? 1.234 : null,
      physicsMs: outcome.trace?.physicsMs ?? 0,
      selectionMs: 0,
    },
  });
}

let classical: DecisionTraceV1;
let hybridLike: DecisionTraceV1;

describe("reasoning overlay: only real values, only earned vocabulary", () => {
  beforeAll(async () => {
    await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
    // The live path: exactly the brain `ui/planner/plan.ts` gets from
    // `getBrain(false, …)`, and exactly the trace it hands the renderer.
    classical = (await classicalTrickOnlyBrain().plan(state, table, 1, config)).decision;
    // A synthetic-but-well-formed prior: this test is about rendering, and the
    // real model is exercised end to end in src/ai/neural/hybrid.test.ts.
    // Confined to [0.60, 0.99] so no synthetic score can *coincide* with the
    // 0.50 reliability threshold, which the panel legitimately prints as part
    // of the rung sentence. Without that gap the "no prior score is rendered"
    // assertion below would fail on a collision rather than on a leak.
    const scores = generateCandidates(board, table, targets).map((_, i) => 0.6 + ((i * 37) % 40) / 100);
    hybridLike = traceFor({
      scores,
      keepTop: 10,
      source: "showboat-ranker-phase2d",
      inferenceMs: 1.234,
    });
  }, 60_000);

  it("the fixture is the live decision, not a reconstruction of one", () => {
    // If `traceFor` ever drifts from what the brain does, the neural fixture
    // below stops being evidence about the shipped renderer. Timing is measured
    // and therefore differs run to run; everything else must match.
    const local = traceFor();
    expect({ ...local, timing: null }).toEqual({ ...classical, timing: null });
    expect(classical.policy).toBe("trick-only");
    expect(classical.selected?.kind).not.toBe("direct");
  });

  it("classical mode says 'Physics search' and shows no model claims", () => {
    const text = panel(classical);
    expect(text).toContain("Physics search");
    expect(text).not.toContain("Neural evaluator");
    for (const re of BANNED) expect(text).not.toMatch(re);
  });

  it("the state slot only ever holds one of the five labels", () => {
    for (const state of REASONING_STATES) {
      const text = panel(classical, { state });
      expect(text).toContain(STATE_LABEL[state]);
      // No other reasoning label leaks in at the same time.
      for (const other of REASONING_STATES) {
        if (other === state) continue;
        if (STATE_LABEL[other].includes(STATE_LABEL[state])) continue;
        expect(text).not.toContain(STATE_LABEL[other]);
      }
    }
  });

  it("outside the reasoning sequence there is no state label at all", () => {
    const text = panel(classical, { state: "SETTLED" });
    for (const state of REASONING_STATES) expect(text).not.toContain(STATE_LABEL[state]);
  });

  it("every number rendered appears in the trace", () => {
    // The replacement for the old "at most three numbers from an allowlist"
    // rule, which was a fact about one revision of the copy. This one is
    // stronger and survives a richer panel: any digit on screen must be
    // traceable to a value the search produced.
    const text = panel(classical, { state: "READY" });
    const numbers = text.match(/\d+(\.\d+)?/g) ?? [];
    const fromTrace = new Set<string>();
    for (const c of classical.candidates) {
      fromTrace.add(String(c.target));
      fromTrace.add(String(c.potId));
    }
    if (classical.selected) {
      fromTrace.add(classical.selected.reliabilityThreshold.toFixed(2));
      fromTrace.add(String(classical.selected.qualifyingTricks));
    }
    for (const n of numbers) expect(fromTrace).toContain(n);
  });

  it("the deleted raw search counters do not reappear", () => {
    const text = panel(classical);
    expect(text).not.toContain("candidates generated");
    expect(text).not.toContain("physics-verified");
    expect(text).not.toContain("scratched in sim");
    expect(text).not.toContain("physics calls spent");
    expect(text).not.toContain("learned ranking");
    expect(text).not.toContain("pruned before physics");
    expect(text).not.toContain("make est.");
    expect(text).not.toMatch(/#\d/);
  });

  it("the chosen shot is named in words a player can read without a key", () => {
    const text = panel(classical);
    const sel = classical.selected!;
    const chosen = classical.candidates.find((c) => c.index === sel.candidateIndex)!;
    expect(text).toContain(`on the ${chosen.target}`);
    expect(text).toMatch(/into the (bottom|top)-(left|right|side) pocket/);
    expect(text).not.toMatch(/\bBL\b|\bSB\b|\bST\b|\bTR\b|\bTL\b|\bBR\b/);
  });

  it("the 'over …' clause only ever names a candidate the trace says lost", () => {
    const text = panel(classical);
    const m = text.match(/over a [a-z- ]+ on the (\d+)/);
    if (m) {
      const named = Number(m[1]);
      const losers = classical.candidates.filter(
        (c) => c.rejection === "lower-utility-than-selected",
      );
      expect(losers.some((c) => c.target === named)).toBe(true);
    }
  });

  it("neural mode adds only fields the prior really supplied", () => {
    const text = panel(hybridLike, {
      badge: { mode: "neural-hybrid", hashVerified: true },
    });
    expect(text).toContain("Neural evaluator and physics search");
    // The prior's calibrated estimates are real decision data, but printed
    // eight-to-a-panel as bare decimals they were a scoreboard, not reasoning.
    const withPrior = hybridLike.candidates.filter((c) => c.neural !== null);
    expect(withPrior.length).toBeGreaterThan(0);
    for (const c of withPrior) expect(text).not.toContain(c.neural!.score.toFixed(2));
    expect(text).not.toContain("showboat-ranker-phase2d");
    for (const re of BANNED) expect(text).not.toMatch(re);
  });

  it("a build whose ranker never loaded cannot name the model", () => {
    // The failure this pins: a trace carries the brain that was asked to plan,
    // so a neural-shaped trace with no fallback set can arrive in a build where
    // the artifact failed preflight and was never fetched. The badge is the
    // half that knows. Reading the trace alone put "Neural evaluator" on screen
    // while the console said the opponent had stayed on the physics search.
    const text = panel(hybridLike, {
      badge: { mode: "classical", fallbackReason: "HTTP 404" },
    });
    expect(text).not.toContain("Neural evaluator");
    expect(text).toContain("Physics search");
  });

  it("a fallback decision is never dressed up as a neural one", () => {
    const fallback: DecisionTraceV1 = {
      ...classical,
      mode: "neural-hybrid",
      fallback: {
        from: "neural-hybrid",
        to: "classical-trick-only",
        cause: "model-absent",
        detail: "model unavailable: HTTP 404",
      },
    };
    const text = panel(fallback, {
      badge: { mode: "neural-hybrid", fallbackReason: "model unavailable: HTTP 404" },
    });
    expect(text).toContain("Physics search");
    expect(text).not.toContain("Neural evaluator");
    expect(text).toContain("classical fallback");
    expect(text).toContain("HTTP 404");
  });

  it("the selection sentence is the ladder's own rung, not prose about it", () => {
    const text = panel(classical);
    const rung = classical.selected!.rung;
    const expected: Record<string, string> = {
      "trick-qualified": "reliability bar",
      "trick-below-threshold": "still pots in simulation",
      "trick-attempt-no-verified-pot": "best legal attempt",
      "non-direct-safety": "safety off the cushion",
      "forced-legal-contact": "shortest legal contact",
    };
    expect(text).toContain(expected[rung]);
  });

  it("the searching state is reachable and says so honestly", () => {
    expect(panel(null, { state: "IDLE", planning: true })).toContain("searching…");
    expect(panel(null, { state: "IDLE", planning: true, modelLoading: true })).toContain(
      "loading the trained model…",
    );
  });

  it("the component source contains no progress animation and no banned words", () => {
    const src = readFileSync(join(__dirname, "OverlayPanel.tsx"), "utf8");
    // Strip the file's own explanatory header, which necessarily *names* the
    // words it forbids. (This used to slice from a marker that no longer
    // existed in the file, so `indexOf` returned -1 and the assertion ran
    // against the last character of the source — i.e. it checked nothing.)
    const marker = "export interface ModelBadge";
    const start = src.indexOf(marker);
    expect(start, "header marker not found — the source guard would be vacuous").toBeGreaterThan(0);
    const body = src.slice(start);
    for (const re of BANNED) expect(body).not.toMatch(re);
    expect(body).not.toContain("thinking-dots");
    expect(body).not.toContain("setTimeout");
    expect(body).not.toContain("setInterval");
    const css = readFileSync(join(APP_ROOT, "src/index.css"), "utf8");
    expect(css).not.toContain("thinking-dots");
    expect(css).not.toContain("dot-pulse");
  });
});
