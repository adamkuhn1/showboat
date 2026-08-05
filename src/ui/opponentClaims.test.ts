// Two regressions that are both, at bottom, about the app never claiming a
// different opponent than the one actually deciding.
//
// 1. The neural-ranking toggle used to be a dependency of the AI-turn effect
//    in App.tsx. Flipping it mid-turn ran the effect cleanup, which cancelled
//    the in-flight search, and then re-entered the effect, which bailed on its
//    `phase !== "aiming"` guard — so the turn sat at "searching…" forever.
//    There is no DOM in this package (see statusMessageVisibility.test.ts for
//    why), so this is asserted against the source: the effect must read the
//    toggle through its ref and must not list it as a dependency.
//
// 2. `brainLabel()` derives the opponent's name from validated model state, so
//    it says "the physics-search opponent" until the artifact is really loaded
//    and ready. That is correct and must stay correct — the flicker it caused
//    in a page header was fixed by not putting a mode claim in the header, not
//    by hardcoding one. These tests fail if anyone "fixes" it the other way.

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { brainLabel, getBrain } from "../ai/brain";
import type { NeuralCandidateEvaluator } from "../ai/neural/evaluator";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_SRC = readFileSync(join(__dirname, "../App.tsx"), "utf8");

/** An evaluator that is present but not yet ready — the mid-download state. */
const notReady = { isReady: () => false } as unknown as NeuralCandidateEvaluator;
const ready = {
  isReady: () => true,
  getState: () => ({ status: "ready" }),
  getManifest: () => ({ artifact: "x.onnx" }),
  score: () => null,
} as unknown as NeuralCandidateEvaluator;

describe("toggling neural ranking mid-turn cannot wedge the opponent", () => {
  it("the AI-turn effect does not depend on `useNeural`", () => {
    // The dependency array of the AI-turn effect, matched by the comment that
    // documents the two intentional exclusions immediately above it.
    const deps = APP_SRC.match(/\}, \[state, vsAI, engineReady[^\]]*\]\);/);
    expect(deps, "AI-turn effect dependency array not found").not.toBeNull();
    expect(deps![0]).not.toContain("useNeural");
  });

  it("the effect still reads the live toggle value, through its ref", () => {
    expect(APP_SRC).toContain("useNeuralRef.current");
    expect(APP_SRC).toContain("getBrain(useNeuralRef.current)");
  });

  it("the exclusion is documented at the dependency array, not just done", () => {
    // A lint autofix or a well-meaning cleanup will re-add `useNeural` unless
    // the reason is written where the change would be made.
    const idx = APP_SRC.search(/\}, \[state, vsAI, engineReady[^\]]*\]\);/);
    const preamble = APP_SRC.slice(Math.max(0, idx - 1200), idx);
    expect(preamble).toContain("useNeural");
    expect(preamble).toMatch(/wedge/i);
  });
});

describe("the opponent's name is never a claim about a model that isn't loaded", () => {
  it("brainLabel says physics-search while the artifact is still loading", () => {
    expect(brainLabel(true, notReady)).toBe("the physics-search opponent");
    expect(brainLabel(true, notReady)).not.toContain("neural");
  });

  it("brainLabel says neural only once the evaluator reports ready", () => {
    expect(brainLabel(true, ready)).toContain("neural");
  });

  it("getBrain falls back to the classical brain on the same condition", () => {
    // Not just the label: the decision itself. A neural label over a classical
    // decision is the failure this pair of functions exists to prevent.
    expect(getBrain(true, notReady)).not.toBe(getBrain(true, ready));
    expect(brainLabel(false, ready)).toBe("the physics-search opponent");
  });

  it("the page header states behaviour, not which implementation is running", () => {
    // The header used to render `brainLabel(useNeural)`, which flipped from
    // "physics-search" to "neural + physics" the moment the runtime finished
    // downloading — read by a cold reviewer as copy that changed between
    // loads. The mode is still stated, on the toggle and on the reasoning
    // panel's title; it is not stated in a masthead.
    const tag = APP_SRC.slice(APP_SRC.indexOf('<p className="tag">'));
    const tagEnd = tag.indexOf("</p>");
    const headerText = tag.slice(0, tagEnd);
    expect(headerText).not.toContain("brainLabel");
    expect(headerText).not.toMatch(/neural/i);
    expect(headerText).not.toMatch(/physics-search/i);
  });
});
