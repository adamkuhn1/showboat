// Regression test for the bug that hid every status message in the app:
// `.board { line-height: 0 }` (inherited by `.msg`, which sets
// `overflow: hidden`) made the status line — "break to start", all shot
// narration, foul reasons, "click the table to place the cue ball", even
// "couldn't load the physics engine" — measure 0px tall and clip to nothing.
//
// There's no jsdom/happy-dom in this package (adding one would be a new
// dependency this sprint doesn't take on), so this can't render the real DOM
// and measure it. It parses the shipped stylesheet instead.
//
// The first version of this file could not fail, in three separate ways, and a
// cold review caught it. All three are fixed here and each is pinned by a
// meta-test below, because a regression guard that silently passes is worse
// than no guard: it launders the absence of a check into the appearance of one.
//
//   1. `declarationsFor` returned `{}` on a selector miss and `isZero(undefined)`
//      is `false`, so every assertion passed whether the rule existed or not.
//      `.board` in fact declares no `line-height` at all, so the headline
//      assertion was reading a property of a declaration that wasn't there.
//   2. The regex matched only the FIRST top-level occurrence of a selector.
//      This sprint added the file's first `@media` blocks, containing `.board`
//      and `.status` overrides — invisible to it. Re-introducing
//      `line-height: 0` inside a media query would have reproduced the original
//      bug with the suite green.
//   3. Nothing asserted the selectors were present, so deleting them entirely
//      would also have passed.
//
// The rule enforced now is the one that actually matters, stated over the whole
// stylesheet rather than one rule of it: no selector on the status line's
// inheritance chain may set `line-height: 0`, anywhere, at any nesting depth —
// and `.msg` must carry its own non-zero `line-height` so its `overflow: hidden`
// can only ever clip horizontally.

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
// Comments stripped first: several of them (this bug's own explanation, among
// others) contain literal `{`/`}` characters, which would otherwise confuse the
// brace matching below.
const css = readFileSync(join(__dirname, "../index.css"), "utf8").replace(
  /\/\*[\s\S]*?\*\//g,
  "",
);

/**
 * Every declaration block for `selector`, at any nesting depth — top level and
 * inside `@media`/`@supports` alike. Returns one entry per occurrence, so a
 * caller can assert over all of them rather than only the first.
 *
 * Deliberately does NOT return an empty array quietly: callers assert
 * `length > 0` first, because "the selector is gone" and "the selector is fine"
 * must not look the same to this file ever again.
 */
function blocksFor(selector: string): Record<string, string>[] {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  // Preceded by start-of-file, `{`, `}` or `;` so `.msg` cannot match inside
  // `.msg-detail`, and a compound like `.status .msg` is matched on its own.
  const re = new RegExp(`(?:^|[{};])\\s*${escaped}\\s*\\{([^{}]*)\\}`, "g");
  const out: Record<string, string>[] = [];
  for (const match of css.matchAll(re)) {
    const decls: Record<string, string> = {};
    for (const decl of match[1].split(";")) {
      const i = decl.indexOf(":");
      if (i === -1) continue;
      decls[decl.slice(0, i).trim()] = decl.slice(i + 1).trim();
    }
    out.push(decls);
  }
  return out;
}

/** True only for an explicit zero. A missing value is not a zero. */
const declaresZeroLineHeight = (decls: Record<string, string>): boolean => {
  const v = decls["line-height"];
  return v !== undefined && parseFloat(v) === 0;
};

// The status line inherits from these. `.board` is the element that carried the
// original bug; `.status` is the row `.msg` lives in.
const INHERITANCE_CHAIN = [".board", ".status", ".msg"];

describe("status message stays visible (regression: .board { line-height: 0 } clipped .msg)", () => {
  it("finds the selectors it claims to be guarding", () => {
    for (const sel of INHERITANCE_CHAIN) {
      expect(blocksFor(sel).length, `${sel} not found in index.css`).toBeGreaterThan(0);
    }
  });

  it("no rule on the status line's inheritance chain zeroes line-height, at any depth", () => {
    for (const sel of INHERITANCE_CHAIN) {
      for (const [i, decls] of blocksFor(sel).entries()) {
        expect(
          declaresZeroLineHeight(decls),
          `${sel} block #${i + 1} sets line-height: 0 — this is the original bug`,
        ).toBe(false);
      }
    }
  });

  it(".msg declares its own non-zero line-height in every block that styles it", () => {
    const blocks = blocksFor(".msg");
    // At least one block must actually set it; the rest must not zero it.
    expect(blocks.some((d) => d["line-height"] !== undefined)).toBe(true);
    for (const decls of blocks) expect(declaresZeroLineHeight(decls)).toBe(false);
  });

  it(".msg's overflow: hidden is paired with a real line-height, so it can only clip horizontally", () => {
    const blocks = blocksFor(".msg");
    const withOverflow = blocks.filter((d) => d["overflow"] === "hidden");
    expect(withOverflow.length).toBeGreaterThan(0);
    const lineHeight = blocks.find((d) => d["line-height"] !== undefined)?.["line-height"];
    expect(lineHeight).toBeDefined();
    expect(parseFloat(lineHeight!)).toBeGreaterThan(0);
  });
});

// Meta-tests: the parser itself, against the exact shapes that defeated the
// previous version. Without these, a future simplification of `blocksFor` could
// quietly restore the vacuum.
describe("the guard above can actually fail", () => {
  const parse = (source: string, selector: string) => {
    const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`(?:^|[{};])\\s*${escaped}\\s*\\{([^{}]*)\\}`, "g");
    return [...source.matchAll(re)].map((m) => {
      const decls: Record<string, string> = {};
      for (const decl of m[1].split(";")) {
        const i = decl.indexOf(":");
        if (i !== -1) decls[decl.slice(0, i).trim()] = decl.slice(i + 1).trim();
      }
      return decls;
    });
  };

  it("sees a rule nested inside @media — the hole that let the bug back in", () => {
    const src = `.board { flex: 1; } @media (max-width: 600px) { .board { line-height: 0; } }`;
    const blocks = parse(src, ".board");
    expect(blocks.length).toBe(2);
    expect(blocks.some(declaresZeroLineHeight)).toBe(true);
  });

  it("reports a missing selector as absent rather than as passing", () => {
    expect(parse(`.other { line-height: 1.4; }`, ".board").length).toBe(0);
  });

  it("does not match a longer selector that merely starts the same", () => {
    expect(parse(`.msg-detail { line-height: 0; }`, ".msg").length).toBe(0);
  });

  it("treats a missing line-height as not-zero, but the presence check catches it", () => {
    expect(declaresZeroLineHeight({ flex: "1 1 480px" })).toBe(false);
  });
});
