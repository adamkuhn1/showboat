// Regression test for the bug that hid every status message in the app:
// `.board { line-height: 0 }` (inherited by `.msg`, which sets
// `overflow: hidden`) made the status line — "break to start", all shot
// narration, foul reasons, "click the table to place the cue ball", even
// "couldn't load the physics engine" — measure 0px tall and clip to nothing.
//
// There's no jsdom/happy-dom in this package (adding one would be a new
// dependency this sprint doesn't take on), so this can't render the real DOM
// and measure it. Instead it parses the actual shipped stylesheet and asserts
// the two properties whose combination caused the bug directly: `.board`
// (an ancestor of `.msg`) must never zero out `line-height` again, and `.msg`
// must always declare its own non-zero `line-height`, so the overflow:hidden
// it also sets can only ever clip horizontal overflow (the ellipsis it's
// there for) and never collapse the whole line to zero height.

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
// Comments stripped first: several of them (this bug's own explanation,
// among others) contain literal `{`/`}` characters, which would otherwise
// confuse the brace-matching below.
const css = readFileSync(join(__dirname, "../index.css"), "utf8").replace(
  /\/\*[\s\S]*?\*\//g,
  "",
);

/** Declarations for the first top-level rule with this exact selector. Good
 * enough for this stylesheet, which has no nested at-rules or comma-separated
 * selector lists on the selectors this test reads. */
function declarationsFor(selector: string): Record<string, string> {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = css.match(new RegExp(`(?:^|\\})\\s*${escaped}\\s*\\{([^}]*)\\}`));
  if (!match) return {};
  const out: Record<string, string> = {};
  for (const decl of match[1].split(";")) {
    const i = decl.indexOf(":");
    if (i === -1) continue;
    out[decl.slice(0, i).trim()] = decl.slice(i + 1).trim();
  }
  return out;
}

const isZero = (value: string | undefined): boolean =>
  value !== undefined && parseFloat(value) === 0;

describe("status message stays visible (regression: .board { line-height: 0 } clipping .msg)", () => {
  it(".board never zeroes the line-height every descendant, including .msg, would inherit", () => {
    const board = declarationsFor(".board");
    expect(isZero(board["line-height"])).toBe(false);
  });

  it(".msg declares its own non-zero line-height, independent of any ancestor", () => {
    const msg = declarationsFor(".msg");
    expect(msg["line-height"]).toBeDefined();
    expect(isZero(msg["line-height"])).toBe(false);
  });

  it(".msg's overflow: hidden is paired with a real line-height, so it can only clip horizontally", () => {
    const msg = declarationsFor(".msg");
    expect(msg["overflow"]).toBe("hidden");
    expect(isZero(msg["line-height"])).toBe(false);
  });

  it(".status (the row .msg lives in) also carries a non-zero line-height", () => {
    const status = declarationsFor(".status");
    expect(isZero(status["line-height"])).toBe(false);
  });
});
