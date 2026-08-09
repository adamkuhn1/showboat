// What a visitor who is not looking at the pixels gets.
//
// Before this, the answer was: an unlabelled `<canvas>`, no announcement of
// anything, and a set of keyboard controls that worked but were written down
// nowhere. The pure parts (the canvas's name, key ownership) are tested as
// functions; the wiring is asserted against `App.tsx`'s source, because this
// package has no DOM to render into and adding one is a dependency this sprint
// does not take on — the same reasoning `statusMessageVisibility.test.ts` sets
// out, and the same discipline: every source assertion below is paired with a
// check that the thing it is looking for is actually there, so deleting the
// feature fails the test rather than passing it vacuously.

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeBall } from "../physics/ball";
import { CUE_ID, EIGHT_ID, SOLIDS, STRIPES } from "../game/rack";
import type { GameState } from "../game/state";
import { boardLabel } from "./boardLabel";
import { focusedControlOwnsKey } from "./keys";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_SRC = readFileSync(join(__dirname, "../App.tsx"), "utf8");
const CSS = readFileSync(join(__dirname, "../index.css"), "utf8");

const AI_PLAYER = 1 as const;

const state = (over: Partial<GameState> = {}): GameState => ({
  balls: [
    makeBall(CUE_ID, -0.5, 0),
    makeBall(1, 0.2, 0.1),
    makeBall(9, 0.3, -0.1),
    makeBall(EIGHT_ID, 0.5, 0),
  ],
  turn: 0,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 2,
  ...over,
});

describe("the table has a name that describes this board", () => {
  it("says what the picture is, whose shot it is and what is left", () => {
    const label = boardLabel(state(), AI_PLAYER);
    expect(label).toContain("Eight-ball table");
    expect(label).toContain("Your shot");
    expect(label).toContain("the table is open");
    expect(label).toContain("1 solid and 1 stripe still up");
    expect(label).toContain("and the 8");
  });

  it("changes when the board changes — it is not a fixed string", () => {
    const open = boardLabel(state(), AI_PLAYER);
    const theirs = boardLabel(state({ turn: 1, groups: { 0: "stripes", 1: "solids" } }), AI_PLAYER);
    expect(theirs).toContain("The opponent's shot");
    expect(theirs).toContain("on solids");
    expect(theirs).not.toBe(open);

    const cleared = boardLabel(
      state({ balls: [makeBall(CUE_ID, 0, 0), makeBall(EIGHT_ID, 0.5, 0)] }),
      AI_PLAYER,
    );
    expect(cleared).toContain("0 solids and 0 stripes still up");
    expect(cleared).not.toBe(open);
  });

  it("reports ball in hand, and the end of the game", () => {
    expect(boardLabel(state({ ballInHand: "anywhere" }), AI_PLAYER)).toContain("Ball in hand.");
    expect(boardLabel(state({ winner: 0 }), AI_PLAYER)).toContain("you win");
    expect(boardLabel(state({ winner: 1 }), AI_PLAYER)).toContain("the opponent wins");
  });

  it("counts only balls still on the table", () => {
    const balls = [makeBall(CUE_ID, -0.5, 0), ...SOLIDS.map((id) => makeBall(id, 0, 0)), ...STRIPES.map((id) => makeBall(id, 0, 0))];
    balls[1].pocketed = true;
    balls[2].pocketed = true;
    const label = boardLabel(state({ balls }), AI_PLAYER);
    expect(label).toContain(`${SOLIDS.length - 2} solids and ${STRIPES.length} stripes still up`);
  });

  it("is wired to the canvas as a real accessible name, with a description", () => {
    expect(APP_SRC).toContain('role="img"');
    expect(APP_SRC).toContain("aria-label={boardLabel(state, AI_PLAYER)}");
    expect(APP_SRC).toContain('aria-describedby="sb-table-help"');
    expect(APP_SRC).toContain('id="sb-table-help"');
    // The description has to say how to play, not just what is drawn.
    const help = APP_SRC.slice(APP_SRC.indexOf('id="sb-table-help"'));
    const text = help.slice(0, help.indexOf("</p>"));
    for (const word of ["arrow", "Shift", "power", "space"]) {
      expect(text.toLowerCase(), `the table description never mentions ${word}`).toContain(
        word.toLowerCase(),
      );
    }
  });
});

describe("turn, foul, result and game over reach a live region", () => {
  it("there is exactly one, and it is a polite status region", () => {
    const regions = APP_SRC.match(/aria-live=/g) ?? [];
    expect(regions.length, "expected a single live region").toBe(1);
    expect(APP_SRC).toContain('role="status"');
    expect(APP_SRC).toContain('aria-live="polite"');
    expect(APP_SRC).toContain("{announcement}");
  });

  it("is fed by the commit message, the shot outcome and the resting cases", () => {
    // `commit` is where turn changes, fouls and game over are worded.
    const commit = APP_SRC.slice(APP_SRC.indexOf("const commit = useCallback"));
    expect(commit.slice(0, commit.indexOf("}, [])")))
      .toContain("setAnnouncement(msg)");
    // The opponent's result, the moment the balls stop.
    expect(APP_SRC).toContain("setAnnouncement(ai.outcome.text)");
    expect(APP_SRC).toMatch(/onNoLegalShot[\s\S]{0,200}setAnnouncement/);
  });

  it("no animation frame ever writes to it", () => {
    // The two per-frame paths in this file. If either learns to announce, a
    // screen reader is handed sixty updates a second.
    const tick = APP_SRC.slice(APP_SRC.indexOf("const tick = (now: number)"));
    expect(tick.slice(0, tick.indexOf("requestAnimationFrame(tick);"))).not.toContain(
      "setAnnouncement",
    );
    const paint = APP_SRC.slice(APP_SRC.indexOf("const paintScene = useCallback"));
    expect(paint.slice(0, paint.indexOf("[table, view]"))).not.toContain("setAnnouncement");
  });

  it("the region is present to assistive tech and absent from the layout", () => {
    expect(APP_SRC).toContain('className="visually-hidden"');
    expect(CSS).toContain(".visually-hidden");
    const block = CSS.slice(CSS.indexOf(".visually-hidden"));
    const decls = block.slice(0, block.indexOf("}"));
    expect(decls).toContain("position: absolute");
    expect(decls).toContain("clip-path");
    // `display: none` would take it out of the accessibility tree entirely,
    // which is the classic way to ship a live region that never speaks.
    expect(decls).not.toContain("display: none");
  });
});

describe("a focused control keeps its own keys", () => {
  const el = (tag: string, attrs: Record<string, string> = {}) =>
    ({
      tagName: tag.toUpperCase(),
      isContentEditable: false,
      getAttribute: (k: string) => attrs[k] ?? null,
    }) as unknown as EventTarget;

  it("a slider or a field owns every key, including the arrows", () => {
    // The regression this rule was written for: the English and Draw·Follow
    // sliders drew a focus ring and then ignored the arrows, because the aim
    // handler had already called preventDefault.
    for (const tag of ["input", "select", "textarea"]) {
      for (const code of ["Space", "Enter", "ArrowLeft", "ArrowUp", "KeyA"]) {
        expect(focusedControlOwnsKey(el(tag), code), `${tag}/${code}`).toBe(true);
      }
    }
    const editable = { tagName: "DIV", isContentEditable: true, getAttribute: () => null };
    expect(focusedControlOwnsKey(editable as unknown as EventTarget, "ArrowLeft")).toBe(true);
  });

  it("a button or a disclosure owns Space and Enter, and nothing else", () => {
    for (const tag of ["button", "summary", "a"]) {
      expect(focusedControlOwnsKey(el(tag), "Space"), `${tag}/Space`).toBe(true);
      expect(focusedControlOwnsKey(el(tag), "Enter"), `${tag}/Enter`).toBe(true);
      // Arrows have no native behaviour on any of them, so aiming stays live.
      expect(focusedControlOwnsKey(el(tag), "ArrowLeft"), `${tag}/ArrowLeft`).toBe(false);
      expect(focusedControlOwnsKey(el(tag), "ArrowUp"), `${tag}/ArrowUp`).toBe(false);
    }
    expect(focusedControlOwnsKey(el("div", { role: "button" }), "Space")).toBe(true);
    expect(focusedControlOwnsKey(el("div", { role: "button" }), "ArrowLeft")).toBe(false);
  });

  it("nothing else owns anything, so the game keeps its shortcuts", () => {
    expect(focusedControlOwnsKey(el("body"), "Space")).toBe(false);
    expect(focusedControlOwnsKey(el("canvas"), "Space")).toBe(false);
    expect(focusedControlOwnsKey(null, "Space")).toBe(false);
  });

  it("the window handler defers to it before touching a key", () => {
    const handler = APP_SRC.slice(APP_SRC.indexOf("const handler = (e: KeyboardEvent)"));
    const guard = handler.indexOf("focusedControlOwnsKey(e.target, e.code)");
    expect(guard, "the key handler does not consult the rule").toBeGreaterThan(0);
    // Before any actual call, or the guard is decorative. (Matched on the call
    // and not on the bare word: the handler's own comment explains the bug in
    // terms of `preventDefault`, and a substring search finds the prose first.)
    const firstCall = handler.indexOf("e.preventDefault();");
    expect(firstCall).toBeGreaterThan(0);
    expect(guard).toBeLessThan(firstCall);
  });

  it("the felt takes focus on pointer-down, so space belongs to the shot again", () => {
    // Without this, clicking a button and then pressing space re-activates the
    // button — correctly, and confusingly, since space is advertised as shoot.
    const down = APP_SRC.slice(APP_SRC.indexOf("const onPointerDown ="));
    expect(down.slice(0, down.indexOf("};"))).toContain("focus({ preventScroll: true })");
    expect(APP_SRC).toContain("tabIndex={-1}");
  });
});

describe("the keyboard controls are discoverable without reading the source", () => {
  it("a disclosure lists them", () => {
    expect(APP_SRC).toContain('className="keyboard-help"');
    expect(APP_SRC).toContain("<summary>Keyboard</summary>");
    expect(CSS).toContain(".keyboard-help");
  });

  it("it lists every key the handler actually implements", () => {
    const disclosure = APP_SRC.slice(APP_SRC.indexOf('className="keyboard-help"'));
    const body = disclosure.slice(0, disclosure.indexOf("</details>"));
    expect(body).toContain("Shift");
    expect(body).toContain("Space");
    expect(body).toContain("Esc");
    expect(body).toMatch(/aim/);
    expect(body).toMatch(/power/);
    // And each of those is a key the handler really handles.
    for (const code of ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Space", "Escape"]) {
      expect(APP_SRC, `${code} is advertised but not handled`).toContain(`e.code === "${code}"`);
    }
    expect(APP_SRC).toContain("e.shiftKey");
  });
});

describe("reduced motion is still honoured", () => {
  it("the stylesheet still collapses transitions rather than the content", () => {
    expect(CSS).toContain("@media (prefers-reduced-motion: reduce)");
    const block = CSS.slice(CSS.indexOf("@media (prefers-reduced-motion: reduce)"));
    expect(block).toContain("transition-duration: 0.001ms");
    expect(block).toContain("animation-duration: 0.001ms");
  });

  it("the app still reads the preference and hands it to the presentation", () => {
    expect(APP_SRC).toContain("usePrefersReducedMotion");
    expect(APP_SRC).toContain("(prefers-reduced-motion: reduce)");
    expect(APP_SRC).toMatch(/reducedMotion,/);
  });
});
