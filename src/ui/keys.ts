// Who owns a key press: the focused control, or the game.
//
// Showboat's shortcuts are registered on `window`, which is the only place they
// can be if space is to shoot from anywhere on the page. The cost is that the
// listener also sees keys that were delivered to a focused control first, and
// `preventDefault()` there cancels that control's own behaviour — the key
// appears to do nothing at all.
//
// That has already happened once: the English and Draw·Follow sliders took
// focus, drew a focus ring, and then ignored every arrow key, because the aim
// handler had called `preventDefault` before the range input saw it. The rule
// below is the general form of that fix.
//
//   - A text or range control owns EVERY key. Arrows step a slider, arrows and
//     characters edit a field; there is no key the game may take from one.
//   - A button, a link or a `<summary>` owns Space and Enter, which are how it
//     is activated, and nothing else. Arrows have no native behaviour on any of
//     them, so aiming stays live while one has focus.
//
// The second rule is why pointer-down on the felt moves focus to the canvas
// (see App.tsx): without it, clicking "New rack" and then pressing space would
// rack again rather than shoot, because the button would still be the focused
// control and would still, correctly, own the key.

const ACTIVATION_KEYS = new Set(["Space", "Enter", "NumpadEnter"]);
const TEXT_LIKE = new Set(["input", "select", "textarea"]);
const ACTIVATABLE = new Set(["button", "summary", "a", "option"]);

export function focusedControlOwnsKey(target: EventTarget | null, code: string): boolean {
  const el = target as (Element & { isContentEditable?: boolean }) | null;
  if (!el || typeof el.tagName !== "string") return false;
  const tag = el.tagName.toLowerCase();
  if (TEXT_LIKE.has(tag) || el.isContentEditable === true) return true;
  const activatable = ACTIVATABLE.has(tag) || el.getAttribute?.("role") === "button";
  return activatable && ACTIVATION_KEYS.has(code);
}
