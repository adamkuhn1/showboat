import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { isEmbedMode, observeHeight, postHeight, postReady } from "./embed";
import "./index.css";

const embed = isEmbedMode();
// Set before the first render so embed styling (index.css `[data-embed]`)
// applies from the first paint.
if (embed) document.documentElement.dataset.embed = "1";

const root = document.getElementById("root")!;
createRoot(root).render(
  <StrictMode>
    <App embed={embed} />
  </StrictMode>,
);

// The portfolio shell's embed contract: an iframe is only "ready" when it
// says so via postMessage (see the portfolio's src/lib/embedProtocol.ts) --
// there's no timer-based fallback, so this handshake is required for the
// inline embed to ever leave its loading state. Double rAF waits for the
// table's first real paint, not just React's commit. No-op outside an iframe.
// In embed mode the content height goes out with it (and after every layout
// change) so the host can size the frame to the content.
if (window.parent !== window) {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      postReady();
      if (embed) {
        postHeight(root, true);
        observeHeight(root);
      }
    });
  });
}
