import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

// The portfolio shell's embed contract: an iframe is only "ready" when it
// says so via postMessage (see apps/portfolio/src/lib/embedProtocol.ts) --
// there's no timer-based fallback, so this handshake is required for the
// inline embed to ever leave its loading state. Double rAF waits for the
// table's first real paint, not just React's commit. No-op outside an iframe.
if (window.parent !== window) {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      window.parent.postMessage({ source: "portfolio-embed", type: "ready", id: "showboat" }, "*");
    });
  });
}
