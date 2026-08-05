import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
// The same self-hosted EB Garamond (OFL) the shell bundles, at the same three
// weights, imported the same way. Showboat is shown inside the portfolio page;
// declaring a Garamond stack without shipping the face meant the frame fell
// through to Georgia while its host rendered EB Garamond, and the same string
// at 40px measured 464.8px here against 408.5px there. See
// apps/portfolio/src/main.tsx.
import "@fontsource/eb-garamond/latin-400.css";
import "@fontsource/eb-garamond/latin-500.css";
import "@fontsource/eb-garamond/latin-600.css";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
