import { defineConfig } from "vitest/config";

export default defineConfig({
  // No React plugin and no JSX override: Vitest 4's own (oxc) transform already
  // handles the automatic JSX runtime for the single .tsx suite
  // (overlayTruthfulness, which renders the real OverlayPanel through
  // react-dom/server). Adding @vitejs/plugin-react here only produced
  // deprecation warnings and changed nothing.
  test: {
    include: ["src/**/*.test.ts", "src/**/*.test.tsx", "training/**/*.test.ts"],
    environment: "node",
  },
});
