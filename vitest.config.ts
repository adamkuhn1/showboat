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
    // Several suites run real WASM physics over full candidate sets (the
    // budget-accounting fixtures, the neural-vs-classical influence fixtures).
    // A 200-unit budget on a busy board is legitimately multi-second, and the
    // suites now run concurrently with each other, so the 5s default started
    // failing on timing rather than on behaviour. Raised deliberately — no
    // test here sleeps or polls; the time is spent in the physics engine.
    testTimeout: 60_000,
  },
});
