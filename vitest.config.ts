import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      // `src/ai/onnx.ts` imports `onnxruntime-web/wasm`, the WASM-execution-
      // -provider-only build. That subpath is browser-only: it has no `node`
      // export condition, and both of its variants hand the ESM loader a
      // `blob:` URL for the runtime glue, which Node refuses outright
      // (ERR_UNSUPPORTED_ESM_URL_SCHEME). Verified directly, with a `file:`-
      // capable fetch shim in place and with the extern-wasm condition
      // selected; neither gets past it.
      //
      // WHAT THIS ALIAS DOES AND DOES NOT COST.
      //
      // It does not weaken anything, because the suite never ran the browser's
      // build to begin with: `onnxruntime-web`'s `.` export has a `node`
      // condition, so under Vitest it has always resolved to
      // `dist/ort.node.min.mjs` while the browser got `dist/ort.bundle.min.mjs`.
      // The Node suite has therefore always been testing *our loader, our
      // validation, our fallbacks and the real committed graph* against a
      // Node-side ORT, and it continues to do exactly that. The one thing it
      // cannot prove — that ORT loads and infers inside a real browser — it
      // could not prove before either, and is covered by `qa/ranker-browser.mjs`
      // driving real Chrome against both the dev server and the production
      // build.
      "onnxruntime-web/wasm": "onnxruntime-web",
    },
  },
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
