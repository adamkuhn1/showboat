import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: "./" keeps built asset paths relative so each app works when served
// standalone AND when embedded in the portfolio shell via iframe.
export default defineConfig({
  plugins: [react()],
  base: "./",

  optimizeDeps: {
    // Do NOT let Vite's dependency pre-bundler rewrite onnxruntime-web.
    //
    // Root cause this fixes (reproduced in Chrome against `vite dev`, see
    // docs/repair/release-candidate/showboat/REPORT.md §1):
    //
    // onnxruntime-web's browser bundle locates its own ~27 MB WASM runtime with
    //     new URL("ort-wasm-simd-threaded.jsep.wasm", import.meta.url)
    // i.e. *relative to the module's own URL*. Vite's dep optimizer copies the
    // JS into `node_modules/.vite/deps/onnxruntime-web.js` but does not copy the
    // sibling `.wasm` binary, so at runtime `import.meta.url` points at
    // `.vite/deps/` and the runtime requests
    //     /node_modules/.vite/deps/ort-wasm-simd-threaded.jsep.wasm
    // which does not exist. Vite's SPA fallback answers 200 text/html with
    // index.html, so ORT gets an HTML document where a WASM binary should be —
    // the observed `expected magic word 00 61 73 6d, found 3c 21 64 6f` (`<!do`).
    //
    // Production was never affected: Rollup statically resolves the same
    // `new URL(..., import.meta.url)` against the real package directory and
    // emits the binary as a build asset with a rewritten URL.
    //
    // Excluding the package from pre-bundling makes dev serve it from its real
    // location, so `import.meta.url` resolves to the actual `dist/` directory
    // where the `.wasm` sits and Vite serves it as `application/wasm`.
    //
    // Chosen over the alternative fix (importing the `.wasm` with `?url` and
    // assigning `ort.env.wasm.wasmPaths`) because that one hard-codes *which*
    // runtime variant ORT is allowed to use — a decision ORT makes at runtime
    // from feature detection (SIMD / threads / JSEP). Excluding the dep leaves
    // that decision where it belongs and keeps dev and prod resolving the asset
    // by the same mechanism. Regression-tested in
    // `src/ai/neural/devServerAssets.test.ts`, which boots a real dev server and
    // asserts the runtime's own WASM URL returns `application/wasm`.
    exclude: ["onnxruntime-web"],
  },
});
