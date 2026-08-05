// Regression test for the `npm run dev` neural-model load failure.
//
// The bug (found in live Chrome QA, see
// docs/repair/release-candidate/showboat/REPORT.md §1): Vite's dependency
// pre-bundler rewrote `onnxruntime-web` into `node_modules/.vite/deps/` without
// copying the sibling `ort-wasm-simd-threaded.jsep.wasm` that the bundle
// resolves with `new URL(<name>, import.meta.url)`. The dev server's SPA
// fallback then answered that request with `index.html`, so ORT was handed an
// HTML document where a WASM binary belonged and every neural decision fell
// back to classical.
//
// This test boots a REAL Vite dev server from the app's real `vite.config.ts`,
// asks it (exactly as the browser does) which URL `import("onnxruntime-web")`
// resolves to, derives the runtime's own WASM URL from that module URL with the
// same `new URL(name, moduleUrl)` rule ORT uses, and fetches it over real HTTP.
//
// It fails on the pre-fix config: the derived URL was
// `/node_modules/.vite/deps/ort-wasm-simd-threaded.jsep.wasm`, which returned
// 200 `text/html` starting with `<!do`.

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { createServer, type ViteDevServer } from "vite";

const APP_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");

// onnxruntime-web's package.json is not an exported subpath, so resolve the
// package entry itself — every published entry lives in `dist/`.
const ORT_DIST = dirname(createRequire(import.meta.url).resolve("onnxruntime-web"));

/**
 * The WASM binaries onnxruntime-web can ask for. Read out of the installed
 * package rather than hard-coded, so a package upgrade that renames the runtime
 * makes this test fail loudly instead of passing vacuously.
 *
 * Preferred source is the exact entry file the dev server resolved to (only
 * that build's own runtime is relevant). When the dev server resolved to a
 * pre-bundled copy that has no on-disk sibling — i.e. the bug this file
 * guards — fall back to every `.wasm` in the real package, so the test still
 * reaches the HTTP/MIME assertion that names the actual failure instead of
 * dying on a file read.
 */
const ortRuntimeWasmNames = (entryFile: string): string[] => {
  if (existsSync(entryFile)) {
    const src = readFileSync(entryFile, "utf8");
    const names = new Set<string>();
    for (const m of src.matchAll(/ort-wasm[a-zA-Z0-9.\-_]*\.wasm/g)) names.add(m[0]);
    if (names.size > 0) return [...names];
  }
  return readdirSync(ORT_DIST).filter((f) => f.endsWith(".wasm"));
};

let server: ViteDevServer;
let origin: string;

beforeAll(async () => {
  server = await createServer({
    root: APP_ROOT,
    logLevel: "silent",
    // Not the app's dev port: this suite must not collide with a developer's
    // running `npm run dev`. strictPort:false lets Vite walk to a free one.
    server: { port: 5399, strictPort: false },
  });
  await server.listen();
  // Vite binds ::1 by default, so derive the origin from the server's own
  // resolved URL rather than assuming 127.0.0.1.
  const local = server.resolvedUrls?.local?.[0];
  if (!local) throw new Error("dev server did not report a local URL");
  origin = local.replace(/\/$/, "");
}, 120_000);

afterAll(async () => {
  await server?.close();
});

describe("vite dev server serves onnxruntime-web's own WASM runtime", () => {
  it("resolves onnxruntime-web to a module URL outside .vite/deps", async () => {
    // `onnx.ts` is the only module that imports onnxruntime-web, and it does so
    // dynamically. Transforming it through the real dev server yields the exact
    // specifier the browser will import.
    const out = await server.transformRequest("/src/ai/onnx.ts");
    expect(out?.code, "onnx.ts should transform").toBeTruthy();
    const m = out!.code.match(/import\(\s*["']([^"']*onnxruntime-web[^"']*)["']/);
    expect(m, `no onnxruntime-web import found in transformed onnx.ts`).toBeTruthy();
    const moduleUrl = m![1];

    // The whole bug in one assertion: `.vite/deps` has no `.wasm` sibling.
    expect(
      moduleUrl.includes("/.vite/deps/"),
      `onnxruntime-web was pre-bundled to ${moduleUrl}; its sibling .wasm runtime ` +
        `is not copied there, so ORT's new URL(name, import.meta.url) 404s into ` +
        `the SPA fallback. Keep optimizeDeps.exclude = ["onnxruntime-web"].`,
    ).toBe(false);
  });

  it("serves every referenced ort-*.wasm as application/wasm with real WASM bytes", async () => {
    const out = await server.transformRequest("/src/ai/onnx.ts");
    const moduleUrl = out!.code.match(/import\(\s*["']([^"']*onnxruntime-web[^"']*)["']/)![1];

    // Resolve the on-disk entry the dev URL points at, so we can read which
    // runtime binaries that exact build references.
    const fsPrefix = "/@fs";
    const withoutQuery = moduleUrl.split("?")[0];
    const entryFile = withoutQuery.startsWith(fsPrefix)
      ? withoutQuery.slice(fsPrefix.length)
      : resolve(APP_ROOT, "." + withoutQuery);
    const names = ortRuntimeWasmNames(entryFile);
    expect(names.length, `no ort-*.wasm references found in ${entryFile}`).toBeGreaterThan(0);

    for (const name of names) {
      // Exactly ORT's own resolution rule.
      const wasmUrl = new URL(name, origin + moduleUrl).href;
      const res = await fetch(wasmUrl);
      expect(res.status, `${name} status`).toBe(200);
      expect(res.headers.get("content-type"), `${name} content-type`).toContain("application/wasm");
      const head = new Uint8Array((await res.arrayBuffer()).slice(0, 4));
      // \0asm — the magic word ORT's error message complained about.
      expect([...head], `${name} magic bytes`).toEqual([0x00, 0x61, 0x73, 0x6d]);
    }
  });

  it("serves the committed model + manifest byte-identically to the repo copy", async () => {
    const manifestRes = await fetch(`${origin}/model/ranker/manifest.json`);
    expect(manifestRes.status).toBe(200);
    expect(manifestRes.headers.get("content-type")).toContain("application/json");
    const manifest = await manifestRes.json();

    const onDisk = readFileSync(resolve(APP_ROOT, "public/model/ranker/manifest.json"), "utf8");
    expect(JSON.parse(onDisk)).toEqual(manifest);

    const artifactRes = await fetch(`${origin}/model/ranker/${manifest.artifact}`);
    expect(artifactRes.status).toBe(200);
    const served = new Uint8Array(await artifactRes.arrayBuffer());
    expect(served.byteLength).toBe(manifest.bytes);
    // Not HTML — the same check the runtime loader makes.
    expect(served[0]).not.toBe(0x3c);

    const diskBytes = new Uint8Array(
      readFileSync(resolve(APP_ROOT, `public/model/ranker/${manifest.artifact}`)),
    );
    expect(Buffer.from(served).equals(Buffer.from(diskBytes))).toBe(true);
  });

  it("still answers a missing model path with the SPA fallback the loader reads as absent", async () => {
    const res = await fetch(`${origin}/model/ranker/definitely-not-here.onnx`);
    // Dev server SPA fallback: the loader classifies this "absent", never "loaded".
    const text = await res.text();
    expect(res.headers.get("content-type") ?? "").toContain("text/html");
    expect(text.trimStart().startsWith("<")).toBe(true);
  });
});
