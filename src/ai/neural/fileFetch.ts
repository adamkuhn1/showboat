// Test-only helper: a `fetch` implementation backed by a directory on disk.
//
// It exists so the vitest suites exercise the *real* loader — the same
// `NeuralCandidateEvaluator.load()` -> `tryLoadRankerModel()` -> WebCrypto
// hash check -> `onnxruntime-web` session path the browser runs — against the
// actually-committed artifact, instead of a mock that would prove nothing.
//
// Only ever imported from `*.test.ts`; no production module references it.

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

export interface FileFetchOverrides {
  /** Serve these exact bytes/text for a path instead of what's on disk. */
  [relPath: string]: Uint8Array | string | null;
}

/**
 * @param root       directory the relative URLs resolve against
 * @param overrides  per-path overrides; `null` means "respond 404"
 */
export function makeFileFetch(root: string, overrides: FileFetchOverrides = {}): typeof fetch {
  return (async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    const rel = url.replace(/^\/+/, "");

    if (Object.prototype.hasOwnProperty.call(overrides, rel)) {
      const o = overrides[rel];
      if (o === null) return new Response("not found", { status: 404 });
      // Pass the view itself, not `.buffer`: Node Buffers are slices of a
      // shared allocation pool, so `.buffer` is the whole pool rather than the
      // file's bytes.
      const body: BodyInit = typeof o === "string" ? o : o;
      return new Response(body, {
        status: 200,
        headers: { "content-type": typeof o === "string" ? "application/json" : "application/octet-stream" },
      });
    }

    const path = join(root, rel);
    if (!existsSync(path)) return new Response("not found", { status: 404 });
    const buf = new Uint8Array(readFileSync(path));
    return new Response(buf, {
      status: 200,
      headers: {
        "content-type": path.endsWith(".json") ? "application/json" : "application/octet-stream",
      },
    });
  }) as typeof fetch;
}
