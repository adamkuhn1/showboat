// ONNX loading and inference for the candidate ranker.
//
// onnxruntime-web is a ~27 MB WASM runtime. It is imported *dynamically*, and
// only after the artifact's bytes have been fetched and validated, so a visitor
// who never reaches an opponent turn never downloads it.
//
// A second model slot used to live here: a whole-board policy/value net from
// the LightZero self-play pipeline, with its own loader, status enum and
// `evaluate()`. That net was never trained, nothing ever imported any of it,
// and it is deleted rather than kept as a plausible-looking stub. What remains
// of it is the observation encoder below, which the ranker's feature builder
// genuinely uses.

type Ort = typeof import("onnxruntime-web/wasm");
type InferenceSession = import("onnxruntime-web/wasm").InferenceSession;

/**
 * Which onnxruntime-web build to load, and why there are two.
 *
 * The browser gets `onnxruntime-web/wasm`: the WASM-execution-provider-only
 * build. It is 73 KB of JS against 405 KB, and it pulls
 * `ort-wasm-simd-threaded.wasm` (13.5 MB) instead of the JSEP runtime
 * (26.8 MB) — 13.7 MB of WebGPU/WebNN support that this app can never reach,
 * because the only session it creates asks for `executionProviders: ["wasm"]`.
 *
 * That subpath does not run under Node: it has no `node` export condition, and
 * both of its variants hand the ESM loader a `blob:` URL for the runtime glue,
 * which Node refuses (ERR_UNSUPPORTED_ESM_URL_SCHEME). Verified directly, with
 * a `file:`-capable fetch shim in place and with the extern-wasm condition
 * selected. So Node — the vitest suite and the `eval/` harnesses, which run
 * real inference through this exact module — takes the package's own `node`
 * entry instead. That is the build it has always used: `.`'s `node` condition
 * resolves to `ort.node.min.mjs`, so nothing about the test story changes here.
 *
 * The specifier is computed and `@vite-ignore`d so the bundler cannot see it:
 * a statically analysable `import("onnxruntime-web")` here would pull the full
 * 405 KB entry back into the browser build and undo the whole point.
 */
const ORT_NODE_ENTRY = "onnxruntime-web";
const isNode = typeof process !== "undefined" && Boolean(process.versions?.node);

const importOrt = async (): Promise<Ort> =>
  isNode
    ? ((await import(/* @vite-ignore */ ORT_NODE_ENTRY)) as unknown as Ort)
    : await import("onnxruntime-web/wasm");

// Board-observation layout, kept in one place so training and inference agree:
// normalized (x,y) for the 16 balls + a pocketed flag each = 48 floats. Read by
// `ranker/encode.ts`, which asserts its own board block matches `OBS_DIM`.
export const OBS_BALLS = 16;
export const OBS_DIM = OBS_BALLS * 3;

// ---------------------------------------------------------------------------
// Candidate-ranker model: load, validate, and batch-score candidate rows.
// ---------------------------------------------------------------------------

// "absent"  — no artifact at the URL at all (dev clone that hasn't staged one).
// "invalid" — an artifact IS there but failed integrity/contract validation.
//             Deliberately distinct from "absent": a corrupted or mismatched
//             model is a loud failure, not a quiet "no model configured".
export type RankerStatus = "absent" | "loading" | "loaded" | "invalid" | "error";

let rankerStatus: RankerStatus = "absent";
let rankerOrt: Ort | null = null;
let rankerSession: InferenceSession | null = null;
let rankerLoadPromise: Promise<void> | null = null;
let rankerLoadedUrl: string | null = null;
let rankerError: string | null = null;

export interface RankerLoadOptions {
  /**
   * sha256 the fetched bytes must hash to (from the model manifest). When
   * given and the environment exposes WebCrypto (`crypto.subtle`, available in
   * any secure context and in Node), a mismatch is a hard failure. When
   * WebCrypto is unavailable the byte length is still checked against
   * `expectedBytes` and `hashVerified()` reports false, rather than pretending
   * the artifact was verified.
   */
  expectedSha256?: string;
  expectedBytes?: number;
  /**
   * Feature-vector width the caller's encoder produces. Verified by running a
   * real zero-filled probe inference through the loaded graph: a graph built
   * for a different input width throws here instead of silently broadcasting
   * or truncating at the first live decision.
   */
  expectedInputDim?: number;
  /**
   * Injected only so tests can serve the real committed artifact off disk
   * through the identical code path the browser uses. Defaults to global
   * `fetch` in every non-test caller.
   */
  fetchImpl?: typeof fetch;
}

let rankerHashVerified = false;

const sha256Hex = async (buf: ArrayBuffer): Promise<string | null> => {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const digest = await subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
};

/**
 * Load and *validate* the candidate-ranker artifact. Every failure path sets a
 * status the caller can distinguish and an error string it can surface — this
 * function never resolves into a state where the app believes it has a model
 * it does not have, and never leaves a validation failure looking like a
 * benign "no model configured".
 */
export const tryLoadRankerModel = async (url: string, opts: RankerLoadOptions = {}): Promise<void> => {
  if (rankerLoadPromise && rankerLoadedUrl === url) return rankerLoadPromise;
  rankerLoadedUrl = url;
  rankerLoadPromise = (async () => {
    rankerStatus = "loading";
    rankerError = null;
    rankerHashVerified = false;
    try {
      const res = await (opts.fetchImpl ?? fetch)(url, { method: "GET" });
      if (!res.ok) {
        rankerStatus = "absent";
        rankerError = `HTTP ${res.status} fetching ${url}`;
        return;
      }
      const ct = res.headers.get("content-type") ?? "";
      if (ct.includes("text/html")) {
        // SPA index.html fallback for a missing file — genuinely absent.
        rankerStatus = "absent";
        rankerError = `${url} returned text/html (SPA fallback) — no artifact at that path`;
        return;
      }
      const buf = await res.arrayBuffer();
      const first = new Uint8Array(buf.slice(0, 1))[0];
      if (first === 0x3c) {
        rankerStatus = "absent";
        rankerError = `${url} starts with '<' — HTML, not an ONNX graph`;
        return;
      }

      if (opts.expectedBytes !== undefined && buf.byteLength !== opts.expectedBytes) {
        rankerStatus = "invalid";
        rankerError = `artifact is ${buf.byteLength} bytes, manifest declares ${opts.expectedBytes}`;
        return;
      }
      if (opts.expectedSha256) {
        const actual = await sha256Hex(buf);
        if (actual === null) {
          rankerHashVerified = false;
        } else if (actual !== opts.expectedSha256) {
          rankerStatus = "invalid";
          rankerError =
            `artifact sha256 ${actual} != manifest ${opts.expectedSha256} — refusing to load ` +
            `an artifact whose provenance cannot be established`;
          return;
        } else {
          rankerHashVerified = true;
        }
      }

      rankerOrt = rankerOrt ?? (await importOrt());
      const candidateSession = await rankerOrt.InferenceSession.create(buf, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });

      if (opts.expectedInputDim !== undefined) {
        // Real probe inference, not metadata introspection: this is the only
        // check that actually proves the graph accepts the encoder's rows.
        const probe = new rankerOrt.Tensor(
          "float32",
          new Float32Array(opts.expectedInputDim),
          [1, opts.expectedInputDim],
        );
        const out = await candidateSession.run({ [candidateSession.inputNames[0]]: probe });
        const data = out[candidateSession.outputNames[0]].data as Float32Array;
        if (data.length !== 1 || !Number.isFinite(data[0])) {
          rankerStatus = "invalid";
          rankerError = `probe inference returned ${data.length} value(s) [${data[0]}], expected 1 finite logit`;
          return;
        }
      }

      rankerSession = candidateSession;
      rankerStatus = "loaded";
    } catch (e) {
      // A throw here means the bytes were present but unusable (unparseable
      // graph, wrong input width caught by the probe, runtime failure).
      // That's "invalid", not "absent" — the difference matters to the UI.
      rankerSession = null;
      rankerStatus = "invalid";
      rankerError = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
    }
  })();
  return rankerLoadPromise;
};

export const rankerModelStatus = (): RankerStatus => rankerStatus;
export const rankerLoadError = (): string | null => rankerError;
export const rankerHashWasVerified = (): boolean => rankerHashVerified;
export const hasRankerModel = (): boolean => rankerStatus === "loaded" && rankerSession !== null;

/** Test-only: drop the loaded session so a different artifact can be loaded. */
export const _resetRankerForTests = (): void => {
  rankerStatus = "absent";
  rankerSession = null;
  rankerLoadPromise = null;
  rankerLoadedUrl = null;
  rankerError = null;
  rankerHashVerified = false;
};

/**
 * Score a batch of candidate rows in one session.run() call (not N calls —
 * see docs/repair/showboat-ml/04-browser-integration.md §8 on per-call
 * overhead). `rows` is `count` rows of `dim` floats each, row-major. Returns
 * one score per row (raw logit; caller applies sigmoid if a probability-style
 * display is wanted — see EVALUATION_SPEC.md on not doing that before
 * calibration is measured).
 */
export const evaluateCandidateRows = async (
  rows: Float32Array,
  count: number,
  dim: number,
): Promise<Float32Array | null> => {
  if (!hasRankerModel() || !rankerSession || !rankerOrt) return null;
  const input = new rankerOrt.Tensor("float32", rows, [count, dim]);
  const feeds: Record<string, import("onnxruntime-web/wasm").Tensor> = {};
  feeds[rankerSession.inputNames[0]] = input;
  const out = await rankerSession.run(feeds);
  return out[rankerSession.outputNames[0]].data as Float32Array;
};

// Encode a ball list into the observation vector the net expects. Positions are
// normalized to [-1,1] by half the table dimensions; pocketed balls report 0s.
export const encodeObservation = (
  balls: { id: number; pos: { x: number; y: number }; pocketed: boolean }[],
  halfLen: number,
  halfWid: number,
): Float32Array => {
  const obs = new Float32Array(OBS_DIM);
  for (const b of balls) {
    if (b.id >= OBS_BALLS) continue;
    const o = b.id * 3;
    if (b.pocketed) {
      obs[o] = 0;
      obs[o + 1] = 0;
      obs[o + 2] = 1; // pocketed flag
    } else {
      obs[o] = b.pos.x / halfLen;
      obs[o + 1] = b.pos.y / halfWid;
      obs[o + 2] = 0;
    }
  }
  return obs;
};
