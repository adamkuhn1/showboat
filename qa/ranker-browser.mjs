// The ranker, in a real browser, through the real production modules.
//
// Two things the Node suite structurally cannot prove, because
// `onnxruntime-web`'s `.` export resolves to a Node build there and the
// `./wasm` subpath does not run under Node at all:
//
//   1. that onnxruntime-web loads and infers in Chrome with the entry point the
//      app actually imports, and
//   2. what logits it produces — so the entry-point change can be shown to be
//      numerically identical rather than assumed to be.
//
// It gets both by importing the app's own modules out of the Vite dev server's
// module graph, so the code under test is `src/ai/neural/evaluator.ts` and
// `src/ai/onnx.ts` exactly as shipped, not a copy.
//
// Also records which ORT runtime binary the browser actually fetched and how
// large it was, which is the bundle claim measured rather than read off a build
// log.
//
//   node qa/ranker-browser.mjs [url]

import { launchChrome, attachToPage } from "../../portfolio/qa/cdp.mjs";

const URL_ = process.argv[2] ?? "http://localhost:5311/";

/** Runs in the page. Imports the real modules and scores two fixed boards. */
const SCORE = `(async () => {
  const [{ makeTable }, { makeBall }, { CUE_ID }, physics, { generateCandidates }, { NeuralCandidateEvaluator }] =
    await Promise.all([
      import("/src/physics/table.ts"),
      import("/src/physics/ball.ts"),
      import("/src/game/rack.ts"),
      import("/src/physics/wasm-bridge.ts"),
      import("/src/ai/candidates.ts"),
      import("/src/ai/neural/evaluator.ts"),
    ]);
  await physics.initPhysics();
  const table = makeTable();
  const boards = {
    open: {
      balls: [makeBall(CUE_ID, -0.6, -0.1), makeBall(1, 0.3, 0.02), makeBall(2, 0.38, 0.1),
              makeBall(4, 0.1, -0.25), makeBall(5, -0.2, 0.3)],
      targets: [1, 2, 4, 5],
    },
    tight: {
      balls: [makeBall(CUE_ID, -0.75, 0.22), makeBall(3, 0.45, -0.28), makeBall(6, -0.1, 0.3)],
      targets: [3, 6],
    },
  };
  const ev = new NeuralCandidateEvaluator(new URL("model/ranker", document.baseURI).href);
  const state = await ev.load();
  if (state.status !== "ready") return { error: "model not ready: " + JSON.stringify(state) };
  const out = { manifest: ev.getManifest(), hashVerified: state.hashVerified, boards: {} };
  for (const [name, b] of Object.entries(boards)) {
    const cands = generateCandidates(b.balls, table, b.targets);
    const scored = await ev.score(b.balls, table, cands);
    if (!scored) return { error: "no scores for " + name };
    out.boards[name] = {
      candidates: cands.length,
      logits: scored.logits.map((v) => v.toFixed(9)),
      scores: scored.scores.map((v) => v.toFixed(9)),
    };
  }
  return out;
})()`;

const main = async () => {
  const chrome = await launchChrome({ headless: true });
  try {
    const page = await attachToPage(chrome.port);

    // Every network request, so the runtime binary can be identified by name
    // and size rather than inferred.
    const requests = [];
    await page.send("Network.enable");
    page.on("Network.responseReceived", (p) => {
      requests.push({ url: p.response.url, status: p.response.status, mime: p.response.mimeType });
    });
    page.on("Network.loadingFinished", (p) => {
      const r = requests.find((x) => x.id === undefined && x.pending === undefined);
      void r;
      void p;
    });

    await page.goto(URL_);
    const result = await page.eval(SCORE);

    const ort = requests.filter((r) => /ort[-.]/.test(r.url));
    // Sizes over the wire, asked of the browser rather than the build log.
    const sizes = await page.eval(`(async () => {
      const out = {};
      for (const e of performance.getEntriesByType("resource")) {
        if (!/ort[-.]/.test(e.name)) continue;
        out[e.name.split("/").pop()] = { transferSize: e.transferSize, decodedBodySize: e.decodedBodySize };
      }
      return out;
    })()`);

    console.log(
      JSON.stringify(
        {
          url: URL_,
          model: result.error
            ? { error: result.error }
            : { artifact: result.manifest.artifact, sha256: result.manifest.onnx_sha256, hashVerified: result.hashVerified },
          boards: result.boards ?? null,
          ortRequests: ort.map((r) => ({ file: r.url.split("/").pop(), status: r.status, mime: r.mime })),
          ortSizes: sizes,
        },
        null,
        1,
      ),
    );
  } finally {
    await chrome.close();
  }
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
