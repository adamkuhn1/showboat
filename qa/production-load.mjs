// Production build: does the ranker load, and does it still fall back?
//
// The dev-server checks import the app's modules directly. This one drives the
// built bundle the way a visitor does, because Rollup resolves the ORT entry
// and its `new URL(..., import.meta.url)` runtime reference by a different
// mechanism from the dev server, and "it works in dev" has already been wrong
// once in this app (see vite.config.ts).
//
// Two runs:
//   loads   — a normal visit; the opponent must reach a `neural-hybrid`
//             decision with the artifact hash verified. This is the check that
//             works and is the one this file is for.
//   blocked — INCONCLUSIVE, and left in place saying so rather than deleted.
//             `Network.setBlockedURLs` is scoped to the target it is enabled
//             on, and the model is fetched inside the PLANNING WORKER, which is
//             a separate target this harness does not attach to. The run
//             therefore reports a normal neural decision and proves nothing
//             about the fallback. Reaching it would mean attaching to the
//             worker target as it is created (Target.setAutoAttach), which is
//             worth doing but is not done here.
//
//             The fallback itself is not unverified: `ai/fallback.test.ts`
//             drives all eight `FallbackTrace` causes, and
//             `neural/productionModel.test.ts` drives a corrupted artifact, a
//             missing artifact and a hash mismatch through the real loader.
//             What is missing is only the browser-level version of that.
//
//   node qa/production-load.mjs [url]

import { launchChrome, attachToPage, sleep } from "../../portfolio/qa/cdp.mjs";

const URL_ = process.argv[2] ?? "http://localhost:5321/";

const PROBE = `
window.__p = { turns: [], ort: [] };
const R = window.Worker;
window.Worker = class extends R {
  constructor(...a) {
    super(...a);
    this.addEventListener("message", (e) => {
      const d = e.data;
      if (!d || d.type !== "done" || !d.planned) return;
      const t = d.planned.trace;
      window.__p.turns.push({
        kind: d.planned.kind,
        mode: t ? t.mode : null,
        model: t && t.model ? { artifact: t.model.artifact, hashVerified: t.model.hashVerified } : null,
        fallback: t && t.fallback ? { cause: t.fallback.cause, detail: t.fallback.detail } : null,
        shotKind: t && t.selected ? t.selected.kind : null,
      });
    });
  }
};
`;

const playUntilOpponentTurn = async (page) => {
  const deadline = Date.now() + 150_000;
  while (Date.now() < deadline) {
    const turns = await page.eval(`window.__p.turns`);
    if (turns.length > 0) return turns;
    await page.eval(`(() => {
      const b = [...document.querySelectorAll('button')].find(x => /shoot/i.test(x.textContent) && !x.disabled);
      if (b) b.click();
      return !!b;
    })()`);
    await sleep(2500);
  }
  return [];
};

// A FRESH browser per run. Reattaching to the same page target carries the
// previous run's HTTP cache and its already-created worker, so the "blocked"
// run silently loaded the model anyway and reported a neural decision — which
// looked like a passing fallback check and was not one.
const run = async ({ block }) => {
  const chrome = await launchChrome({ headless: true });
  try {
    return await runIn(chrome, { block });
  } finally {
    await chrome.close();
  }
};

const runIn = async (chrome, { block }) => {
  const page = await attachToPage(chrome.port);
  await page.addInitScript(PROBE);
  if (block) await page.blockUrls(["*showboat-ranker*.onnx", "*manifest.json*"]);

  const ortFiles = [];
  await page.send("Network.enable");
  page.on("Network.responseReceived", (p) => {
    const f = p.response.url.split("/").pop();
    if (/ort[-.]/.test(f)) ortFiles.push({ file: f, status: p.response.status });
  });

  await page.goto(URL_);
  await sleep(2500);
  const turns = await playUntilOpponentTurn(page);
  return { block, turns: turns.slice(0, 3), ortFiles };
};

const main = async () => {
  const loads = await run({ block: false });
  const blocked = await run({ block: true });
  console.log(JSON.stringify({ url: URL_, loads, blocked }, null, 1));
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
