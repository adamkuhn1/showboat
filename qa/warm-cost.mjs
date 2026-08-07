// What does a visitor who opens Showboat and leaves actually download?
//
// `App.tsx` warms the ranker on first idle, so the onnxruntime runtime is
// fetched whether or not anyone plays. The question is whether that is the
// right trade, and it has two halves:
//
//   - HOW MUCH: measured separately and directly by `qa/ranker-browser.mjs`,
//     which reads the browser's own resource timings for the runtime.
//   - WHEN: measured here, by recording the instant the page posts `warm` to
//     the planning worker on a page nobody touches.
//
// Why not simply total the worker's bytes here: the runtime is fetched INSIDE
// the planning worker, and `Network.enable` is scoped to the target it is
// enabled on. Seeing those requests means routing CDP messages to the worker's
// own session, which the shared harness (`apps/portfolio/qa/cdp.mjs`, owned by
// another team this sprint) does not expose. Recording the trigger and pairing
// it with the separately-measured size answers the same question without
// reaching outside this app.
//
//   node qa/warm-cost.mjs [url]

import { launchChrome, attachToPage, sleep } from "../../portfolio/qa/cdp.mjs";

const URL_ = process.argv[2] ?? "http://localhost:5321/";

/**
 * Records every message the page sends to the planning worker, with the time
 * since navigation start. `warm` is the one that pulls the runtime.
 */
const PROBE = `
window.__w = { posted: [], workers: 0 };
const R = window.Worker;
window.Worker = class extends R {
  constructor(...a) {
    super(...a);
    window.__w.workers++;
    const realPost = this.postMessage.bind(this);
    this.postMessage = (msg, ...rest) => {
      window.__w.posted.push({ type: msg && msg.type, atMs: Math.round(performance.now()) });
      return realPost(msg, ...rest);
    };
  }
};
`;

const run = async ({ interact, seconds }) => {
  const chrome = await launchChrome({ headless: true });
  try {
    const page = await attachToPage(chrome.port);
    await page.addInitScript(PROBE);
    await page.goto(URL_);
    if (interact) {
      await sleep(1500);
      await page.eval(`(() => {
        const b = [...document.querySelectorAll('button')].find(x => /shoot/i.test(x.textContent) && !x.disabled);
        if (b) b.click();
        return !!b;
      })()`);
    }
    await sleep(seconds * 1000);
    return await page.eval(`window.__w`);
  } finally {
    await chrome.close();
  }
};

const main = async () => {
  const idleOnly = await run({ interact: false, seconds: 12 });
  const played = await run({ interact: true, seconds: 12 });
  console.log(
    JSON.stringify(
      {
        url: URL_,
        "opens and leaves": idleOnly,
        "opens and plays": played,
        note: "a `warm` entry on the first run is the runtime being fetched for a visitor who never played",
      },
      null,
      1,
    ),
  );
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
