// Recovery paths, in a real browser.
//
// The unit suite drives all eight fallback causes directly (ai/fallback.test.ts)
// with stubbed evaluators. This checks the two that are worth seeing a real
// browser survive, using the network layer rather than a stub:
//
//   1. the ranker artifact cannot be fetched at all  -> model-absent
//   2. the artifact request never answers            -> model-load-timeout
//
// In both cases the opponent must still take its turn, and the panel must say
// "Physics search" with the reason — never a neural label over a classical
// decision.
//
//   node qa/fallback.mjs <baseUrl> <outDir>

import { execSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { launchChrome, attachToPage } from "../../portfolio/qa/cdp.mjs";

const BASE = process.argv[2] ?? "http://localhost:5187/";
const OUT = process.argv[3] ?? "/tmp/showboat-qa-fallback";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = [];
const say = (...a) => {
  const l = a.join(" ");
  log.push(l);
  console.log(l);
};

/** Play one opponent turn and report what the panel claimed. */
const PLAY_ONE = `(async () => {
  const wait = (pred, ms) => new Promise(res => {
    const t0 = Date.now();
    const tick = () => {
      if (pred()) return res(true);
      if (Date.now() - t0 > ms) return res(false);
      setTimeout(tick, 100);
    };
    tick();
  });
  const btn = () => [...document.querySelectorAll('.buttons button')][0];
  const idle = () => btn() && !/Rolling|Opponent/.test(btn().textContent);
  const canvas = () => document.querySelector('canvas.table');
  const aimAndShoot = (i) => {
    const c = canvas(), r = c.getBoundingClientRect();
    const a = (i * 53 % 360) * Math.PI / 180;
    const x = r.left + r.width / 2 + Math.cos(a) * 180;
    const y = r.top + r.height / 2 - Math.sin(a) * 90;
    const o = { bubbles: true, clientX: x, clientY: y, pointerId: 1, isPrimary: true };
    c.dispatchEvent(new PointerEvent('pointerdown', o));
    c.dispatchEvent(new PointerEvent('pointermove', o));
    c.dispatchEvent(new PointerEvent('pointerup', o));
    if (btn() && !btn().disabled) { btn().click(); return true; }
    return false;
  };

  await wait(() => btn() && !btn().disabled, 30000);
  // Keep shooting until the turn actually passes to the opponent and it plans.
  // A break that keeps the table is the common case and is not the thing under
  // test here.
  let appeared = false;
  for (let i = 0; i < 10 && !appeared; i++) {
    aimAndShoot(i);
    await wait(() => !idle(), 5000);
    await wait(idle, 90000);
    appeared = !!document.querySelector('.overlay');
    if (appeared) await wait(idle, 90000);
  }
  const a = document.querySelector('.overlay');
  return {
    panelAppeared: appeared,
    title: a ? (a.querySelector('.overlay-title') || {}).textContent : null,
    warn: a ? (a.querySelector('.overlay-warn') || {}).textContent : null,
    lines: a ? [...a.querySelectorAll('.overlay-line')].map(e => e.textContent) : [],
    message: (document.querySelector('.msg') || {}).textContent,
    turnAdvanced: !/opponent/.test(document.querySelector('.turn').textContent),
  };
})()`;

async function scenario(name, setup) {
  const chrome = await launchChrome({ headless: true });
  const page = await attachToPage(chrome.port);
  const errors = [];
  await page.send("Runtime.enable");
  page.on("Runtime.consoleAPICalled", (p) => {
    if (p.type === "error") errors.push(p.args.map((a) => a.value ?? a.description ?? "").join(" "));
  });
  await page.viewport(1400, 950, { dpr: 2 });
  await setup(page);
  await page.goto(BASE);
  await sleep(1500);
  const out = await page.eval(PLAY_ONE);
  const { data } = await page.send("Page.captureScreenshot", { format: "png" });
  await writeFile(`${OUT}/${name}.png`, Buffer.from(data, "base64"));
  say(`${name.padEnd(20)} ${JSON.stringify(out)}`);
  say(`${"".padEnd(20)} console errors: ${errors.length}${errors[0] ? " — " + errors[0].slice(0, 140) : ""}`);
  await chrome.close();
  return out;
}

async function main() {
  await mkdir(OUT, { recursive: true });
  say(`commit           ${execSync("git rev-parse HEAD", { encoding: "utf8" }).trim()}`);
  say(`base url         ${BASE}`);

  // Control: nothing blocked.
  const ok = await scenario("00-healthy", async () => {});

  // 1. The artifact cannot be fetched. `Network.setBlockedURLs` is NOT enough
  //    and the first version of this file was wrong to use it: it applies to
  //    the page target, and the ranker is loaded by the planning WORKER, so the
  //    page's preflight failed while the worker loaded the model perfectly
  //    happily and the run proved nothing. `Fetch` interception does reach the
  //    worker — scenario 2 below demonstrates that by name in the panel text.
  const absent = await scenario("01-model-absent", async (page) => {
    await page.send("Fetch.enable", {
      patterns: [{ urlPattern: "*model/ranker*", requestStage: "Request" }],
    });
    page.on("Fetch.requestPaused", (p) => {
      void page.send("Fetch.failRequest", { requestId: p.requestId, errorReason: "Failed" });
    });
  });

  // 2. The artifact request never answers. This is the one that used to wedge
  //    a turn at "searching…" forever; it must now bound and recover.
  const stalled = await scenario("02-model-load-timeout", async (page) => {
    await page.send("Fetch.enable", {
      patterns: [{ urlPattern: "*model/ranker*", requestStage: "Request" }],
    });
    // Intercept and never continue: the request hangs open.
    page.on("Fetch.requestPaused", () => {});
  });

  const verdicts = [
    ["healthy run took a turn", ok.turnAdvanced || ok.panelAppeared],
    ["model-absent still took a turn", absent.turnAdvanced || absent.panelAppeared],
    ["model-absent labelled classical", absent.title === "Physics search"],
    // The reason must name the MODEL, not a board that happened to offer no
    // candidates. Without this the run passes on a snookered table while the
    // model is loading fine — which is how the first version of this file was
    // fooled.
    [
      "model-absent blamed the model, not the board",
      /model|ranker|artifact|load/i.test(absent.warn ?? ""),
    ],
    ["stalled load still took a turn", stalled.turnAdvanced || stalled.panelAppeared],
    ["stalled load labelled classical", stalled.title === "Physics search"],
    [
      "stalled load named the deadline",
      /deadline|loading/i.test(stalled.warn ?? ""),
    ],
  ];
  for (const [what, pass] of verdicts) say(`${pass ? "PASS" : "FAIL"}  ${what}`);

  await writeFile(`${OUT}/fallback.log`, log.join("\n") + "\n");
  if (verdicts.some(([, p]) => !p)) process.exitCode = 1;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
