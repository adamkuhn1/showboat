// Browser verification that the reasoning display is an observation.
//
// The unit suite proves the search publishes a correct stream. It cannot prove
// the page RECEIVES that stream while the search is still running — that is a
// property of the worker boundary, the message channel and the paint loop, and
// it needs a real browser and a real worker.
//
// So this instruments `Worker.prototype.postMessage`'s receiving end from the
// page: it records the wall-clock arrival time of every `progress` message and
// of the final `done`, then reports how much of the search was observable
// before the answer landed. A reconstruction would show every event arriving in
// one burst at, or after, `done`.
//
// Usage: node qa/live-search.mjs [url]   (default http://localhost:5311/)

import { launchChrome, attachToPage, sleep } from "../../portfolio/qa/cdp.mjs";

const URL = process.argv[2] ?? "http://localhost:5311/";

/**
 * Installed before the app's own scripts. It wraps the page-side `Worker` so
 * every message the planning worker posts is timestamped on arrival, without
 * touching a line of shipped code.
 */
const PROBE = `
window.__sb = { msgs: [], turns: [] };
const RealWorker = window.Worker;
window.Worker = class extends RealWorker {
  constructor(...args) {
    super(...args);
    this.addEventListener("message", (e) => {
      const d = e.data;
      if (!d || typeof d !== "object") return;
      const at = performance.now();
      if (d.type === "progress") {
        window.__sb.msgs.push({ at, id: d.id, kind: d.event.kind, seq: d.event.seq, atMs: d.event.atMs,
          index: d.event.index ?? null, reason: d.event.reason ?? null,
          n: d.event.candidates ? d.event.candidates.length : null });
      } else if (d.type === "done") {
        window.__sb.msgs.push({ at, id: d.id, kind: "__done__", seq: -1, atMs: -1 });
        const sel = d.planned.trace && d.planned.trace.selected;
        window.__sb.turns.push({
          id: d.id,
          kind: d.planned.kind,
          shotKind: sel ? sel.kind : null,
          rung: sel ? sel.rung : null,
          mode: d.planned.trace ? d.planned.trace.mode : null,
        });
      }
    });
  }
};
`;

const summarise = (msgs, id) => {
  const mine = msgs.filter((m) => m.id === id).sort((a, b) => a.at - b.at);
  if (mine.length === 0) return null;
  const done = mine.find((m) => m.kind === "__done__");
  const events = mine.filter((m) => m.kind !== "__done__");
  if (events.length === 0 || !done) return null;
  const t0 = events[0].at;
  const beforeDone = events.filter((m) => m.at < done.at);
  return {
    id,
    events: events.length,
    // How long the page spent receiving events, in wall-clock ms.
    observableMs: +(events[events.length - 1].at - t0).toFixed(1),
    // The gap between the last event and the answer arriving.
    doneAfterLastEventMs: +(done.at - events[events.length - 1].at).toFixed(1),
    eventsBeforeDone: beforeDone.length,
    // The decisive number: a reconstruction can only be 0.
    fractionBeforeDone: +(beforeDone.length / events.length).toFixed(3),
    kinds: events.reduce((a, m) => ((a[m.kind] = (a[m.kind] ?? 0) + 1), a), {}),
    // Spacing between consecutive simulation results — is it readable?
    verifyGapsMs: (() => {
      const v = events.filter((m) => m.kind === "candidate-verified").map((m) => m.at);
      return v.slice(1).map((t, i) => +(t - v[i]).toFixed(1));
    })(),
  };
};

const main = async () => {
  const chrome = await launchChrome({ headless: true });
  try {
    const page = await attachToPage(chrome.port);
    await page.addInitScript(PROBE);
    await page.viewport(1440, 900, { dpr: 1 });
    await page.goto(URL);
    await page.eval(`new Promise(r => setTimeout(r, 1500))`);

    // Break, so the opponent gets a turn. The canvas owns aiming; the Shoot
    // button is the one control that does not need a pointer gesture.
    const shoot = async () => {
      const ok = await page.eval(`(() => {
        const b = [...document.querySelectorAll('button')].find(x => /shoot/i.test(x.textContent) && !x.disabled);
        if (!b) return false;
        b.click();
        return true;
      })()`);
      return ok;
    };

    // The human keeps the table whenever the break (or a later shot) pots, so
    // shooting once does not reliably hand the opponent a turn. Shoot whenever
    // the button is live, and collect every opponent decision that results,
    // until enough turns have been observed.
    const WANT = Number(process.env.SB_TURNS ?? 5);
    const results = [];
    const seen = new Set();
    const deadline = Date.now() + 240_000;

    while (results.length < WANT && Date.now() < deadline) {
      const turns = await page.eval(`window.__sb.turns`);
      for (const t of turns) {
        if (seen.has(t.id)) continue;
        seen.add(t.id);
        const msgs = await page.eval(`window.__sb.msgs`);
        const s = summarise(msgs, t.id);
        if (s) results.push({ ...s, ...t });
      }
      if (results.length >= WANT) break;

      // Nudge the aim between shots so the rack does not repeat one line.
      await page.eval(`(() => {
        const n = [...document.querySelectorAll('button')].filter(b => /0\\.25/.test(b.textContent));
        for (let i = 0; i < 1 + Math.floor(Math.random() * 12); i++) n[Math.random() < 0.5 ? 0 : 1]?.click();
        return true;
      })()`);
      await shoot();
      await sleep(2500);
    }

    console.log(JSON.stringify({ url: URL, turns: results }, null, 2));
  } finally {
    await chrome.close();
  }
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
