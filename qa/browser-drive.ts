// Shared browser plumbing for the Showboat QA harnesses.
//
// One copy of "start the app's own server", "install the observation probe",
// "take a shot", "wait for the table to settle" — so two harnesses that drive
// the same page cannot drift into driving it differently and reporting
// different things about the same build.
//
// The probe below is QA-side only. It wraps the `Worker` constructor before the
// app boots and copies the planning worker's messages onto `window.__showboatQA`.
// Nothing in `src/` knows it exists, and it changes no behaviour: every message
// is forwarded to the real handler exactly as it arrived.

import { spawn, type ChildProcess } from "node:child_process";
import {
  createServer,
  type IncomingMessage,
  type Server as HttpServer,
  type ServerResponse,
} from "node:http";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const APP_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Installed with `Page.addScriptToEvaluateOnNewDocument`, before app scripts. */
export const PROBE = `
(() => {
  const qa = { requests: [], responses: [], titles: [], outcomes: [], panels: [], frames: [] };
  window.__showboatQA = qa;
  const NativeWorker = window.Worker;
  window.Worker = function (url, opts) {
    const w = new NativeWorker(url, opts);
    const post = w.postMessage.bind(w);
    w.postMessage = (msg, ...rest) => {
      try { qa.requests.push({ at: performance.now(), msg: JSON.parse(JSON.stringify(msg)) }); } catch {}
      return post(msg, ...rest);
    };
    w.addEventListener("message", (e) => {
      try {
        const d = e.data;
        if (d && (d.type === "done" || d.type === "model-status" || d.type === "error")) {
          qa.responses.push({ at: performance.now(), msg: JSON.parse(JSON.stringify(d)) });
        }
      } catch {}
    });
    return w;
  };
  window.Worker.prototype = NativeWorker.prototype;
  // Panel text, sampled. Two questions need a timestamp rather than an end
  // state: when the title first claimed neural, and whether a result line was
  // on screen before the next turn started.
  const txt = (sel) => { const e = document.querySelector(sel); return e ? e.textContent : null; };
  setInterval(() => {
    const now = performance.now();
    const text = txt(".overlay-title");
    const last = qa.titles[qa.titles.length - 1];
    if (text && (!last || last.text !== text)) qa.titles.push({ at: now, text });
    const otext = txt(".overlay-outcome");
    const lastO = qa.outcomes[qa.outcomes.length - 1];
    if (otext && (!lastO || lastO.text !== otext)) qa.outcomes.push({ at: now, text: otext });
    // Every distinct state of the panel's four claim lines, so a turn the
    // sampler did not happen to be looking at when it settled is still
    // checkable afterwards. Presence of the result line is recorded on every
    // tick, not only when its wording changes, because two consecutive shots
    // can legitimately produce the same sentence.
    const p = {
      title: text,
      chosen: txt(".chosen-why"),
      reconcile: txt(".overlay-reconcile"),
      reason: txt(".overlay-reason"),
      outcome: otext,
      // The fallback banner. Sampled with the rest because the recovery gate
      // has to read it off a turn that has already been replaced by the next
      // one by the time the table settles.
      warn: txt(".overlay-warn"),
    };
    const lastP = qa.panels[qa.panels.length - 1];
    const same =
      lastP &&
      lastP.title === p.title &&
      lastP.chosen === p.chosen &&
      lastP.reconcile === p.reconcile &&
      lastP.reason === p.reason &&
      lastP.outcome === p.outcome &&
      lastP.warn === p.warn;
    if (same) lastP.until = now;
    else qa.panels.push({ at: now, until: now, ...p });
  }, 40);
  // Frame timestamps, for the playback-rate check. rAF only; no drawing.
  const tick = (t) => { qa.frames.push(t); if (qa.frames.length > 20000) qa.frames.shift(); requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
})();
`;

/** Resolves once the app has booted, or with a TIMEOUT string. */
export const READY = `new Promise(res => {
  const t0 = Date.now();
  const tick = () => {
    const m = document.querySelector('.msg');
    if (m && /break to start|couldn't load/.test(m.textContent)) return res(m.textContent);
    if (Date.now() - t0 > 60000) return res('TIMEOUT: ' + (m ? m.textContent : 'no .msg'));
    setTimeout(tick, 100);
  };
  tick();
})`;

export interface Server {
  proc: ChildProcess;
  origin: string;
  stop: () => void;
}

/**
 * Start one of the app's own npm scripts on a given port. The port is passed
 * through rather than hard-coded so a QA run cannot collide with a developer's
 * running server; everything else is the script as it ships.
 */
export async function startServer(script: string, port: number, label: string): Promise<Server> {
  const proc = spawn("npm", ["run", script, "--", "--port", String(port), "--strictPort"], {
    cwd: APP_ROOT,
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, BROWSER: "none" },
  });
  let out = "";
  // Read the origin off the server's own banner rather than assuming
  // 127.0.0.1: Vite binds `localhost`, which resolves to ::1 first on macOS, so
  // a hand-built IPv4 origin is refused even though the server is up.
  const origin = await new Promise<string>((res, rej) => {
    const t = setTimeout(() => rej(new Error(`${label} did not start:\n${out}`)), 180_000);
    const onData = (b: Buffer) => {
      out += b.toString();
      // eslint-disable-next-line no-control-regex
      const clean = out.replace(/\[[0-9;]*m/g, "");
      const m = clean.match(new RegExp(`(https?://[^\\s/]+:${port})/?`));
      if (m) {
        clearTimeout(t);
        res(m[1]);
      }
    };
    proc.stdout?.on("data", onData);
    proc.stderr?.on("data", onData);
  });
  // The banner prints before the server is necessarily answering.
  let answered = false;
  for (let i = 0; i < 150 && !answered; i++) {
    try {
      answered = (await fetch(origin + "/")).ok;
    } catch {
      /* not up yet */
    }
    if (!answered) await sleep(200);
  }
  if (!answered) throw new Error(`${label} announced ${origin} but never answered:\n${out}`);
  return { proc, origin, stop: () => proc.kill("SIGTERM") };
}

/** The subset of the CDP client these helpers use. */
export interface DrivablePage {
  eval(expression: string): Promise<unknown>;
}

/** How a proxied path is broken. */
export type BreakMode = "spa-fallback" | "corrupt";

export interface BreakRule {
  match: RegExp;
  mode: BreakMode;
}

export interface Proxy {
  origin: string;
  /** Requests seen per path, including the ones that were broken. */
  hits: Map<string, number>;
  stop: () => Promise<void>;
}

/**
 * A pass-through HTTP proxy in front of a running server, with rules that break
 * chosen paths.
 *
 * The page's own network cannot be used for this. The model artifact and ORT's
 * runtime are fetched by the PLANNING WORKER, a separate DevTools target, so
 * `Network.setBlockedURLs` on the page session leaves them untouched — the
 * turns keep running the model and the "broken" condition never happens.
 * Breaking the response at the origin reaches whoever asks for it.
 *
 * Two failure shapes, because the loader distinguishes them:
 *   `spa-fallback` answers with the dev server's own `index.html`, which is the
 *      shape a missing file takes on a single-page dev server;
 *   `corrupt`      answers with the real bytes, one of them changed, which is
 *      the shape an artifact that fails its integrity check takes.
 */
export async function startBlockingProxy(
  target: string,
  port: number,
  rules: BreakRule[],
): Promise<Proxy> {
  const hits = new Map<string, number>();
  const server: HttpServer = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    const path = req.url ?? "/";
    hits.set(path, (hits.get(path) ?? 0) + 1);
    const rule = rules.find((r) => r.match.test(path));
    try {
      if (rule?.mode === "spa-fallback") {
        const html = await (await fetch(`${target}/`)).text();
        res.writeHead(200, { "content-type": "text/html" });
        res.end(html);
        return;
      }
      const upstream = await fetch(target + path, {
        method: req.method,
        headers: { accept: String(req.headers.accept ?? "*/*") },
      });
      const buf = Buffer.from(await upstream.arrayBuffer());
      if (rule?.mode === "corrupt" && buf.length > 64) buf[Math.floor(buf.length / 2)] ^= 0xff;
      const headers: Record<string, string> = {};
      const ct = upstream.headers.get("content-type");
      if (ct) headers["content-type"] = ct;
      res.writeHead(upstream.status, headers);
      res.end(buf);
    } catch (e) {
      res.writeHead(502, { "content-type": "text/plain" });
      res.end(String(e));
    }
  });
  await new Promise<void>((r) => server.listen(port, "127.0.0.1", r));
  return {
    origin: `http://127.0.0.1:${port}`,
    hits,
    stop: () => new Promise<void>((r) => server.close(() => r())),
  };
}

/**
 * Wait for the table to stop being busy.
 *
 * A stall is "busy AND unchanging", not "busy for a while": an opponent that
 * keeps potting stays at the table legitimately. The page is downsampled to a
 * 32x18 checksum every 100 ms; a running animation changes it every frame and a
 * wedged turn does not change it at all.
 */
export async function waitIdle(
  page: DrivablePage,
  stallMs = 15_000,
  capMs = 120_000,
): Promise<string> {
  return (await page.eval(`new Promise(res => {
    const started = Date.now();
    let lastSig = null, lastChange = Date.now();
    const thumb = document.createElement('canvas');
    thumb.width = 32; thumb.height = 18;
    const tg = thumb.getContext('2d', { willReadFrequently: true });
    const sig = () => {
      const c = document.querySelector('canvas.table');
      if (!c) return 'no-canvas';
      tg.drawImage(c, 0, 0, 32, 18);
      const d = tg.getImageData(0, 0, 32, 18).data;
      let h = 0;
      for (let i = 0; i < d.length; i += 4) h = (h * 31 + d[i] + d[i+1] * 3 + d[i+2] * 7) | 0;
      const btn = [...document.querySelectorAll('.buttons button')][0];
      const st = document.querySelector('.overlay-state');
      return h + '|' + (btn ? btn.textContent : '') + '|' + (st ? st.textContent : '');
    };
    const tick = () => {
      const btn = [...document.querySelectorAll('.buttons button')][0];
      const busy = btn && /Rolling|Opponent/.test(btn.textContent);
      if (!busy) return res('idle:' + (Date.now() - started));
      const s = sig();
      if (s !== lastSig) { lastSig = s; lastChange = Date.now(); }
      if (Date.now() - lastChange > ${stallMs}) return res('STALL:' + (Date.now() - started));
      if (Date.now() - started > ${capMs}) return res('LONG:' + (Date.now() - started));
      setTimeout(tick, 100);
    };
    tick();
  })`)) as string;
}

/** One human shot at a varying angle, then settle. */
export async function humanShot(page: DrivablePage, seed: number): Promise<void> {
  const angle = ((seed * 47) % 360) * (Math.PI / 180);
  await page.eval(`(() => {
    const c = document.querySelector('canvas.table');
    const r = c.getBoundingClientRect();
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const x = cx + Math.cos(${angle}) * 180, y = cy - Math.sin(${angle}) * 90;
    const opts = () => ({ bubbles: true, clientX: x, clientY: y, pointerId: 1, isPrimary: true });
    c.dispatchEvent(new PointerEvent('pointerdown', opts()));
    c.dispatchEvent(new PointerEvent('pointermove', opts()));
    c.dispatchEvent(new PointerEvent('pointerup', opts()));
  })()`);
  await sleep(60);
  const clicked = await page.eval(`(() => {
    const b = [...document.querySelectorAll('.buttons button')][0];
    if (!b || b.disabled) return false;
    b.click();
    return true;
  })()`);
  if (!clicked) {
    // Ball in hand, or the opponent is up. Place the cue and let the caller
    // come round again.
    await page.eval(`(() => {
      const c = document.querySelector('canvas.table');
      const r = c.getBoundingClientRect();
      const o = { bubbles: true, clientX: r.left + r.width * 0.3, clientY: r.top + r.height / 2, pointerId: 1, isPrimary: true };
      c.dispatchEvent(new PointerEvent('pointerdown', o));
      c.dispatchEvent(new PointerEvent('pointerup', o));
    })()`);
    await sleep(150);
    return;
  }
  await waitIdle(page);
}

/** One distinct state of the panel's claim lines, with the span it held for. */
export interface PanelSample {
  at: number;
  until: number;
  title: string | null;
  chosen: string | null;
  reconcile: string | null;
  reason: string | null;
  outcome: string | null;
  warn: string | null;
}

export interface QaCapture {
  requests: { at: number; msg: Record<string, unknown> }[];
  responses: { at: number; msg: Record<string, unknown> }[];
  titles: { at: number; text: string }[];
  outcomes: { at: number; text: string }[];
  panels: PanelSample[];
  frames: number[];
}

export const capture = (page: DrivablePage): Promise<QaCapture> =>
  page.eval(`JSON.parse(JSON.stringify(window.__showboatQA))`) as Promise<QaCapture>;

export const newRack = async (page: DrivablePage): Promise<void> => {
  await page.eval(
    `(() => { const b = [...document.querySelectorAll('.buttons button')].find(x => /New rack/.test(x.textContent)); if (b) b.click(); })()`,
  );
  await sleep(400);
};

/** Play until at least `want` opponent turns have completed, or rounds run out. */
export async function playForOpponentTurns(
  page: DrivablePage,
  want: number,
  maxRounds = 40,
): Promise<QaCapture> {
  for (let round = 1; round <= maxRounds; round++) {
    await humanShot(page, round);
    await waitIdle(page);
    const cap = await capture(page);
    if (cap.responses.filter((r) => r.msg.type === "done").length >= want) return cap;
    if (await page.eval(`!!document.querySelector('.win-screen')`)) await newRack(page);
  }
  return capture(page);
}

/** One `done` message, reduced to the fields a gate reads. */
export interface TurnEvidence {
  mode: string;
  modelArtifact: string | null;
  hashVerified: boolean | null;
  neuralRunMs: number | null;
  fallback: string | null;
  logitsPresent: number;
  kind: string | null;
  plannedKind: string | null;
  measured: {
    classification: string;
    rails: number;
    trickVerified: boolean;
    contactChain: number[];
    pottedBall: number | null;
  } | null;
  rung: string | null;
  executedPublished: boolean;
  at: number;
}

export function turnsFrom(cap: QaCapture): TurnEvidence[] {
  const out: TurnEvidence[] = [];
  for (const r of cap.responses) {
    if (r.msg.type !== "done") continue;
    const trace = (r.msg as { planned: { trace: Record<string, unknown> } }).planned.trace as {
      mode: string;
      model: { artifact: string; hashVerified: boolean } | null;
      timing: { neuralRunMs: number | null };
      fallback: { detail: string } | null;
      candidates: { neural: unknown | null }[];
      selected: {
        kind: string;
        plannedKind: string;
        rung: string;
        executed: unknown | null;
        measured: TurnEvidence["measured"];
      } | null;
    };
    out.push({
      mode: trace.mode,
      modelArtifact: trace.model?.artifact ?? null,
      hashVerified: trace.model?.hashVerified ?? null,
      neuralRunMs: trace.timing.neuralRunMs,
      fallback: trace.fallback?.detail ?? null,
      logitsPresent: trace.candidates.filter((c) => c.neural !== null).length,
      kind: trace.selected?.kind ?? null,
      plannedKind: trace.selected?.plannedKind ?? null,
      measured: trace.selected?.measured ?? null,
      rung: trace.selected?.rung ?? null,
      executedPublished: trace.selected?.executed != null,
      at: r.at,
    });
  }
  return out;
}
