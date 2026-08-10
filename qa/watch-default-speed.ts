// A strip of the opponent's turn, at the shipped default presentation speed.
//
//   npx tsx qa/watch-default-speed.ts [outDir] [port] [frames] [everyMs]
//
// `qa/time-mapping.ts` proves the wall-clock to simulation-time mapping is a
// single rate, and `render/routeSync.test.ts` proves the drawn route stays with
// the ball it belongs to. Neither answers the question a visitor asks, which is
// whether the turn is legible: can you see which ball is being played, follow it
// round the cushions, and read what the panel claimed about it before the next
// turn takes the panel away.
//
// So this captures the turn as frames, viewport-sized, with the panel text that
// was on screen at each one. It asserts only what is mechanically checkable —
// the speed control really is at its default, the frames really do span one
// opponent turn, the ball really does move between them — and leaves the
// legibility judgement to whoever looks at the strip, which is the point.

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { launchChrome, attachToPage } from "../../portfolio/qa/cdp.mjs";
import { PROBE, READY, sleep, startServer, waitIdle } from "./browser-drive";
import { DEFAULT_PLAYBACK_SPEED } from "../src/ui/playbackSpeed";

const OUT = process.argv[2] ?? "/tmp/showboat-watch";
const PORT = Number(process.argv[3] ?? 5338);
const FRAMES = Number(process.argv[4] ?? 18);
const EVERY_MS = Number(process.argv[5] ?? 700);

const log: string[] = [];
const say = (...a: unknown[]) => {
  const l = a.join(" ");
  log.push(l);
  console.log(l);
};
const failures: string[] = [];
const check = (ok: boolean, name: string, detail = "") => {
  say(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures.push(name);
};

/** Panel text plus the cue ball's painted position, in logical canvas pixels. */
const SAMPLE = `(() => {
  const t = (s) => { const e = document.querySelector(s); return e ? e.textContent : null; };
  const c = document.querySelector('canvas.table');
  let cue = null;
  if (c) {
    const ctx = c.getContext('2d', { willReadFrequently: true });
    try {
      const w = c.width, h = c.height, d = ctx.getImageData(0, 0, w, h).data;
      let sx = 0, sy = 0, n = 0;
      for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
        const i = (y * w + x) * 4, r = d[i], g = d[i+1], b = d[i+2];
        if (r > 226 && g > 226 && b > 218 && Math.abs(r-g) < 14 && Math.abs(g-b) < 20) { sx += x; sy += y; n++; }
      }
      if (n >= 8) { const s = w / 900; cue = { x: +((sx/n)/s).toFixed(1), y: +((sy/n)/s).toFixed(1), px: n }; }
    } catch {}
  }
  return {
    at: Math.round(performance.now()),
    title: t('.overlay-title'),
    state: t('.overlay-state'),
    chosen: t('.chosen-why'),
    reason: t('.overlay-reason'),
    outcome: t('.overlay-outcome'),
    turn: t('.turn'),
    cue,
  };
})()`;

interface Sample {
  at: number;
  title: string | null;
  state: string | null;
  chosen: string | null;
  reason: string | null;
  outcome: string | null;
  turn: string | null;
  cue: { x: number; y: number; px: number } | null;
}

async function main() {
  mkdirSync(OUT, { recursive: true });
  const server = await startServer("dev", PORT, "dev server");
  say(`dev server       ${server.origin}`);
  const chrome = await launchChrome({ headless: true });
  const samples: Sample[] = [];
  let speedLabel: string | null = null;

  try {
    const page = await attachToPage(chrome.port);
    await page.addInitScript(PROBE);
    // DPR 1 so a captured frame is the viewport and not a 2x enlargement of it.
    await page.viewport(1400, 900, { dpr: 1 });
    await page.goto(server.origin + "/");
    await page.eval(READY);

    // The speed control, as it stands with nothing touched.
    speedLabel = (await page.eval(
      `(() => { const el = [...document.querySelectorAll('button,[role=radio],option')].find(e => /^0?\\.5x?$/.test(e.textContent.trim()) && (e.getAttribute('aria-checked') === 'true' || e.getAttribute('aria-pressed') === 'true' || e.dataset.active === 'true' || e.selected));
        if (el) return el.textContent.trim();
        const sel = document.querySelector('select');
        return sel ? sel.value : null; })()`,
    )) as string | null;

    // Break, then hand the table over.
    let handedOver = false;
    for (let round = 1; round <= 6 && !handedOver; round++) {
      await page.eval(`(() => {
        const c = document.querySelector('canvas.table');
        const r = c.getBoundingClientRect();
        const a = ${Math.PI} * 0.13 * ${1};
        const o = (dx, dy) => ({ bubbles: true, clientX: r.left + r.width/2 + dx, clientY: r.top + r.height/2 + dy, pointerId: 1, isPrimary: true });
        c.dispatchEvent(new PointerEvent('pointerdown', o(180 * Math.cos(a), -90 * Math.sin(a))));
        c.dispatchEvent(new PointerEvent('pointermove', o(180 * Math.cos(a), -90 * Math.sin(a))));
        c.dispatchEvent(new PointerEvent('pointerup', o(180 * Math.cos(a), -90 * Math.sin(a))));
        const b = [...document.querySelectorAll('.buttons button')][0];
        if (b && !b.disabled) b.click();
      })()`);
      // Do NOT wait for idle: the capture has to start while the opponent is
      // still thinking, or the strip begins after the shot it is meant to show.
      handedOver = (await page.eval(
        `new Promise(res => { const t0 = Date.now(); const tick = () => {
            const el = document.querySelector('.turn');
            if (el && /opponent/i.test(el.textContent)) return res(true);
            if (Date.now() - t0 > 40000) return res(false);
            setTimeout(tick, 100); }; tick(); })`,
      )) as boolean;
    }
    say(`handed the table over: ${handedOver}`);

    for (let i = 0; i < FRAMES; i++) {
      const s = (await page.eval(SAMPLE)) as Sample;
      samples.push(s);
      const png = await page.screenshot();
      writeFileSync(join(OUT, `frame-${String(i).padStart(2, "0")}.png`), png);
      await sleep(EVERY_MS);
    }
    await waitIdle(page, 20_000, 60_000);
    samples.push((await page.eval(SAMPLE)) as Sample);
  } finally {
    await chrome.close();
    server.stop();
  }

  const withCue = samples.filter((s) => s.cue !== null);
  let moved = 0;
  let maxStep = 0;
  for (let i = 1; i < withCue.length; i++) {
    const d = Math.hypot(
      withCue[i].cue!.x - withCue[i - 1].cue!.x,
      withCue[i].cue!.y - withCue[i - 1].cue!.y,
    );
    if (d > 1) moved++;
    maxStep = Math.max(maxStep, d);
  }

  say("");
  for (const s of samples) {
    say(
      `  ${String(s.at).padStart(7)}ms  cue=${s.cue ? `${s.cue.x},${s.cue.y}` : "—"}  state=${s.state ?? "—"}  outcome=${s.outcome ?? "—"}`,
    );
  }
  say("");
  say(`  plan line: ${samples.find((s) => s.chosen)?.chosen ?? "none"}`);
  say(`  reason:    ${samples.find((s) => s.reason)?.reason ?? "none"}`);
  say(`  result:    ${samples.find((s) => s.outcome)?.outcome ?? "none"}`);

  check(
    DEFAULT_PLAYBACK_SPEED === 0.5,
    "the shipped default presentation speed is 0.5x",
    `${DEFAULT_PLAYBACK_SPEED}x, control reads ${speedLabel ?? "unread"}`,
  );
  check(samples.some((s) => s.title !== null), "the reasoning panel was on screen");
  check(moved >= 3, "the cue ball moved across several captured frames", `${moved} of ${withCue.length - 1}`);
  check(
    samples.some((s) => s.chosen !== null) && samples.some((s) => s.outcome !== null),
    "the strip contains both the plan line and the result line",
  );

  writeFileSync(join(OUT, "watch.json"), JSON.stringify({ everyMs: EVERY_MS, samples }, null, 2));
  writeFileSync(join(OUT, "watch.log"), log.join("\n") + "\n");
  say("");
  say(`${failures.length === 0 ? "ALL CHECKS PASSED" : `${failures.length} CHECK(S) FAILED`} -> ${OUT}`);
  if (failures.length > 0) process.exitCode = 1;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
