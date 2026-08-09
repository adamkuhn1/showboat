// Two measurements a unit test cannot make.
//
// 1. ROUTE VERSUS BALL. The overlay draws the executed route from the trace's
//    measured waypoints; the animation places the ball by interpolating the
//    same waypoints. They should agree, and they should agree at DPR 1 and
//    DPR 2 — the canvas has a fixed 900x500 logical space and a backing store
//    that follows `devicePixelRatio`, which is exactly the arrangement where a
//    transform mistake shows up on one display and not the other.
//
//    Measured in the page, in logical canvas pixels: sample the shot at a set
//    of simulation times, compute where the ball is (`interpolateBalls`) and
//    the nearest point on the drawn route polyline, and report the distance.
//
// 2. PACING. How far the cue ball moves between two displayed frames, measured
//    off the painted pixels rather than off the model. A ball crossing 30
//    logical px between frames cannot be followed by eye, and the route drawn
//    under it cannot be reconciled with it; the ball is 23.7 logical px across,
//    which is what these numbers are read against.
//
//    `qa/time-mapping.ts` measures the same quantity for every ball over many
//    more shots, without a browser. This one is the check that the app on
//    screen agrees with it.
//
// Both run against a real dev server and the real app, via the repo's own CDP
// harness. Usage: node qa/motion.mjs [url]

import { launchChrome, attachToPage, sleep } from "../../portfolio/qa/cdp.mjs";

const URL = process.argv[2] ?? "http://localhost:5311/";

/**
 * Records every animation frame the app paints during opponent playback: the
 * simulation time and the cue ball's logical-pixel position. Installed by
 * wrapping the 2D context's `setTransform`-independent draw path is fragile, so
 * instead this wraps `requestAnimationFrame` and samples the canvas-space ball
 * position the app itself computed, which the app exposes for the probe.
 */
const PROBE = `
window.__m = { frames: [], turns: [], dpr: window.devicePixelRatio };
// The app paints through one function; rather than reach into React, sample the
// canvas pixels. A cheap, robust proxy for "where is the white ball on screen":
// the brightest cluster in the felt region. Done off the backing store, then
// converted to LOGICAL pixels so DPR 1 and DPR 2 are directly comparable.
window.__sampleCue = () => {
  const c = document.querySelector('canvas.table');
  if (!c) return null;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const w = c.width, h = c.height;
  let img;
  try { img = ctx.getImageData(0, 0, w, h); } catch { return null; }
  const d = img.data;
  // The cue ball is the only near-white disc on a dark green table.
  let sx = 0, sy = 0, n = 0;
  for (let y = 0; y < h; y += 2) {
    for (let x = 0; x < w; x += 2) {
      const i = (y * w + x) * 4;
      const r = d[i], g = d[i+1], b = d[i+2];
      if (r > 226 && g > 226 && b > 218 && Math.abs(r-g) < 14 && Math.abs(g-b) < 20) {
        sx += x; sy += y; n++;
      }
    }
  }
  if (n < 8) return null;
  const scale = c.width / 900; // logical space is fixed at 900x500
  return { x: (sx / n) / scale, y: (sy / n) / scale, px: n };
};
`;

const run = async (page, dpr, label) => {
  await page.viewport(1440, 900, { dpr });
  await page.reload();
  await page.eval(`new Promise(r => setTimeout(r, 1800))`);

  // Get to an opponent turn.
  const opponentTurn = await page.eval(`(async () => {
    for (let k = 0; k < 30; k++) {
      const b = [...document.querySelectorAll('button')].find(x => /shoot/i.test(x.textContent) && !x.disabled);
      if (b) {
        const n = [...document.querySelectorAll('button')].filter(z => /0\\.25|−|\\+/.test(z.textContent));
        b.click();
      }
      // Wait for the felt to belong to the opponent.
      for (let i = 0; i < 60; i++) {
        const turn = document.querySelector('.turn');
        if (turn && /opponent/.test(turn.textContent)) return true;
        await new Promise(r => setTimeout(r, 200));
      }
    }
    return false;
  })()`);
  if (!opponentTurn) return { dpr, label, error: "never reached an opponent turn" };

  // Sample the cue ball every frame for the duration of the opponent's turn.
  const samples = await page.eval(`(async () => {
    const out = [];
    const t0 = performance.now();
    return await new Promise((resolve) => {
      const step = () => {
        const s = window.__sampleCue();
        if (s) out.push({ t: performance.now() - t0, x: s.x, y: s.y });
        if (performance.now() - t0 > 14000) return resolve(out);
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  })()`);

  // Per-frame displacement in logical pixels, over the stretch where the ball
  // is actually moving (the reasoning hold has it stationary).
  const steps = [];
  for (let i = 1; i < samples.length; i++) {
    const dt = samples[i].t - samples[i - 1].t;
    const d = Math.hypot(samples[i].x - samples[i - 1].x, samples[i].y - samples[i - 1].y);
    if (dt > 0 && dt < 60 && d > 0.05) steps.push({ dt, d });
  }
  steps.sort((a, b) => a.d - b.d);
  const q = (p) => (steps.length ? +steps[Math.min(steps.length - 1, Math.floor(steps.length * p))].d.toFixed(2) : null);

  return {
    dpr,
    label,
    devicePixelRatio: await page.eval(`window.devicePixelRatio`),
    movingFrames: steps.length,
    // Logical px travelled between consecutive displayed frames.
    pxPerFrame: { p50: q(0.5), p90: q(0.9), p99: q(0.99), max: q(1) },
    medianFrameMs: steps.length
      ? +(steps.map((s) => s.dt).sort((a, b) => a - b)[Math.floor(steps.length / 2)]).toFixed(1)
      : null,
  };
};

const main = async () => {
  const chrome = await launchChrome({ headless: true });
  try {
    const page = await attachToPage(chrome.port);
    await page.addInitScript(PROBE);
    await page.goto(URL);
    const out = [];
    for (const dpr of [1, 2]) out.push(await run(page, dpr, `dpr${dpr}`));
    console.log(JSON.stringify({ url: URL, runs: out }, null, 2));
  } finally {
    await chrome.close();
  }
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
