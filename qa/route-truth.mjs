// Browser verification for the route-truth and pacing sprint.
//
// Drives the app in a real Chrome through the repo's own CDP client (there is
// no Playwright here, by design — see apps/portfolio/qa/cdp.mjs). It never
// adopts a server it found running: the caller starts one, and this script
// records the commit and the served bundle so the report can say WHICH build
// the numbers came from.
//
//   node qa/route-truth.mjs <baseUrl> <outDir>

import { execSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { launchChrome, attachToPage } from "../../portfolio/qa/cdp.mjs";

const BASE = process.argv[2] ?? "http://localhost:5186/";
const OUT = process.argv[3] ?? "/tmp/showboat-qa";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = [];
const say = (...a) => {
  const line = a.join(" ");
  log.push(line);
  console.log(line);
};

async function shot(page, name) {
  const { data } = await page.send("Page.captureScreenshot", { format: "png" });
  await writeFile(`${OUT}/${name}.png`, Buffer.from(data, "base64"));
}

/** Fraction of the table canvas that is not the empty-cloth colour. */
const INK = `(() => {
  const c = document.querySelector('canvas.table');
  if (!c) return null;
  const g = c.getContext('2d');
  const d = g.getImageData(0, 0, c.width, c.height).data;
  let ink = 0, n = 0;
  for (let i = 0; i < d.length; i += 4 * 37) {
    n++;
    const r = d[i], gg = d[i+1], b = d[i+2];
    if (Math.abs(r - 17) > 12 || Math.abs(gg - 96) > 12 || Math.abs(b - 58) > 12) ink++;
  }
  return n ? ink / n : null;
})()`;

async function main() {
  await mkdir(OUT, { recursive: true });

  const sha = execSync("git rev-parse HEAD", { encoding: "utf8" }).trim();
  const dirty = execSync("git status --porcelain -- .", { encoding: "utf8" }).trim();
  say(`commit           ${sha}${dirty ? "  (WORKING TREE DIRTY)" : ""}`);
  say(`base url         ${BASE}`);

  const chrome = await launchChrome({ headless: true });
  const page = await attachToPage(chrome.port);
  say(`chrome           ${chrome.version.Browser}`);

  const errors = [];
  await page.send("Runtime.enable");
  page.on("Runtime.consoleAPICalled", (p) => {
    if (p.type === "error") {
      errors.push(p.args.map((a) => a.value ?? a.description ?? "").join(" "));
    }
  });
  page.on("Runtime.exceptionThrown", (p) =>
    errors.push(`EXCEPTION ${p.exceptionDetails?.exception?.description ?? "?"}`),
  );

  await page.viewport(1400, 950, { dpr: 2 });
  await page.goto(BASE);

  // What was actually served. In a preview build these are hashed asset names;
  // in dev they are module URLs, and the report says which run is which.
  const served = await page.eval(`(() => {
    const s = [...document.querySelectorAll('script[src],link[href]')]
      .map(e => e.src || e.href).filter(u => u.includes('/assets/') || u.includes('/src/'));
    return s;
  })()`);
  say(`served entries   ${JSON.stringify(served)}`);

  // Wait for the engine.
  const ready = await page.eval(`new Promise(res => {
    const t0 = Date.now();
    const tick = () => {
      const m = document.querySelector('.msg');
      if (m && /break to start|couldn't load/.test(m.textContent)) return res(m.textContent);
      if (Date.now() - t0 > 30000) return res('TIMEOUT: ' + (m ? m.textContent : 'no .msg'));
      setTimeout(tick, 100);
    };
    tick();
  })`);
  say(`engine           ${ready}`);
  if (!/break to start/.test(ready)) throw new Error(`engine never became ready: ${ready}`);

  await shot(page, "01-idle");
  say(`ink at rest      ${((await page.eval(INK)) * 100).toFixed(1)}%`);

  // ---- the presentation speed control -----------------------------------
  const speeds = await page.eval(`(() => {
    const g = document.querySelector('.speed');
    if (!g) return null;
    return {
      label: g.getAttribute('aria-label'),
      text: g.querySelector('.speed-label').textContent,
      options: [...g.querySelectorAll('.speed-btn')].map(b => ({
        t: b.textContent, on: b.getAttribute('aria-pressed') === 'true'
      })),
    };
  })()`);
  say(`speed control    ${JSON.stringify(speeds)}`);

  // It persists for the session.
  await page.eval(
    `[...document.querySelectorAll('.speed-btn')].find(b => b.textContent.startsWith('0.35')).click()`,
  );
  await sleep(120);
  const stored = await page.eval(`sessionStorage.getItem('showboat.playbackSpeed')`);
  say(`speed persisted  ${stored}`);
  await page.eval(
    `[...document.querySelectorAll('.speed-btn')].find(b => b.textContent.startsWith('0.6')).click()`,
  );
  await sleep(120);

  // ---- five racks --------------------------------------------------------
  //
  // A rack is played until someone wins or the shot cap is reached, then reset.
  // What is being watched for is degradation and hangs, not who wins: every
  // turn must complete, the canvas must stay inked, and no turn may sit in a
  // busy state past the deadline the app gives itself.
  // Five racks is the floor, not the target. Random aim loses a lot of racks
  // in two shots (the 8 goes down on the break, which ends the game), and five
  // two-shot racks is not sustained play. The loop keeps racking until it has
  // seen enough opponent turns to mean something.
  const MIN_RACKS = 5;
  const MAX_RACKS = 5;
  const MIN_OPPONENT_TURNS = 0;
  const SHOT_CAP = 14;
  const TURN_TIMEOUT_MS = 90_000;
  let totalShots = 0;
  let opponentTurns = 0;
  let hangs = 0;
  let longTurns = 0;
  let midShotTaken = false;
  const inks = [];
  const turnMs = [];

  // A hang is "busy and NOT CHANGING", not "busy for a while".
  //
  // The first version of this counted any wait past a fixed cap as a hang and
  // reported 22 of them, which was a harness artifact: an opponent that keeps
  // potting stays at the table legitimately, and at 0.6x a turn is genuinely
  // several seconds. What distinguishes a wedged turn is that nothing moves —
  // so the probe downsamples the table into a 32x18 thumbnail every 100 ms and
  // watches the checksum. A running animation changes it every frame; a turn
  // stuck on "searching…" does not change it at all.
  const STALL_MS = 12_000;
  const waitIdle = async () => {
    const t0 = Date.now();
    const res = await page.eval(`new Promise(res => {
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
        return h + '|' + (btn ? btn.textContent : '') + '|' + (st ? st.textContent : '') +
               '|' + (document.querySelector('.msg') || {}).textContent;
      };
      const tick = () => {
        const btn = [...document.querySelectorAll('.buttons button')][0];
        const busy = btn && /Rolling|Opponent/.test(btn.textContent);
        if (!busy) return res('idle:' + (Date.now() - started));
        const s = sig();
        if (s !== lastSig) { lastSig = s; lastChange = Date.now(); }
        if (Date.now() - lastChange > ${STALL_MS}) {
          return res('HANG:' + (Date.now() - started) + ':' + (btn ? btn.textContent : '') +
                     ':' + ((document.querySelector('.overlay-state')||{}).textContent || 'no-state'));
        }
        if (Date.now() - started > ${TURN_TIMEOUT_MS}) {
          return res('LONG:' + (Date.now() - started));
        }
        setTimeout(tick, 100);
      };
      tick();
    })`);
    turnMs.push(Date.now() - t0);
    if (res.startsWith("HANG")) {
      hangs++;
      say(`  HANG  ${res}`);
    } else if (res.startsWith("LONG")) {
      longTurns++;
    }
    return res;
  };

  let rack = 0;
  while (rack < MAX_RACKS && (rack < MIN_RACKS || opponentTurns < MIN_OPPONENT_TURNS)) {
    rack++;
    await page.eval(
      `[...document.querySelectorAll('.buttons button')].find(b => /New rack/.test(b.textContent)).click()`,
    );
    await sleep(300);
    let shots = 0;
    let winner = false;
    for (; shots < SHOT_CAP; shots++) {
      // Aim: a drag across the felt at a rack-dependent angle, so the five
      // racks are not the same game five times.
      const angle = ((rack * 37 + shots * 61) % 360) * (Math.PI / 180);
      await page.eval(`(() => {
        const c = document.querySelector('canvas.table');
        const r = c.getBoundingClientRect();
        const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
        const x = cx + Math.cos(${angle}) * 180, y = cy - Math.sin(${angle}) * 90;
        const opts = b => ({ bubbles: true, clientX: x, clientY: y, pointerId: 1, isPrimary: true, button: b ?? 0 });
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
        // Ball in hand, or the opponent is up. Place and let it settle.
        await waitIdle();
        await page.eval(`(() => {
          const c = document.querySelector('canvas.table');
          const r = c.getBoundingClientRect();
          const opts = { bubbles: true, clientX: r.left + r.width * 0.3, clientY: r.top + r.height / 2, pointerId: 1, isPrimary: true };
          c.dispatchEvent(new PointerEvent('pointerdown', opts));
          c.dispatchEvent(new PointerEvent('pointerup', opts));
        })()`);
        await sleep(120);
        continue;
      }
      totalShots++;
      await waitIdle();

      // One frame from the middle of an opponent turn, so the report can show
      // the measured route on the felt rather than only end-of-rack boards.
      if (!midShotTaken) {
        const caught = await page.eval(`new Promise(res => {
          const t0 = Date.now();
          const tick = () => {
            const st = document.querySelector('.overlay-state');
            if (st && /selected|ready to shoot/.test(st.textContent)) return res(st.textContent);
            if (Date.now() - t0 > 12000) return res(null);
            setTimeout(tick, 40);
          };
          tick();
        })`);
        if (caught) {
          await shot(page, "05-selected-route");
          say(`mid-turn frame   captured at "${caught}"`);
          midShotTaken = true;
        }
      }

      // The opponent's turn, if it is now up.
      const aiUp = await page.eval(`/opponent/.test(document.querySelector('.turn').textContent)`);
      if (aiUp) {
        opponentTurns++;
        await waitIdle();
      }
      inks.push(await page.eval(INK));
      winner = await page.eval(`!!document.querySelector('.win-screen')`);
      if (winner) break;
    }
    say(
      `rack ${rack}           shots=${shots + 1} winner=${winner} ink=${((inks[inks.length - 1] ?? 0) * 100).toFixed(1)}%`,
    );
    await shot(page, `02-rack-${rack}`);
  }

  const finite = inks.filter((v) => typeof v === "number");
  say(`racks            ${rack}`);
  say(`player shots     ${totalShots}`);
  say(`opponent turns   ${opponentTurns}`);
  say(`turn hangs       ${hangs}   (busy AND unchanging for ${12}s+)`);
  say(`turns over 90s   ${longTurns}`);
  say(
    `ink min/max      ${(Math.min(...finite) * 100).toFixed(1)}% / ${(Math.max(...finite) * 100).toFixed(1)}%`,
  );
  say(
    `turn wall-clock  median ${Math.round(turnMs.sort((a, b) => a - b)[Math.floor(turnMs.length / 2)])} ms, max ${Math.max(...turnMs)} ms`,
  );

  // ---- the panel's own claims -------------------------------------------
  const panel = await page.eval(`(() => {
    const a = document.querySelector('.overlay');
    if (!a) return null;
    return {
      title: a.querySelector('.overlay-title')?.textContent ?? null,
      warn: a.querySelector('.overlay-warn')?.textContent ?? null,
      lines: [...a.querySelectorAll('.overlay-line')].map(e => e.textContent),
      notes: [...a.querySelectorAll('.overlay-note')].map(e => e.textContent),
      neuralChecked: a.querySelector('input[type=checkbox]')?.checked ?? null,
    };
  })()`);
  say(`panel            ${JSON.stringify(panel)}`);

  // Classical mode, on the toggle. The label must follow the decision.
  await page.eval(`document.querySelector('.overlay input[type=checkbox]').click()`);
  await sleep(200);
  await page.eval(
    `[...document.querySelectorAll('.buttons button')].find(b => /New rack/.test(b.textContent)).click()`,
  );
  await sleep(300);
  for (let i = 0; i < 3; i++) {
    await page.eval(`(() => { const b=[...document.querySelectorAll('.buttons button')][0]; if(b && !b.disabled) b.click(); })()`);
    await waitIdle();
    await waitIdle();
  }
  const classical = await page.eval(`(() => {
    const a = document.querySelector('.overlay');
    return a ? {
      title: a.querySelector('.overlay-title')?.textContent ?? null,
      warn: a.querySelector('.overlay-warn')?.textContent ?? null,
    } : null;
  })()`);
  say(`classical panel  ${JSON.stringify(classical)}`);
  await shot(page, "03-classical");

  // ---- reduced motion ----------------------------------------------------
  await page.emulateReducedMotion(true);
  await page.goto(BASE);
  await sleep(2500);
  await page.eval(`(() => { const b=[...document.querySelectorAll('.buttons button')][0]; if(b && !b.disabled) b.click(); })()`);
  await waitIdle();
  await waitIdle();
  say(`reduced motion   ink=${((await page.eval(INK)) * 100).toFixed(1)}%  errors so far=${errors.length}`);
  await shot(page, "04-reduced-motion");
  await page.emulateReducedMotion(false);

  say(`console errors   ${errors.length}`);
  for (const e of errors.slice(0, 12)) say(`  ! ${e.slice(0, 200)}`);

  await writeFile(`${OUT}/route-truth.log`, log.join("\n") + "\n");
  await chrome.close();
}

main().catch(async (e) => {
  console.error(e);
  process.exit(1);
});
