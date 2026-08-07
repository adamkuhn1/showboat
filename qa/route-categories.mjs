// Every trick-shot category, drawn on the real canvas by the real renderer.
//
// A previous sprint found that `combo` and `rail-combo` were generated and
// eligible in every trace but never naturally SELECTED across 76 sampled
// decisions. So this does not wait for the policy to choose one: it builds a
// deterministic fixture per category — a real board, a real WASM simulation —
// and pushes it through `extractExecutedMotion` -> `measuredRoute` ->
// `drawPresentation` onto the page's own canvas, then measures what landed.
//
// It runs against the DEV server, because it imports the app's own modules by
// URL to do it. The five-rack / fallback / pacing run is the production
// preview; the report says which is which.
//
//   node qa/route-categories.mjs <devUrl> <outDir>

import { execSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { launchChrome, attachToPage } from "../../portfolio/qa/cdp.mjs";

const BASE = process.argv[2] ?? "http://localhost:5186/";
const OUT = process.argv[3] ?? "/tmp/showboat-qa-cat";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = [];
const say = (...a) => {
  const l = a.join(" ");
  log.push(l);
  console.log(l);
};

// Everything below runs in the page. It is one expression so the CDP client
// can await it as a promise.
const FIXTURES = `(async () => {
  const [{ extractExecutedMotion }, { measuredRoute }, { drawPresentation },
         { render, computeView }, { makeTable }, { makeBall },
         { initPhysics, simulateShotWasm }, { generateCandidates },
         { contactMarksFromExecuted }] = await Promise.all([
    import('/src/ai/trace/executed.ts'),
    import('/src/render/presentation.ts'),
    import('/src/render/overlay.ts'),
    import('/src/render/renderer.ts'),
    import('/src/physics/table.ts'),
    import('/src/physics/ball.ts'),
    import('/src/physics/wasm-bridge.ts'),
    import('/src/ai/candidates.ts'),
    import('/src/render/annotate.ts'),
  ]);
  await initPhysics();

  const table = makeTable();
  const canvas = document.querySelector('canvas.table');
  const ctx = canvas.getContext('2d');
  const view = computeView(900, 500, table);

  const CUE = 0;
  const clone = b => ({ ...b, pos: { ...b.pos }, vel: { ...b.vel }, roll: { ...b.roll } });
  const MARK = new Set(['ball-ball','ball-cushion','pocket']);

  const shapeOf = sim => {
    const evs = sim.events.filter(e => MARK.has(e.kind));
    const pot = evs.find(e => e.kind === 'pocket' && e.balls[0] !== CUE);
    const potted = pot ? pot.balls[0] : null;
    const fb = evs.findIndex(e => e.kind === 'ball-ball' && e.balls.includes(CUE));
    const fc = evs.findIndex(e => e.kind === 'ball-cushion' && e.balls[0] === CUE);
    const kickFirst = fc >= 0 && (fb < 0 || fc < fb);
    let cushions = 0, combo = false;
    if (potted !== null) {
      const pi = evs.indexOf(pot);
      for (let i = 0; i < pi; i++) {
        if (evs[i].kind === 'ball-cushion' && evs[i].balls[0] === potted) cushions++;
      }
      const sm = evs.find(e => e.kind === 'ball-ball' && e.balls.includes(potted));
      combo = !!sm && !sm.balls.includes(CUE);
    }
    return { cushions, kickFirst, combo, potted };
  };
  const want = {
    'single-bank':   s => s.potted !== null && !s.combo && s.cushions === 1,
    'double-bank':   s => s.potted !== null && !s.combo && s.cushions === 2,
    'multi-cushion': s => s.potted !== null && s.cushions >= 3,
    // Pure combination vs one with a cushion in the potted ball's route.
    // Split so the two cannot resolve to the same shot and count twice.
    'combo':         s => s.potted !== null && s.combo && s.cushions === 0,
    'rail-combo':    s => s.potted !== null && s.combo && s.cushions >= 1,
    'safety-kick':   s => s.kickFirst,
  };

  const BOARDS = [
    { n:'openSpread', b:[makeBall(CUE,-0.35,0.05),makeBall(1,0.25,0.15),makeBall(3,-0.05,-0.28),makeBall(9,0.5,-0.2)], t:[1,3,9] },
    { n:'railHeavy',  b:[makeBall(CUE,0,0),makeBall(1,-0.7,0.28),makeBall(3,0.72,-0.29),makeBall(6,-0.4,-0.3)], t:[1,3,6] },
    { n:'clustered',  b:[makeBall(CUE,-0.75,-0.05),makeBall(2,0.1,0.02),makeBall(4,0.22,0.06),makeBall(5,0.45,-0.24),makeBall(7,-0.2,0.3)], t:[2,4,5,7] },
    { n:'longRail',   b:[makeBall(CUE,-0.85,0.34),makeBall(1,0.62,0.3),makeBall(8,0,-0.36),makeBall(11,0.35,0.36)], t:[1,11] },
  ];

  const found = {};
  for (const board of BOARDS) {
    const cands = generateCandidates(board.b, table, board.t);
    for (const c of cands) {
      for (const power of [c.action.power, 0.7, 0.9]) {
        const action = { ...c.action, power };
        const working = board.b.map(clone);
        const sim = simulateShotWasm(working, action);
        const s = shapeOf(sim);
        for (const k of Object.keys(want)) {
          if (found[k] || !want[k](s)) continue;
          const motion = extractExecutedMotion(sim);
          if (!motion) continue;
          found[k] = { board: board.n, pre: board.b.map(clone), sim, motion };
        }
      }
    }
  }

  // A deliberate kick, if no generated candidate produced one. The candidate
  // generator aims at balls; the safety rung is what aims at a rail, and it is
  // not what this fixture pass walks.
  if (!found['safety-kick']) {
    const b = [makeBall(CUE,0,0), makeBall(1,-0.6,0.2), makeBall(3,0.5,0.3)];
    const action = { phi: -Math.PI/2.2, power: 0.75, sideSpin: 0, topSpin: 0 };
    const sim = simulateShotWasm(b.map(clone), action);
    const motion = extractExecutedMotion(sim);
    if (motion && shapeOf(sim).kickFirst) {
      found['safety-kick'] = { board: 'kick', pre: b.map(clone), sim, motion };
    }
  }

  window.__qa = { found, table, view, ctx, canvas, measuredRoute, render,
                  drawPresentation, contactMarksFromExecuted };
  return Object.keys(found);
})()`;

/** Count pixels of the selected-route stroke colour, 126/233/174. */
const ROUTE_PIXELS = `(() => {
  const c = document.querySelector('canvas.table');
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let n = 0;
  for (let i = 0; i < d.length; i += 4) {
    if (Math.abs(d[i]-126) < 55 && Math.abs(d[i+1]-233) < 55 && Math.abs(d[i+2]-174) < 55) n++;
  }
  return n;
})()`;

async function main() {
  await mkdir(OUT, { recursive: true });
  const sha = execSync("git rev-parse HEAD", { encoding: "utf8" }).trim();
  say(`commit           ${sha}`);
  say(`base url         ${BASE}`);

  const chrome = await launchChrome({ headless: true });
  const page = await attachToPage(chrome.port);
  say(`chrome           ${chrome.version.Browser}`);
  const errors = [];
  await page.send("Runtime.enable");
  page.on("Runtime.consoleAPICalled", (p) => {
    if (p.type === "error") errors.push(p.args.map((a) => a.value ?? a.description ?? "").join(" "));
  });

  await page.viewport(1400, 950, { dpr: 2 });
  await page.goto(BASE);
  await sleep(2500);

  const categories = await page.eval(FIXTURES);
  say(`categories found ${JSON.stringify(categories)}`);

  const CATS = ["single-bank", "double-bank", "multi-cushion", "combo", "rail-combo", "safety-kick"];
  const missing = CATS.filter((c) => !categories.includes(c));
  if (missing.length) throw new Error(`no fixture for: ${missing.join(", ")}`);

  const rows = [];
  for (const cat of CATS) {
    // Draw the verified route as the app draws it before the stroke, then part
    // way through playback so the time-based reveal is exercised too.
    const stats = await page.eval(`(() => {
      const q = window.__qa;
      const f = q.found[${JSON.stringify(cat)}];
      const m = q.measuredRoute(f.motion);
      const frame = {
        state: 'READY', label: null, progress: 1, showContacts: true, strokeProgress: null,
        routes: [{
          index: 0, kind: 'bank',
          cueLeg: m.cue ? m.cue.points : null,
          objectLeg: m.object ? m.object.points : [],
          source: 'simulated', measured: m,
          reveal: 1, weight: 1, alpha: 1, role: 'selected',
          reason: null, resolving: false, justResolved: false,
        }],
      };
      const state = { balls: f.pre, turn: 1, groups: {0:null,1:null}, ballInHand: false,
                      winner: null, broken: true, shotCount: 3 };
      q.render(q.ctx, state, q.table, q.view);
      const marks = q.contactMarksFromExecuted(f.motion, 8);
      q.drawPresentation(q.ctx, frame, q.view, marks, null);
      return {
        board: f.board,
        durationSec: +f.motion.durationSec.toFixed(3),
        balls: f.motion.trajectories.length,
        points: f.motion.trajectories.reduce((n,t)=>n+t.points.length,0),
        rawPoints: f.motion.trajectories.reduce((n,t)=>n + f.sim.waypoints.filter(w=>{
          const b = w.balls.find(x=>x.id===t.ballId); return !!b && !b.pocketed;
        }).length, 0),
        contacts: f.motion.contactSequence.length,
        marks: marks.length,
        maxDeviationMm: +(f.motion.maxDeviationM * 1000).toFixed(4),
        toleranceMm: +(f.motion.simplifyToleranceM * 1000).toFixed(4),
        roles: f.motion.trajectories.map(t => t.ballId + ':' + t.roles.join('+')),
      };
    })()`);
    const routePx = await page.eval(ROUTE_PIXELS);
    const { data } = await page.send("Page.captureScreenshot", { format: "png" });
    await writeFile(`${OUT}/cat-${cat}.png`, Buffer.from(data, "base64"));

    // Mid-playback: the trail must be SHORTER than the full route.
    const midPx = await page.eval(`(() => {
      const q = window.__qa;
      const f = q.found[${JSON.stringify(cat)}];
      const m = q.measuredRoute(f.motion);
      const half = f.motion.durationSec * 0.45;
      const frame = {
        state: 'SHOOTING', label: null, progress: 1, showContacts: true, strokeProgress: null,
        routes: [{ index: 0, kind: 'bank',
          cueLeg: m.cue ? m.cue.points : null, objectLeg: m.object ? m.object.points : [],
          source: 'simulated', measured: m, reveal: 1, weight: 1, alpha: 1,
          role: 'selected', reason: null, resolving: false, justResolved: false }],
      };
      const state = { balls: f.pre, turn: 1, groups: {0:null,1:null}, ballInHand: false,
                      winner: null, broken: true, shotCount: 3 };
      q.render(q.ctx, state, q.table, q.view);
      q.drawPresentation(q.ctx, frame, q.view, q.contactMarksFromExecuted(f.motion, 8), half);
      return null;
    })()` ).then(() => page.eval(ROUTE_PIXELS));
    const { data: d2 } = await page.send("Page.captureScreenshot", { format: "png" });
    await writeFile(`${OUT}/cat-${cat}-mid.png`, Buffer.from(d2, "base64"));

    rows.push({ cat, routePx, midPx, ...stats });
    say(
      `${cat.padEnd(14)} board=${stats.board.padEnd(11)} balls=${stats.balls} ` +
        `pts=${String(stats.points).padStart(3)}/${String(stats.rawPoints).padStart(3)} ` +
        `contacts=${stats.contacts} marks=${stats.marks} ` +
        `dev=${stats.maxDeviationMm}mm/${stats.toleranceMm}mm ` +
        `routePx=${routePx} midPx=${midPx} ${stats.roles.join(" ")}`,
    );
  }

  const bad = rows.filter((r) => r.routePx < 200);
  say(`categories drawn ${rows.length}, none blank: ${bad.length === 0}`);
  const shrink = rows.filter((r) => r.midPx >= r.routePx);
  say(`trail reveals over time on all: ${shrink.length === 0}`);
  say(`console errors   ${errors.length}`);
  for (const e of errors.slice(0, 8)) say(`  ! ${e.slice(0, 200)}`);

  await writeFile(`${OUT}/categories.log`, log.join("\n") + "\n");
  await writeFile(`${OUT}/categories.json`, JSON.stringify(rows, null, 2));
  await chrome.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
