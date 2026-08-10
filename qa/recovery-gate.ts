// Two recovery paths that need a real worker and a real clock.
//
//   npx tsx qa/recovery-gate.ts [outDir] [port]
//
// `qa/fallback.mjs` covers the third — the ranker artifact failing to load —
// by breaking the network in front of the worker. The two here cannot be
// reached that way:
//
//   * A SILENT WORKER. A module worker reclaimed under memory pressure stays a
//     live object that never posts again and never fires `error`. Simulated by
//     wrapping `Worker` before the app boots and refusing to attach the app's
//     own `message` listener to the first worker it builds, which is
//     indistinguishable from the worker going quiet. The watchdog in
//     `ui/planner/workerPlan.ts` must terminate it and replan on the main
//     thread, labelled `planner-timeout`.
//   * A STARVED SEARCH. The decision has a wall clock
//     (`DECISION_DEADLINE_MS`), and the physics search polls it between
//     synchronous WASM calls. Simulated with DevTools CPU throttling, which
//     slows the real work rather than shortening the budget, so what is
//     measured is the shipped path running out of time.
//
// Both are asked the same question: does the turn finish, honestly labelled,
// with a result on screen — or does the page sit at "searching…".

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { launchChrome, attachToPage } from "../../portfolio/qa/cdp.mjs";
import {
  PROBE,
  READY,
  capture,
  humanShot,
  newRack,
  startServer,
  turnsFrom,
  waitIdle,
  type QaCapture,
} from "./browser-drive";

const OUT = process.argv[2] ?? "/tmp/showboat-recovery";
const PORT = Number(process.argv[3] ?? 5334);

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

/**
 * Installed after `PROBE`, so it wraps the probe's wrapper: the probe still
 * sees every message the worker posts, and the app sees none of them. Only the
 * first worker is muted — the watchdog drops its reference to a dead worker and
 * the next turn builds a fresh one, which must work normally.
 */
const MUTE_FIRST_WORKER = `
(() => {
  const Wrapped = window.Worker;
  let armed = true;
  window.__sbMutedWorker = false;
  window.Worker = function (url, opts) {
    const w = new Wrapped(url, opts);
    if (armed) {
      armed = false;
      window.__sbMutedWorker = true;
      const add = w.addEventListener.bind(w);
      w.addEventListener = (type, fn, o) => (type === "message" ? undefined : add(type, fn, o));
    }
    return w;
  };
  window.Worker.prototype = Wrapped.prototype;
})();
`;

/** The panel's claim lines as they stand right now. */
const PANEL = `(() => {
  const t = (s) => { const e = document.querySelector(s); return e ? e.textContent : null; };
  return {
    title: t('.overlay-title'),
    warn: t('.overlay-warn'),
    chosen: t('.chosen-why'),
    reason: t('.overlay-reason'),
    outcome: t('.overlay-outcome'),
    state: t('.overlay-state'),
    stuck: !!document.querySelector('.overlay-state') && /searching/i.test(t('.overlay-state') || ''),
  };
})()`;

interface Panel {
  title: string | null;
  warn: string | null;
  chosen: string | null;
  reason: string | null;
  outcome: string | null;
  state: string | null;
  stuck: boolean;
}

async function main() {
  mkdirSync(OUT, { recursive: true });
  const server = await startServer("dev", PORT, "dev server");
  say(`dev server       ${server.origin}`);
  const record: Record<string, unknown> = {};

  try {
    // --- 1. the planning worker goes silent --------------------------------
    {
      const chrome = await launchChrome({ headless: true });
      const page = await attachToPage(chrome.port);
      const errors: string[] = [];
      await page.send("Runtime.enable");
      page.on("Runtime.consoleAPICalled", (p: { type: string; args: { value?: string }[] }) => {
        if (p.type === "error") errors.push(p.args.map((a) => a.value ?? "").join(" "));
      });
      await page.addInitScript(PROBE);
      await page.addInitScript(MUTE_FIRST_WORKER);
      await page.viewport(1400, 950, { dpr: 1 });
      await page.goto(server.origin + "/");
      await page.eval(READY);

      let cap: QaCapture | null = null;
      let stalls = 0;
      for (let round = 1; round <= 3; round++) {
        await humanShot(page, round);
        // The watchdog is an 8 s silence timer; the settle wait has to outlast
        // it or the harness would call the recovery a stall.
        const settle = await waitIdle(page, 30_000, 180_000);
        if (/^STALL|^LONG/.test(settle)) {
          stalls++;
          say(`  settle: ${settle}`);
        }
        cap = await capture(page);
        if (await page.eval(`!!document.querySelector('.win-screen')`)) await newRack(page);
      }
      const muted = await page.eval(`window.__sbMutedWorker === true`);
      const last = (await page.eval(PANEL)) as Panel;
      // The panel history, not a snapshot: the rescued turn has usually been
      // replaced by the next one before the table settles.
      const panels = cap!.panels;
      const withWarn = panels.filter((p) => p.warn !== null);
      const shot = await page.screenshot();
      writeFileSync(join(OUT, "muted-worker.png"), shot);

      say("");
      say("1  the planning worker goes silent");
      say(`   muted a worker: ${muted}; panel states recorded: ${panels.length}`);
      for (const p of panels) {
        if (p.warn !== null || p.title !== null) {
          say(`   panel: title=${p.title} warn=${p.warn} outcome=${p.outcome}`);
        }
      }
      check(muted === true, "the first worker really was muted");
      check(stalls === 0, "no turn wedged", `${stalls} stalls`);
      check(
        panels.some((p) => p.title === "Physics search"),
        "the rescued turn is labelled classical",
        [...new Set(panels.map((p) => p.title))].join(" | "),
      );
      check(
        withWarn.some((p) => /worker|responding/i.test(p.warn ?? "")),
        "and the panel names the worker as the reason",
        [...new Set(withWarn.map((p) => p.warn))].join(" | "),
      );
      check(
        panels.some((p) => p.outcome !== null),
        "a result line reached the screen",
        [...new Set(panels.map((p) => p.outcome))].filter(Boolean).join(" | "),
      );
      check(!last.stuck, "the panel is not left at searching", JSON.stringify(last));
      record.mutedWorker = { muted, stalls, panels, last, errors };
      await chrome.close();
    }

    // --- 2. the search is starved of CPU -----------------------------------
    {
      const chrome = await launchChrome({ headless: true });
      const page = await attachToPage(chrome.port);
      const errors: string[] = [];
      await page.send("Runtime.enable");
      page.on("Runtime.consoleAPICalled", (p: { type: string; args: { value?: string }[] }) => {
        if (p.type === "error") errors.push(p.args.map((a) => a.value ?? "").join(" "));
      });
      await page.addInitScript(PROBE);
      await page.viewport(1400, 950, { dpr: 1 });
      await page.goto(server.origin + "/");
      await page.eval(READY);
      // 20x slower than this machine. The decision budget is 5 s of wall clock
      // and the physics is unchanged, so the search runs out of time doing real
      // work rather than being told it has none.
      await page.send("Emulation.setCPUThrottlingRate", { rate: 20 });
      say("");
      say("2  the search is starved of CPU (throttle 20x)");

      let cap: QaCapture | null = null;
      let stalls = 0;
      const panels: Panel[] = [];
      for (let round = 1; round <= 12 && (cap?.responses.length ?? 0) < 4; round++) {
        await humanShot(page, round);
        const settle = await waitIdle(page, 60_000, 300_000);
        if (/^STALL|^LONG/.test(settle)) {
          stalls++;
          say(`  settle: ${settle}`);
        }
        panels.push((await page.eval(PANEL)) as Panel);
        cap = await capture(page);
        if (await page.eval(`!!document.querySelector('.win-screen')`)) await newRack(page);
      }
      const turns = turnsFrom(cap!);
      const traces = cap!.responses
        .filter((r) => r.msg.type === "done")
        .map(
          (r) =>
            (r.msg as { planned: { trace: Record<string, unknown> } }).planned.trace as {
              budget: { physicsUnitsAllowed: number; physicsUnitsSpent: number; seedTimedOut: boolean };
              timing: { totalMs: number };
              selected: { kind: string; measured: { classification: string } | null } | null;
            },
        );
      await page.send("Emulation.setCPUThrottlingRate", { rate: 1 });
      const shot = await page.screenshot();
      writeFileSync(join(OUT, "starved-search.png"), shot);

      for (const t of traces) {
        say(
          `   budget ${t.budget.physicsUnitsSpent}/${t.budget.physicsUnitsAllowed} seedTimedOut=${t.budget.seedTimedOut} total=${Math.round(t.timing.totalMs)}ms kind=${t.selected?.kind ?? "none"} measured=${t.selected?.measured?.classification ?? "none"}`,
        );
      }
      check(turns.length > 0, "the opponent still decided", `${turns.length} turns`);
      check(stalls === 0, "no turn wedged under starvation", `${stalls} stalls`);
      check(
        traces.some((t) => t.budget.physicsUnitsSpent < t.budget.physicsUnitsAllowed),
        "at least one search really did run short of its budget",
        traces.map((t) => `${t.budget.physicsUnitsSpent}/${t.budget.physicsUnitsAllowed}`).join(" "),
      );
      check(
        turns.every((t) => t.kind === "safety-kick" || (t.measured?.trickVerified ?? false)),
        "every shot played under starvation is still a measured trick or a safety",
      );
      check(
        turns.every((t) => t.measured?.classification !== "direct"),
        "and none of them is a direct pot",
      );
      check(
        panels.some((p) => p.outcome !== null),
        "a result line reached the screen",
      );
      record.starvedSearch = { stalls, panels, traces, turns: turns.length, errors };
      await chrome.close();
    }
  } finally {
    server.stop();
  }

  writeFileSync(join(OUT, "recovery.json"), JSON.stringify(record, null, 2));
  writeFileSync(join(OUT, "recovery.log"), log.join("\n") + "\n");
  say("");
  say(`${failures.length === 0 ? "ALL CHECKS PASSED" : `${failures.length} CHECK(S) FAILED`} -> ${OUT}`);
  if (failures.length > 0) process.exitCode = 1;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
