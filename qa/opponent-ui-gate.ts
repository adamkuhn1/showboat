// The interface half of the measured-shot gate.
//
//   npx tsx qa/opponent-ui-gate.ts [outDir] [port] [opponentTurns]
//
// `qa/measured-gate.ts` proves the DECISIONS are honest. This proves the page
// showing them is: over a long session against the real app in a real browser,
//
//   * no turn wedges — every one of them settles;
//   * every claim line the panel shows for a turn is character-identical to the
//     string the trace's own render functions produce from the trace the worker
//     sent, so "the wording matches the trace" is an equality and not a reading;
//   * a result line for the shot is on screen before the next turn is asked for;
//   * nothing on screen claims a bank, a combination or a model that the trace
//     does not support.
//
// The panel is read from a change log the probe keeps, not from a snapshot: the
// opponent can take several turns in a row, and a harness that only looks after
// the table settles sees the last of them and misses the rest.
//
// Playback rate is measured by `qa/time-mapping.ts`, which fits the wall-clock
// to simulation-time mapping over many shots; it is not re-measured here.

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { launchChrome, attachToPage } from "../../portfolio/qa/cdp.mjs";
import {
  PROBE,
  READY,
  capture,
  humanShot,
  newRack,
  sleep,
  startServer,
  turnsFrom,
  waitIdle,
  type PanelSample,
  type QaCapture,
} from "./browser-drive";
import type { DecisionTraceV1 } from "../src/ai/trace/contract";
import { plannedVsMeasured, rungText, shotSentence } from "../src/ui/shotSentence";

const OUT = process.argv[2] ?? "/tmp/showboat-ui-gate";
const PORT = Number(process.argv[3] ?? 5324);
const OPPONENT_TURNS = Number(process.argv[4] ?? 14);

const log: string[] = [];
const say = (...a: unknown[]) => {
  const line = a.join(" ");
  log.push(line);
  console.log(line);
};
const failures: string[] = [];
const check = (ok: boolean, name: string, detail = "") => {
  say(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  if (!ok) failures.push(`${name}${detail ? `: ${detail}` : ""}`);
};

const tracesOf = (cap: QaCapture): DecisionTraceV1[] =>
  cap.responses
    .filter((r) => r.msg.type === "done")
    .map((r) => (r.msg as { planned: { trace: DecisionTraceV1 } }).planned.trace);

/** Panel states that were on screen while this turn owned the panel. */
const windowOf = (panels: PanelSample[], from: number, to: number): PanelSample[] =>
  panels.filter((p) => p.until > from && p.at < to);

async function main() {
  mkdirSync(OUT, { recursive: true });
  const chrome = await launchChrome({ headless: true });
  const page = await attachToPage(chrome.port);
  say(`chrome           ${chrome.version.Browser}`);

  const consoleErrors: string[] = [];
  await page.send("Runtime.enable");
  page.on("Runtime.consoleAPICalled", (p: { type: string; args: { value?: string }[] }) => {
    if (p.type === "error") consoleErrors.push(p.args.map((a) => a.value ?? "").join(" "));
  });

  const server = await startServer("dev", PORT, "dev server");
  say(`dev server       ${server.origin}`);
  let stalls = 0;
  let cap: QaCapture | null = null;
  try {
    await page.addInitScript(PROBE);
    await page.viewport(1400, 950, { dpr: 1 });
    await page.goto(server.origin + "/");
    const ready = await page.eval(READY);
    check(/break to start/.test(String(ready)), "the app boots", String(ready));

    let seenTurns = 0;
    let round = 0;
    // Half the session with the model, half without, so the classical-fallback
    // claims are checked on the same board distribution as the neural ones.
    let toggled = false;
    while (seenTurns < OPPONENT_TURNS && round < OPPONENT_TURNS * 4) {
      round++;
      if (!toggled && seenTurns >= Math.floor(OPPONENT_TURNS / 2)) {
        await page.eval(`document.querySelector('.overlay input[type=checkbox]').click()`);
        await sleep(200);
        toggled = true;
        say(`--- model toggled off after ${seenTurns} turns ---`);
      }
      await humanShot(page, round);
      const settle = await waitIdle(page);
      if (/^STALL|^LONG/.test(settle)) {
        stalls++;
        say(`  STUCK TURN  ${settle}`);
      }
      cap = await capture(page);
      seenTurns = tracesOf(cap).length;
      if (await page.eval(`!!document.querySelector('.win-screen')`)) await newRack(page);
    }
    // A last settle so the final turn's result line is inside the record.
    await sleep(1500);
    cap = await capture(page);
  } finally {
    server.stop();
    await chrome.close();
  }

  const traces = tracesOf(cap!);
  const turns = turnsFrom(cap!);
  const planAts = cap!.requests.filter((r) => r.msg.type === "plan").map((r) => r.at);
  const rows: Record<string, unknown>[] = [];
  let wordingMatched = 0;
  let resultShown = 0;

  for (let i = 0; i < turns.length; i++) {
    const trace = traces[i];
    const sel = trace.selected;
    const from = turns[i].at;
    // The turn owns the panel until the next turn's answer arrives.
    const to = turns[i + 1]?.at ?? Number.POSITIVE_INFINITY;
    const win = windowOf(cap!.panels, from, to);

    const expectedChosen = shotSentence(trace)?.text ?? null;
    const expectedReconcile = plannedVsMeasured(trace);
    const expectedReason = rungText(trace);
    const claimsNeural = trace.mode === "neural-hybrid" && trace.fallback === null;

    const wordingOk = win.some(
      (p) =>
        p.chosen === expectedChosen &&
        (p.reconcile ?? null) === expectedReconcile &&
        p.reason === expectedReason,
    );
    // The next turn cannot be asked for before the result of this one is up.
    const nextPlan = planAts.find((a) => a > from) ?? Number.POSITIVE_INFINITY;
    const resultOk = win.some((p) => p.outcome !== null && p.at < Math.min(to, nextPlan) + 1);
    const neuralClaimed = win.some((p) => /Neural evaluator/.test(p.title ?? ""));

    wordingMatched += Number(wordingOk);
    resultShown += Number(resultOk);

    rows.push({
      turn: i + 1,
      mode: trace.mode,
      fallback: trace.fallback?.detail ?? null,
      kind: sel?.kind ?? null,
      plannedKind: sel?.plannedKind ?? null,
      measuredClass: sel?.measured?.classification ?? null,
      measuredRails: sel?.measured?.rails ?? null,
      trickVerified: sel?.measured?.trickVerified ?? null,
      rung: sel?.rung ?? null,
      expectedChosen,
      expectedReconcile,
      expectedReason,
      wordingOk,
      resultOk,
      neuralClaimed,
      panelStates: win.length,
    });

    if (!wordingOk) {
      check(false, `turn ${i + 1}: the panel's lines match the trace`, JSON.stringify({ expectedChosen, expectedReconcile, expectedReason, saw: win.map((p) => [p.chosen, p.reconcile, p.reason]) }).slice(0, 600));
    }
    if (!claimsNeural && neuralClaimed) {
      check(false, `turn ${i + 1}: no neural claim on a classical decision`, trace.fallback?.cause ?? trace.mode);
    }
    if (sel && sel.kind !== "safety-kick" && !(sel.measured?.trickVerified ?? false)) {
      check(false, `turn ${i + 1}: the played trick is measured`, JSON.stringify(sel.measured));
    }
    if (sel?.measured?.classification === "direct") {
      check(false, `turn ${i + 1}: no direct pot is played`, "measured as a direct");
    }
  }

  say("");
  check(turns.length >= OPPONENT_TURNS, "the session reached its opponent-turn target", `${turns.length}`);
  check(stalls === 0, "zero stuck turns", `${stalls} stalls over ${turns.length} turns`);
  check(
    wordingMatched === turns.length,
    "every turn's panel lines are character-identical to the trace's own",
    `${wordingMatched}/${turns.length}`,
  );
  check(
    resultShown === turns.length,
    "every shot's result line is on screen before the next turn is asked for",
    `${resultShown}/${turns.length}`,
  );
  check(
    turns.every((t) => t.kind === "safety-kick" || (t.measured?.trickVerified ?? false)),
    "every played trick is measured",
  );
  check(turns.every((t) => t.measured?.classification !== "direct"), "zero direct pots played");
  check(turns.every((t) => t.executedPublished), "every shot published its measured route");
  check(
    turns.some((t) => t.mode === "neural-hybrid" && t.fallback === null),
    "at least one turn really ran the model",
  );
  check(
    turns.some((t) => t.mode === "classical-trick-only"),
    "at least one turn ran classical, for the fallback claims",
  );
  check(consoleErrors.length === 0, "no console errors", consoleErrors.slice(0, 3).join(" | "));

  writeFileSync(
    join(OUT, "ui-gate.json"),
    JSON.stringify(
      { turns: rows, panels: cap!.panels, summary: { turns: turns.length, stalls, consoleErrors } },
      null,
      2,
    ),
  );
  writeFileSync(join(OUT, "ui-gate.log"), log.join("\n") + "\n");
  say("");
  say(`${failures.length === 0 ? "ALL CHECKS PASSED" : `${failures.length} CHECK(S) FAILED`} -> ${OUT}`);
  if (failures.length > 0) process.exitCode = 1;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
