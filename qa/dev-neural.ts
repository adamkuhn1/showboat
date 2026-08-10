// Proof that an opponent turn on the DEV SERVER runs the trained model.
//
//   npx tsx qa/dev-neural.ts [outDir] [devPort] [previewPort] [proxyPort]
//
// `devServerAssets.test.ts` proves the artifact and ORT's own runtime are
// SERVED. That is a different claim from "an opponent turn ran inference", and
// only the second one is what the interface asserts when it says "Neural
// evaluator and physics search".
//
// WHAT COUNTS AS PROOF HERE
//
// Not the badge, and not the trace's own `mode` field: both are claims the page
// makes about itself. The evidence is the model's RAW PRE-CALIBRATION LOGITS.
// One is emitted per candidate, they are the graph's output tensor, and they
// reach the page only by `session.run()` returning. This harness captures them
// off the planning worker's own message together with the board they were
// computed from, then re-runs the identical encoder and the identical committed
// artifact under Node's onnxruntime and requires the two sets of numbers to
// agree. A page that had not run the graph has no way to produce numbers that
// match.
//
// The same run checks the failure side three times — the artifact answered with
// the dev server's SPA fallback, the artifact answered with corrupted bytes, and
// ORT's own runtime answered with the SPA fallback — because those fail in three
// different places. It then repeats the inference proof against `vite preview`
// on the production build, so the dev-server fix cannot have been bought with a
// production regression.

import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

import { launchChrome, attachToPage } from "../../portfolio/qa/cdp.mjs";
import {
  APP_ROOT,
  PROBE,
  READY,
  playForOpponentTurns,
  startBlockingProxy,
  startServer,
  turnsFrom,
  type BreakRule,
} from "./browser-drive";
import { makeTable } from "../src/physics/table";
import type { GameState, PlayerId } from "../src/game/state";
import { legalTargets } from "../src/ai/turn";
import { generateCandidates } from "../src/ai/candidates";
import { encodeRow, TOTAL_DIM } from "../src/ai/ranker/encode";
import { evaluateCandidateRows, tryLoadRankerModel, _resetRankerForTests } from "../src/ai/onnx";
import { makeFileFetch } from "../src/ai/neural/fileFetch";
import { validateManifest } from "../src/ai/neural/manifest";

const OUT = process.argv[2] ?? "/tmp/showboat-dev-neural";
const DEV_PORT = Number(process.argv[3] ?? 5322);
const PREVIEW_PORT = Number(process.argv[4] ?? 5323);
/** Where the breaking proxy sits in front of the dev server. */
const BROKEN_PORT = Number(process.argv[5] ?? 5325);

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
  return ok;
};

// ---------------------------------------------------------------------------
// Independent recomputation of the model's output, under Node
// ---------------------------------------------------------------------------

interface LogitCheck {
  candidates: number;
  maxAbsDelta: number;
  distinctBrowserLogits: number;
  browserSample: number[];
  nodeSample: number[];
}

async function recomputeLogits(
  request: { state: GameState; player: PlayerId },
  browserLogits: (number | null)[],
): Promise<LogitCheck> {
  const table = makeTable();
  const targets = legalTargets(request.state, request.player);
  const candidates = generateCandidates(request.state.balls, table, targets);
  if (candidates.length !== browserLogits.length) {
    throw new Error(
      `candidate count differs: node ${candidates.length}, browser ${browserLogits.length}`,
    );
  }
  const rows = new Float32Array(candidates.length * TOTAL_DIM);
  for (let i = 0; i < candidates.length; i++) {
    rows.set(encodeRow(request.state.balls, table, candidates[i]), i * TOTAL_DIM);
  }
  const out = await evaluateCandidateRows(rows, candidates.length, TOTAL_DIM);
  if (!out) throw new Error("node-side ranker produced no output");
  let maxAbsDelta = 0;
  for (let i = 0; i < candidates.length; i++) {
    const b = browserLogits[i];
    if (b === null || !Number.isFinite(b)) throw new Error(`browser logit ${i} is ${b}`);
    maxAbsDelta = Math.max(maxAbsDelta, Math.abs(b - out[i]));
  }
  return {
    candidates: candidates.length,
    maxAbsDelta,
    distinctBrowserLogits: new Set(browserLogits.map((x) => Math.round((x ?? 0) * 1e6))).size,
    browserSample: browserLogits.slice(0, 6).map((x) => Number((x ?? 0).toFixed(6))),
    nodeSample: [...out.slice(0, 6)].map((x) => Number(x.toFixed(6))),
  };
}

async function loadNodeRanker(): Promise<void> {
  _resetRankerForTests();
  const raw = JSON.parse(readFileSync(join(APP_ROOT, "public/model/ranker/manifest.json"), "utf8"));
  const validated = validateManifest(raw);
  if (!validated.ok) throw new Error(`manifest invalid: ${validated.error}`);
  await tryLoadRankerModel(`model/ranker/${validated.manifest.artifact}`, {
    expectedSha256: validated.manifest.onnx_sha256,
    expectedBytes: validated.manifest.bytes,
    expectedInputDim: validated.manifest.total_dim,
    fetchImpl: makeFileFetch(join(APP_ROOT, "public")),
  });
}

/**
 * Every ORT runtime binary this install ships, fetched over real HTTP at the
 * URL ORT's own `new URL(name, import.meta.url)` produces against the dev
 * server — an `/@fs/…` path into the real package directory, which is what
 * `optimizeDeps.exclude` buys.
 */
async function probeOrtRuntime(origin: string) {
  const ortDist = dirname(createRequire(import.meta.url).resolve("onnxruntime-web"));
  const names = readdirSync(ortDist).filter((f) => f.endsWith(".wasm"));
  const out: { name: string; ok: boolean; note: string }[] = [];
  for (const name of names) {
    const url = `${origin}/@fs${join(ortDist, name)}`;
    try {
      const res = await fetch(url);
      const ct = res.headers.get("content-type") ?? "";
      const head = new Uint8Array((await res.arrayBuffer()).slice(0, 4));
      const magic = [...head].map((b) => b.toString(16).padStart(2, "0")).join("");
      out.push({
        name,
        ok: res.status === 200 && ct.includes("application/wasm") && magic === "0061736d",
        note: `${res.status} ${ct} magic=${magic}`,
      });
    } catch (e) {
      out.push({ name, ok: false, note: String(e) });
    }
  }
  return out;
}

async function run() {
  mkdirSync(OUT, { recursive: true });
  await loadNodeRanker();
  say("node-side ranker loaded from the committed artifact");

  const evidence: Record<string, unknown> = {};
  const chrome = await launchChrome({ headless: true });
  const page = await attachToPage(chrome.port);
  say(`chrome           ${chrome.version.Browser}`);

  const consoleErrors: string[] = [];
  await page.send("Runtime.enable");
  page.on("Runtime.consoleAPICalled", (p: { type: string; args: { value?: string }[] }) => {
    if (p.type === "error") consoleErrors.push(p.args.map((a) => a.value ?? "").join(" "));
  });

  const dev = await startServer("dev", DEV_PORT, "dev server");
  say(`dev server       ${dev.origin}`);
  try {
    // ---- 1. the runtime assets, over real HTTP ----------------------------
    const ort = await probeOrtRuntime(dev.origin);
    check(
      ort.length > 0 && ort.every((r) => r.ok),
      "no SPA fallback for any ORT runtime binary",
      ort.map((r) => `${r.name} ${r.note}`).join("; "),
    );
    evidence.ortRuntime = ort;

    const manifest = (await (await fetch(`${dev.origin}/model/ranker/manifest.json`)).json()) as {
      artifact: string;
      bytes: number;
    };
    const artifactRes = await fetch(`${dev.origin}/model/ranker/${manifest.artifact}`);
    const artifactBytes = new Uint8Array(await artifactRes.arrayBuffer());
    check(
      artifactRes.status === 200 &&
        artifactBytes.byteLength === manifest.bytes &&
        artifactBytes[0] !== 0x3c,
      "no SPA fallback for the model artifact",
      `${artifactRes.status} ${artifactBytes.byteLength}/${manifest.bytes} bytes`,
    );

    // ---- 2. a real opponent turn runs the graph ---------------------------
    await page.addInitScript(PROBE);
    await page.viewport(1400, 950, { dpr: 1 });
    await page.goto(dev.origin + "/");
    const ready = await page.eval(READY);
    check(/break to start/.test(String(ready)), "the dev server boots the app", String(ready));

    const cap = await playForOpponentTurns(page, 2, 20);
    const turns = turnsFrom(cap);
    check(turns.length >= 1, "an opponent turn completed on the dev server", `${turns.length} turns`);

    const neuralTurn = turns.find((t) => t.mode === "neural-hybrid" && t.fallback === null);
    check(neuralTurn !== undefined, "the opponent turn ran in neural-hybrid mode");
    check(
      (neuralTurn?.logitsPresent ?? 0) > 0 && (neuralTurn?.neuralRunMs ?? 0) > 0,
      "the turn carries per-candidate logits and a measured session.run time",
      `${neuralTurn?.logitsPresent} logits, runMs=${neuralTurn?.neuralRunMs}`,
    );
    check(neuralTurn?.hashVerified === true, "the artifact's sha256 was verified in the browser");

    const doneMsgs = cap.responses.filter((r) => r.msg.type === "done");
    const reqs = cap.requests.filter((r) => r.msg.type === "plan");
    let logitCheck: LogitCheck | null = null;
    let logitError: string | null = null;
    for (let i = 0; i < doneMsgs.length && logitCheck === null; i++) {
      const trace = (doneMsgs[i].msg as { planned: { trace: Record<string, unknown> } }).planned
        .trace as { candidates: { neural: { logit: number | null } | null }[] };
      const logits = trace.candidates.map((c) => c.neural?.logit ?? null);
      if (logits.length === 0 || logits.some((l) => l === null)) continue;
      const req = reqs[i]?.msg as unknown as { state: GameState; player: PlayerId } | undefined;
      if (!req) continue;
      try {
        logitCheck = await recomputeLogits(req, logits);
      } catch (e) {
        logitError = String(e);
      }
    }
    evidence.logitCheck = logitCheck ?? logitError;
    check(
      logitCheck !== null && logitCheck.maxAbsDelta < 1e-3 && logitCheck.distinctBrowserLogits > 1,
      "the browser's logits reproduce against the committed artifact under Node",
      logitCheck
        ? `n=${logitCheck.candidates} maxDelta=${logitCheck.maxAbsDelta.toExponential(2)} distinct=${logitCheck.distinctBrowserLogits}`
        : (logitError ?? "no neural turn produced logits"),
    );

    // ---- 3. the claim never precedes the initialisation -------------------
    const firstNeuralTitle = cap.titles.find((t) => /Neural evaluator/.test(t.text));
    const firstReadyStatus = cap.responses.find(
      (r) => r.msg.type === "model-status" && (r.msg as { ok: boolean }).ok === true,
    );
    check(
      firstNeuralTitle === undefined ||
        (firstReadyStatus !== undefined && firstReadyStatus.at <= firstNeuralTitle.at),
      "the panel claims neural only after the model reported ready",
      `ready@${firstReadyStatus?.at?.toFixed(0) ?? "never"} title@${firstNeuralTitle?.at?.toFixed(0) ?? "never"}`,
    );
    evidence.titles = cap.titles;
    evidence.devTurns = turns;

    // ---- 4. the broken condition: the proof must fail ---------------------
    //
    // Broken at the ORIGIN, through a proxy, not with `Network.setBlockedURLs`:
    // the artifact and ORT's runtime are fetched by the planning worker, which
    // is a separate DevTools target, so a block installed on the page session
    // does not reach them and the "broken" run keeps running the model. That
    // was observed before this proxy existed — three turns with the artifact
    // nominally blocked still returned 46 and 48 live logits.
    const breakages: [string, BreakRule[]][] = [
      ["artifact returns the SPA fallback", [{ match: /model\/ranker\/.*\.onnx/, mode: "spa-fallback" }]],
      ["artifact is corrupted", [{ match: /model\/ranker\/.*\.onnx/, mode: "corrupt" }]],
      ["ORT runtime returns the SPA fallback", [{ match: /ort-wasm.*\.wasm/, mode: "spa-fallback" }]],
    ];
    for (const [name, rules] of breakages) {
      const proxy = await startBlockingProxy(dev.origin, BROKEN_PORT, rules);
      const brokenHits = () =>
        [...proxy.hits.entries()]
          .filter(([p]) => rules.some((r) => r.match.test(p)))
          .reduce((a, [, n]) => a + n, 0);
      await page.goto(proxy.origin + "/");
      await page.eval(READY);
      // One turn first, then several more, so "falls back once" can be stated
      // as "the request count did not move", which is the property that matters
      // and the one a per-turn retry loop would break.
      await playForOpponentTurns(page, 1, 10);
      const hitsAfterFirst = brokenHits();
      const broken = await playForOpponentTurns(page, 4, 16);
      const brokenTurns = turnsFrom(broken);
      const anyNeural = brokenTurns.some((t) => t.mode === "neural-hybrid" && t.fallback === null);
      const anyLogits = brokenTurns.some((t) => t.logitsPresent > 0);
      check(
        brokenTurns.length >= 2 && !anyNeural && !anyLogits,
        `[${name}] the inference proof fails`,
        `${brokenTurns.length} turns, neural=${anyNeural}, logits=${anyLogits}`,
      );

      const panel = (await page.eval(`(() => {
        const a = document.querySelector('.overlay');
        return a ? { title: a.querySelector('.overlay-title')?.textContent ?? null,
                     warn: a.querySelector('.overlay-warn')?.textContent ?? null } : null;
      })()`)) as { title: string | null; warn: string | null } | null;
      check(
        panel !== null &&
          panel.title === "Physics search" &&
          typeof panel.warn === "string" &&
          panel.warn.length > 0,
        `[${name}] the failure is visible and never labelled neural`,
        JSON.stringify(panel),
      );
      check(
        brokenTurns.length >= 2 && brokenTurns.every((t) => t.fallback !== null),
        `[${name}] every turn still plays, each one labelled a fallback`,
        `${brokenTurns.length} turns`,
      );
      // Falls back ONCE, not per turn: each realm's evaluator memoises its own
      // failure, so after the first turn has paid for it the request count stops
      // moving however many turns follow. Counted at the proxy, which is where
      // every request really lands — including the worker's, which is the whole
      // reason the proxy exists.
      const attempts = brokenHits();
      check(
        attempts === hitsAfterFirst && brokenTurns.length > 1,
        `[${name}] the broken asset is not re-fetched every turn`,
        `${hitsAfterFirst} requests after 1 turn, ${attempts} after ${brokenTurns.length}`,
      );
      evidence[`broken:${name}`] = { turns: brokenTurns, panel, hitsAfterFirst, attempts };
      await proxy.stop();
    }
  } finally {
    dev.stop();
  }

  // ---- 5. production preview still works ---------------------------------
  say("");
  say("building for production preview…");
  await new Promise<void>((res, rej) => {
    const b = spawn("npm", ["run", "build"], { cwd: APP_ROOT, stdio: "ignore" });
    b.on("exit", (c) => (c === 0 ? res() : rej(new Error(`build exited ${c}`))));
  });
  const preview = await startServer("preview", PREVIEW_PORT, "preview server");
  say(`preview server   ${preview.origin}`);
  try {
    await page.goto(preview.origin + "/");
    const ready = await page.eval(READY);
    check(/break to start/.test(String(ready)), "the production preview boots", String(ready));
    const cap = await playForOpponentTurns(page, 1, 20);
    const turns = turnsFrom(cap);
    const neural = turns.find((t) => t.mode === "neural-hybrid" && t.fallback === null);
    check(
      neural !== undefined && neural.logitsPresent > 0 && (neural.neuralRunMs ?? 0) > 0,
      "the production preview runs neural inference on an opponent turn",
      JSON.stringify(neural ?? turns),
    );
    evidence.preview = turns;
  } finally {
    preview.stop();
  }

  await chrome.close();

  evidence.consoleErrors = consoleErrors;
  evidence.failures = failures;
  writeFileSync(join(OUT, "dev-neural.json"), JSON.stringify(evidence, null, 2));
  writeFileSync(join(OUT, "dev-neural.log"), log.join("\n") + "\n");
  say("");
  say(`${failures.length === 0 ? "ALL CHECKS PASSED" : `${failures.length} CHECK(S) FAILED`} -> ${OUT}`);
  if (failures.length > 0) process.exitCode = 1;
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
