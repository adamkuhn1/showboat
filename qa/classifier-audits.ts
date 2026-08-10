// Four structural questions about the measured classifier, asked of real
// physics rather than of a hand-built log.
//
//   npx tsx qa/classifier-audits.ts [outDir]
//
// `src/ai/measure/classify.test.ts` states each rule against a log written by
// hand, which is the right way to pin a rule. It cannot show that the rule has
// anything to do. These four boards are placed by hand, shot with a stated
// action, and run through the shipped WASM engine; each one produces the event
// shape the rule exists for, and each case reports both what the classifier
// answered and what it would have answered with the rule switched off.
//
// The layouts and actions below were found by construction and then frozen. No
// search runs here: the same five numbers produce the same event log every time,
// because the engine is deterministic.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { makeBall, cloneBall, type Ball } from "../src/physics/ball";
import { initPhysics, simulateShotWasm } from "../src/physics/wasm-bridge";
import type { ShotEvent } from "../src/physics/engine";
import { CUE_ID } from "../src/game/rack";
import {
  classifyMeasuredShot,
  measuredRouteLabel,
  type MeasuredShot,
  type ShotIntent,
} from "../src/ai/measure/classify";

const APP_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = process.argv[2] ?? "/tmp/showboat-classifier-audits";
const D = 0.05715;

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

interface Shot {
  balls: [number, number, number][];
  phi: number;
  power: number;
  intent: ShotIntent;
}

const render = (events: readonly ShotEvent[], potAt: number): string =>
  events
    .map((e, i) => {
      const t = `@${e.time.toFixed(3)}`;
      const mark = i === potAt ? ">>>" : "";
      if (e.kind === "ball-ball") return `${mark}${e.balls[0]}-${e.balls[1]}${t}`;
      if (e.kind === "ball-cushion") return `${mark}${e.balls[0]}|${e.cushion}${t}`;
      if (e.kind === "pocket") return `${mark}POT ${e.balls[0]}@${e.pocket}${t}`;
      return "";
    })
    .filter(Boolean)
    .join("  ");

const run = (s: Shot): { m: MeasuredShot; events: ShotEvent[]; line: string } => {
  const balls: Ball[] = s.balls.map(([id, x, y]) => makeBall(id, x, y));
  const sim = simulateShotWasm(balls.map(cloneBall), {
    phi: s.phi,
    power: s.power,
    sideSpin: 0,
    topSpin: 0,
  });
  const m = classifyMeasuredShot(sim, s.intent);
  return { m, events: sim.events, line: render(sim.events, m.potEventIndex) };
};

/**
 * The label the same chain would have carried at a different rail count, using
 * the classifier's own rule for turning a chain length and a rail count into a
 * class. This is how each case below states its counterfactual.
 */
const relabel = (m: MeasuredShot, rails: number): string => {
  const n = m.contactChain.length;
  const classification =
    n > 2
      ? rails >= 1
        ? ("rail-combination" as const)
        : ("combination" as const)
      : rails === 0
        ? ("direct" as const)
        : rails === 1
          ? ("one-rail-bank" as const)
          : ("multi-rail-bank" as const);
  return measuredRouteLabel({ ...m, classification, rails });
};

/** Cushions taken by chain balls, counted over a slice of the log. */
const chainCushions = (events: readonly ShotEvent[], chain: readonly number[]): number => {
  const set = new Set(chain.filter((id) => id !== CUE_ID));
  return events.filter((e) => e.kind === "ball-cushion" && set.has(e.balls[0])).length;
};

// ---------------------------------------------------------------------------

async function main() {
  mkdirSync(OUT, { recursive: true });
  await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));

  const record: Record<string, unknown> = {};

  // --- 1. A cushion struck after the pot cannot inflate the classification ---
  //
  // A cut combination into the top-side pocket. The 2 drops at 0.15 s; the 1,
  // which is on the chain, carries on and takes two cushions afterwards.
  const one: Shot = {
    balls: [
      [CUE_ID, -0.6340275154942367, 0.18809438198497935],
      [1, -0.25, 0.3],
      [2, 0, 0.43],
    ],
    phi: 0.2835483315959372,
    power: 0.5,
    intent: { target: 1, potId: 2, legalTargets: [1, 2] },
  };
  {
    const { m, events, line } = run(one);
    const post = events.slice(m.potEventIndex + 1);
    const postOnChain = chainCushions(post, m.contactChain);
    const withoutTruncation = relabel(m, m.rails + postOnChain);
    say("");
    say("1  post-pot cushions");
    say(`   ${line}`);
    say(
      `   measured: ${m.classification} rails=${m.rails} chain=${JSON.stringify(m.contactChain)} -> "${measuredRouteLabel(m)}"`,
    );
    say(
      `   without the truncation rule: rails=${m.rails + postOnChain} -> "${withoutTruncation}"`,
    );
    check(postOnChain >= 2, "the constructed shot really does take cushions after the pot", `${postOnChain}`);
    check(m.rails === 0, "none of them is counted", `rails=${m.rails}`);
    check(m.classification === "combination", "the label is the untruncated route's", m.classification);
    check(
      withoutTruncation !== measuredRouteLabel(m),
      "and the rule is load-bearing: without it the label changes",
      `${measuredRouteLabel(m)} vs ${withoutTruncation}`,
    );
    record.postPotCushions = { events: line, measured: m, postOnChain };
  }

  // --- 2. A cushion struck by an unrelated ball cannot inflate it ------------
  //
  // The same combination, further from the pocket, with a third ball parked on
  // the 1's onward path. The 3 reaches the right cushion at 0.26 s, well before
  // the 2 drops at 0.38 s, and the 3 is on no part of the causal chain.
  const two: Shot = {
    balls: [
      [CUE_ID, -0.6340275154942367, -0.1919056180150206],
      [1, -0.25, -0.08],
      [2, 0, 0.05],
      [3, 0.8, -0.00715],
    ],
    phi: 0.2835483315959371,
    power: 0.75,
    intent: { target: 1, potId: 2, legalTargets: [1, 2, 3] },
  };
  {
    const { m, events, line } = run(two);
    const pre = events.slice(0, m.potEventIndex);
    const offChain = pre.filter(
      (e) =>
        e.kind === "ball-cushion" &&
        e.balls[0] !== CUE_ID &&
        !m.contactChain.includes(e.balls[0]),
    );
    const withoutFilter = relabel(m, m.rails + offChain.length);
    say("");
    say("2  cushions by a ball off the causal chain");
    say(`   ${line}`);
    say(
      `   measured: ${m.classification} rails=${m.rails} chain=${JSON.stringify(m.contactChain)} -> "${measuredRouteLabel(m)}"`,
    );
    say(
      `   off-chain cushions before the pot: ${offChain.map((e) => `${e.balls[0]}|${e.cushion}`).join(", ")}`,
    );
    say(`   without the causal filter: rails=${m.rails + offChain.length} -> "${withoutFilter}"`);
    check(offChain.length >= 1, "an uninvolved ball really does reach a cushion first", `${offChain.length}`);
    check(m.rails === 0, "none of them is counted", `rails=${m.rails}`);
    check(m.classification === "combination", "the label is the causal route's", m.classification);
    check(
      withoutFilter !== measuredRouteLabel(m),
      "and the rule is load-bearing: without it the label changes",
      `${measuredRouteLabel(m)} vs ${withoutFilter}`,
    );
    record.offChainCushions = { events: line, measured: m, offChain: offChain.length };
  }

  // --- 3. Collision chatter -------------------------------------------------
  //
  // Two balls frozen together against the bottom cushion. The cue drives into
  // the pair and the 1 is trapped between the cue and the rail: four cue-1
  // contacts, and two bottom-cushion contacts 50 ms apart.
  const three: Shot = {
    balls: [
      [CUE_ID, -0.75, -0.4953 + D / 2 + 0.0001 + 0.004],
      [1, -0.45, -0.4953 + D / 2 + 0.0001],
      [2, -0.45 + D + 0.0002, -0.4953 + D / 2 + 0.0001],
    ],
    phi: 0,
    power: 0.5,
    intent: { target: 1, potId: 2, legalTargets: [1, 2] },
  };
  {
    const { m, events, line } = run(three);
    const pre = events.slice(0, m.potEventIndex);
    const pairs = new Map<string, number[]>();
    for (const e of pre) {
      if (e.kind !== "ball-ball" || e.balls.length < 2) continue;
      const k = [e.balls[0], e.balls[1]].sort((a, b) => a - b).join("-");
      pairs.set(k, [...(pairs.get(k) ?? []), e.time]);
    }
    const repeats = [...pairs.entries()].filter(([, t]) => t.length > 1);
    const tightest = Math.min(
      ...repeats.flatMap(([, t]) => t.slice(1).map((x, i) => (x - t[i]) * 1000)),
    );
    // A rattle: the same ball meeting the same cushion again within 50 ms.
    const seen = new Map<string, number>();
    let rattles = 0;
    for (const e of pre) {
      if (e.kind !== "ball-cushion" || e.balls[0] === CUE_ID) continue;
      const k = `${e.balls[0]}:${e.cushion}`;
      const prev = seen.get(k);
      seen.set(k, e.time);
      if (prev !== undefined && (e.time - prev) * 1000 <= 50) rattles++;
    }
    say("");
    say("3  collision chatter");
    say(`   ${line}`);
    say(
      `   measured: ${m.classification} rails=${m.rails} chain=${JSON.stringify(m.contactChain)} -> "${measuredRouteLabel(m)}"`,
    );
    say(
      `   repeated pairs: ${repeats.map(([k, t]) => `${k}x${t.length}`).join(", ")}; tightest ${tightest.toFixed(1)} ms`,
    );
    say(`   published rails that are a re-contact on the same rail within 50 ms: ${rattles}`);
    check(repeats.length >= 1 && tightest < 60, "the constructed shot really does chatter", `${tightest.toFixed(1)} ms`);
    check(
      m.contactChain.join(",") === "0,1,2",
      "repeated contacts between the same pair leave the chain intact",
      JSON.stringify(m.contactChain),
    );
    check(
      m.rails === pre.filter((e) => e.kind === "ball-cushion" && m.contactChain.includes(e.balls[0]) && e.balls[0] !== CUE_ID).length,
      "the rail count is one per cushion event and repeated ball contacts add none",
      `rails=${m.rails}`,
    );
    // The safety property: a rattle can raise a rail count but can never create
    // the first rail, because the first contact of a repeat is never a repeat.
    check(m.rails - rattles >= 1, "collapsing every rattle still leaves a real cushion", `${m.rails} - ${rattles}`);
    record.chatter = { events: line, measured: m, rattles, tightestMs: tightest };
  }

  // --- 4. Contact with a ball that is already moving ------------------------
  //
  // The cue kicks off the top rail into the 1; the 1 banks twice and sets the 2
  // moving; the 2 takes a cushion; and then the CUE catches the 2 and puts it
  // in. The chain credits the 1, which is the ball that started the 2 — not the
  // cue, which is the ball that finished it.
  const four: Shot = {
    balls: [
      [CUE_ID, -0.1042, -0.0842],
      [1, 0.586, 0.1524],
      [2, 0.8985, 0.1689],
      [3, 0.4978, 0.1234],
    ],
    phi: 0.841488,
    power: 0.7778,
    intent: { target: 1, potId: 2, legalTargets: [1, 2, 3] },
  };
  {
    const { m, events, line } = run(four);
    const pre = events.slice(0, m.potEventIndex);
    const potted = m.contactChain[m.contactChain.length - 1] ?? null;
    const credited = m.contactChain[m.contactChain.length - 2] ?? null;
    let lastToucher: number | null = null;
    for (const e of pre) {
      if (e.kind !== "ball-ball" || e.balls.length < 2) continue;
      if (e.balls[0] === potted) lastToucher = e.balls[1];
      else if (e.balls[1] === potted) lastToucher = e.balls[0];
    }
    say("");
    say("4  contact with an already-moving ball");
    say(`   ${line}`);
    say(
      `   measured: ${m.classification} rails=${m.rails} chain=${JSON.stringify(m.contactChain)} verified=${m.trickVerified} -> "${measuredRouteLabel(m)}"`,
    );
    say(`   the ${potted} was started by the ${credited} and last touched by the ${lastToucher}`);
    check(lastToucher !== credited, "the constructed shot really is ambiguous", `${lastToucher} vs ${credited}`);
    check(
      m.contactChain[0] === CUE_ID && m.contactChain.length >= 3,
      "the chain still walks back to the cue ball",
      JSON.stringify(m.contactChain),
    );
    // The property that matters: an ambiguous driver cannot manufacture a trick
    // out of a straight pot. A chain of three or more requires the cue's first
    // contact to be some other ball, and every counted cushion is an event the
    // potted ball really produced after it started moving.
    check(
      m.firstContact !== potted,
      "a chain of three or more means the cue did not strike the potted ball first",
      `firstContact=${m.firstContact}`,
    );
    record.alreadyMoving = { events: line, measured: m, credited, lastToucher };
  }

  writeFileSync(join(OUT, "classifier-audits.json"), JSON.stringify(record, null, 2));
  writeFileSync(join(OUT, "classifier-audits.log"), log.join("\n") + "\n");
  say("");
  say(`${failures.length === 0 ? "ALL CHECKS PASSED" : `${failures.length} CHECK(S) FAILED`} -> ${OUT}`);
  if (failures.length > 0) process.exitCode = 1;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
