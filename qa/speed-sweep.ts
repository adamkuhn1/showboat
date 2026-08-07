// Which default presentation speed, and why that one.
//
// The old default (0.6x) was chosen against a FLAT multiplier and Adam reports
// it is still too fast to follow. `ui/pacing.ts` changed the shape of the
// curve, so the default has to be re-derived rather than inherited.
//
// The metric is the one the eye actually cares about: how far a ball moves
// across the screen between two displayed frames. Smooth pursuit of a small
// object breaks down when it jumps more than roughly its own diameter per
// frame; the ball is ~14 logical px across at the shipped view, so that is the
// bar this reports against.
//
// Computed over real simulations, at the real view transform, for every
// candidate speed — then one of them is confirmed in a browser by qa/motion.mjs.
//
//   npx tsx qa/speed-sweep.ts

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../src/physics/table";
import { makeBall, cloneBall, type Ball } from "../src/physics/ball";
import { CUE_ID } from "../src/game/rack";
import { BALL_RADIUS } from "../src/physics/constants";
import { initPhysics, simulateShotWasm } from "../src/physics/wasm-bridge";
import { generateCandidates } from "../src/ai/candidates";
import { contactTimes, interpolateBalls, type AnimTrack } from "../src/render/animate";
import { computeView } from "../src/render/renderer";
import { buildPacing, screenSeconds, CONTACT_WINDOW_SEC } from "../src/ui/pacing";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "..");
const table = makeTable();
const view = computeView(900, 500, table);
const BALL_PX = 2 * BALL_RADIUS * view.scale;
const FRAME_MS = 1000 / 60;

const BOARDS: { name: string; balls: Ball[]; targets: number[] }[] = [
  {
    name: "open",
    balls: [makeBall(CUE_ID, -0.6, -0.1), makeBall(1, 0.3, 0.02), makeBall(2, 0.38, 0.1), makeBall(4, 0.1, -0.25)],
    targets: [1, 2, 4],
  },
  {
    name: "spread",
    balls: [makeBall(CUE_ID, -0.75, 0.22), makeBall(3, 0.45, -0.28), makeBall(6, -0.1, 0.3), makeBall(9, 0.2, 0.25)],
    targets: [3, 6, 9],
  },
];

const main = async () => {
  await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));

  const shots: { track: AnimTrack; contacts: number[] }[] = [];
  for (const board of BOARDS) {
    for (const c of generateCandidates(board.balls, table, board.targets)) {
      for (const power of [c.action.power, 0.85]) {
        const sim = simulateShotWasm(board.balls.map(cloneBall), { ...c.action, power });
        const wps = sim.waypoints ?? [];
        const contacts = contactTimes(sim);
        if (wps.length < 5 || contacts.length < 2) continue;
        shots.push({
          track: { waypoints: wps.map((w) => ({ simTime: w.time, balls: w.balls })), duration: sim.duration },
          contacts,
        });
        if (shots.length >= 60) break;
      }
      if (shots.length >= 60) break;
    }
  }

  console.log(`shots: ${shots.length}   ball diameter: ${BALL_PX.toFixed(1)} logical px\n`);
  console.log(
    ["speed", "screen s", "px/frame p50", "p90", "p99", "max", "at contacts p90", "frames > 1 ball"]
      .map((h) => h.padStart(16))
      .join(""),
  );

  for (const speed of [0.35, 0.5, 0.6, 1]) {
    const all: number[] = [];
    const nearContact: number[] = [];
    let screenTotal = 0;

    for (const { track, contacts } of shots) {
      const pacing = buildPacing({ contactTimes: contacts, durationSec: track.duration, speed });
      screenTotal += screenSeconds(pacing, track.duration);

      // Walk the shot the way the animation does: integrate the rate frame by
      // frame and measure how far each ball moved on screen in that frame.
      let simTime = 0;
      let prev = interpolateBalls(track, 0);
      let guard = 0;
      while (simTime < track.duration && guard++ < 6000) {
        const rate = pacing.rateAt(simTime);
        const next = simTime + (FRAME_MS / 1000) * rate;
        const now = interpolateBalls(track, Math.min(next, track.duration));
        let frameMax = 0;
        for (const b of now) {
          const p = prev.find((x) => x.id === b.id);
          if (!p || b.pocketed || p.pocketed) continue;
          frameMax = Math.max(frameMax, Math.hypot(b.pos.x - p.pos.x, b.pos.y - p.pos.y) * view.scale);
        }
        if (frameMax > 0.01) {
          all.push(frameMax);
          if (contacts.some((c) => Math.abs(simTime - c) < CONTACT_WINDOW_SEC)) nearContact.push(frameMax);
        }
        prev = now;
        simTime = next;
      }
    }

    all.sort((a, b) => a - b);
    nearContact.sort((a, b) => a - b);
    const q = (arr: number[], p: number) => (arr.length ? arr[Math.min(arr.length - 1, Math.floor(arr.length * p))] : 0);
    const over = all.filter((d) => d > BALL_PX).length / all.length;

    console.log(
      [
        `${speed}x`,
        (screenTotal / shots.length).toFixed(2),
        q(all, 0.5).toFixed(1),
        q(all, 0.9).toFixed(1),
        q(all, 0.99).toFixed(1),
        q(all, 1).toFixed(1),
        q(nearContact, 0.9).toFixed(1),
        `${(over * 100).toFixed(1)}%`,
      ]
        .map((s) => s.padStart(16))
        .join(""),
    );
  }
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
