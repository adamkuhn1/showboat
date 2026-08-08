// Does the screen show the shot that was simulated, at one speed?
//
// The complaint this answers: "all of the balls speed up and slow down
// randomly, and the cue ball goes crazy". A unit test cannot see that, because
// every ball position on screen is a correct sample of a correct simulation —
// what is wrong is WHICH simulation instant each displayed frame samples.
//
// So this measures the mapping itself, over real shots run through the real
// WASM engine and the real playback loop:
//
//   1. AFFINE. Sample (wall-clock elapsed, simulation time) every frame and fit
//      simTime = a*wall + b by least squares. A single-rate playback is exactly
//      affine; the residual is the amount of time-warping in the shot, in
//      simulation seconds. Also reported as the ratio between the fastest and
//      slowest instantaneous rate, which is the swing a viewer perceives.
//
//   2. PER-BALL IMPLIED RATE. For each ball at each frame, divide the distance
//      it moved on screen by the distance it moved in the simulation over the
//      same interval. That ratio IS the playback rate, measured from that one
//      ball's motion. Under a single-rate mapping every ball reports the same
//      constant. Under a time warp the ratio moves — and the interesting
//      question is whether it moves for reasons that ball had anything to do
//      with, so each excursion is labelled with the time to that BALL'S OWN
//      nearest contact.
//
//   3. FLING. The largest apparent speed-up relative to real time. A ball
//      played faster than it physically rolled is the "goes crazy" reading.
//
//   4. PX PER FRAME. How far a ball jumps between two displayed frames, in
//      logical canvas pixels, against the ball's own 23.7 px diameter. This is
//      the legibility number the presentation speed is chosen against.
//
//   npx tsx qa/time-mapping.ts [--json]

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../src/physics/table";
import { makeBall, cloneBall, type Ball } from "../src/physics/ball";
import { CUE_ID, rackEightBall } from "../src/game/rack";
import { BALL_RADIUS } from "../src/physics/constants";
import { initPhysics, simulateShotWasm } from "../src/physics/wasm-bridge";
import { generateCandidates } from "../src/ai/candidates";
import { contactTimes, interpolateBalls, type AnimTrack } from "../src/render/animate";
import { computeView } from "../src/render/renderer";
import { simulationRate } from "../src/ui/pacing";
import type { SimResult } from "../src/physics/engine";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "..");
const table = makeTable();
const view = computeView(900, 500, table);
const BALL_PX = 2 * BALL_RADIUS * view.scale;
const FRAME_SEC = 1 / 60;

/** Simulation times of the contacts that involved one specific ball. */
const contactsFor = (sim: SimResult, id: number): number[] =>
  sim.events
    .filter(
      (e) =>
        (e.kind === "ball-ball" || e.kind === "ball-cushion" || e.kind === "pocket") &&
        e.balls.includes(id),
    )
    .map((e) => e.time)
    .sort((a, b) => a - b);

interface Shot {
  name: string;
  track: AnimTrack;
  contacts: number[];
  ownContacts: Map<number, number[]>;
}

const buildShots = (): Shot[] => {
  const shots: Shot[] = [];

  const add = (name: string, balls: Ball[], action: Parameters<typeof simulateShotWasm>[1]) => {
    const sim = simulateShotWasm(balls.map(cloneBall), action);
    const wps = sim.waypoints ?? [];
    if (wps.length < 5) return;
    const own = new Map<number, number[]>();
    for (const b of balls) own.set(b.id, contactsFor(sim, b.id));
    shots.push({
      name,
      track: {
        waypoints: wps.map((w) => ({ simTime: w.time, balls: w.balls })),
        duration: sim.duration,
      },
      contacts: contactTimes(sim),
      ownContacts: own,
    });
  };

  // The break: the one shot every visitor sees first.
  add("break", rackEightBall(), { phi: 0, power: 1, sideSpin: 0, topSpin: 0 });

  // Two open boards, driven by the opponent's own candidate generator so the
  // shots are ones the product actually plays rather than ones chosen to look
  // good here.
  const boards: { name: string; balls: Ball[]; targets: number[] }[] = [
    {
      name: "open",
      balls: [
        makeBall(CUE_ID, -0.6, -0.1),
        makeBall(1, 0.3, 0.02),
        makeBall(2, 0.38, 0.1),
        makeBall(4, 0.1, -0.25),
      ],
      targets: [1, 2, 4],
    },
    {
      name: "spread",
      balls: [
        makeBall(CUE_ID, -0.75, 0.22),
        makeBall(3, 0.45, -0.28),
        makeBall(6, -0.1, 0.3),
        makeBall(9, 0.2, 0.25),
      ],
      targets: [3, 6, 9],
    },
  ];
  for (const board of boards) {
    let n = 0;
    for (const c of generateCandidates(board.balls, table, board.targets)) {
      for (const power of [c.action.power, 0.85]) {
        add(`${board.name}#${n}`, board.balls, { ...c.action, power });
        n++;
      }
      if (n >= 40) break;
    }
  }

  // A shot with NO contact at all: the cue ball rolls the length of the table
  // and stops. Nothing happens in it, which makes it the cleanest possible test
  // of whether the mapping is uniform.
  add("no-contact", [makeBall(CUE_ID, -0.8, 0)], {
    phi: Math.PI,
    power: 0.35,
    sideSpin: 0,
    topSpin: 0,
  });

  return shots;
};

interface Frame {
  wall: number;
  simTime: number;
  rate: number;
}

/** Walk one shot exactly as the animation loop does, at a fixed 60 Hz. */
const walk = (shot: Shot, speed: number): Frame[] => {
  const rate = simulationRate(speed);
  const frames: Frame[] = [];
  let simTime = 0;
  let wall = 0;
  let guard = 0;
  frames.push({ wall: 0, simTime: 0, rate });
  while (simTime < shot.track.duration && guard++ < 20000) {
    simTime = Math.min(simTime + FRAME_SEC * rate, shot.track.duration);
    wall += FRAME_SEC;
    frames.push({ wall, simTime, rate });
  }
  return frames;
};

const quantile = (sorted: number[], p: number): number =>
  sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] : 0;

interface Excursion {
  /** Playback rate implied by this ball's own screen motion. */
  implied: number;
  /** Seconds to this ball's nearest own contact. */
  toOwnContact: number;
}

const measure = (shots: Shot[], speed: number) => {
  let worstAffineResidual = 0;
  let worstAffineShot = "";
  let worstSwing = 1;
  let worstSwingShot = "";
  let fastestRate = 0;
  let fastestShot = "";

  const pxPerFrame: number[] = [];
  const excursions: Excursion[] = [];
  let screenTotal = 0;

  for (const shot of shots) {
    const frames = walk(shot, speed);
    if (frames.length < 4) continue;
    screenTotal += frames[frames.length - 1].wall;

    // 1. Least-squares fit of simTime against wall-clock elapsed.
    const n = frames.length;
    let sw = 0, ss = 0, sww = 0, sws = 0;
    for (const f of frames) {
      sw += f.wall;
      ss += f.simTime;
      sww += f.wall * f.wall;
      sws += f.wall * f.simTime;
    }
    const denom = n * sww - sw * sw;
    const a = denom === 0 ? 0 : (n * sws - sw * ss) / denom;
    const b = (ss - a * sw) / n;
    let residual = 0;
    for (const f of frames) residual = Math.max(residual, Math.abs(f.simTime - (a * f.wall + b)));
    if (residual > worstAffineResidual) {
      worstAffineResidual = residual;
      worstAffineShot = shot.name;
    }

    // Instantaneous rate swing within this one shot.
    const rates = frames.map((f) => f.rate);
    const lo = Math.min(...rates);
    const hi = Math.max(...rates);
    if (lo > 0 && hi / lo > worstSwing) {
      worstSwing = hi / lo;
      worstSwingShot = shot.name;
    }
    if (hi > fastestRate) {
      fastestRate = hi;
      fastestShot = shot.name;
    }

    // 2 & 4. Per-ball implied rate and screen displacement, frame by frame.
    let prev = interpolateBalls(shot.track, frames[0].simTime);
    for (let i = 1; i < frames.length; i++) {
      const now = interpolateBalls(shot.track, frames[i].simTime);
      const dSim = frames[i].simTime - frames[i - 1].simTime;
      const dWall = frames[i].wall - frames[i - 1].wall;
      for (const ball of now) {
        const p = prev.find((x) => x.id === ball.id);
        if (!p || ball.pocketed || p.pocketed) continue;
        const moved = Math.hypot(ball.pos.x - p.pos.x, ball.pos.y - p.pos.y);
        const px = moved * view.scale;
        if (px > 0.01) pxPerFrame.push(px);
        // Implied rate: simulation seconds consumed per wall-clock second, as
        // read off this ball. Only meaningful while the ball is actually
        // moving, so a stationary ball does not report a 0/0.
        if (moved > 1e-4 && dWall > 0) {
          const own = shot.ownContacts.get(ball.id) ?? [];
          const t = frames[i].simTime;
          let nearest = Infinity;
          for (const c of own) nearest = Math.min(nearest, Math.abs(t - c));
          excursions.push({ implied: dSim / dWall, toOwnContact: nearest });
        }
      }
      prev = now;
    }
  }

  pxPerFrame.sort((x, y) => x - y);
  const implied = excursions.map((e) => e.implied).sort((x, y) => x - y);
  // The excursions that a viewer cannot explain: the playback rate is far from
  // the shot's mean, and this ball had no contact anywhere near this instant.
  const mean = implied.reduce((s, v) => s + v, 0) / (implied.length || 1);
  const unexplained = excursions.filter(
    (e) => Math.abs(e.implied - mean) / mean > 0.1 && e.toOwnContact > 0.35,
  );

  return {
    speed,
    shots: shots.length,
    meanScreenSec: +(screenTotal / shots.length).toFixed(2),
    affine: {
      worstResidualSimSec: +worstAffineResidual.toFixed(6),
      worstShot: worstAffineShot,
    },
    rate: {
      worstWithinShotSwing: +worstSwing.toFixed(2),
      worstSwingShot,
      fastestRate: +fastestRate.toFixed(2),
      fastestShot,
    },
    impliedRate: {
      min: +quantile(implied, 0).toFixed(3),
      p50: +quantile(implied, 0.5).toFixed(3),
      max: +quantile(implied, 1).toFixed(3),
      spread: +(quantile(implied, 1) / Math.max(1e-9, quantile(implied, 0))).toFixed(2),
    },
    unexplainedFrames: {
      count: unexplained.length,
      fraction: +(unexplained.length / (excursions.length || 1)).toFixed(4),
    },
    pxPerFrame: {
      ballPx: +BALL_PX.toFixed(1),
      p50: +quantile(pxPerFrame, 0.5).toFixed(1),
      p90: +quantile(pxPerFrame, 0.9).toFixed(1),
      p99: +quantile(pxPerFrame, 0.99).toFixed(1),
      max: +quantile(pxPerFrame, 1).toFixed(1),
      overOneBall: +(pxPerFrame.filter((d) => d > BALL_PX).length / (pxPerFrame.length || 1)).toFixed(4),
    },
  };
};

const main = async () => {
  await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
  const shots = buildShots();
  const out = [0.35, 0.5, 0.6, 0.75, 1].map((s) => measure(shots, s));

  if (process.argv.includes("--json")) {
    console.log(JSON.stringify({ shots: shots.length, results: out }, null, 2));
    return;
  }

  console.log(`shots: ${shots.length}   ball diameter: ${BALL_PX.toFixed(1)} logical px\n`);
  const cols = [
    "speed",
    "screen s",
    "affine resid",
    "rate swing",
    "fastest rate",
    "implied min",
    "implied max",
    "unexplained",
    "px/f p90",
    "px/f max",
  ];
  console.log(cols.map((c) => c.padStart(14)).join(""));
  for (const r of out) {
    console.log(
      [
        `${r.speed}x`,
        r.meanScreenSec.toFixed(2),
        r.affine.worstResidualSimSec.toFixed(4),
        `${r.rate.worstWithinShotSwing.toFixed(2)}x`,
        `${r.rate.fastestRate.toFixed(2)}x`,
        r.impliedRate.min.toFixed(3),
        r.impliedRate.max.toFixed(3),
        `${(r.unexplainedFrames.fraction * 100).toFixed(1)}%`,
        r.pxPerFrame.p90.toFixed(1),
        r.pxPerFrame.max.toFixed(1),
      ]
        .map((s) => s.padStart(14))
        .join(""),
    );
  }
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
