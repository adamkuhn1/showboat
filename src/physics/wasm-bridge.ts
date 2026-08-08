// Typed bridge from the TypeScript game/UI layer to the Rust→WASM physics core.
//
// The heavy simulation and the UCB search rollout hot loop live in Rust (see
// physics-core/); this module is the thin boundary that (de)serializes ball
// state across the flat Float64Array interface and re-materializes the event
// trace into the same ShotEvent/SimResult shapes the TS engine produced, so the
// rest of the app is agnostic to which engine ran.
//
// The pure-TS engine in ./engine.ts remains the reference implementation and the
// vitest oracle; this bridge is what ships to the browser for speed and for the
// native rollout loop.

import init, {
  simulateShot as wasmSimulateShot,
  rolloutValue as wasmRolloutValue,
} from "../wasm/showboat_physics.js";
import { type Ball, Motion, classifyMotion } from "./ball";
import { type SimResult, type SimWaypoint, type ShotEvent, type ShotEventKind } from "./engine";
import { type CueAction } from "./cue";
import { BALL_RADIUS } from "./constants";

const STRIDE = 8;

let ready: Promise<void> | null = null;

// Initialize the WASM module exactly once. Vite resolves the ?url import to the
// hashed asset path, so this works both standalone and embedded in the shell.
//
// `source` lets a non-Vite caller (the Node/tsx headless dataset generator in
// training/ranker/gen_dataset.ts) supply the same .wasm bytes directly via
// fs.readFileSync instead of the Vite-only `?url` import — so training data
// and the browser build load the literal same compiled artifact through the
// literal same init/simulate code path, not a re-implementation of it.
//
// The `?url` import is loaded with a *dynamic* import, evaluated only on the
// no-`source` (browser) path. A static top-level `?url` import is eagerly
// resolved at module-load time even when unused, and Node/tsx's ESM loader
// doesn't understand Vite's `?url` query convention — it tries to load the
// referenced .wasm file itself as a native WASM ES module and fails. Vite
// still statically analyzes this dynamic import (the specifier is a literal
// string) and code-splits/hashes the asset exactly as it would a static one.
export const initPhysics = (source?: BufferSource | string): Promise<void> => {
  if (!ready) {
    ready = (async () => {
      const modulePath =
        source ?? (await import("../wasm/showboat_physics_bg.wasm?url")).default;
      await init({ module_or_path: modulePath });
    })();
  }
  return ready;
};

// Flatten ball state into the [id, px, py, vx, vy, wz, rx, ry] layout. Pocketed
// balls report NaN x (the sentinel the Rust side reads).
export const flattenBalls = (balls: Ball[]): Float64Array => {
  const out = new Float64Array(balls.length * STRIDE);
  balls.forEach((b, i) => {
    const o = i * STRIDE;
    out[o] = b.id;
    out[o + 1] = b.pocketed ? NaN : b.pos.x;
    out[o + 2] = b.pos.y;
    out[o + 3] = b.vel.x;
    out[o + 4] = b.vel.y;
    out[o + 5] = b.wz;
    out[o + 6] = b.roll.x;
    out[o + 7] = b.roll.y;
  });
  return out;
};

// Ball separation is the Rust engine's job, not this bridge's.
//
// `simulate_shot` (physics-core/src/engine.rs) runs a multi-pass separation
// before its first event scan, again after every ball-ball resolve, and once
// more at the end, so no input state can produce the t≈0 event storm that used
// to block a call for 10–30 s. Measured in `engineInput.test.ts`: a rack at
// exact contact distance, a rack penetrating by 1 mm and fifteen coincident
// balls all simulate in ~100 ms. Separating on this side as well would only
// move the caller's balls before a simulation that is about to do the same
// thing to its own copy.
//
// What stays here is a guard on the state that becomes the committed board:
// a pair that is genuinely interpenetrating is pushed to touching distance and
// no further. Expanding a pair that is merely close would move balls the
// simulation never moved, which is visible on the felt and cumulative across
// shots.

/** Overlap below this is float noise on a metre-scale coordinate, not penetration. */
const PENETRATION_EPS = 1e-9;

/**
 * Push genuinely interpenetrating pairs apart to exactly touching.
 *
 * Symmetric (each ball moves half the overlap) and iterated, because resolving
 * A–B can push A into C. Returns the number of passes used; 0 means nothing was
 * interpenetrating, which is the normal case.
 *
 * The pass budget is sized against the worst board that can exist: all fifteen
 * object balls racked and penetrating by a full millimetre on every edge
 * converges in 33 passes (`engineInput.test.ts`). O(n²) per pass with n ≤ 16.
 */
export const resolvePenetration = (balls: Ball[]): number => {
  const TOUCHING = 2 * BALL_RADIUS;
  const live = balls.filter((b) => !b.pocketed);
  const MAX_PASSES = 48;
  let pass = 0;
  for (; pass < MAX_PASSES; pass++) {
    let any = false;
    for (let i = 0; i < live.length; i++) {
      for (let j = i + 1; j < live.length; j++) {
        const a = live[i], b = live[j];
        const dx = b.pos.x - a.pos.x;
        const dy = b.pos.y - a.pos.y;
        const dist = Math.hypot(dx, dy);
        const overlap = TOUCHING - dist;
        if (overlap <= PENETRATION_EPS) continue;
        any = true;
        // Coincident centres have no separating direction; +x is arbitrary and
        // only has to be consistent.
        const [nx, ny] = dist > PENETRATION_EPS ? [dx / dist, dy / dist] : [1, 0];
        a.pos = { x: a.pos.x - (nx * overlap) / 2, y: a.pos.y - (ny * overlap) / 2 };
        b.pos = { x: b.pos.x + (nx * overlap) / 2, y: b.pos.y + (ny * overlap) / 2 };
      }
    }
    if (!any) break;
  }
  return pass;
};

// Write a flat array back onto an existing ball list (preserving id order).
const applyFlat = (balls: Ball[], flat: Float64Array): void => {
  balls.forEach((b, i) => {
    const o = i * STRIDE;
    const px = flat[o + 1];
    b.pocketed = Number.isNaN(px);
    if (!b.pocketed) {
      b.pos = { x: px, y: flat[o + 2] };
    }
    b.vel = { x: flat[o + 3], y: flat[o + 4] };
    b.wz = flat[o + 5];
    b.roll = { x: flat[o + 6], y: flat[o + 7] };
    b.motion = b.pocketed ? Motion.Stationary : Motion.Sliding;
  });
};

// Build a fresh Ball[] from a flat waypoint block — unlike applyFlat (which
// mutates an existing array in place), each waypoint snapshot needs its own
// independent ball objects.
const ballsFromFlat = (flat: Float64Array, offset: number, count: number): Ball[] => {
  const out: Ball[] = [];
  for (let i = 0; i < count; i++) {
    const o = offset + i * STRIDE;
    const px = flat[o + 1];
    const pocketed = Number.isNaN(px);
    const b: Ball = {
      id: flat[o],
      pos: { x: pocketed ? 0 : px, y: flat[o + 2] },
      vel: { x: flat[o + 3], y: flat[o + 4] },
      wz: flat[o + 5],
      roll: { x: flat[o + 6], y: flat[o + 7] },
      motion: Motion.Stationary,
      pocketed,
    };
    b.motion = pocketed ? Motion.Stationary : classifyMotion(b);
    out.push(b);
  }
  return out;
};

const KIND: ShotEventKind[] = ["ball-ball", "ball-cushion", "pocket", "stop"];

// Run a full shot through the WASM engine. `initPhysics()` must have resolved.
// Mutates `balls` to the resting state and returns the reconstructed SimResult.
export const simulateShotWasm = (
  balls: Ball[],
  action: CueAction,
): SimResult => {
  // The input goes to Rust exactly as the caller holds it; the engine does its
  // own separation on its own copy (see the note above `resolvePenetration`).
  const flat = flattenBalls(balls);
  const res = wasmSimulateShot(
    flat,
    action.phi,
    action.power,
    action.sideSpin,
    action.topSpin,
  );

  applyFlat(balls, res.balls);
  // The engine's own final pass already leaves no pair interpenetrating, so on
  // every measured shot this is a no-op. It stays because the engine can also
  // exit on its iteration guard, and a resting board with one ball inside
  // another is the one state the next shot must never start from.
  resolvePenetration(balls);

  const kinds = res.eventKinds;
  const eballs = res.eventBalls; // pairs
  const pockets = res.eventPockets;
  const cushions = res.eventCushions as string[];
  const times = res.eventTimes;

  // Cap event parsing: a degenerate simulation can return tens of thousands of
  // events (two balls bouncing to the Rust engine's internal collision ceiling).
  // We never need more than ~200 events to compute potsTarget / rail counts /
  // shot description — pockets always happen within the first ~50 events for any
  // real shot.  Capping protects the JS parsing loop from multi-second stalls.
  const MAX_EVENTS = 200;
  const events: ShotEvent[] = [];
  for (let k = 0; k < Math.min(kinds.length, MAX_EVENTS); k++) {
    const kind = KIND[kinds[k]];
    const a = eballs[k * 2];
    const b = eballs[k * 2 + 1];
    const ev: ShotEvent = {
      time: times[k],
      kind,
      balls: b >= 0 ? [a, b] : a >= 0 ? [a] : [],
    };
    if (kind === "ball-cushion") ev.cushion = cushions[k];
    if (kind === "pocket") ev.pocket = pocketIdToName(pockets[k]);
    events.push(ev);
  }

  const pocketed = Array.from(res.pocketed);
  const firstContact = res.firstContact >= 0 ? res.firstContact : null;
  // Extract duration/waypoints before free() — accessing wasm-bindgen
  // properties on a freed struct throws "null pointer passed to rust".
  const duration = res.duration;
  const waypointTimes = res.waypointTimes;
  const waypointBalls = res.waypointBalls;
  const ballCount = res.ballCount;

  const waypoints: SimWaypoint[] = [];
  const blockSize = ballCount * STRIDE;
  for (let w = 0; w < waypointTimes.length; w++) {
    waypoints.push({
      time: waypointTimes[w],
      balls: ballsFromFlat(waypointBalls, w * blockSize, ballCount),
    });
  }

  res.free();

  return {
    balls,
    events,
    pocketed,
    firstContact,
    duration,
    waypoints,
  };
};

// The Rust side returns numeric pocket ids (0..5); map them to the string ids
// the TS renderer/trace use.
const POCKET_NAMES = ["bl", "tl", "br", "tr", "sb", "st"];
const pocketIdToName = (id: number): string => POCKET_NAMES[id] ?? String(id);

// Native rollout hot loop: mean target-balls-pocketed value for a candidate.
export const rolloutValueWasm = (
  balls: Ball[],
  action: CueAction,
  targets: number[],
  depth: number,
  nRollouts: number,
  seed: number,
): number => {
  const flat = flattenBalls(balls);
  return wasmRolloutValue(
    flat,
    action.phi,
    action.power,
    action.sideSpin,
    action.topSpin,
    new Uint8Array(targets),
    depth,
    nRollouts,
    seed,
  );
};

export { STRIDE };
