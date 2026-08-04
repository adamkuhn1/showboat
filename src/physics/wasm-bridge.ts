// Typed bridge from the TypeScript game/UI layer to the Rust→WASM physics core.
//
// The heavy simulation and the MCTS rollout hot loop live in Rust (see
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
} from "../wasm/showboat_physics";
import wasmUrl from "../wasm/showboat_physics_bg.wasm?url";
import { type Ball, Motion, classifyMotion } from "./ball";
import { type SimResult, type SimWaypoint, type ShotEvent, type ShotEventKind } from "./engine";
import { type CueAction } from "./cue";
import { BALL_RADIUS } from "./constants";

const STRIDE = 8;

let ready: Promise<void> | null = null;

// Initialize the WASM module exactly once. Vite resolves the ?url import to the
// hashed asset path, so this works both standalone and embedded in the shell.
export const initPhysics = (): Promise<void> => {
  if (!ready) {
    ready = init({ module_or_path: wasmUrl }).then(() => undefined);
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

// After a WASM sim, the Rust engine can leave balls in a numerically overlapping
// state (distance < 2*BALL_RADIUS) due to floating-point precision at high event
// counts. Any overlap causes the NEXT simulation that uses this state to generate
// thousands of immediate ball-ball events before it can terminate, making the
// search seeding phase extremely slow. Separate overlapping pairs by the minimum
// distance needed to bring them to exact contact — this is a physics correction,
// not a foul or position change visible in gameplay.
// Exported so the MCTS layer can pre-sanitise the game-state copy before any
// WASM call (the TS engine's break simulation can leave balls numerically at
// contact distance; passing such a state to Rust causes thousands of t≈0
// collision events and multi-second hangs).
export const separateOverlaps = (balls: Ball[]): number => {
  // 2 mm clearance (≈7% of ball diameter) gives Rust's event-detection enough
  // room that even after a collision the balls won't immediately re-overlap on
  // the next timestep.  1e-5 m (0.01 mm) was too tight.
  const MIN = 2 * BALL_RADIUS + 0.002;
  const live = balls.filter((b) => !b.pocketed);
  // Iterate until fully converged: a single pass is not enough for tight clusters
  // (pushing pair A-B can re-overlap A with C if they share ball A).  O(n²) per
  // pass, n≤15, so 10–20 passes ≈ 2250–4500 comparisons — negligible cost in JS.
  const MAX_PASSES = 20;
  let pass = 0;
  for (; pass < MAX_PASSES; pass++) {
    let anyOverlap = false;
    for (let i = 0; i < live.length; i++) {
      for (let j = i + 1; j < live.length; j++) {
        const a = live[i], b = live[j];
        const dx = b.pos.x - a.pos.x;
        const dy = b.pos.y - a.pos.y;
        const dist = Math.hypot(dx, dy);
        if (dist < MIN) {
          anyOverlap = true;
          const overlap = MIN - dist;
          if (dist < 1e-9) {
            // Coincident: push apart along x
            a.pos = { x: a.pos.x - MIN / 2, y: a.pos.y };
            b.pos = { x: b.pos.x + MIN / 2, y: b.pos.y };
          } else {
            const nx = dx / dist, ny = dy / dist;
            a.pos = { x: a.pos.x - nx * overlap / 2, y: a.pos.y - ny * overlap / 2 };
            b.pos = { x: b.pos.x + nx * overlap / 2, y: b.pos.y + ny * overlap / 2 };
          }
        }
      }
    }
    if (!anyOverlap) break;
  }
  return pass; // number of passes needed (0 = no overlaps found)
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
  // Fix any overlapping balls in the INPUT before handing to Rust.  The caller
  // may have cloned state from the TS engine (e.g. post-break) where balls
  // ended up numerically touching; without this Rust generates thousands of
  // t=0 events and the call blocks for 10-30 s.
  separateOverlaps(balls);
  const flat = flattenBalls(balls);
  const res = wasmSimulateShot(
    flat,
    action.phi,
    action.power,
    action.sideSpin,
    action.topSpin,
  );

  applyFlat(balls, res.balls);
  separateOverlaps(balls);

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
  separateOverlaps(balls);
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
