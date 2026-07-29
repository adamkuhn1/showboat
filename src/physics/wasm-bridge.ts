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
import { type Ball, Motion } from "./ball";
import { type SimResult, type ShotEvent, type ShotEventKind } from "./engine";
import { type CueAction } from "./cue";

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

const KIND: ShotEventKind[] = ["ball-ball", "ball-cushion", "pocket", "stop"];

// Run a full shot through the WASM engine. `initPhysics()` must have resolved.
// Mutates `balls` to the resting state and returns the reconstructed SimResult.
export const simulateShotWasm = (
  balls: Ball[],
  action: CueAction,
): SimResult => {
  const flat = flattenBalls(balls);
  const res = wasmSimulateShot(
    flat,
    action.phi,
    action.power,
    action.sideSpin,
    action.topSpin,
  );

  applyFlat(balls, res.balls);

  const kinds = res.eventKinds;
  const eballs = res.eventBalls; // pairs
  const pockets = res.eventPockets;
  const cushions = res.eventCushions as string[];
  const times = res.eventTimes;

  const events: ShotEvent[] = [];
  for (let k = 0; k < kinds.length; k++) {
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

  // free the wasm-owned struct.
  res.free();

  return {
    balls,
    events,
    pocketed,
    firstContact,
    duration: res.duration,
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
