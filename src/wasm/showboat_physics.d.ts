/* tslint:disable */
/* eslint-disable */

/**
 * Result of a single shot returned to JS. Events are packed into a flat i32
 * array (kind, ball_a, ball_b, pocket) plus a parallel f64 time array so the
 * overlay can reconstruct the trace without a struct-of-arrays dance in JS.
 */
export class ShotResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly ballCount: number;
    readonly balls: Float64Array;
    readonly duration: number;
    readonly eventBalls: Int32Array;
    readonly eventCushions: any[];
    readonly eventKinds: Int32Array;
    readonly eventPockets: Int32Array;
    readonly eventTimes: Float64Array;
    readonly firstContact: number;
    readonly pocketed: Uint8Array;
    readonly waypointBalls: Float64Array;
    readonly waypointTimes: Float64Array;
}

/**
 * Evaluate a candidate shot with the native rollout hot loop. Returns the mean
 * target-balls-pocketed value across `n_rollouts` playouts of `depth` shots.
 * This is the pure-search baseline value the MCTS uses before the ONNX net
 * supplies a learned value.
 */
export function rolloutValue(flat: Float64Array, phi: number, power: number, side_spin: number, top_spin: number, targets: Uint8Array, depth: number, n_rollouts: number, seed: number): number;

/**
 * Simulate one shot to its resting state. `flat` is the ball state; the cue
 * action is applied to ball id 0. Returns the final state + event trace.
 */
export function simulateShot(flat: Float64Array, phi: number, power: number, side_spin: number, top_spin: number): ShotResult;

/**
 * Table geometry constants exposed so the TS renderer/overlay never hardcodes
 * numbers that could drift from the physics core.
 */
export function tableDims(): Float64Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_shotresult_free: (a: number, b: number) => void;
    readonly rolloutValue: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => number;
    readonly shotresult_ballCount: (a: number) => number;
    readonly shotresult_balls: (a: number) => [number, number];
    readonly shotresult_duration: (a: number) => number;
    readonly shotresult_eventBalls: (a: number) => [number, number];
    readonly shotresult_eventCushions: (a: number) => [number, number];
    readonly shotresult_eventKinds: (a: number) => [number, number];
    readonly shotresult_eventPockets: (a: number) => [number, number];
    readonly shotresult_eventTimes: (a: number) => [number, number];
    readonly shotresult_firstContact: (a: number) => number;
    readonly shotresult_pocketed: (a: number) => [number, number];
    readonly shotresult_waypointBalls: (a: number) => [number, number];
    readonly shotresult_waypointTimes: (a: number) => [number, number];
    readonly simulateShot: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly tableDims: () => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_drop_slice: (a: number, b: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
