// Presentation speed changes the screen and nothing else.
//
// That is a claim about a change to a shipped animation, so it is proved three
// ways rather than asserted once:
//
//  1. **Reparameterisation.** Playback maps wall-clock onto simulation time by
//     one multiplication. Two speeds that reach the same simulation time
//     produce the identical ball states, and every speed ends at the identical
//     resting state, which is the outcome the game commits.
//  2. **Reach.** The speed value is read in exactly the two animation loops and
//     nowhere near a simulation, a candidate action, or the game rules — checked
//     against the source, so a future edit that widens its reach fails here.
//  3. **Physics constants.** Every exported constant of `physics/constants.ts`
//     is pinned to its committed value. A pacing change that moved one — the
//     tempting way to "slow the shot down" — cannot pass.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { join, dirname, relative } from "node:path";
import { fileURLToPath } from "node:url";

import { initPhysics, simulateShotWasm } from "../physics/wasm-bridge";
import { makeBall, type Ball } from "../physics/ball";
import { CUE_ID } from "../game/rack";
import { interpolateBalls, lastContactSec, type AnimTrack } from "../render/animate";
import type { SimResult } from "../physics/engine";
import * as C from "../physics/constants";
import {
  DEFAULT_PLAYBACK_SPEED,
  PLAYBACK_SPEEDS,
  PLAYBACK_SPEED_KEY,
  PLAYBACK_SPEED_LABEL,
  SETTLE_MAX_SEC,
  loadPlaybackSpeed,
  savePlaybackSpeed,
  settleRate,
} from "./playbackSpeed";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../..");
const SRC = join(APP_ROOT, "src");

let sim: SimResult;
let track: AnimTrack;

const posOf = (balls: Ball[]) =>
  balls.map((b) => ({ id: b.id, x: b.pos.x, y: b.pos.y, pocketed: b.pocketed }));

describe("playback speed is presentation only", () => {
  beforeAll(async () => {
    await initPhysics(readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm")));
    const balls = [makeBall(CUE_ID, -0.7, 0.0), makeBall(3, 0.2, 0.05), makeBall(5, 0.5, -0.3)];
    sim = simulateShotWasm(balls, { phi: 0.06, power: 0.85, sideSpin: 0, topSpin: 0 });
    track = {
      waypoints: (sim.waypoints ?? []).map((wp) => ({ simTime: wp.time, balls: wp.balls })),
      duration: sim.duration,
    };
    expect(track.waypoints.length).toBeGreaterThan(2);
  }, 60_000);

  it("two speeds that reach the same simulation time show the same table", () => {
    // The loops integrate `simTime += dt * speed`. Feed each speed the wall
    // clock it needs to arrive at the same simulation instant and the frames
    // must be identical — not merely close.
    for (const at of [0.05, 0.2, 0.5, 0.9]) {
      const target = track.duration * at;
      const frames = PLAYBACK_SPEEDS.map((speed) => {
        const wallSeconds = target / speed;
        // The reconstruction the loop performs, at that speed.
        return posOf(interpolateBalls(track, wallSeconds * speed));
      });
      for (let i = 1; i < frames.length; i++) expect(frames[i]).toEqual(frames[0]);
    }
  });

  it("every speed ends at the simulation's own resting state", () => {
    const resting = posOf(sim.balls);
    // Run each speed the way the loop does — 60 Hz frames, integrating
    // `dt * speed` — until it reaches the end, and look at the last frame.
    for (const speed of PLAYBACK_SPEEDS) {
      let t = 0;
      let frames = 0;
      while (t < track.duration && frames < 100_000) {
        t += (1 / 60) * speed;
        frames++;
      }
      const last = posOf(interpolateBalls(track, t));
      expect(frames, `speed ${speed} never finished`).toBeLessThan(100_000);
      for (const b of last) {
        const truth = resting.find((r) => r.id === b.id)!;
        expect(b.pocketed, `speed ${speed}: ball ${b.id} pocketed`).toBe(truth.pocketed);
        if (truth.pocketed) continue;
        expect(
          Math.hypot(b.x - truth.x, b.y - truth.y),
          `speed ${speed}: ball ${b.id} settles elsewhere`,
        ).toBeLessThan(1e-6);
      }
      // Slower means more frames, which is the entire difference between them.
      expect(frames).toBe(Math.ceil(track.duration / ((1 / 60) * speed)));
    }
  });

  it("a speed change part-way through does not move the balls", () => {
    // The loops integrate rather than recompute `elapsed * speed`, so switching
    // speed at a given instant continues from where the balls are. Modelled
    // here exactly as the loops do it.
    const step = 1 / 60;
    const integrate = (speeds: number[]): number => {
      let t = 0;
      for (const s of speeds) t += step * s;
      return t;
    };
    // Compared to a micron rather than bit-for-bit: summing the same total in
    // a different order differs in the last bits of a double, and a claim that
    // it does not would be a claim about IEEE 754 rather than about the game.
    const same = (t1: number, t2: number) => {
      const A = posOf(interpolateBalls(track, t1));
      const B = posOf(interpolateBalls(track, t2));
      expect(A.length).toBe(B.length);
      for (let i = 0; i < A.length; i++) {
        expect(A[i].id).toBe(B[i].id);
        expect(A[i].pocketed).toBe(B[i].pocketed);
        expect(Math.hypot(A[i].x - B[i].x, A[i].y - B[i].y)).toBeLessThan(1e-6);
      }
    };

    // 30 frames at 1.0x then 30 at 0.35x reaches the same place as the reverse.
    const a = integrate([...Array(30).fill(1), ...Array(30).fill(0.35)]);
    const b = integrate([...Array(30).fill(0.35), ...Array(30).fill(1)]);
    expect(a).toBeCloseTo(b, 12);
    same(a, b);

    // And the frame at the moment of the switch is continuous: the state at
    // t is the state at t, regardless of how the clock got there.
    same(integrate(Array(30).fill(1)), integrate(Array(60).fill(0.5)));
  });

  it("skipping to the end lands on the same state as watching it", () => {
    // What `skip` does: `simTime = track.duration`. It must not be a different
    // outcome from letting the animation run.
    expect(posOf(interpolateBalls(track, track.duration))).toEqual(
      posOf(interpolateBalls(track, track.duration * 4)),
    );
  });

  it("the speed is read in the animation loops and reaches nothing else", () => {
    const files: string[] = [];
    const walk = (dir: string): void => {
      for (const e of readdirSync(dir, { withFileTypes: true })) {
        const p = join(dir, e.name);
        if (e.isDirectory()) {
          if (e.name === "wasm" || e.name === "node_modules") continue;
          walk(p);
        } else if (/\.tsx?$/.test(e.name) && !/\.test\.tsx?$/.test(e.name)) {
          files.push(p);
        }
      }
    };
    walk(SRC);

    // Comments are stripped first. `render/presentation.ts` *mentions* the
    // module in a header note explaining where the rate comes from, which is
    // documentation, not a read — counting it would make this test unable to
    // tell the two apart.
    const codeOf = (p: string) =>
      readFileSync(p, "utf8")
        .replace(/\/\*[\s\S]*?\*\//g, " ")
        .split("\n")
        .filter((l) => !/^\s*(\/\/|\*)/.test(l))
        .join("\n");
    const readers = files
      .filter((p) => /\b(speedRef|playbackSpeed|PlaybackSpeed|PLAYBACK_SPEED)/.test(codeOf(p)))
      .map((p) => relative(SRC, p))
      .sort();
    // The module, the two hosts that own an animation loop, and the hook seam
    // that carries the value between them. Nothing in `ai/`, `physics/` or
    // `game/`.
    expect(readers).toEqual(["App.tsx", "ui/playbackSpeed.ts", "ui/useAiTurn.ts"]);
    for (const r of readers) {
      expect(r.startsWith("ai/"), `${r} reads the presentation speed`).toBe(false);
      expect(r.startsWith("physics/"), `${r} reads the presentation speed`).toBe(false);
      expect(r.startsWith("game/"), `${r} reads the presentation speed`).toBe(false);
    }
  });

  it("the animation is the only thing the speed multiplies", () => {
    // Both loops must scale a wall-clock delta and nothing else. If a future
    // edit multiplies an action, a power or a duration by it, the shape below
    // stops matching.
    for (const name of ["ui/useAiTurn.ts", "App.tsx"]) {
      const src = readFileSync(join(SRC, name), "utf8");
      const uses = src.match(/speedRef\.current/g) ?? [];
      expect(uses.length, `${name} does not read the speed`).toBeGreaterThan(0);
      for (const m of src.matchAll(/[^\n]*speedRef\.current[^\n]*/g)) {
        const line = m[0].trim();
        // Three legal shapes and no others: storing the live value into the
        // ref, and the two halves of the rate the loop integrates.
        const ok =
          /^speedRef\.current = playbackSpeed;$/.test(line) ||
          /^simTime < contactEnd \? speedRef\.current : settleRate\(tailSec, speedRef\.current\);$/.test(line) ||
          /^const rate = simTime < contactEnd \? speedRef\.current : settleRate\(tailSec, speedRef\.current\);$/.test(line);
        expect(ok, `${name}: unexpected use of the speed: ${line}`).toBe(true);
      }
      // And the integration itself is a wall-clock delta times that rate.
      expect(src).toContain("simTime += ((now - last) / 1000) * rate;");
    }
  });

  it("the settle is capped and never slower than the speed asked for", () => {
    // 37% of a median shot is coasting after the last contact. The chosen speed
    // governs the part with contacts in it; this is the rest.
    for (const speed of PLAYBACK_SPEEDS) {
      for (const tail of [0, 0.2, 0.9, 1.99, 3.72, 5.9]) {
        const r = settleRate(tail, speed);
        // Asking for slow motion must never make the coast faster than asked
        // when the coast is already short.
        expect(r, `speed ${speed}, tail ${tail}`).toBeGreaterThanOrEqual(speed);
        // And a long coast always fits inside the cap.
        const screenSec = tail === 0 ? 0 : tail / r;
        expect(screenSec, `speed ${speed}, tail ${tail}`).toBeLessThanOrEqual(SETTLE_MAX_SEC + 1e-9);
      }
    }
    // A shot with no contact at all has nothing to slow down for and settles
    // whole at the chosen speed.
    expect(settleRate(0, 0.6)).toBe(0.6);
  });

  it("the two-phase rate lands the median shot where the module claims", () => {
    // The numbers in `playbackSpeed.ts`'s header, recomputed. If the cap or the
    // default moves, the comment stops being true and this fails.
    const medianTotal = 5.39;
    const medianContactEnd = 3.04;
    const tail = medianTotal - medianContactEnd;
    const screen =
      medianContactEnd / DEFAULT_PLAYBACK_SPEED + tail / settleRate(tail, DEFAULT_PLAYBACK_SPEED);
    expect(screen).toBeGreaterThan(5.9);
    expect(screen).toBeLessThan(6.3);
    // Longer than the old flat 1.0x, but not by the 67% a flat 0.6x would cost.
    expect(screen / medianTotal).toBeLessThan(1.2);
    expect(medianContactEnd / DEFAULT_PLAYBACK_SPEED / medianContactEnd).toBeCloseTo(1 / 0.6, 6);
  });

  it("the last contact is read off the event log, and a shot with none has no slow part", () => {
    expect(lastContactSec(sim)).toBeGreaterThan(0);
    const marked = sim.events.filter((e) =>
      ["ball-ball", "ball-cushion", "pocket"].includes(e.kind),
    );
    expect(lastContactSec(sim)).toBe(marked[marked.length - 1].time);
    expect(lastContactSec({ ...sim, events: [] })).toBe(0);
    // `stop` is not a contact.
    expect(lastContactSec({ ...sim, events: [{ time: 9, kind: "stop", balls: [] }] })).toBe(0);
  });

  it("no simulation, action or rule is reachable from the speed module", () => {
    const src = readFileSync(join(SRC, "ui/playbackSpeed.ts"), "utf8");
    for (const banned of ["simulate", "takeShot", "CueAction", "physics/", "game/", "ai/"]) {
      expect(src, `playbackSpeed.ts references ${banned}`).not.toContain(banned);
    }
  });

  it("the physics constants are exactly what they were", () => {
    // A pinned snapshot, not a range. Slowing a shot down by shaving friction
    // would change every simulated outcome in the app, and it is the obvious
    // wrong way to solve "the animation is too fast".
    expect({
      G: C.G,
      BALL_RADIUS: C.BALL_RADIUS,
      BALL_DIAMETER: C.BALL_DIAMETER,
      BALL_MASS: C.BALL_MASS,
      BALL_INERTIA: C.BALL_INERTIA,
      MU_SLIDING: C.MU_SLIDING,
      MU_ROLLING: C.MU_ROLLING,
      MU_SPINNING: C.MU_SPINNING,
      E_BALL_BALL: C.E_BALL_BALL,
      E_BALL_CUSHION: C.E_BALL_CUSHION,
      MU_BALL_BALL: C.MU_BALL_BALL,
      MU_BALL_CUSHION: C.MU_BALL_CUSHION,
      CUSHION_HEIGHT_FRACTION: C.CUSHION_HEIGHT_FRACTION,
      TABLE_LENGTH: C.TABLE_LENGTH,
      TABLE_WIDTH: C.TABLE_WIDTH,
      CORNER_POCKET_RADIUS: C.CORNER_POCKET_RADIUS,
      SIDE_POCKET_RADIUS: C.SIDE_POCKET_RADIUS,
      STOP_SPEED: C.STOP_SPEED,
      STOP_SPIN: C.STOP_SPIN,
      EPS: C.EPS,
    }).toEqual({
      G: 9.8,
      BALL_RADIUS: 0.028575,
      BALL_DIAMETER: 0.05715,
      BALL_MASS: 0.17,
      BALL_INERTIA: (2 / 5) * 0.17 * 0.028575 * 0.028575,
      MU_SLIDING: 0.2,
      MU_ROLLING: 0.01,
      MU_SPINNING: 0.044,
      E_BALL_BALL: 0.95,
      E_BALL_CUSHION: 0.85,
      MU_BALL_BALL: 0.06,
      MU_BALL_CUSHION: 0.2,
      CUSHION_HEIGHT_FRACTION: 0.635,
      TABLE_LENGTH: 1.9812,
      TABLE_WIDTH: 0.9906,
      CORNER_POCKET_RADIUS: 0.0605,
      SIDE_POCKET_RADIUS: 0.055,
      STOP_SPEED: 0.005,
      STOP_SPIN: 0.05,
      EPS: 1e-9,
    });
  });
});

describe("the speed control itself", () => {
  it("offers a short list, slowest first, with a labelled default in range", () => {
    expect([...PLAYBACK_SPEEDS]).toEqual([0.35, 0.6, 1]);
    expect(PLAYBACK_SPEEDS).toContain(DEFAULT_PLAYBACK_SPEED);
    // The brief's window for a first-play default that can be followed without
    // turning every shot into an event.
    expect(DEFAULT_PLAYBACK_SPEED).toBeGreaterThanOrEqual(0.5);
    expect(DEFAULT_PLAYBACK_SPEED).toBeLessThanOrEqual(0.65);
    // Named as a property of the screen, never of the simulation.
    expect(PLAYBACK_SPEED_LABEL).toBe("presentation speed");
  });

  it("persists for the session and rejects anything it did not offer", () => {
    const store = new Map<string, string>();
    (globalThis as unknown as { sessionStorage: Storage }).sessionStorage = {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
      removeItem: (k: string) => void store.delete(k),
      clear: () => store.clear(),
      key: () => null,
      length: 0,
    } as unknown as Storage;

    expect(loadPlaybackSpeed()).toBe(DEFAULT_PLAYBACK_SPEED);
    savePlaybackSpeed(0.35);
    expect(store.get(PLAYBACK_SPEED_KEY)).toBe("0.35");
    expect(loadPlaybackSpeed()).toBe(0.35);

    for (const junk of ["0", "-1", "12", "fast", "", "NaN"]) {
      store.set(PLAYBACK_SPEED_KEY, junk);
      expect(loadPlaybackSpeed(), `"${junk}" was accepted as a speed`).toBe(
        DEFAULT_PLAYBACK_SPEED,
      );
    }
    delete (globalThis as unknown as { sessionStorage?: Storage }).sessionStorage;
  });

  it("survives storage being unavailable, which is a real browser state", () => {
    (globalThis as unknown as { sessionStorage: Storage }).sessionStorage = {
      getItem: () => {
        throw new Error("storage is disabled");
      },
      setItem: () => {
        throw new Error("storage is disabled");
      },
    } as unknown as Storage;
    expect(loadPlaybackSpeed()).toBe(DEFAULT_PLAYBACK_SPEED);
    expect(() => savePlaybackSpeed(1)).not.toThrow();
    delete (globalThis as unknown as { sessionStorage?: Storage }).sessionStorage;
  });
});
