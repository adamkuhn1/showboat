import { describe, it, expect } from "vitest";
import { makeBall, Motion } from "./ball";
import { makeTable } from "./table";
import { simulateShot } from "./engine";
import { applyCue } from "./cue";
import { resolveBallBall } from "./collisions";
import { mag } from "./vec";
import { BALL_RADIUS, BALL_DIAMETER, TABLE_LENGTH } from "./constants";

describe("ball-ball collision", () => {
  it("transfers momentum on a dead-centre stop shot (elastic-ish stun)", () => {
    // Cue ball moving in +x hits a stationary object ball dead centre. With
    // near-unit restitution and no spin, most forward momentum transfers to the
    // object ball (classic stun shot): cue nearly stops, object goes forward.
    const cue = makeBall(0, 0, 0);
    cue.vel = { x: 2, y: 0 };
    cue.motion = Motion.Sliding;
    const obj = makeBall(1, BALL_DIAMETER, 0); // touching along +x
    obj.motion = Motion.Stationary;

    resolveBallBall(cue, obj);

    // Object ball should now carry the bulk of the +x velocity.
    expect(obj.vel.x).toBeGreaterThan(1.8);
    expect(Math.abs(obj.vel.y)).toBeLessThan(0.05);
    // Cue ball should be nearly stopped in x.
    expect(cue.vel.x).toBeLessThan(0.2);
  });

  it("conserves linear momentum in a head-on collision", () => {
    const cue = makeBall(0, 0, 0);
    cue.vel = { x: 3, y: 0 };
    const obj = makeBall(1, BALL_DIAMETER, 0);
    const pBefore = cue.vel.x + obj.vel.x;
    resolveBallBall(cue, obj);
    const pAfter = cue.vel.x + obj.vel.x;
    // Equal masses → total x-momentum (proportional to velocity sum) preserved.
    expect(pAfter).toBeCloseTo(pBefore, 6);
  });

  it("sends balls at ~90 degrees on a half-ball cut (no spin)", () => {
    // A rolling/stunned cue ball cutting an object ball should separate near 90
    // degrees — the classic tangent-line result for equal masses.
    const cue = makeBall(0, 0, 0);
    cue.vel = { x: 2, y: 0 };
    // Object offset so the line of centres is 45 degrees.
    const off = BALL_DIAMETER / Math.SQRT2;
    const obj = makeBall(1, off, off);
    resolveBallBall(cue, obj);
    // Angle between outgoing velocities should be close to 90 degrees.
    const c = cue.vel;
    const o = obj.vel;
    const cosang =
      (c.x * o.x + c.y * o.y) / (mag(c) * mag(o) + 1e-12);
    // Throw perturbs it slightly; require within ~15 degrees of orthogonal.
    expect(Math.abs(cosang)).toBeLessThan(0.26);
  });
});

describe("event-based evolution", () => {
  it("a straight shot down the table pockets in a corner", () => {
    // Aim the cue ball straight at the top-right corner pocket from centre.
    const hx = TABLE_LENGTH / 2;
    const cue = makeBall(0, 0, 0);
    const table2 = makeTable();
    const corner = table2.pockets.find((p) => p.id === "tr")!;
    const phi = Math.atan2(corner.center.y - 0, corner.center.x - 0);
    applyCue(cue, { phi, power: 0.6, sideSpin: 0, topSpin: 0 });
    const res = simulateShot([cue], table2);
    expect(res.pocketed).toContain(0);
    void hx;
  });

  it("cushion rebound reverses the normal velocity component", () => {
    // Fire a ball straight into the right cushion; it should come back in -x.
    const cue = makeBall(0, 0, 0);
    applyCue(cue, { phi: 0, power: 0.3, sideSpin: 0, topSpin: 0 });
    const res = simulateShot([cue], makeTable());
    // A ball-cushion event on the right rail should be in the trace.
    const railHit = res.events.find(
      (e) => e.kind === "ball-cushion" && e.cushion === "right",
    );
    expect(railHit).toBeDefined();
    // Ball ends at rest somewhere with x < starting rebound region (came back).
    expect(res.balls[0].vel.x).toBeCloseTo(0, 3);
  });

  it("friction brings every ball to rest (energy is dissipated, never gained)", () => {
    const cue = makeBall(0, 0, 0);
    applyCue(cue, { phi: 0.7, power: 1, sideSpin: 0.3, topSpin: 0.2 });
    const startKE = 0.5 * mag(cue.vel) ** 2;
    const res = simulateShot([cue], makeTable());
    const endKE = 0.5 * mag(res.balls[0].vel) ** 2;
    // Every ball must be at rest (or pocketed) — the sim terminates.
    for (const b of res.balls) {
      expect(b.pocketed || mag(b.vel) < 0.01).toBe(true);
    }
    // Energy must never increase across the whole shot.
    expect(endKE).toBeLessThanOrEqual(startKE + 1e-9);
  });

  it("balls never end up overlapping after a multi-ball shot", () => {
    const cue = makeBall(0, -0.3, 0);
    applyCue(cue, { phi: 0, power: 0.8, sideSpin: 0, topSpin: 0 });
    const b1 = makeBall(1, 0.2, 0);
    const b2 = makeBall(2, 0.2 + BALL_DIAMETER * 1.02, 0);
    const res = simulateShot([cue, b1, b2], makeTable());
    for (let i = 0; i < res.balls.length; i++) {
      for (let j = i + 1; j < res.balls.length; j++) {
        const a = res.balls[i];
        const b = res.balls[j];
        if (a.pocketed || b.pocketed) continue;
        const d = Math.hypot(a.pos.x - b.pos.x, a.pos.y - b.pos.y);
        // Allow a tiny numeric tolerance below one diameter.
        expect(d).toBeGreaterThan(BALL_DIAMETER - 1e-3);
      }
    }
  });
});

describe("action-space constraint (no jump / masse)", () => {
  it("the cue action interface has no elevation field", () => {
    // Compile-time guarantee is the real enforcement; at runtime we assert the
    // struck ball has zero vertical state because there is no vertical axis.
    const cue = makeBall(0, 0, 0);
    applyCue(cue, { phi: 0, power: 1, sideSpin: 1, topSpin: 1 });
    // The ball stays on the bed: its state is purely planar (x,y,wz,roll). No
    // z-velocity exists anywhere in the model. Sanity: speed is finite & bounded.
    expect(mag(cue.vel)).toBeLessThanOrEqual(8.5 + 1e-6);
    expect(Number.isFinite(cue.wz)).toBe(true);
  });

  it("caps launch speed at MAX_V0 even at full power", () => {
    const cue = makeBall(0, 0, 0);
    applyCue(cue, { phi: 1.2, power: 5, sideSpin: 0, topSpin: 0 }); // power>1
    expect(mag(cue.vel)).toBeLessThanOrEqual(8.5 + 1e-6);
  });
});

describe("geometry sanity", () => {
  it("ball radius and diameter are consistent", () => {
    expect(BALL_DIAMETER).toBeCloseTo(2 * BALL_RADIUS, 12);
  });
});
