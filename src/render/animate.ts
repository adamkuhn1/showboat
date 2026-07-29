import { type Ball, Motion, classifyMotion } from "../physics/ball";
import { type Table } from "../physics/table";
import { advanceBall } from "../physics/motion";
import { timeToBallBall, timeToCushion, timeToPocket } from "../physics/predict";
import { resolveBallBall, resolveBallCushion } from "../physics/collisions";
import { timeToPhaseChange } from "../physics/motion";

// A stepping wrapper around the physics engine so the canvas can ANIMATE the
// shot in real time instead of jumping to the resting state. It advances the
// same event-based model by a wall-clock dt, resolving any events that fall
// inside that slice. This is the exact physics used by simulateShot — just
// surfaced incrementally for display. (The authoritative outcome still comes
// from simulateShot in the game controller; this is purely visual playback of
// the same equations, kept in lockstep by using identical solvers.)

const LOOKAHEAD = 0.05;

// Advance the world by up to dtRemaining seconds, resolving events. Mutates
// balls. Returns true if anything is still moving.
export const stepWorld = (
  balls: Ball[],
  table: Table,
  dtRemaining: number,
): boolean => {
  let remaining = dtRemaining;
  for (const b of balls) if (!b.pocketed) b.motion = classifyMotion(b);

  let guard = 0;
  while (remaining > 1e-6 && guard < 2000) {
    guard++;
    const window = Math.min(LOOKAHEAD, remaining);

    let bestT = window;
    let apply: (() => void) | null = null;

    for (const b of balls) {
      const tp = timeToPhaseChange(b);
      if (tp < bestT) {
        bestT = tp;
        apply = null;
      }
    }
    for (let i = 0; i < balls.length; i++) {
      for (let j = i + 1; j < balls.length; j++) {
        const tc = timeToBallBall(balls[i], balls[j], bestT);
        if (isFinite(tc) && tc < bestT) {
          bestT = tc;
          const a = balls[i];
          const b = balls[j];
          apply = () => resolveBallBall(a, b);
        }
      }
    }
    for (const b of balls) {
      for (const c of table.cushions) {
        const tc = timeToCushion(b, c, bestT);
        if (isFinite(tc) && tc < bestT) {
          bestT = tc;
          apply = () => resolveBallCushion(b, c);
        }
      }
    }
    for (const b of balls) {
      const p = timeToPocket(b, table, bestT);
      if (p && isFinite(p.t) && p.t < bestT) {
        bestT = p.t;
        apply = () => {
          b.pocketed = true;
          b.vel = { x: 0, y: 0 };
          b.roll = { x: 0, y: 0 };
          b.wz = 0;
          b.motion = Motion.Stationary;
        };
      }
    }

    const step = Math.max(bestT, 0);
    for (const b of balls) advanceBall(b, step);
    remaining -= step;
    if (apply) apply();
    for (const b of balls) if (!b.pocketed) b.motion = classifyMotion(b);

    if (step <= 1e-9 && !apply) break;
  }

  return balls.some((b) => !b.pocketed && b.motion !== Motion.Stationary);
};
