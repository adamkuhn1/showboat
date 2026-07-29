// Public surface of the in-house event-based pool physics engine.
//
// Architecture (Han-2005 lineage, mirroring pooltool's train-time sim so the
// ONNX policy sees consistent physics at play time):
//   - vec.ts         2D vector math
//   - constants.ts   SI physical constants + table geometry
//   - ball.ts        ball state + motion-phase classification
//   - table.ts       cushions, pockets, spots
//   - motion.ts      closed-form trajectory integration per phase
//   - predict.ts     analytic time-to-next-event solvers
//   - collisions.ts  ball-ball throw + Han-2005 cushion rebound with spin
//   - cue.ts         action space (theta absent -> jump/masse unrepresentable)
//   - engine.ts      event loop -> resting state + event trace

export * from "./vec";
export * from "./constants";
export * from "./ball";
export * from "./table";
export * from "./motion";
export * from "./collisions";
export * from "./predict";
export * from "./cue";
export * from "./engine";
