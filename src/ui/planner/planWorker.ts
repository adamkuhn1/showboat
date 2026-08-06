/// <reference lib="webworker" />
//
// The opponent thinks here, not on the page.
//
// Measured before this existed: every AI turn blocked the main thread for
// 3.4–5.0 s in dev (12.3 s worst case immediately after a break, and five
// plans back to back crashed the renderer process); independent profiling of a
// production build recorded 11 long tasks totalling 11.3 s over 81.5 s, p50
// 1.03 s, max 2.13 s. Nothing repainted, nothing hovered, nothing clicked. It
// is also why the "searching…" state had never once been painted: `plan()`
// blocked before React could commit it.
//
// Feasibility was checked by inspection before this was written, not assumed:
// the physics bridge, the wasm-bindgen glue and the neural evaluator contain
// zero `document`/`window` references, and nothing here needs
// `SharedArrayBuffer` — the bridge copies through flat `Float64Array`s rather
// than sharing memory. The cost is that the physics WASM is instantiated twice
// (the page still needs it for `takeShot` and the waypoint capture), which is
// one compile of a small module.

import { planTurnTraced } from "./plan";
import type { PlanRequest, PlanResponse } from "./protocol";

const post = (msg: PlanResponse) => (self as unknown as Worker).postMessage(msg);

self.onmessage = async (e: MessageEvent<PlanRequest>) => {
  const req = e.data;
  if (!req || req.type !== "plan") return;
  try {
    const planned = await planTurnTraced(
      {
        state: req.state,
        table: req.table,
        player: req.player,
        useNeural: req.useNeural,
      },
      {
        onModelLoadStart: () => post({ type: "model-loading", id: req.id }),
        onModelStatus: (s) =>
          post({
            type: "model-status",
            id: req.id,
            ok: s.status === "ready",
            reason: s.reason,
            hashVerified: s.hashVerified,
          }),
      },
    );
    post({
      type: "done",
      id: req.id,
      trace: planned.trace,
      action: planned.action,
      report: planned.report,
    });
  } catch (err) {
    post({ type: "error", id: req.id, message: err instanceof Error ? err.message : String(err) });
  }
};
