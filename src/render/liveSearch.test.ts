// The live observation, driven by a real search's real event stream.
//
// `liveSearch.ts` is the only thing standing between the search's events and
// the felt, so the properties that matter are properties of this fold: it must
// never draw a route the search has not described, never claim a physics result
// the search has not returned, and never lose a candidate the search has
// finished with. All three are checked here against a stream collected from a
// real `brain.plan()` — no hand-built events except where a specific ordering
// has to be forced.

import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { makeTable } from "../physics/table";
import { makeBall, type Ball } from "../physics/ball";
import { CUE_ID } from "../game/rack";
import type { GameState } from "../game/state";
import { initPhysics } from "../physics/wasm-bridge";
import { defaultConfig } from "../ai/shotSearch";
import { classicalTrickOnlyBrain } from "../ai/brain";
import { createProgressSink, type SearchProgressEvent } from "../ai/search/progress";
import {
  applyProgress,
  CAPTION_DWELL_MS,
  createLiveSearch,
  liveFrame,
  liveState,
} from "./liveSearch";

const __dirname = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = join(__dirname, "../..");
const WASM = readFileSync(join(APP_ROOT, "src/wasm/showboat_physics_bg.wasm"));
const table = makeTable();

beforeAll(async () => {
  await initPhysics(WASM);
}, 60_000);

const BOARD: Ball[] = [
  makeBall(CUE_ID, -0.6, -0.1),
  makeBall(1, 0.3, 0.02),
  makeBall(2, 0.38, 0.1),
  makeBall(4, 0.1, -0.25),
  makeBall(5, -0.2, 0.3),
];
const state: GameState = {
  balls: BOARD,
  turn: 1,
  groups: { 0: null, 1: null },
  ballInHand: false,
  winner: null,
  broken: true,
  shotCount: 3,
};
const cfg = { ...defaultConfig, seed: 20260807, seedTimeoutMs: Infinity, searchTimeoutMs: Infinity };
const geom = { cuePos: { x: -0.6, y: -0.1 } };

let stream: SearchProgressEvent[] = [];
beforeAll(async () => {
  const events: SearchProgressEvent[] = [];
  await classicalTrickOnlyBrain().plan(
    state,
    table,
    1,
    cfg,
    undefined,
    createProgressSink((e) => events.push(e)),
  );
  stream = events;
}, 60_000);

/** Fold the first `n` events of the real stream. */
const foldTo = (n: number) => {
  const live = createLiveSearch();
  for (const e of stream.slice(0, n)) applyProgress(live, e);
  return live;
};

describe("a route may not appear before the search says it exists", () => {
  it("nothing is drawn before the geometry event arrives", () => {
    const genAt = stream.findIndex((e) => e.kind === "candidates-generated");
    expect(genAt).toBeGreaterThanOrEqual(0);
    const live = foldTo(genAt); // everything strictly before the geometry
    expect(liveFrame(live, geom).routes).toEqual([]);
    expect(liveState(live)).toBe("IDLE");
  });

  it("once geometry arrives, exactly the published candidates are drawn", () => {
    const genAt = stream.findIndex((e) => e.kind === "candidates-generated");
    const gen = stream[genAt] as Extract<SearchProgressEvent, { kind: "candidates-generated" }>;
    const live = foldTo(genAt + 1);
    const routes = liveFrame(live, geom).routes;
    expect(routes.length).toBe(gen.candidates.length);
    expect(routes.map((r) => r.index).sort((a, b) => a - b)).toEqual(
      gen.candidates.map((c) => c.index).sort((a, b) => a - b),
    );
    // Drawn from the published geometry, not from anything reconstructed.
    for (const r of routes) {
      const c = gen.candidates.find((x) => x.index === r.index)!;
      expect(r.objectLeg).toEqual(c.path);
      expect(r.cueLeg![1]).toEqual(c.aimPoint);
    }
  });

  it("a live route is always plan geometry — the measured motion does not exist yet", () => {
    const live = foldTo(stream.length);
    for (const r of liveFrame(live, geom).routes) {
      expect(r.source).toBe("plan");
      expect(r.measured).toBeNull();
    }
  });
});

describe("a verification may not appear before its simulation completes", () => {
  it("a candidate in the simulator has no physics result attached yet", () => {
    // Find a simulation event and fold up to (not including) its verification.
    const simAt = stream.findIndex((e) => e.kind === "candidate-simulating");
    const idx = (stream[simAt] as { index: number }).index;
    const live = foldTo(simAt + 1);
    const c = live.candidates[idx];
    expect(c.simulating).toBe(true);
    expect(c.physics).toBeNull();
    // And it is the one route the frame marks as resolving.
    const r = liveFrame(live, geom).routes.find((x) => x.index === idx)!;
    expect(r.resolving).toBe(true);
  });

  it("the physics result attaches only when the verification event arrives", () => {
    const simAt = stream.findIndex((e) => e.kind === "candidate-simulating");
    const idx = (stream[simAt] as { index: number }).index;
    const verAt = stream.findIndex((e) => e.kind === "candidate-verified" && e.index === idx);
    expect(verAt).toBeGreaterThan(simAt);
    const live = foldTo(verAt + 1);
    expect(live.candidates[idx].simulating).toBe(false);
    expect(live.candidates[idx].physics).not.toBeNull();
  });
});

describe("completed events are retained, not flashed", () => {
  it("a candidate resolved early is still on the felt at the end of the search", () => {
    const firstResolved = stream.find(
      (e) => e.kind === "candidate-rejected" && e.reason !== "direct-excluded-by-policy",
    ) as { index: number; reason: string } | undefined;
    expect(firstResolved).toBeDefined();
    const live = foldTo(stream.length);
    const r = liveFrame(live, geom).routes.find((x) => x.index === firstResolved!.index)!;
    expect(r).toBeDefined();
    // Still drawn, still carrying the reason it lost.
    expect(r.role === "rejected" || r.role === "selected").toBe(true);
    if (r.role === "rejected") expect(r.reason).not.toBeNull();
  });

  it("every candidate the search resolved keeps a role and a reason to the end", () => {
    const live = foldTo(stream.length);
    const routes = liveFrame(live, geom).routes;
    for (const r of routes) {
      if (r.role === "rejected") expect(r.reason).not.toBeNull();
      if (r.role === "selected" || r.role === "verified") expect(r.reason).toBeNull();
    }
  });

  it("the whole event stream is retained for replay and inspection", () => {
    const live = foldTo(stream.length);
    expect(live.events.length).toBe(stream.length);
    expect(live.outOfOrder).toBe(0);
  });
});

describe("the fold reports only what the search published", () => {
  it("reaches the selected state and names the shot the search chose", () => {
    const live = foldTo(stream.length);
    const sel = stream.find((e) => e.kind === "selected") as
      | Extract<SearchProgressEvent, { kind: "selected" }>
      | undefined;
    expect(sel).toBeDefined();
    expect(live.selectedIndex).toBe(sel!.index);
    expect(live.selectedRung).toBe(sel!.rung);
    expect(live.phase).toBe("completed");
    // Exactly one route is the answer.
    const chosen = liveFrame(live, geom).routes.filter((r) => r.role === "selected");
    expect(chosen.length).toBe(1);
  });

  it("counts are counts of real events, and only ever go up", () => {
    let prev = { generated: 0, simulated: 0, retained: 0, rejected: 0 };
    const live = createLiveSearch();
    for (const e of stream) {
      applyProgress(live, e);
      expect(live.counts.simulated).toBeGreaterThanOrEqual(prev.simulated);
      expect(live.counts.rejected).toBeGreaterThanOrEqual(prev.rejected);
      expect(live.counts.retained).toBeGreaterThanOrEqual(prev.retained);
      prev = { ...live.counts };
    }
    // The simulated count is the number of verification events, exactly.
    expect(live.counts.simulated).toBe(
      stream.filter((e) => e.kind === "candidate-verified").length,
    );
  });

  it("the frame exposes no progress fraction for a live search", () => {
    // There is no honest denominator: the search does not know how many
    // candidates it will reach. `progress` is fixed at 0 and drives nothing.
    for (let n = 0; n <= stream.length; n += Math.max(1, Math.floor(stream.length / 10))) {
      expect(liveFrame(foldTo(n), geom).progress).toBe(0);
    }
  });

  it("a classical search never reaches the neural ranking state", () => {
    const live = foldTo(stream.length);
    expect(stream.some((e) => e.kind === "neural-scored")).toBe(false);
    // The fold advances straight from enumerated to verifying.
    const seen = new Set<string>();
    const l2 = createLiveSearch();
    for (const e of stream) {
      applyProgress(l2, e);
      seen.add(liveState(l2));
    }
    expect(seen.has("RANKING")).toBe(false);
    expect(seen.has("VERIFYING")).toBe(true);
    expect(live.modelId).toBeNull();
  });

  it("the rejection caption is rate limited, and only ever holds a real reason", () => {
    const live = createLiveSearch();
    const captions: { index: number; atMs: number }[] = [];
    for (const e of stream) {
      const before = live.captionIndex;
      applyProgress(live, e);
      if (live.captionIndex !== null && live.captionIndex !== before) {
        captions.push({ index: live.captionIndex, atMs: live.captionAtMs });
      }
    }
    // Never faster than the dwell.
    for (let i = 1; i < captions.length; i++) {
      expect(captions[i].atMs - captions[i - 1].atMs).toBeGreaterThanOrEqual(CAPTION_DWELL_MS);
    }
    // And every one of them names a candidate the search really rejected, with
    // a reason it really gave — a rate limit may drop a caption, never invent one.
    const rejected = new Map(
      stream
        .filter((e) => e.kind === "candidate-rejected")
        .map((e) => [(e as { index: number }).index, (e as { reason: string }).reason]),
    );
    for (const c of captions) {
      expect(rejected.has(c.index)).toBe(true);
      expect(rejected.get(c.index)).not.toBe("direct-excluded-by-policy");
      expect(rejected.get(c.index)).not.toBe("pruned-by-prior");
    }
  });

  it("the phase never moves backwards, even though rejections arrive after selection", () => {
    const order = ["started", "enumerated", "ranked", "verifying", "selected", "completed"];
    const live = createLiveSearch();
    let high = 0;
    for (const e of stream) {
      applyProgress(live, e);
      const at = order.indexOf(live.phase);
      expect(at).toBeGreaterThanOrEqual(high);
      high = at;
    }
    expect(live.phase).toBe("completed");
  });
});
