// The measured classifier, against hand-built event logs.
//
// Hand-built on purpose: the interesting cases are the ones real physics
// produces rarely and the plan gets wrong often — a bank that misses its
// cushion, a cue ball that rattles three rails after the pot has already
// dropped, a ball that happens to touch a cushion on the other side of the
// table while an unrelated combination goes in. A fixture log states each of
// those in four lines and leaves no doubt about what was being asked.
//
// One rule governs every case here: the answer comes from the ordered log, and
// the candidate's planned kind is never an input.

import { describe, it, expect } from "vitest";
import type { ShotEvent } from "../../physics/engine";
import {
  classifyMeasuredShot,
  measuredRouteLabel,
  measuredTrickKind,
  type ShotIntent,
} from "./classify";

const CUE = 0;

const hit = (time: number, a: number, b: number): ShotEvent => ({
  time,
  kind: "ball-ball",
  balls: [a, b],
});
const rail = (time: number, b: number, cushion = "top"): ShotEvent => ({
  time,
  kind: "ball-cushion",
  balls: [b],
  cushion,
});
const pot = (time: number, b: number, pocket = "tr"): ShotEvent => ({
  time,
  kind: "pocket",
  balls: [b],
  pocket,
});
const stop = (time: number): ShotEvent => ({ time, kind: "stop", balls: [] });

const intent = (over: Partial<ShotIntent> = {}): ShotIntent => ({
  target: 1,
  potId: 1,
  legalTargets: [1, 2, 3],
  ...over,
});

const classify = (events: ShotEvent[], i: ShotIntent = intent()) =>
  classifyMeasuredShot({ events }, i);

describe("the planned kind is never an input", () => {
  it("a route that touches no cushion is a direct pot, whatever it was generated as", () => {
    const m = classify([hit(0.1, CUE, 1), pot(0.5, 1), stop(2)]);
    expect(m.classification).toBe("direct");
    expect(m.rails).toBe(0);
    expect(m.trickVerified).toBe(false);
    expect(measuredTrickKind(m)).toBeNull();
  });

  it("a route planned as one bank that takes two cushions is a two-rail bank", () => {
    const m = classify([
      hit(0.1, CUE, 1),
      rail(0.3, 1, "top"),
      rail(0.45, 1, "right"),
      pot(0.7, 1),
      stop(2),
    ]);
    expect(m.classification).toBe("multi-rail-bank");
    expect(m.rails).toBe(2);
    expect(m.railCushions).toEqual(["top", "right"]);
    expect(m.trickVerified).toBe(true);
    expect(measuredTrickKind(m)).toBe("double-bank");
    expect(measuredRouteLabel(m)).toBe("two-rail bank");
  });

  it("three cushions are named as three, not rounded to the nearest planned label", () => {
    const m = classify([
      hit(0.1, CUE, 1),
      rail(0.2, 1, "top"),
      rail(0.3, 1, "right"),
      rail(0.4, 1, "bottom"),
      pot(0.8, 1),
    ]);
    expect(m.rails).toBe(3);
    expect(measuredRouteLabel(m)).toBe("three-rail bank");
  });
});

describe("nothing after the decisive pot counts", () => {
  it("cue-ball cushions after the pot are not part of the route at all", () => {
    const m = classify([
      hit(0.1, CUE, 1),
      pot(0.5, 1),
      rail(0.7, CUE, "top"),
      rail(0.9, CUE, "left"),
      rail(1.1, CUE, "bottom"),
      stop(2),
    ]);
    expect(m.classification).toBe("direct");
    expect(m.rails).toBe(0);
    expect(m.cueRails, "three cushions, all of them after the ball dropped").toBe(0);
  });

  it("a chain ball's cushion after the pot does not promote a combination", () => {
    // The cue drives 1 into 2, 2 drops, and 1 carries on into a cushion. Ball 1
    // is genuinely on the chain, so only the truncation keeps this a plain
    // combination rather than a rail combination.
    const m = classify(
      [hit(0.1, CUE, 1), hit(0.25, 1, 2), pot(0.6, 2), rail(0.9, 1, "left"), stop(2)],
      intent({ potId: 2 }),
    );
    expect(m.classification).toBe("combination");
    expect(m.rails).toBe(0);
  });

  it("a cushion by a ball that is not on the chain is not a rail", () => {
    const m = classify([
      hit(0.1, CUE, 1),
      rail(0.3, 1, "top"),
      pot(0.6, 1),
      rail(0.8, 2, "left"),
      stop(2),
    ]);
    expect(m.classification).toBe("one-rail-bank");
    expect(m.rails).toBe(1);
  });
});

describe("rails are credited causally, not by co-occurrence", () => {
  it("a cushion taken by an uninvolved ball is not a rail in this route", () => {
    // Ball 5 was already moving from an earlier contact elsewhere on the table
    // and takes a cushion while the shot is in flight. It is not on the chain.
    const m = classify([
      hit(0.1, CUE, 1),
      hit(0.15, 1, 5),
      rail(0.3, 5, "left"),
      pot(0.6, 1),
    ]);
    expect(m.classification).toBe("direct");
    expect(m.rails).toBe(0);
    expect(m.contactChain).toEqual([CUE, 1]);
  });

  it("a combination's chain is the balls that set each other moving", () => {
    const m = classify([hit(0.1, CUE, 1), hit(0.25, 1, 2), pot(0.6, 2)], intent({ potId: 2 }));
    expect(m.classification).toBe("combination");
    expect(m.contactChain).toEqual([CUE, 1, 2]);
    expect(m.trickVerified).toBe(true);
    expect(measuredTrickKind(m)).toBe("combo");
  });

  it("a rail combination needs the cushion on the chain, and the chain in order", () => {
    const m = classify(
      [hit(0.1, CUE, 1), hit(0.25, 1, 2), rail(0.4, 2, "left"), pot(0.7, 2)],
      intent({ potId: 2 }),
    );
    expect(m.classification).toBe("rail-combination");
    expect(m.rails).toBe(1);
    expect(measuredTrickKind(m)).toBe("rail-combo");
    expect(measuredRouteLabel(m)).toBe("rail combination");
  });

  it("the cue ball's own cushions are a kick and never a bank", () => {
    const m = classify([rail(0.1, CUE, "left"), hit(0.3, CUE, 1), pot(0.7, 1)]);
    expect(m.classification).toBe("direct");
    expect(m.rails).toBe(0);
    expect(m.cueRails).toBe(1);
  });
});

describe("which pot is decisive", () => {
  it("the intended ball's pot decides, even when another ball fell first", () => {
    const m = classify(
      [
        hit(0.1, CUE, 1),
        hit(0.2, 1, 3),
        pot(0.3, 3, "bl"),
        rail(0.4, 1, "top"),
        pot(0.8, 1, "tr"),
      ],
      intent({ potId: 1 }),
    );
    expect(m.pottedBall).toBe(1);
    expect(m.pocket).toBe("tr");
    expect(m.classification).toBe("one-rail-bank");
    expect(m.rails).toBe(1);
  });

  it("when the intended ball never drops, the shot is judged on what did", () => {
    const m = classify(
      [hit(0.1, CUE, 1), hit(0.2, 1, 3), pot(0.5, 3, "bl")],
      intent({ potId: 1 }),
    );
    expect(m.pottedBall).toBe(3);
    expect(m.classification).toBe("combination");
    // It executed a combination, but not the one it was for, so it is not a
    // trick this policy may play.
    expect(m.trickVerified).toBe(false);
  });
});

describe("fouls, safeties and misses", () => {
  it("a scratch is a foul whatever else the shot did", () => {
    const m = classify([hit(0.1, CUE, 1), rail(0.3, 1), pot(0.6, 1), pot(0.9, CUE, "bl")]);
    expect(m.scratched).toBe(true);
    expect(m.classification).toBe("foul");
    expect(m.foulReason).toBe("scratch");
    expect(m.trickVerified).toBe(false);
  });

  it("no ball contact at all is a foul", () => {
    const m = classify([rail(0.2, CUE, "top"), stop(2)]);
    expect(m.classification).toBe("foul");
    expect(m.foulReason).toBe("no contact");
  });

  it("striking a ball that is not a legal target is a foul", () => {
    const m = classify([hit(0.1, CUE, 9), rail(0.3, 9), pot(0.7, 9)], intent({ legalTargets: [1, 2] }));
    expect(m.classification).toBe("foul");
    expect(m.foulReason).toBe("illegal first contact");
    expect(m.firstContactLegal).toBe(false);
  });

  it("legal contact with nothing potted and no cushion is the no-rail foul", () => {
    const m = classify([hit(0.1, CUE, 1), stop(2)]);
    expect(m.classification).toBe("foul");
    expect(m.foulReason).toBe("no rail");
  });

  it("a shot that planned no pot and did not foul is a safety", () => {
    const m = classify(
      [rail(0.1, CUE, "left"), hit(0.4, CUE, 1), rail(0.6, 1, "top"), stop(2)],
      intent({ target: null, potId: null }),
    );
    expect(m.classification).toBe("safety");
  });

  it("a shot that planned a pot and did not make one is a miss", () => {
    const m = classify([hit(0.1, CUE, 1), rail(0.4, 1, "top"), stop(2)]);
    expect(m.classification).toBe("miss");
  });
});

describe("the intended ball has to be the one that drops, off the intended contact", () => {
  it("a bank that pots by striking the wrong ball first is not verified", () => {
    const m = classify(
      [hit(0.1, CUE, 2), hit(0.2, 2, 1), rail(0.4, 1, "top"), pot(0.7, 1)],
      intent({ target: 1, potId: 1 }),
    );
    expect(m.firstContact).toBe(2);
    expect(m.trickVerified).toBe(false);
  });
});
