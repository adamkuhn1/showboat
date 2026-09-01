import { type SimResult } from "../physics/engine";
import { CUE_ID } from "../game/rack";

// Measured shot classification. The kind of a shot ("bank", "combo", ...)
// shown anywhere in the UI is derived HERE, from the event log of the actual
// simulation — which balls and cushions were struck, in what order — never
// from what the candidate generator intended.

export interface MeasuredShot {
  // What physically happened, per the event log:
  cueRailsBeforeContact: number; // cushions the cue ball took before first contact
  objectRails: number; // cushions the target ball took before being potted (or settling)
  chainLength: number; // object balls struck in sequence before the target moved
  targetPotted: boolean;
  pocketId: string | null;
  railsBeforePot: number; // all cushion events before the target dropped
  kind: "direct" | "bank" | "kick" | "combo" | "no-pot";
  // A short human label composed only from the measurements above.
  label: string;
}

export const classifyShot = (sim: SimResult, targetBall: number): MeasuredShot => {
  let firstContactIdx = -1;
  let cueRailsBeforeContact = 0;
  for (let i = 0; i < sim.events.length; i++) {
    const e = sim.events[i];
    if (e.kind === "ball-cushion" && e.balls.includes(CUE_ID) && firstContactIdx < 0) {
      cueRailsBeforeContact++;
    }
    if (e.kind === "ball-ball" && e.balls.includes(CUE_ID) && firstContactIdx < 0) {
      firstContactIdx = i;
      break;
    }
  }

  // Chain: distinct object balls involved in ball-ball events from first
  // contact until the target ball is first struck.
  const chain: number[] = [];
  let targetStruckIdx = -1;
  if (firstContactIdx >= 0) {
    for (let i = firstContactIdx; i < sim.events.length; i++) {
      const e = sim.events[i];
      if (e.kind !== "ball-ball") continue;
      for (const id of e.balls) {
        if (id !== CUE_ID && !chain.includes(id)) chain.push(id);
      }
      if (e.balls.includes(targetBall)) {
        targetStruckIdx = i;
        break;
      }
    }
  }

  // Target's cushion contacts between being struck and being potted (or the
  // end of the shot), and the pocket it fell into.
  let objectRails = 0;
  let railsBeforePot = 0;
  let pocketId: string | null = null;
  let targetPotted = false;
  if (targetStruckIdx >= 0) {
    for (let i = targetStruckIdx; i < sim.events.length; i++) {
      const e = sim.events[i];
      if (e.kind === "ball-cushion") {
        railsBeforePot++;
        if (e.balls.includes(targetBall)) objectRails++;
      }
      if (e.kind === "pocket" && e.balls.includes(targetBall)) {
        targetPotted = true;
        pocketId = e.pocket ?? null;
        break;
      }
    }
  }

  const chainLength = chain.length;
  let kind: MeasuredShot["kind"];
  if (!targetPotted) kind = "no-pot";
  else if (chainLength >= 2) kind = "combo";
  else if (objectRails >= 1) kind = "bank";
  else if (cueRailsBeforeContact >= 1) kind = "kick";
  else kind = "direct";

  const label = buildLabel(kind, cueRailsBeforeContact, objectRails, chainLength);

  return {
    cueRailsBeforeContact,
    objectRails,
    chainLength,
    targetPotted,
    pocketId,
    railsBeforePot,
    kind,
    label,
  };
};

const buildLabel = (
  kind: MeasuredShot["kind"],
  cueRails: number,
  objRails: number,
  chain: number,
): string => {
  switch (kind) {
    case "bank":
      return objRails >= 2 ? `${objRails}-rail bank` : "bank";
    case "kick":
      return cueRails >= 2 ? `${cueRails}-rail kick` : "kick";
    case "combo":
      return objRails >= 1
        ? `combo off ${chain - 1} ball${chain - 1 > 1 ? "s" : ""}, banked`
        : `combo off ${chain - 1} ball${chain - 1 > 1 ? "s" : ""}`;
    case "direct":
      return "direct shot";
    case "no-pot":
      return "no pot";
  }
};

// True when the measured shot involved at least one cushion or a combo chain —
// the "trick shot" family the AI prefers. Read from measurements only.
export const isTrickShot = (m: MeasuredShot): boolean =>
  m.targetPotted && (m.kind === "bank" || m.kind === "kick" || m.kind === "combo");
