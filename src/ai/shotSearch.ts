import { type Ball, cloneBall } from "../physics/ball";
import { type Table } from "../physics/table";
import { type CueAction } from "../physics/cue";
import { type ShotEvent, type SimResult } from "../physics/engine";
import { type Candidate, type CandidateKind, generateCandidates } from "./candidates";
import { rolloutValueWasm, separateOverlaps, simulateShotWasm } from "../physics/wasm-bridge";
import { railsBeforePot } from "./trace";

/**
 * A legal pot requires: the cue's first contact matches the candidate's
 * intended legal ball (`sim.firstContact` — an obstruction the geometry check
 * missed, or a deflection, can make the cue strike something else first,
 * which is an illegal first contact regardless of what ends up pocketed),
 * the ball actually meant to drop (`candidate.potId` — for a combo/
 * rail-combo this is the driven intermediate ball, NOT `candidate.target`,
 * the first-contact ball; see `Candidate.potId`'s doc comment in
 * candidates.ts) is the one that's pocketed, AND — when `potId !== target`
 * (combo/rail-combo, where a second object ball is involved) — that the
 * intended intermediate ball actually struck the pocketed ball, not just
 * that both events happened to occur somewhere in the same shot.
 *
 * The third condition closes a real endpoint-only labeling gap (found during
 * Phase 2C Stage A closure): checking only `firstContact` + final `pocketed`
 * set cannot distinguish "the combo worked as intended" from "the cue hit
 * `target` first, and `potId` was *separately* pocketed by some unrelated
 * contact chain in the same shot" — both produce an identical
 * (firstContact, pocketed) endpoint. `SimResult.events` (from the real WASM
 * simulator) records every ball-ball collision in order, which is exactly
 * the information needed to tell these apart. Direct/bank/double-bank never
 * need this check (`potId === target`, one object ball throughout — no
 * intermediate contact to misattribute; a ball reaching a pocket via more or
 * fewer cushion bounces than the candidate generator planned is still a
 * completely legal pot under real 8-ball rules, not a labeling error).
 *
 * Extracted as its own function (rather than inlined in the seeding loop) so
 * it's directly unit-testable against hand-built SimResult fixtures without
 * needing real physics to organically produce an illegal-first-contact case.
 */
export function isLegalPot(sim: Pick<SimResult, "firstContact" | "pocketed" | "events">, candidate: Candidate): boolean {
  const legalFirstContact = sim.firstContact === candidate.target;
  if (!legalFirstContact || !sim.pocketed.includes(candidate.potId)) return false;
  if (candidate.potId === candidate.target) return true;
  return sim.events.some(
    (e) => e.kind === "ball-ball" && e.balls.includes(candidate.target) && e.balls.includes(candidate.potId),
  );
}

// Flat UCB bandit over the candidate shot set. One level of MCTS is enough for
// a pool turn: a shot is a single continuous action, so the branching factor
// is already handled by candidate generation. Physics rollouts give the value.

export interface CandidateStat {
  candidate: Candidate;
  visits: number;
  value: number;
  // A monotonic, displayable transform of `value` in [0,1) — NOT a
  // calibrated win/pot probability. Nothing has ever measured this against
  // real outcome frequencies (see docs/repair/showboat-ml/EVALUATION_SPEC.md
  // on what "probability" is allowed to mean once that measurement exists).
  // Named `strength` deliberately, not `winProb`/`confidence`, so no UI
  // consumer is tempted to render it with a "%" as if it were one.
  strength: number;
  rails: number;
  potsTarget: boolean;
  styleScore: number;
  // --- learned-prior fields; undefined/false in pure classical mode ---
  // Calibrated make-estimate from the trained ranker (sigmoid(a*logit+b),
  // blended per the manifest's per-kind confidence). This is a *prior over
  // which candidates deserve physics*, never a substitute for the physics
  // result — `value`/`strength`/`potsTarget` are always physics-derived.
  priorScore?: number;
  // 1-based rank by priorScore, i.e. the order the model wanted candidates
  // examined in, before any simulation ran.
  priorRank?: number;
  // Raw pre-calibration model output for this candidate. Carried only so the
  // decision trace can show what the model actually emitted alongside the
  // calibrated score; nothing reads it for a decision.
  priorLogit?: number;
  // Did the authoritative physics search actually simulate this candidate?
  // False for candidates pruned by the prior or cut off by the budget — those
  // can never be selected, since selectBest only sees visits>0 stats.
  verified: boolean;
}

/**
 * Everything the reasoning overlay is allowed to display, and nothing else.
 * Every field here is read directly off the real search that just ran; there
 * is no second, decorative copy of any of it.
 */
export interface DecisionTrace {
  mode: "classical" | "neural-hybrid";
  candidatesGenerated: number;
  /** Candidates handed to the authoritative physics search. */
  candidatesConsidered: number;
  /** Candidates dropped by the learned prior before any physics ran. */
  prunedByPrior: number;
  /** Candidates that actually got a real WASM shot simulation. */
  physicsVerified: number;
  /**
   * Indices (into the candidate list the search was given) of exactly those
   * candidates a real simulation was run on. `stats` cannot answer this: it
   * drops every `visits === 0` row, which includes candidates whose seeding
   * simulation scratched — those were verified, and the verification is why
   * they're gone. The evaluation harness scores candidate recall against this.
   */
  verifiedIndices: number[];
  /** Of those, how many the real simulation showed legally pot their ball. */
  legalPots: number;
  /** Candidates whose seeding simulation scratched the cue (excluded outright). */
  scratched: number;
  /** Physics-engine call units actually spent, against `config.simulations`. */
  physicsCalls: number;
  /**
   * How many candidates the kind reserve rescued — i.e. survived pruning that
   * the model's global ranking alone would have dropped. 0 whenever the model's
   * own top-K already satisfied the reserve, which is the common case.
   */
  reservePromotions: number;
  /** Encode + batched ONNX inference, ms. Only set in neural-hybrid mode. */
  neuralInferenceMs?: number;
  /** Model identity, so the overlay never claims a model it isn't running. */
  modelId?: string;
  /** Why the classical path was taken, when it was taken involuntarily. */
  fallbackReason?: string;
  /** Which branch of `selectBestWithReason` actually chose the shot. */
  selectionReason?: SelectionReason;
  /** Verified tricks that pot AND clear `TRICK_RELIABILITY_THRESHOLD`. */
  qualifyingTricks?: number;
  /** Candidates the `eligible` predicate excluded before physics ran. */
  ineligible?: number;
  /**
   * Indices the search intended to verify, in visit order — i.e. what survived
   * the eligibility filter and the prior's pruning. An index that is here but
   * not in `verifiedIndices` ran out of budget or hit the seed timeout; an
   * eligible index that is in neither was pruned by the prior. That
   * distinction is what lets the published trace give each candidate its real
   * rejection reason instead of a plausible one.
   */
  consideredIndices?: number[];
  /** Did `seedTimeoutMs` truncate the seeding loop? */
  seedTimedOut?: boolean;
  /** Did `searchTimeoutMs` truncate the seeding or the refinement loop? */
  searchTimedOut?: boolean;
  /** Wall-clock ms spent inside the seeding + refinement loops. */
  physicsMs?: number;
}

/**
 * The real simulation record for one candidate, kept so the decision trace can
 * publish what the physics actually did instead of a summary of it. Populated
 * only for candidates a physics call was really spent on — which is exactly
 * what makes the trace's anti-fabrication check possible.
 */
export interface CandidateVerification {
  /** Index into the candidate list the search was given (generation order). */
  index: number;
  firstContact: number | null;
  legalFirstContact: boolean;
  scratched: boolean;
  legalPot: boolean;
  pocketed: number[];
  railsBeforePot: number;
  events: ShotEvent[];
}

/**
 * What `searchCandidates` returns. Note what is NOT here: a chosen shot.
 * The search generates, prunes, verifies and values; it does not decide.
 * Selection lives in `policy/trickOnly.ts` for live play and in
 * `selectBestWithReason` for the evaluation baseline.
 */
export interface SearchOutcome {
  /** Seeded candidates, sorted by visits then value. Historic display order. */
  stats: CandidateStat[];
  /**
   * EVERY candidate, in generation order, seeded or not. `stats` drops
   * `visits === 0` rows — including candidates whose simulation scratched,
   * which the trace must still be able to explain. Index here IS the
   * candidate's stable trace id.
   */
  allStats: CandidateStat[];
  /** Index-aligned with `allStats`; null where no simulation was run. */
  verifications: (CandidateVerification | null)[];
  simulations: number;
  trace?: DecisionTrace;
}

/** A `SearchOutcome` plus the legacy mixed policy's choice. Baseline only. */
export interface SearchResult extends SearchOutcome {
  best: CandidateStat | null;
}

const UCB_C = 1.2;

// Map mean-balls-pocketed (~0..a few) to a bounded [0,1) display score. This
// is an ad-hoc monotonic squash of the rollout/model value, not a calibrated
// probability — see `strength`'s doc comment above.
const toStrength = (v: number): number => 1 - Math.exp(-v);

export interface SearchConfig {
  simulations: number;
  rolloutDepth: number;
  rolloutsPerEval: number;
  seed: number;
  // When the trained net is loaded, it supplies a position value here so the
  // seeding phase can skip per-candidate rollouts and spend the whole budget
  // on UCB refinement instead.
  netSeedValue?: number;
  // Phase 2A candidate ranker: one score per candidate, indexed identically
  // to `generateCandidates`'s output order. Distinct from `netSeedValue`
  // (a single whole-board scalar applied to every candidate, from the older
  // parked policy/value net) — this is the per-candidate signal that was
  // missing before (see docs/repair/showboat-ml/ARCHITECTURE_DECISION.md).
  // Takes priority over `netSeedValue` when both are present.
  netSeedScores?: number[];
  /**
   * Wall-clock guard on the seeding loop, ms. Defaults to
   * `DEFAULT_SEED_TIMEOUT_MS` and exists purely so a pathological board cannot
   * freeze the UI; the binding constraint on work done is `simulations`, not
   * this.
   *
   * Configurable because it makes the search **non-deterministic under CPU
   * load**: the same fixture truncates at a different candidate depending on
   * what else the machine is doing. That was found empirically — a dev
   * evaluation sweep run as four concurrent processes produced *different
   * classical-arm results across runs that should have been identical*, which
   * is a benchmark measuring machine load rather than search policy. Evaluation
   * therefore sets this to `Infinity` (both arms, equally), so results depend
   * only on the physics budget. Production keeps the guard.
   */
  seedTimeoutMs?: number;
  /**
   * Wall-clock guard on the WHOLE search — seeding *and* UCB refinement — ms,
   * measured from the same instant as `seedTimeoutMs`.
   *
   * `seedTimeoutMs` bounds only the seeding loop. When it fired, the physics
   * budget it left unspent was handed to the refinement loop below, which had
   * no clock at all: a turn that recorded `seedTimedOut: true` was measured at
   * 5,309 ms end to end on an idle machine, over the 5,000 ms the evaluation's
   * T3 criterion allows. This is the guard that was missing. Callers that
   * disable `seedTimeoutMs` for determinism must disable this too — the
   * evaluation harness sets both to `Infinity`, in every arm.
   */
  searchTimeoutMs?: number;
  // Phase 2F hybrid agent: a learned prior over the candidate list. Unlike
  // `netSeedScores` above, this NEVER supplies a candidate's value — it only
  // decides the order candidates are physics-verified in, and which ones get
  // dropped before physics runs at all. See `CandidatePrior`.
  prior?: CandidatePrior;
  /**
   * Optional budget filter: candidates this returns false for never receive a
   * physics call. The trick-only policy passes `isTrickCandidate` here so the
   * budget is not spent verifying shots it can never select.
   *
   * **This is an efficiency filter, not the exclusion mechanism.** The
   * exclusion is enforced by the type of the selection ladder in
   * `policy/trickOnly.ts`; `trickOnly.test.ts` A7 runs with this filter
   * deliberately disabled and proves selection alone is sufficient. Excluded
   * candidates still appear in the trace, with `physics: null` and an honest
   * rejection reason.
   */
  eligible?: (c: Candidate) => boolean;
}

/**
 * A learned ordering over `generateCandidates`'s output. The whole contract:
 *
 *  - `scores[i]` corresponds to candidate `i` in generation order.
 *  - Candidates are sorted by score descending and the top `keepTop` are the
 *    only ones the authoritative physics search spends budget on.
 *  - Everything after that is unchanged classical search: real WASM shot
 *    simulation, real `isLegalPot`, real rollout values, the same
 *    `selectBest` trick-preference rule against the same physics-derived
 *    reliability threshold.
 *
 * So the model can change *which shots get examined* and *how much of the
 * budget each gets*, and through that the final choice — but it cannot make
 * an illegal shot legal, cannot make an unreliable trick clear the threshold,
 * and cannot put a number on screen that the physics didn't produce.
 */
export interface CandidatePrior {
  scores: number[];
  keepTop: number;
  /** Raw pre-calibration logits, index-aligned with `scores`. Trace only. */
  logits?: number[];
  /** Model identity for the trace, e.g. "showboat-ranker-phase2d". */
  source: string;
  /** Encode + inference cost, ms, for the trace. */
  inferenceMs?: number;
  /**
   * Minimum number of candidates of a given kind that must survive pruning,
   * regardless of where the model ranked them. See `DEFAULT_PRIOR_RESERVE`.
   * Omitted / empty = pure global top-K, the pre-fix behaviour.
   */
  reserve?: Partial<Record<CandidateKind, number>>;
}

/**
 * The search-policy half of the hybrid: how a FIXED physics budget is split
 * across candidate kinds after the model has ranked them.
 *
 * Why this exists. The prior sprint's equal-budget evaluation found one clear,
 * repeatable regression: pruning to the model's *global* top-K crowds `direct`
 * candidates out. Measured on 120 fixtures, direct recall fell 99.1% (classical
 * order) to 83.0% (global top-16) while every trick kind gained 9-10 points.
 * The cause is structural, not a model defect: the generator emits roughly two
 * banks for every direct, so a global top-16 is mostly banks even when the
 * model scores directs perfectly sensibly.
 *
 * That matters because of Showboat's actual selection rule
 * (`selectBestWithReason`): a trick is played whenever one clears the
 * physics-derived reliability threshold, and *otherwise a reliable direct pot
 * is the fallback*. If no direct ever reaches physics, that fallback branch has
 * nothing to choose from.
 *
 * The fix reserves a small number of the K slots for the model's own
 * highest-scoring `direct` candidates, and only when the global ranking has not
 * already kept that many — so it costs nothing in the common case and cannot
 * silently override the model where the model was already doing the right
 * thing. Everything else stays allocated by learned rank.
 *
 * Deliberately NOT the per-kind floor the prior sprint prototyped in the eval
 * harness (`applyPerKindFloor`, floor=3 for all five kinds): that consumed 15
 * of 16 slots on reserves, and measured worse — overall recall 61.9% -> 54.9%
 * and direct recall 85.6% -> 64.6%, because it displaced the learned ranking
 * almost entirely. A reserve is a floor of 2 on ONE kind, applied only on
 * shortfall.
 *
 * **How large the effect actually is, stated plainly: small.** Swept over
 * R ∈ {0,1,2,3} on dev fixture seed 777001 (50 fixtures, disjoint from both
 * final evaluation seeds), the reserve fires on 0.00 / 0.00 / 0.02 / 0.14
 * decisions per turn respectively. It is close to a no-op on the currently
 * shipped model, because the Deep Sets ranker — unlike the Phase 2D MLP this
 * problem was originally diagnosed against — already keeps a mean of 4.4
 * `direct` candidates inside its own top-16. This is insurance, not a
 * performance win, and it is not presented as one.
 *
 * Value: 2, not 0, 1 or 3.
 *   - 0 provides no guarantee at all, and the failure mode is real rather than
 *     hypothetical: `searchPolicy.test.ts` drives a prior that ranks every
 *     non-direct above every direct and shows zero directs reach physics.
 *   - 1 gives a single fallback with no redundancy — if that one candidate
 *     scratches in its seeding simulation it is dropped outright (scratches are
 *     removed from `stats`), leaving the fallback branch empty again.
 *   - 3 measurably starts costing the trick budget that is the point of the
 *     hybrid: dev trick-attempt rate fell 98% -> 96% at R=3, with no
 *     corresponding gain in direct-fallback preservation (90.2% at every R).
 *   - 2 costs 0.02 promotions/turn on dev — indistinguishable from free — and
 *     guarantees a spare.
 *
 * Chosen on dev only, and frozen before the final evaluation ran; the freeze
 * and the full dev table are in docs/repair/release-candidate/showboat/.
 */
export const DEFAULT_PRIOR_RESERVE: Partial<Record<CandidateKind, number>> = { direct: 2 };

// How many candidates the learned prior forwards to physics by default.
//
// Derivation, not a guess: seeding costs 1 shot simulation + `rolloutsPerEval`
// (2) rollout units per candidate = 3 units, against `defaultConfig.simulations`
// = 60. Classical search therefore reaches at most 20 candidates in raw
// generation order and never looks at the rest. Keeping 16 leaves 60 - 48 = 12
// units — six UCB refinement rounds — for the candidates the model rated
// highest, instead of spending the entire budget on a fixed prefix of the
// generation order. Same total physics budget either way; different
// allocation of it. `eval/hybridEval.ts` measures whether that trade is worth
// it rather than assuming.
export const DEFAULT_PRIOR_KEEP_TOP = 16;

/** UI-responsiveness guard on the seeding loop. See `SearchConfig.seedTimeoutMs`. */
export const DEFAULT_SEED_TIMEOUT_MS = 2000;

/**
 * Wall-clock guard on the whole search. See `SearchConfig.searchTimeoutMs`.
 *
 * 3500 ms, not 2000 and not 5000. The binding constraint on *work* is still
 * `simulations` (60 units); this only stops the clock running past the point
 * where the turn stops feeling like a turn. Measured on an idle machine, seven
 * unhurried production turns spent 2,222-5,309 ms in physics with the budget
 * fully spent in all but the truncated ones, so 3,500 ms leaves the median turn
 * untouched, truncates the slowest ~quarter by a fraction of the refinement
 * loop, and keeps the decision total under the 5,000 ms cap with room for the
 * safety rung's simulations and the neural deadline on top.
 */
export const DEFAULT_SEARCH_TIMEOUT_MS = 3500;

export const defaultConfig: SearchConfig = {
  simulations: 60,
  rolloutDepth: 1,
  rolloutsPerEval: 2,
  seed: 12345,
  /**
   * DELIBERATELY UNBOUNDED. This config is shared by the unit suite and the
   * evaluation harness, and a wall-clock guard here makes both depend on how
   * loaded the machine is rather than on the physics budget.
   *
   * That is not hypothetical: when `DEFAULT_SEARCH_TIMEOUT_MS` (3,500 ms)
   * applied here, the Showboat suite failed roughly three runs in four, a
   * DIFFERENT test each time — `searchPolicy`, `overlayTruthfulness` and
   * `rankerIntegration` all run a real search, and a truncated search is a
   * different decision. A suite that fails at random trains people to ignore
   * it. An earlier sprint hit the same class of bug in the evaluation harness
   * and fixed it the same way.
   *
   * Live play is still bounded, and by a tighter clock: `withinDeadline`
   * (ai/brain.ts) folds the turn's remaining time in with `Math.min`, so a real
   * turn gets whatever is left of `DECISION_DEADLINE_MS` and `Infinity` never
   * reaches the search. `Deadline.none()` leaves this value alone, which is
   * what keeps tests and the harness deterministic.
   */
  searchTimeoutMs: Number.POSITIVE_INFINITY,
};

/**
 * LEGACY MIXED POLICY — evaluation baseline and training/self-play path only.
 * **Not the live policy.** See `selectBestWithReason`'s doc comment for why it
 * still exists and why it must not change.
 */
export const searchWithLegacySelection = (
  candidates: Candidate[],
  balls: Ball[],
  targets: number[],
  config: SearchConfig = defaultConfig,
): SearchResult => {
  const outcome = searchCandidates(candidates, balls, targets, config);
  const selection = selectBestWithReason(outcome.stats);
  return {
    ...outcome,
    best: selection.best,
    trace: outcome.trace && {
      ...outcome.trace,
      selectionReason: selection.reason,
      qualifyingTricks: selection.qualifyingTricks,
    },
  };
};

/**
 * LEGACY MIXED POLICY entry point (generates candidates, then selects with
 * `selectBestWithReason`). Retained unchanged because
 * `training/ranker/phase2c/selfplay.ts` generates the committed model's
 * training-state distribution through it: changing this policy would change
 * that distribution and break reproducibility of the shipped artifact.
 */
export const searchBaseline = (
  balls: Ball[],
  table: Table,
  targets: number[],
  config: SearchConfig = defaultConfig,
): SearchResult =>
  searchWithLegacySelection(generateCandidates(balls, table, targets), balls, targets, config);

/**
 * The authoritative shot search, over an already-generated candidate list.
 *
 * Split out from `searchBaseline` so the hybrid agent can generate candidates,
 * score them with the trained ranker, and hand the same list back here —
 * without the search having to know or care where the prior came from. Every
 * legality, reliability and value judgement below is made by the real physics
 * engine on this exact list, identically in both modes.
 *
 * **It does not choose a shot.** It used to; selection moved out to
 * `policy/trickOnly.ts` so that there is exactly one place in the codebase
 * where a playable shot is produced, and that place cannot produce a direct.
 */
export const searchCandidates = (
  candidates: Candidate[],
  balls: Ball[],
  targets: number[],
  config: SearchConfig = defaultConfig,
): SearchOutcome => {
  const prior = config.prior;
  const usePrior = prior !== undefined && prior.scores.length === candidates.length;
  const baseTrace = (): DecisionTrace => ({
    mode: usePrior ? "neural-hybrid" : "classical",
    candidatesGenerated: candidates.length,
    candidatesConsidered: 0,
    prunedByPrior: 0,
    physicsVerified: 0,
    verifiedIndices: [],
    legalPots: 0,
    scratched: 0,
    physicsCalls: 0,
    reservePromotions: 0,
    ineligible: 0,
    seedTimedOut: false,
    searchTimedOut: false,
    consideredIndices: [],
    physicsMs: 0,
    neuralInferenceMs: usePrior ? prior!.inferenceMs : undefined,
    modelId: usePrior ? prior!.source : undefined,
  });

  if (candidates.length === 0) {
    return { stats: [], allStats: [], verifications: [], simulations: 0, trace: baseTrace() };
  }

  // Separate any overlapping balls before handing to Rust — the TS animation
  // engine can leave balls at exact contact distance.
  const workBalls = balls.map(cloneBall);
  separateOverlaps(workBalls);

  const stats: CandidateStat[] = candidates.map((c) => ({
    candidate: c,
    visits: 0,
    value: 0,
    strength: 0,
    rails: c.banks,
    potsTarget: false,
    styleScore: 0,
    verified: false,
  }));

  // Visit order + pruning. Classical mode walks the raw generation order (by
  // kind, then shortest path first) and simply runs out of budget partway
  // through on a busy table. Hybrid mode walks the learned prior's order and
  // hands physics only the top `keepTop` — the same budget, spent on the
  // candidates the trained model rates highest instead of on a fixed prefix.
  //
  // `config.eligible`, when supplied, removes candidates from the visit order
  // entirely: no physics call is ever spent on a shot the policy could not
  // select. They keep their slot in `allStats` and in the trace, with
  // `verified: false`, so nothing disappears — it is only budget that moves.
  const isEligible = config.eligible ?? (() => true);
  const eligibleIdx = candidates.map((c, i) => (isEligible(c) ? i : -1)).filter((i) => i >= 0);
  const ineligible = candidates.length - eligibleIdx.length;
  let order: number[] = eligibleIdx;
  let prunedByPrior = 0;
  let reservePromotions = 0;
  if (usePrior) {
    const scores = prior!.scores;
    const logits = prior!.logits;
    // Global learned ranking, over EVERY candidate including ineligible ones —
    // `priorRank` reports what the model said, not what the policy allowed.
    const rankedAll = candidates.map((_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b);
    rankedAll.forEach((ci, rank) => {
      stats[ci].priorScore = scores[ci];
      stats[ci].priorRank = rank + 1;
      if (logits && logits.length === candidates.length) stats[ci].priorLogit = logits[ci];
    });
    // Only eligible candidates compete for the physics budget below.
    const ranked = rankedAll.filter((ci) => isEligible(candidates[ci]));

    const keep = Math.max(1, Math.min(prior!.keepTop, ranked.length));
    const globalTop = ranked.slice(0, keep);

    // Kind reserve (see DEFAULT_PRIOR_RESERVE): top up any kind that the global
    // top-K under-represents, using that kind's OWN highest-scoring candidates,
    // and evicting the lowest-ranked non-reserved survivor to pay for it. Total
    // kept — and therefore total physics budget — is unchanged.
    const kept = new Set(globalTop);
    const reserve = prior!.reserve;
    if (reserve) {
      const reservedIds = new Set<number>();
      for (const [kind, want] of Object.entries(reserve) as [CandidateKind, number][]) {
        if (!want || want <= 0) continue;
        const ofKind = ranked.filter((ci) => candidates[ci].kind === kind);
        const target = Math.min(want, ofKind.length);
        // Already-surviving members of this kind count toward the reserve, so
        // no slot is spent when the model kept enough of them on its own.
        const already = ofKind.filter((ci) => kept.has(ci));
        for (const ci of already.slice(0, target)) reservedIds.add(ci);
        let shortfall = target - already.length;
        if (shortfall <= 0) continue;
        for (const ci of ofKind) {
          if (shortfall <= 0) break;
          if (kept.has(ci)) continue;
          // Evict the worst-ranked survivor that is not itself reserved.
          let evict: number | null = null;
          for (let r = ranked.length - 1; r >= 0; r--) {
            const cj = ranked[r];
            if (kept.has(cj) && !reservedIds.has(cj)) {
              evict = cj;
              break;
            }
          }
          if (evict === null) break; // every survivor is reserved; nothing to trade
          kept.delete(evict);
          kept.add(ci);
          reservedIds.add(ci);
          reservePromotions++;
          shortfall--;
        }
      }
    }

    prunedByPrior = ranked.length - kept.size;
    // Visit the survivors in the model's own order. The reserve decides WHICH
    // candidates get physics; the model still decides in what order, so if the
    // budget runs short mid-seeding it runs short on the candidates the model
    // rated lowest.
    order = ranked.filter((ci) => kept.has(ci));
  }

  let seed = config.seed >>> 0;
  const nextSeed = (): number => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed;
  };

  // Strict budget accounting: `sims` counts every physics-engine call as it
  // actually happens (a single simulateShotWasm = 1 unit; a rolloutValueWasm
  // call = rolloutsPerEval units, since it runs that many continuations
  // internally), and the seed+refinement loop together are hard-capped at
  // `config.simulations` total units. The previous version estimated seeding
  // cost analytically from `stats.length * rolloutsPerEval` *after* the loop
  // ran — which silently diverged from what actually happened whenever
  // SEED_TIMEOUT_MS truncated the loop early or a scratch skipped a
  // candidate (both make the real seeded count less than `stats.length`),
  // and it didn't count the always-run simulateShotWasm call in the
  // net-seeded path at all — so total physics calls could exceed
  // `config.simulations` without the code ever knowing. Tracking real cost
  // as it's spent, against one shared ceiling, makes that structurally
  // impossible instead of relying on an estimate matching reality.
  let sims = 0;
  const BUDGET = config.simulations;
  const netScores = config.netSeedScores;
  const useNetScores = netScores !== undefined && netScores.length === candidates.length;
  const useNetSeed = useNetScores || config.netSeedValue !== undefined;
  const SEED_TIMEOUT_MS = config.seedTimeoutMs ?? DEFAULT_SEED_TIMEOUT_MS;
  const SEARCH_TIMEOUT_MS = config.searchTimeoutMs ?? DEFAULT_SEARCH_TIMEOUT_MS;
  const seedStart = performance.now();
  let searchTimedOut = false;
  const outOfTime = (): boolean => {
    if (performance.now() - seedStart <= SEARCH_TIMEOUT_MS) return false;
    searchTimedOut = true;
    return true;
  };

  const verifiedIndices: number[] = [];
  const verifications: (CandidateVerification | null)[] = candidates.map(() => null);
  let legalPots = 0;
  let scratched = 0;
  let seedTimedOut = false;

  for (const ci of order) {
    if (outOfTime()) break;
    if (performance.now() - seedStart > SEED_TIMEOUT_MS) {
      seedTimedOut = true;
      break;
    }
    if (sims + 1 > BUDGET) break; // can't even afford the seeding shot sim
    const s = stats[ci];
    const copy = workBalls.map(cloneBall);
    const sim = simulateShotWasm(copy, s.candidate.action);
    sims += 1;
    s.verified = true;
    verifiedIndices.push(ci);
    // Keep the real simulation record. This is the ONLY source of a non-null
    // `physics` block in the published trace, which is what makes "the trace
    // cannot claim physics that did not run" a mechanical property rather
    // than a convention.
    const scratchedHere = sim.pocketed.includes(0);
    verifications[ci] = {
      index: ci,
      firstContact: sim.firstContact,
      legalFirstContact: sim.firstContact === s.candidate.target,
      scratched: scratchedHere,
      legalPot: !scratchedHere && isLegalPot(sim, s.candidate),
      pocketed: [...sim.pocketed],
      railsBeforePot: railsBeforePot(sim),
      events: sim.events,
    };
    // Cue ball id is always 0. A scratch is a foul regardless of what else was
    // pocketed — skip the candidate entirely so it can't win UCB selection.
    if (scratchedHere) {
      scratched++;
      continue;
    }
    // Legality is decided here, by the real simulation, in both modes. No
    // prior score reaches this line — a candidate the model loved and a
    // candidate it hated are judged by exactly the same physics.
    s.potsTarget = isLegalPot(sim, s.candidate);
    if (s.potsTarget) legalPots++;
    s.rails = railsBeforePot(sim);
    const isComboLike =
      s.candidate.kind === "combo" ||
      s.candidate.kind === "double-bank" ||
      s.candidate.kind === "rail-combo";
    s.styleScore = s.rails + (isComboLike ? 1 : 0);

    if (useNetSeed) {
      // Per-candidate score when available (Phase 2A ranker); otherwise the
      // old flat whole-board scalar (parked policy/value net) as a fallback.
      // No additional physics cost beyond the simulateShotWasm call above —
      // the net score itself is not a physics rollout.
      const v = useNetScores ? netScores![ci] : config.netSeedValue!;
      s.value = v;
      s.strength = toStrength(v);
      s.visits = 1;
    } else {
      if (sims + config.rolloutsPerEval > BUDGET) continue; // leave unseeded (visits=0); can't afford it
      const v = rolloutValueWasm(
        workBalls,
        s.candidate.action,
        targets,
        config.rolloutDepth,
        config.rolloutsPerEval,
        nextSeed(),
      );
      sims += config.rolloutsPerEval;
      s.visits = 1;
      s.value = v;
      s.strength = toStrength(v);
    }
  }

  const seeded = stats.filter(s => s.visits > 0);

  // The refinement loop is where the seeding loop's freed budget was being
  // spent with nothing timing it. `outOfTime()` is the same clock, from the
  // same start instant, as the seeding guard above — one wall-clock budget for
  // the whole search rather than one for its first half.
  while (seeded.length > 0 && sims + config.rolloutsPerEval <= BUDGET && !outOfTime()) {
    const totalVisits = seeded.reduce((a, s) => a + s.visits, 0);
    let pick = seeded[0];
    let bestUcb = -Infinity;
    for (const s of seeded) {
      const ucb = s.value + UCB_C * Math.sqrt(Math.log(totalVisits + 1) / s.visits);
      if (ucb > bestUcb) { bestUcb = ucb; pick = s; }
    }
    const v = rolloutValueWasm(
      workBalls, pick.candidate.action, targets,
      config.rolloutDepth, config.rolloutsPerEval, nextSeed(),
    );
    sims += config.rolloutsPerEval;
    pick.value = (pick.value * pick.visits + v) / (pick.visits + 1);
    pick.visits += 1;
    pick.strength = toStrength(pick.value);
  }

  const sorted = [...stats]
    .filter(s => s.visits > 0)
    .sort((a, b) => b.visits - a.visits || b.value - a.value);

  const trace: DecisionTrace = {
    ...baseTrace(),
    candidatesConsidered: order.length,
    prunedByPrior,
    physicsVerified: verifiedIndices.length,
    verifiedIndices,
    legalPots,
    scratched,
    physicsCalls: sims,
    reservePromotions,
    ineligible,
    seedTimedOut,
    searchTimedOut,
    consideredIndices: order,
    physicsMs: performance.now() - seedStart,
  };
  return { stats: sorted, allStats: stats, verifications, simulations: sims, trace };
};

// A trick candidate must clear this strength score to be considered reliable
// enough to attempt over a safe direct pot. Tunable — this is the one number
// that encodes "how much risk Showboat is willing to take for style." 0.5
// means the search's own rollout-estimated success value must be at least as
// likely to work as not, before style preference is allowed to override
// safety. Exported so tests and the overlay can reference the same bar
// rather than a second copy of the number.
export const TRICK_RELIABILITY_THRESHOLD = 0.5;

// How much a style point (see `styleScore` — rails + a combo/double-bank
// bonus) is worth relative to strength when ranking *qualifying* trick
// candidates against each other. Small on purpose: style breaks ties and
// nudges toward flashier routes among comparably-reliable options, it does
// not let a wildly less reliable trick beat a more reliable one outright —
// that job belongs to `TRICK_RELIABILITY_THRESHOLD` above.
export const STYLE_WEIGHT = 0.12;

/**
 * Ranking score among *qualifying* tricks. Exported so the live trick-only
 * policy uses this exact expression rather than a second copy of it — rung 1
 * of the production ladder is byte-for-byte the same objective the evaluation
 * baseline uses, which is what makes the two arms comparable.
 */
export const trickUtility = (s: CandidateStat): number => s.strength + STYLE_WEIGHT * s.styleScore;

/**
 * LEGACY MIXED POLICY — **evaluation baseline only. Not the live policy.**
 *
 * A trick is preferred whenever at least one qualifies (pots its target AND
 * clears `TRICK_RELIABILITY_THRESHOLD`); among qualifying tricks the one
 * maximizing strength + style wins; and — this is the part the product
 * decision retired — a **direct** pot is selected when no trick qualifies.
 *
 * It is deliberately kept, unchanged, for three reasons:
 *
 *  1. `training/ranker/phase2c/selfplay.ts` generates the committed model's
 *     training-state distribution through `planTurn` -> `searchBaseline` ->
 *     this function. Changing it changes that distribution and breaks
 *     reproducibility of `showboat-ranker-phase2e-deepsets.onnx`.
 *  2. `eval/harness.ts` sources self-play fixtures the same way.
 *  3. It is arm A of the trick-only evaluation: the "previous opponent" that
 *     trick-only Showboat is measured against.
 *
 * Nothing on the live path calls it. `policy/trickOnlySourceGuard.test.ts`
 * asserts that mechanically.
 */
export function selectBest(sorted: CandidateStat[]): CandidateStat | null {
  return selectBestWithReason(sorted).best;
}

/** Which branch of the legacy `selectBest` fired. Baseline reporting only. */
export type SelectionReason =
  | "trick-qualified"
  | "no-trick-qualified"
  | "no-verified-pot"
  | "none";

export function selectBestWithReason(sorted: CandidateStat[]): {
  best: CandidateStat | null;
  reason: SelectionReason;
  qualifyingTricks: number;
} {
  const qualifyingTricks = sorted.filter(
    (s) => s.potsTarget && s.candidate.kind !== "direct" && s.strength >= TRICK_RELIABILITY_THRESHOLD,
  );
  if (qualifyingTricks.length > 0) {
    return {
      best: qualifyingTricks.reduce((a, b) => (trickUtility(b) > trickUtility(a) ? b : a)),
      reason: "trick-qualified",
      qualifyingTricks: qualifyingTricks.length,
    };
  }
  const pot = sorted.find((s) => s.potsTarget);
  if (pot) return { best: pot, reason: "no-trick-qualified", qualifyingTricks: 0 };
  return {
    best: sorted[0] ?? null,
    reason: sorted.length > 0 ? "no-verified-pot" : "none",
    qualifyingTricks: 0,
  };
}

export const chooseShot = (
  balls: Ball[],
  table: Table,
  targets: number[],
  config?: SearchConfig,
): CueAction | null => {
  const res = searchBaseline(balls, table, targets, config);
  return res.best ? res.best.candidate.action : null;
};
