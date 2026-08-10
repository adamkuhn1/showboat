import { type GameState, type PlayerId } from "../game/state";
import { type Table } from "../physics/table";
import { simulateShotWasm } from "../physics/wasm-bridge";
import {
  type SearchResult,
  type SearchConfig,
  type SearchOutcome,
  type SelectionReason,
  defaultConfig,
  searchCandidates,
  DEFAULT_PRIOR_KEEP_TOP,
  DEFAULT_SEARCH_TIMEOUT_MS,
} from "./shotSearch";
import { generateCandidates, type Candidate, type CandidateKind } from "./candidates";
import { legalTargets } from "./turn";
import { CUE_ID } from "../game/rack";
import { neuralEvaluator, type NeuralCandidateEvaluator } from "./neural/evaluator";
import {
  isTrickCandidate,
  selectTrickOnly,
  TrickOnlyInvariantError,
  type PlayableShot,
  type TrickOnlyDecision,
} from "./policy/trickOnly";
import { buildDecisionTrace, progressCandidates } from "./trace/build";
import { NO_PROGRESS, type SearchProgressSink } from "./search/progress";
import {
  type DecisionTraceV1,
  type FallbackTrace,
  type ModelIdentity,
  type SelectionRung,
} from "./trace/contract";
import { rankerHashWasVerified } from "./onnx";
import { Deadline } from "./deadline";

/**
 * Two brains, ONE policy.
 *
 * - `neural-hybrid`: `generateCandidates` -> one batched inference through the
 *   trained Deep Sets ranker -> the authoritative physics search, visited in
 *   the model's order and pruned to its top-K -> `selectTrickOnly`.
 * - `classical-trick-only`: the same, minus the prior. This is also the
 *   fallback whenever the model is absent, invalid, throws, times out, or
 *   returns nothing usable.
 *
 * There is no brain, mode, toggle or failure path in which a different
 * selection function runs. Both call `selectTrickOnly`, and there is exactly
 * one `selectTrickOnly` — which is why "Showboat never plays a direct" is a
 * property of the code rather than a claim about it.
 *
 * The model still cannot reach a decision the physics didn't sanction: it
 * supplies a prior over which candidates get simulated, nothing more. Legality
 * (`isLegalPot` on a real simulation), value (real rollouts) and the trick
 * reliability threshold are all downstream of it.
 */
export type BrainKind = "neural-hybrid" | "classical-trick-only";

/**
 * What an AI turn produces.
 *
 * `shot` is the only playable thing here and it is branded — see
 * `policy/trickOnly.ts`. `best` and `stats` are retained for the existing
 * overlay renderer; `best` is the selected candidate's stat, or null when the
 * shot is a generated safety kick that has no candidate. `decision` is the
 * published trace contract and is what new rendering code should read.
 */
export interface AiDecision extends SearchResult {
  shot: PlayableShot | null;
  decision: DecisionTraceV1;
}

export interface Brain {
  kind: BrainKind;
  /** Human-facing description of what is actually running. Never aspirational. */
  label: string;
  plan: (
    state: GameState,
    table: Table,
    player: PlayerId,
    config?: SearchConfig,
    /**
     * The turn's wall clock. Omitted by the evaluation harness and by unit
     * tests, which get `Deadline.none()` so their results depend on the physics
     * budget rather than on how loaded the machine is.
     */
    deadline?: Deadline,
    /**
     * Live observer of this decision. Defaults to `NO_PROGRESS`. The events are
     * published from the code below as it runs; nothing downstream may
     * manufacture one. See `search/progress.ts`.
     */
    progress?: SearchProgressSink,
  ) => Promise<AiDecision>;
}

/**
 * Cap on ONE `session.run()`, on top of whatever the turn's own `Deadline` has
 * left.
 *
 * Not a guess: the corrected gate measured this artifact's inference at a
 * median of 1.63 ms and a p95 of 4.60 ms over 400 fixtures
 * (docs/repair/visual-authorship/showboat/CORRECTED_GATE_RESULT.md section 5).
 * 250 ms is ~54x that p95 — it cannot fire on a slow-but-working machine. The
 * race loser is treated exactly like "the model returned nothing": classical
 * trick-only, labelled `inference-timeout`.
 *
 * What this does NOT cover, and never did, is `evaluator.load()` — where
 * `InferenceSession.create()` runs. That is the turn `Deadline`'s job
 * (`ai/deadline.ts`), applied at `planTurnTraced`, and it is the guard whose
 * absence let a stalled load wedge a turn at "searching…" forever.
 */
export const NEURAL_SCORE_DEADLINE_MS = 250;

const emptySearch = (): SearchOutcome => ({
  stats: [],
  allStats: [],
  verifications: [],
  simulations: 0,
  trace: {
    mode: "classical",
    candidatesGenerated: 0,
    candidatesConsidered: 0,
    prunedByPrior: 0,
    physicsVerified: 0,
    verifiedIndices: [],
    consideredIndices: [],
    legalPots: 0,
    scratched: 0,
    physicsCalls: 0,
    reservePromotions: 0,
    ineligible: 0,
    seedTimedOut: false,
    physicsMs: 0,
  },
});

const emptyDecision = (): TrickOnlyDecision => ({
  shot: null,
  rung: null,
  qualifyingTricks: 0,
  utility: null,
  excludedIndices: [],
  unmeasuredTrickIndices: [],
  verifiedTricks: 0,
  safetySimsSpent: 0,
  safety: null,
  safetyQuality: null,
});

/**
 * Fold the turn's remaining wall clock into the search config.
 *
 * `Deadline.none()` leaves `searchTimeoutMs` exactly as the caller set it, so
 * the evaluation harness's `Infinity` survives. A live turn takes the tighter
 * of the search's own default and whatever the turn has left.
 */
const withinDeadline = (cfg: SearchConfig, deadline: Deadline): SearchConfig => {
  const remaining = deadline.remainingMs();
  if (remaining === Number.POSITIVE_INFINITY) return cfg;
  return {
    ...cfg,
    searchTimeoutMs: Math.min(cfg.searchTimeoutMs ?? DEFAULT_SEARCH_TIMEOUT_MS, remaining),
  };
};

const legacyReasonFor = (rung: SelectionRung | null): SelectionReason => {
  switch (rung) {
    case "trick-qualified":
      return "trick-qualified";
    case "trick-below-threshold":
      return "no-trick-qualified";
    default:
      return "none";
  }
};

/**
 * Assemble the turn result: the branded shot, the published trace, and the
 * legacy `best`/`stats` shape the current overlay renderer still reads.
 *
 * `best` is looked up from the selected candidate's generation index, so the
 * highlighted candidate is by construction the shot being played. It is null
 * for a safety kick, which is not a member of the candidate list — a renderer
 * must read `decision.selected` for that case rather than infer one.
 */
const assemble = (
  outcome: SearchOutcome,
  decision: TrickOnlyDecision,
  state: GameState,
  player: PlayerId,
  targets: number[],
  physicsUnitsAllowed: number,
  model: ModelIdentity | null,
  fallback: FallbackTrace | null,
  timing: { totalMs: number; neuralEncodeMs: number | null; neuralRunMs: number | null; selectionMs: number },
): AiDecision => {
  const idx = decision.shot?.candidateIndex ?? null;
  return {
    ...outcome,
    best: idx === null ? null : outcome.allStats[idx] ?? null,
    // Compatibility shim for the CURRENT overlay panel, which renders a
    // sentence keyed off the legacy `SelectionReason`. The mapping is lossy but
    // not untrue: rung 2 really is "no trick cleared the reliability bar", and
    // the two safety rungs really are "no trick candidate was measurably
    // executed". New rendering should read `decision.selected.rung`, which has
    // all four values; this line goes away when it does.
    trace: outcome.trace && { ...outcome.trace, selectionReason: legacyReasonFor(decision.rung) },
    shot: decision.shot,
    decision: buildDecisionTrace({
      outcome,
      decision,
      state,
      player,
      targets,
      physicsUnitsAllowed,
      model,
      fallback,
      timing: {
        totalMs: timing.totalMs,
        neuralEncodeMs: timing.neuralEncodeMs,
        neuralRunMs: timing.neuralRunMs,
        physicsMs: outcome.trace?.physicsMs ?? 0,
        selectionMs: timing.selectionMs,
      },
    }),
  };
};

/**
 * Run the search and the policy. Shared by both brains, so the fallback path
 * is not a second implementation that could drift from the primary one.
 */
const decide = (
  state: GameState,
  table: Table,
  player: PlayerId,
  targets: number[],
  /**
   * Generated by the caller, so the neural brain scores and searches ONE list.
   * It used to generate its own here as well, which meant the candidate list
   * was built twice on every neural turn and — once the live stream existed —
   * that the model's ranking referred to a list the observer had not been shown.
   */
  candidates: Candidate[],
  cfg: SearchConfig,
  model: ModelIdentity | null,
  fallback: FallbackTrace | null,
  prior: SearchConfig["prior"],
  neuralEncodeMs: number | null,
  neuralRunMs: number | null,
  progress: SearchProgressSink,
): AiDecision => {
  const t0 = performance.now();
  // The eligibility filter is a BUDGET decision, not the exclusion mechanism:
  // it stops the search paying for simulations of shots the policy could never
  // select. Directs still appear in the trace, with `physics: null` and an
  // honest rejection reason. `trickOnly.test.ts` A7 proves selection alone is
  // sufficient by running with this filter off.
  const outcome = searchCandidates(candidates, state.balls, targets, {
    ...cfg,
    prior,
    eligible: isTrickCandidate,
    progress,
  });

  const tSel = performance.now();
  let decision: TrickOnlyDecision;
  try {
    decision = selectTrickOnly(outcome.allStats, outcome.verifications, {
      // `turn: player` explicitly: the safety rung verifies its kicks through
      // the real `applyShotRules`, which decides first-contact legality from
      // `pre.turn`'s group. Deriving the shooter from the argument rather than
      // from the state means a caller that plans for a player who is not to
      // move cannot silently get the other player's legality rules.
      state: { ...state, turn: player },
      table,
      targets,
      simulate: simulateShotWasm,
    });
  } catch (e) {
    // Layer 3. Unreachable unless the type or the partition has been broken by
    // a later edit. Degrade to the safety rung — a worse shot, never a direct,
    // and never a rejected promise that would hang the turn.
    if (!(e instanceof TrickOnlyInvariantError)) throw e;
    console.error(`[showboat] trick-only invariant violated: ${e.message}`);
    decision = selectTrickOnly([], [], {
      state: { ...state, turn: player },
      table,
      targets,
      simulate: simulateShotWasm,
    });
  }
  const selectionMs = performance.now() - tSel;

  const result = assemble(outcome, decision, state, player, targets, cfg.simulations, model, fallback, {
    totalMs: performance.now() - t0,
    neuralEncodeMs,
    neuralRunMs,
    selectionMs,
  });

  // The three rejection reasons the SEARCH cannot know, because they are
  // selection decisions rather than physics results — the measured-route
  // rejection included, since the search records an event log and the policy is
  // what classifies it. Published from the finished trace rather than
  // recomputed, so the live stream and the record cannot disagree about why a
  // route lost.
  for (const c of result.decision.candidates) {
    if (
      c.rejection === "planned-trick-not-measured" ||
      c.rejection === "below-reliability-threshold" ||
      c.rejection === "lower-utility-than-selected"
    ) {
      progress.rejected(c.index, c.rejection);
    }
  }
  const sel = result.decision.selected;
  if (sel) progress.selected(sel.candidateIndex, sel.kind, sel.rung);

  // Last, and only here: the decision is finished, so nothing further will be
  // published. The figures are the search's own, read off the trace rather than
  // recounted.
  const it = outcome.trace;
  progress.completed(
    it?.physicsCalls ?? 0,
    it?.physicsMs ?? 0,
    it?.physicsVerified ?? 0,
    it?.seedTimedOut ?? false,
    it?.searchTimedOut ?? false,
  );

  return result;
};

const noLegalTarget = (
  state: GameState,
  player: PlayerId,
  cfg: SearchConfig,
  model: ModelIdentity | null,
  fallback: FallbackTrace | null,
): AiDecision =>
  assemble(emptySearch(), emptyDecision(), state, player, [], cfg.simulations, model, fallback, {
    totalMs: 0,
    neuralEncodeMs: null,
    neuralRunMs: null,
    selectionMs: 0,
  });

/**
 * The classical trick-only brain.
 *
 * `fallback` is non-null when this brain is standing in for the neural one —
 * i.e. when a neural decision was asked for and could not be made. It used to
 * be unconditionally null, which is why a genuine `model-absent` or
 * `model-invalid` condition produced a trace that recorded no fallback at all:
 * the visitor was told the truth by the badge, and the published trace was not.
 */
export const classicalTrickOnlyBrain = (fallback: FallbackTrace | null = null): Brain => ({
  kind: "classical-trick-only",
  label: "Physics search",
  plan: async (state, table, player, config, deadline, progress = NO_PROGRESS) => {
    const cfg = withinDeadline(config ?? defaultConfig, deadline ?? Deadline.none());
    const targets = legalTargets(state, player);
    const cue = state.balls.find((b) => b.id === CUE_ID);
    if (!cue || targets.length === 0) return noLegalTarget(state, player, cfg, null, fallback);

    progress.started("classical", cfg.simulations);
    // A stand-in for the neural brain says so before it does anything else.
    if (fallback) progress.fallback(fallback.cause, fallback.detail);
    const candidates = generateCandidates(state.balls, table, targets);
    // Geometry first: nothing downstream may refer to a candidate the observer
    // has not been given a route for.
    progress.candidatesGenerated(progressCandidates(candidates, { x: cue.pos.x, y: cue.pos.y }));
    // Directs are excluded by the policy, not by the physics, so the exclusion
    // is published here — at the moment it is applied — rather than inferred.
    candidates.forEach((c, i) => {
      if (!isTrickCandidate(c)) progress.rejected(i, "direct-excluded-by-policy");
    });

    const res = decide(state, table, player, targets, candidates, cfg, null, fallback, undefined, null, null, progress);
    if (!fallback) return res;
    return { ...res, trace: res.trace && { ...res.trace, fallbackReason: fallback.detail } };
  },
});

export const neuralTrickOnlyBrain = (
  evaluator: NeuralCandidateEvaluator = neuralEvaluator,
  keepTop: number = DEFAULT_PRIOR_KEEP_TOP,
  // Trick-only reserves nothing for directs: reserving physics slots for a kind
  // that can never be selected would be spending the budget on nothing. This is
  // part of arm B of the pre-registered evaluation, stated rather than slipped
  // in. Correctness does not depend on it.
  reserve: Partial<Record<CandidateKind, number>> = {},
): Brain => ({
  kind: "neural-hybrid",
  label: "Neural evaluator + physics search",
  plan: async (state, table, player, config, deadline, progress = NO_PROGRESS) => {
    const dl = deadline ?? Deadline.none();
    const cfg = withinDeadline(config ?? defaultConfig, dl);
    const targets = legalTargets(state, player);
    const cue = state.balls.find((b) => b.id === CUE_ID);
    if (!cue || targets.length === 0) return noLegalTarget(state, player, cfg, null, null);

    progress.started("neural-hybrid", cfg.simulations);
    const candidates = generateCandidates(state.balls, table, targets);
    progress.candidatesGenerated(progressCandidates(candidates, { x: cue.pos.x, y: cue.pos.y }));
    candidates.forEach((c, i) => {
      if (!isTrickCandidate(c)) progress.rejected(i, "direct-excluded-by-policy");
    });
    const before = evaluator.getState();

    // Every way inference can fail resolves to the SAME classical trick-only
    // search, with an accurate `fallback.cause`. None of them can hang: the
    // evaluator never throws (it catches ORT errors internally) and the call is
    // raced against both the per-call cap and the turn's remaining budget.
    const raced = await dl.race(
      evaluator.score(state.balls, table, candidates),
      NEURAL_SCORE_DEADLINE_MS,
    );
    const scored = raced.ok ? raced.value : null;

    if (!scored) {
      const after = evaluator.getState();
      // An empty candidate set is not a model failure. The evaluator returns
      // null for it just as it does for a real inference problem, and reading
      // that as "the model gave nothing back" blames the ranker for a board
      // that offered it nothing to rank. Checked first so the more specific
      // cause wins.
      const noCandidates = candidates.length === 0;
      // `score()` downgrades the evaluator to `invalid` when onnxruntime throws
      // *during* a run. Comparing before with after is what separates that —
      // `inference-error`, a session that broke on this call — from an
      // evaluator that was already invalid when it was handed over, which is
      // `model-invalid` and is not this call's fault.
      const brokeDuringRun = before.status === "ready" && after.status === "invalid";
      const [cause, detail]: [FallbackTrace["cause"], string] = noCandidates
        ? ["no-candidates", "no shots to rank from this position"]
        : !raced.ok
          ? ["inference-timeout", `inference exceeded ${Math.round(raced.waitedMs)} ms`]
          : brokeDuringRun
            ? ["inference-error", `model unavailable: ${(after as { reason?: string }).reason ?? "onnx runtime error"}`]
            : after.status === "ready"
              ? ["no-scores", "the model scored nothing for this set of shots"]
              : after.status === "invalid"
                ? ["model-invalid", `model unusable: ${after.reason}`]
                : ["model-absent", `model unavailable: ${after.reason}`];
      // The real cause, published the moment it is established.
      progress.fallback(cause, detail);
      const res = decide(state, table, player, targets, candidates, cfg, null, { from: "neural-hybrid", to: "classical-trick-only", cause, detail }, undefined, null, null, progress);
      return {
        ...res,
        trace: res.trace && { ...res.trace, fallbackReason: detail },
      };
    }

    const manifest = evaluator.getManifest()!;
    const model: ModelIdentity = {
      artifact: manifest.artifact,
      sha256: manifest.onnx_sha256,
      schema: manifest.schema_version,
      hashVerified: rankerHashWasVerified(),
    };
    const modelId = manifest.artifact.replace(/\.onnx$/, "");

    // The model's ORDER, from the model's own output. The ranking is computed
    // here and used verbatim below by `searchCandidates` — this is not a second
    // sort for display, it is the same array, published before any physics ran
    // so that the order on screen is demonstrably the order the search took.
    const ranked = candidates
      .map((_, i) => i)
      .sort((a, b) => scored.scores[b] - scored.scores[a] || a - b);
    progress.neuralScored(
      modelId,
      scored.inferenceMs,
      ranked.map((index, r) => ({
        index,
        rank: r + 1,
        score: scored.scores[index],
        logit: scored.logits?.[index] ?? null,
      })),
    );

    return decide(
      state,
      table,
      player,
      targets,
      candidates,
      cfg,
      model,
      null,
      {
        scores: scored.scores,
        logits: scored.logits,
        keepTop,
        reserve,
        source: modelId,
        inferenceMs: scored.inferenceMs,
      },
      scored.encodeMs,
      scored.runMs,
      progress,
    );
  },
});

/**
 * Why the neural path is unusable right now, as a `FallbackTrace`. Null when it
 * is usable, or when nobody asked for it.
 *
 * Split out from `getBrain` so `plan.ts` can pass a *more specific* reason it
 * already knows — a load that blew its deadline is `model-load-timeout`, not
 * the `model-absent` the evaluator's own state would report a moment later.
 */
export const neuralUnavailability = (
  evaluator: NeuralCandidateEvaluator,
): FallbackTrace | null => {
  if (evaluator.isReady()) return null;
  const state = evaluator.getState();
  const base = { from: "neural-hybrid", to: "classical-trick-only" } as const;
  if (state.status === "invalid") {
    return { ...base, cause: "model-invalid", detail: `model unusable: ${state.reason}` };
  }
  if (state.status === "absent") {
    return { ...base, cause: "model-absent", detail: `model unavailable: ${state.reason}` };
  }
  // Validated manifest, but no ranker session in this realm — the page's own
  // evaluator after `preflight()`, for instance. Absent, not invalid.
  return {
    ...base,
    cause: "model-absent",
    detail: "the ranker session is not available on this thread",
  };
};

/**
 * Pick the brain for this turn. `preferNeural` is the user-facing toggle (the
 * model-disabled comparison mode); the evaluator's own validated readiness is
 * the hard gate. Asking for neural when no valid artifact loaded gets the
 * classical trick-only brain, honestly labelled — never a neural label over a
 * classical decision, and never a policy change either way.
 *
 * That classical stand-in now carries the reason with it. It did not before,
 * and the consequence was a published trace with `fallback: null` on exactly
 * the two conditions — `model-absent`, `model-invalid` — the contract has
 * causes for; both were unreachable in practice because this function routed
 * around them.
 */
export const getBrain = (
  preferNeural: boolean,
  evaluator: NeuralCandidateEvaluator = neuralEvaluator,
  /** A reason the caller already knows, overriding the evaluator's own. */
  unavailable: FallbackTrace | null = null,
): Brain => {
  // `unavailable` still applies with the toggle off: it can describe a failure
  // that has nothing to do with the model — the planning worker going silent,
  // for one — and that failure happened whether or not neural was asked for.
  if (!preferNeural) return classicalTrickOnlyBrain(unavailable);
  const reason = unavailable ?? neuralUnavailability(evaluator);
  return reason === null ? neuralTrickOnlyBrain(evaluator) : classicalTrickOnlyBrain(reason);
};

/**
 * The opponent description. Derived from validated model state.
 *
 * It names which *evaluator* is running, not which policy: the policy is
 * always trick-only, in every mode, so putting it in this string would say
 * nothing. `src/ui/opponentClaims.test.ts` pins these exact strings.
 */
export const brainLabel = (
  preferNeural: boolean,
  evaluator: NeuralCandidateEvaluator = neuralEvaluator,
): string =>
  preferNeural && evaluator.isReady() ? "the neural + physics opponent" : "the physics-search opponent";
