# Showboat

2D bar-pool game with a from-scratch event-based physics engine, a search-driven
opponent, a **trained candidate-ranking model** that ships and runs in the
browser, and a **live reasoning overlay** driven entirely by actual search data.

> **Which brain is playing, today?**
>
> By default, the **trained neural ranker** (Phase 2E Deep Sets, relational)
> orders and prunes candidates before the same physics search runs — physics
> still verifies every shot. The UI says "the neural + physics opponent" and
> the overlay shows the model's per-candidate calibrated make-estimate, its
> ranking, and how many candidates it pruned before physics ran.
>
> The **classical physics search** (`src/ai/shotSearch.ts` — a flat UCB
> bandit over the candidate-shot set, not a mislabeled "MCTS") remains fully
> available as an **explicitly selectable comparison mode** via the
> **neural ranking** toggle — uncheck it to play against physics search alone.
>
> **Why neural is the default.** The original release-candidate gate
> (`docs/repair/release-candidate/showboat/DECISION_GATE.md`), pooled over
> 210 held-out fixtures against seeds `20260805`/`424242`, failed 2 of 7
> measurable criteria (C: foul/scratch ceilings, D: trick-shot benefit) — but
> both failures were **mis-specifications, not real regressions**, honestly
> disclosed by the team that ran it and independently confirmed
> arithmetically by a cold review
> (`docs/repair/release-candidate/reviews/ml-truth.md`). Criterion C's
> threshold (+2.0pp) was narrower than the achievable 95% CI half-width
> (≈2.09pp) at n=210 **even under a true null** — it was expected to fail
> regardless of the model's real behavior. Criterion D measured "any
> non-direct shot," which classical already attempts on 96.2% of turns
> (capping the maximum possible gain at +3.8pp against a +5pp bar) and which
> conflates a plain single-rail bank shot with the "multi-wall combos"
> `CLAUDE.md`/`PLAN.md` actually name as the product's trick-shot target.
>
> A **corrected gate**
> (`docs/repair/visual-authorship/showboat/CORRECTED_GATE.md`) was frozen
> **before** running — fixing C's power (same ±0.02/±0.01 thresholds, n
> increased to 400) and D's construct (restricted to `{double-bank, combo,
> rail-combo}`, same +5pp/CI-excluding-zero bar) — then run **exactly once**
> against the **currently-shipped Phase 2E model**, on a genuinely fresh
> held-out seed (`55508219`, verified unused in any prior sweep). Full
> writeup: `docs/repair/visual-authorship/showboat/CORRECTED_GATE_RESULT.md`.
>
> **Result: every measurable criterion passes** (400 paired fixtures):
>
> - **Shot quality is non-inferior** — legal-pot 92.0% vs 92.3%
>   (-0.25pp, 95% CI **[-2.27pp, +1.77pp]**, includes zero — not a claim of
>   superiority), regret 0.040 vs 0.037, scratch 0.0% in both.
> - **C′ passes**: foul -0.50pp **[-2.58pp, +1.58pp]** (upper bound clears
>   +2.0pp), scratch 0.0% both arms.
> - **D′ passes**: multi-wall-combo (double-bank/combo/rail-combo) selection
>   rate rises 8.8% → 25.5% of all 400 decisions (**+16.75pp, 95% CI
>   [+12.71pp, +20.79pp]** — excludes zero), completion rate +16.50pp
>   [+12.54pp, +20.46pp].
> - Equal budget (0 over-budget either arm, mean call diff +0.03), latency
>   (hybrid p95 0.99× classical, neural inference median 1.63ms), and
>   direct-fallback preservation (95.0% vs classical's 100.0%, within the
>   frozen -5pp allowance) all pass unchanged from the original gate's
>   formulas and thresholds.
>
> Full games remain **descriptive, not gate evidence**, exactly as declared
> in the original gate (the power to detect a game-level win-rate difference
> at this scale is too low to be evidence either way): 40 games, 30 decided,
> hybrid 19–11. Reproduce with `npm run eval:hybrid -- --fixtures 400 --games
> 40 --seed 55508219 --out corrected_55508219.json` then `npx tsx
> eval/correctedGateReport.ts eval/results/corrected_55508219.json`.

## ML status (2026-08-03, redirected — read before touching training/)

Two separate ML tracks exist; do not conflate them:

1. **`training/ranker/` — active, real, small.** A candidate-ranking MLP
   trained on data generated from the authoritative Rust/WASM physics (the
   exact binary this app ships), with a real checkpoint, a real ONNX export,
   and automated tests proving it changes candidate ordering, budget
   allocation, and final shot selection. See `training/ranker/README.md` for
   reproduction commands and honest limitations, and
   `public/model/ranker/README.md` for how the shipped artifact is delivered
   and verified. **Now wired into live gameplay** as an opt-in mode (see the
   banner above).
2. **`training/showboat_env/` — parked.** The original PoolTool+LightZero
   self-play pipeline described further down this README. It has never been
   executed (PoolTool needs Python ≥3.10; the dev box is 3.9) — no checkpoint
   or ONNX file from it exists anywhere in git history. Kept for the record,
   not deleted, not the active path. See `training/README.md`'s status
   banner and `docs/repair/showboat-ml/ARCHITECTURE_DECISION.md` for why
   candidate ranking was chosen over self-play as the active track.

## Run

```bash
npm install                 # at the repo root (workspaces)
npm run dev -w @portfolio-suite/showboat      # http://localhost:5175
```

Play: move the mouse to aim, set power / english / draw-follow, **Shoot**.
Toggle **vs AI** to play the opponent and **overlay** to see its reasoning.

Verify:

```bash
npm run typecheck -w @portfolio-suite/showboat   # tsc, strict
npm test -w @portfolio-suite/showboat            # vitest (TS engine + game + AI + model)
npm run build -w @portfolio-suite/showboat       # vite build + the shipped-model gate
npm run test:wasm -w @portfolio-suite/showboat   # cargo test (Rust physics/rollout)
npm run build:wasm -w @portfolio-suite/showboat  # regenerate src/wasm from Rust
npm run eval:hybrid -w @portfolio-suite/showboat # classical vs neural at equal budget
npm run eval:gate -w @portfolio-suite/showboat -- eval/results/final_20260805_r2.json eval/results/final_424242_r2.json
# run from apps/showboat/ (correctedGateReport.ts is invoked directly, not via a package script):
npx tsx eval/correctedGateReport.ts eval/results/corrected_55508219.json
```

`eval:hybrid` writes one result file per fixture seed. `eval:gate` pools the
per-fixture pairs across seeds and applies the **original, historical**
release-candidate gate
(`docs/repair/release-candidate/showboat/DECISION_GATE.md`) — kept unedited
as the record of what actually ran and failed. `eval/correctedGateReport.ts`
applies the **corrected** gate
(`docs/repair/visual-authorship/showboat/CORRECTED_GATE.md`, the one that
determines today's default — see the banner above) to a single fresh-seed
result file. Both scripts cross-check that the classical arm is
byte-identical across runs of the same seed — the search's wall-clock
seeding guard is disabled during evaluation (`SearchConfig.seedTimeoutMs`)
precisely so results measure policy and not machine load.

**Run evaluations sequentially on an idle machine.** They report real decision
latency, and the runs are long (~20 min per 120-fixture seed; the corrected
gate's 400-fixture/40-game run took ~52 minutes on the machine it was measured
on).

The committed `src/wasm/` pkg means a plain `npm run build` needs **no Rust
toolchain**; `build:wasm` is only for regenerating it after changing the crate.

## Architecture

Polyglot where it's load-bearing (per `PLAN.md §0`): the physics + search hot
loop is **Rust→WASM**, the UI/rules/inference glue is **TypeScript**, the
training pipeline is **Python**.

```
physics-core/  (Rust → WASM)   event-based Han-2005 physics + UCB search rollout loop
  src/physics/                 pure-TS reference engine (the test ORACLE + spec)
  src/wasm/                    committed wasm-pack output (JS build needs no cargo)
src/game/                      8-ball ruleset + game controller (engine-agnostic)
src/ai/                        candidate generator, UCB shot search, ONNX loader, brain seam
src/ai/neural/                 NeuralCandidateEvaluator: manifest validation + batched inference
src/ai/trace/                  the decision-trace contract the renderer reads (types only)
public/model/ranker/           the shipped, hash-verified trained artifact + its manifest
eval/                          equal-budget classical-vs-hybrid evaluation harness
src/render/                    canvas render, the decision presentation machine, contact marks
src/ui/                        the panel, the opponent-turn orchestration, the planning worker
training/      (Python)        pooltool + LightZero self-play → ONNX
```

### Physics engine (event-based, Han-2005 lineage)

Not a fixed-timestep simulation: within each **motion phase** (sliding / rolling
/ spinning) a ball follows a closed-form trajectory, so the engine **solves for
the time of the next event** (ball-ball, ball-cushion, pocket, phase change) and
advances directly to it. Ball-ball collisions model **throw**; cushion rebounds
use the **Han-2005** coupling where the nose contacts the ball above its equator,
so **english changes the rebound angle and speed** — banks behave realistically.
This mirrors the training simulator (**pooltool**, same Han-2005 lineage) so a
policy trained in Python behaves consistently at play time. The physics was
written in-house (referencing pooltool/tailuge ideas, copying **no** GPL code).

The engine lives twice, on purpose:
- **Rust** (`physics-core/`) is what ships — an AI decision runs thousands of
  full-shot rollouts, a hot loop that honestly wants a compiled language.
- **TypeScript** (`src/physics/`) is the reference implementation and the vitest
  oracle. Both share the same equations and a final de-overlap pass, kept in
  parity; `cargo test` and `vitest` assert the same cases (momentum
  conservation, ~90° cut separation, energy-never-gained, no resting overlap,
  cushion rebound, action-space caps).

### No jump / no massé — impossible at the action level

The cue action is `{ phi, power, sideSpin, topSpin }`. There is **no elevation
(`theta`) parameter anywhere** — not in `src/physics/cue.ts`, not in
`physics-core/src/cue.rs`, not in `training/showboat_env/action_space.py`. A jump
needs downward elevation into the ball; a massé needs steep elevation + extreme
english. With the parameter simply absent from the type, those shots are
**unrepresentable**, not filtered. `V0` is additionally capped so raw speed can't
launch the ball airborne.

## Train vs. play split

**Active track (`training/ranker/`):** the dataset generator loads the exact
WASM physics binary this app ships (`src/wasm/showboat_physics_bg.wasm`) from
Node, reusing `candidates.ts`/`rules.ts`/`trace.ts` unmodified, so training
labels and play-time physics are not just "the same lineage" — they're the
same compiled artifact. Training itself is Python/PyTorch, reading the
already-encoded rows the Node generator wrote (no second feature encoder to
drift out of sync). See `training/ranker/README.md`.

**Parked track (`training/showboat_env/`), described for the historical
record:** `pooltool` simulates (a *different*, independent physics engine
from the Rust core above — not verified equivalent, see
`docs/repair/showboat-ml/01-current-ml-audit.md`); `LightZero` (Sampled
EfficientZero → EfficientZero V2) runs self-play over the 4-D continuous
action (`phi, V0, a, b`, **no theta**); reward is **outcome-only**. The
policy/value MLP exports to **ONNX**. Never executed on this dev box
(PoolTool needs Python 3.10+; the box is 3.9) or anywhere else — see
`training/README.md`'s status banner.

**Play (browser), today:** `src/ai/shotSearch.ts`'s flat UCB bandit searches
over the candidate-shot set using the Rust physics for rollouts. Default mode
is uniform priors + a physics rollout value, no model involved.

With **neural ranking** enabled, `src/ai/neural/evaluator.ts` encodes every
generated candidate with the same `encode.ts` the training data was built with,
runs one batched `session.run()` through the committed ONNX artifact, applies
the manifest's Platt calibration, blends `double-bank` scores halfway toward
the training-split kind mean (the artifact's own documented weak kind), and
hands the result to `searchCandidates` as a **prior only**. The prior decides
the order candidates are physics-verified in, and which ones are dropped before
any simulation runs. It never supplies a value: `potsTarget` still comes from
`isLegalPot()` on a real WASM simulation, `strength` still comes from real
rollouts, and the trick-reliability threshold is still applied to that
physics-derived strength. `src/ai/neural/hybrid.test.ts` proves those
guarantees with an adversarial prior, so they hold for any score vector the
model could ever emit — not just for the ones it happens to emit today.

## How the opponent's turn is presented

The opponent's decision is shown as a sequence, not as a finished panel that
appears all at once:

```
IDLE → ENUMERATING → RANKING → VERIFYING → SELECTED → READY → STROKE → SHOOTING → SETTLED
        evaluating    neural     physics     selected   ready
                      ranking  verification            to shoot
```

**RANKING is absent, not greyed, when no model ran.** Turning the ranker off in
the reasoning panel produces a structurally shorter sequence, which is the most
honest rendering of the toggle and makes the model's contribution legible
without a sentence of claim.

`src/render/presentation.ts` is the whole machine and it is **pure** — it takes
a completed `DecisionTraceV1` and a wall-clock offset and returns what belongs
on the felt. No canvas, no React, no timers, so `presentation.test.ts` can test
the sequence as arithmetic.

### What is measured and what is paced

The search finishes before the sequence begins. Everything drawn — every route,
every rejection reason, every contact — is read off the trace the opponent
actually decided with; what the presentation adds is the *order and speed* of
the reveal. That is stated in the panel on the first opponent turn of a session,
and here.

Durations derive from trace size, never from fixed waits:

| state | length |
|---|---|
| ENUMERATING | `clamp(candidatesGenerated × 18 ms, 400, 900)` |
| RANKING | `600 ms`, or the state does not exist |
| VERIFYING | the search's **real** physics time (`timing.physicsMs`), revealed one candidate per measured average |
| SELECTED | `clamp(250 + losers × 20 ms, 350, 800)` |
| READY | `clamp(500 + words × 28 ms, 700, 1400)` |
| STROKE | `350 ms`, direction and power from the real `CueAction` |
| SHOOTING | the authoritative simulation at **1.0×** |

A 6.5 s ceiling is applied as a uniform scale over the *hold* states only —
never over VERIFYING, because scaling preserves relative cost and truncating
would misrepresent it. A per-session decay shortens the holds from the third
opponent turn onward, and pointer-down / `Space` skips to READY in 250 ms.

**Known gap, stated rather than papered over:** the trace contract carries no
per-candidate verification time and no progress stream, so VERIFYING is paced
by the *average* real physics cost rather than by each candidate's own. It makes
no claim about any individual candidate. If a progress channel is added,
`verifyMs()` in `presentation.ts` is the single function that changes.

### How the drawing maps to real decisions

`src/ui/overlayTruthfulness.test.tsx` renders the real panel against a real
trace built from a real physics search, and fails if a number appears that the
search did not produce, or if the copy uses vocabulary the algorithm has not
earned (MCTS, win probability, confidence, "thinking").

- **Every generated candidate is drawn**, in generation order, directs included.
  The old overlay drew `stats.slice(0, 6)` ordered by *visits* while selection
  was by *trick utility*, so the route about to be played could be missing from
  the picture entirely. The winner is now addressed by generation index, which
  makes that unrepresentable.
- **Routes are drawn cue-first.** `candidate.path` starts at the *object* ball,
  so the chosen line used to float unconnected to the white ball; the cue's own
  leg (cue ball → ghost-ball contact) is drawn ahead of it.
- **Rejection reasons come only from the trace.** `TracedCandidate.rejection` is
  printed on the felt beside the route it belongs to. Where the trace supplies
  no reason, the route dims and nothing is said — the renderer decides where a
  line goes, never why a line died.
- **Cushion and combination marks** come from the *executed* simulation's event
  trace and waypoints (`src/render/annotate.ts`), not from the candidate
  generator's planned mirror geometry. The plan is the search's intention; the
  event trace is the physics' result.
- **Line weight** is driven by real ordering (the model's rank, then physics
  strength) and is never printed as a number. `strength` is a bounded monotonic
  transform of the rollout value, not a probability, and nothing renders it.
- **The sentence** ("Playing a two-rail bank on the 13, into the bottom-side
  pocket, over a direct pot on the 11") restates the selected candidate and one
  the trace itself labelled `lower-utility-than-selected`. The line under it is
  the selection ladder's own rung, mapped one-to-one to a phrase.

### The opponent thinks in a worker

`src/ui/planner/` moves the whole turn — search, selection, and the
authoritative shot simulation — off the main thread. Measured before: 3.4–5.0 s
main-thread freezes on every opponent turn, 12.3 s worst case after a break, and
a renderer crash after five plans back to back; independently profiled on a
production build at 11 long tasks totalling 11,278 ms over 81.5 s. Measured
after, same rAF-gap method on a production build at the 1180 px embed width over
~50 s and four opponent turns: **one gap above 60 ms, max 67 ms.**

It is also why `searching…` is now reachable at all. It never once painted
before: `plan()` blocked the main thread before React could commit the frame
that carried it.

`planner/plan.ts` is imported by both the worker and the inline fallback, so the
fallback (a `file://` page, a hardened CSP, a test runner with no `Worker`)
cannot drift from the worker. Failure to construct the worker is logged loudly
rather than downgraded silently, because a silent downgrade would hide the
freeze coming back.

### Controls

Aiming is a **drag** on Pointer Events — one code path for mouse, pen and touch,
with `setPointerCapture` so a drag that leaves the canvas keeps tracking and
`touch-action: none` so it does not scroll the page out from under the embed.
Two ±0.25° nudges exist because a fingertip covers ~26 mm of felt at embed
scale. `Space` shoots, or skips the opponent's reasoning while it is running;
`Escape` asks the portfolio shell to leave the focused embed.

## Metrics

Live, in the decision trace the presentation reads: candidates generated,
candidates pruned before physics, candidates physics-verified, legal pots and
scratches seen in simulation, physics calls spent, and (neural mode) inference
milliseconds. The panel deliberately prints almost none of them — they are
process telemetry, and they drive the drawing instead.

Offline, reproducible with `npm run eval:hybrid` and written to
`eval/results/*.json`: legal-pot / foul / scratch rate, trick attempt
and success rate, candidate recall against an exhaustive physics oracle, final
regret, direct-fallback preservation, kind-reserve promotions, physics calls per
turn, decision and inference latency — each broken out per candidate kind,
paired between the two modes at an identical budget, with 95% CIs — plus
full-game win rate and shots-to-win over fixed seeds. `npm run eval:gate` adds
pooled cross-seed paired differences, exact McNemar tests on the discordant
pairs, and the frozen accept/reject verdict.

Training-time metrics for the original MLP (5 seeds, held-out test,
calibration, ablations, per-kind breakdown, sanity controls including the
full-scale zeroed-feature control) live in `docs/repair/showboat-ml/phase-2d/`
and `training/ranker/phase2d/results/*.json` — kept as history, not deleted,
since the Phase 2E report compares against it directly. The currently-staged
model's own training-time metrics live in
`docs/repair/product-proof-sprint/showboat-model-research/REPORT.md` and
`training/ranker/phase2e/results/*.json`.
