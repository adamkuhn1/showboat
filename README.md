# Showboat

2D bar-pool game with a from-scratch event-based physics engine, a search-driven
opponent, a **trained candidate-ranking model** that ships and runs in the
browser, and a **live reasoning overlay** driven entirely by actual search data.

> **Which brain is playing, today?**
>
> By default, the **classical physics search** (`src/ai/shotSearch.ts` — a flat
> UCB bandit over the candidate-shot set, not a mislabeled "MCTS"). The UI says
> "the physics-search opponent".
>
> A **trained neural ranker** also ships, is hash-verified at build time and
> at load time, and can be switched on with the **neural ranking** toggle.
> When it is on, the UI says "the neural + physics opponent" and the overlay
> shows the model's per-candidate calibrated make-estimate, its ranking, and
> how many candidates it pruned before physics ran. **Currently staged: the
> Phase 2E Deep Sets (relational) ranker**, adopted in place of the original
> Phase 2D MLP — see
> `docs/repair/product-proof-sprint/showboat-model-research/REPORT.md`.
>
> It is **off by default because the evidence says so**, not because it isn't
> wired up — though that evidence needs an important caveat after the model
> swap below. Against the **original Phase 2D MLP**, two independent fixture
> seeds, paired, at an identical 60-unit physics budget (120 fixtures + 24
> games, and 90 fixtures + 16 games):
>
> - **Consistent:** the hybrid attempts more tricks (+10.8pp and +5.6pp) and
>   gets meaningfully more makeable *trick* candidates in front of the physics
>   engine (combo recall 21.0% vs 11.3% and 10.5% vs 2.6%; similar for bank,
>   double-bank and rail-combo).
> - **Not consistent:** shot quality. Legal-pot rate came out **-3.3pp** on one
>   seed and **+1.1pp** on the other; every 95% CI includes zero and the sign
>   flips. The full-game record is 11–19 to classical across both seeds
>   (p ≈ 0.10, not significant), and game-level pot rate flips sign too.
>
> **After swapping in the Phase 2E model** (which measurably improves offline
> ranking quality, especially on the previously-weak double-bank kind — see
> the model-research report), a small **30-fixture / 8-game spot-check**
> (`eval/results/hybrid_eval_phase2e_spotcheck_seed20260805.json`) shows
> broadly similar behavior (legal-pot -3.3pp, trick-attempt rate now nearly
> identical between modes at ~93-94% vs the MLP's +10.8pp gap) but **is far
> too small to draw a conclusion from** — it exists so this README doesn't
> cite stale evidence for the currently-staged model, not as a replacement
> for the full 120-fixture/2-seed protocol. **Re-running that full protocol
> against the Phase 2E model is the recommended next step before revisiting
> the default.** Full numbers and method:
> `docs/repair/product-proof-sprint/showboat-live/REPORT.md` and
> `eval/results/*.json`. Reproduce with `npm run eval:hybrid`.

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
```

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
public/model/ranker/           the shipped, hash-verified trained artifact + its manifest
eval/                          equal-budget classical-vs-hybrid evaluation harness
src/render/ + src/ui/          canvas render + reasoning overlay
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

## How the overlay maps to real decisions

Every number and line the overlay draws is **actual search output** — nothing is
decorative (the `qa-audit` hard constraint). `src/ui/overlayTruthfulness.test.tsx`
renders the real component against a real search result and fails if a number
appears that the search didn't produce, or if the copy uses vocabulary the
algorithm hasn't earned (MCTS, win probability, confidence, "thinking").

- **Decision stages** come from `SearchResult.trace` (`DecisionTrace`), which
  `searchCandidates` fills in as it runs: candidates generated, candidates
  scored by the learned ranker and how many milliseconds that took, candidates
  pruned before physics, candidates physics-verified and how many of them
  legally potted, how many scratched the cue in simulation, and physics-engine
  call units actually spent.
- **Per-candidate strength** is the squashed rollout value the search optimized
  — a relative, uncalibrated score (`CandidateStat.strength`), never displayed
  as a percentage and never called a probability.
- **Per-candidate make estimate** (neural mode only) is the model's *calibrated*
  legal-pot probability. Calling it a probability is allowed here because it was
  measured: ECE 0.0067–0.0155 post-Platt on the held-out test split. The
  ranker's rank for each candidate is shown as `#n`.
- **Ghost candidate paths** are the geometric aiming routes the search
  enumerated. Line weight/opacity tracks each candidate's real UCB visit share.
- The chosen shot is whatever `selectBestWithReason` returned, and the one line
  of prose under it is that function's own `selectionReason` — the display
  cannot describe a different decision than the one played.
- The shot caption ("cue → rail → 3-ball → corner") is generated from the real
  physics event trace (`src/ai/trace.ts`).

There is one timing concession, stated plainly: after the search finishes, the
overlay is held on screen for a bounded 350–1100 ms before the balls move, so
the decision is readable. Nothing is computed during that hold and nothing on
screen animates as if it were.

## Metrics

Live, in the overlay: candidates generated, candidates pruned before physics,
candidates physics-verified, legal pots and scratches seen in simulation,
physics calls spent, and (neural mode) inference milliseconds.

Offline, reproducible with `npm run eval:hybrid` and written to
`eval/results/hybrid_eval.json`: legal-pot / foul / scratch rate, trick attempt
and success rate, candidate recall against an exhaustive physics oracle, final
regret, physics calls per turn, decision and inference latency — each broken
out per candidate kind, paired between the two modes at an identical budget,
with 95% CIs — plus full-game win rate and shots-to-win over fixed seeds.

Training-time metrics for the original MLP (5 seeds, held-out test,
calibration, ablations, per-kind breakdown, sanity controls including the
full-scale zeroed-feature control) live in `docs/repair/showboat-ml/phase-2d/`
and `training/ranker/phase2d/results/*.json` — kept as history, not deleted,
since the Phase 2E report compares against it directly. The currently-staged
model's own training-time metrics live in
`docs/repair/product-proof-sprint/showboat-model-research/REPORT.md` and
`training/ranker/phase2e/results/*.json`.
