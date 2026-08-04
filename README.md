# Showboat

2D bar-pool game with a from-scratch event-based physics engine, a search-driven
opponent, and a **live reasoning overlay** driven by actual search data. A real
ML foundation (a trained candidate-ranking model) is in progress — see "ML
status" below before repeating any claim about a trained AI.

> **Which brain is playing, today?** The shipped app runs the **pure-search
> baseline** (`src/ai/mcts.ts`, a flat UCB bandit over the candidate-shot set —
> not a mislabeled "MCTS") and says so in the UI ("search baseline"/"the AI").
> No model file ships to `public/model/`, so this is the only thing a visitor
> ever plays against right now.

## ML status (2026-08-03, redirected — read before touching training/)

Two separate ML tracks exist; do not conflate them:

1. **`training/ranker/` — active, real, small.** A candidate-ranking MLP
   trained on data generated from the authoritative Rust/WASM physics (the
   exact binary this app ships), with a real checkpoint, a real ONNX export,
   and an automated test proving it changes candidate ordering. See
   `training/ranker/README.md` for reproduction commands and honest
   limitations. **Not yet wired into live gameplay** — Phase 2F does that.
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
npm test -w @portfolio-suite/showboat            # vitest (TS engine + game + AI)
npm run test:wasm -w @portfolio-suite/showboat   # cargo test (Rust physics/rollout)
npm run build:wasm -w @portfolio-suite/showboat  # regenerate src/wasm from Rust
```

The committed `src/wasm/` pkg means a plain `npm run build` needs **no Rust
toolchain**; `build:wasm` is only for regenerating it after changing the crate.

## Architecture

Polyglot where it's load-bearing (per `PLAN.md §0`): the physics + search hot
loop is **Rust→WASM**, the UI/rules/inference glue is **TypeScript**, the
training pipeline is **Python**.

```
physics-core/  (Rust → WASM)   event-based Han-2005 physics + MCTS rollout loop
  src/physics/                 pure-TS reference engine (the test ORACLE + spec)
  src/wasm/                    committed wasm-pack output (JS build needs no cargo)
src/game/                      8-ball ruleset + game controller (engine-agnostic)
src/ai/                        candidate generator, MCTS, ONNX loader, brain seam
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

**Play (browser), today:** `src/ai/mcts.ts`'s flat UCB bandit searches over
the candidate-shot set using the Rust physics for rollouts — **uniform
priors + a physics rollout value**, no model involved. This is the pure-search
baseline every visitor plays against right now.

## How the overlay maps to real decisions

Every number and line the overlay draws is **actual search output** — nothing is
decorative (the `qa-audit` hard constraint):

- **Ghost candidate paths** are the geometric aiming routes the search
  enumerated: **direct** pot, **single-cushion bank** (pocket mirrored across a
  rail), and **combo** (through an intermediate ball). Line weight/opacity tracks
  each candidate's **MCTS visit share**.
- **Per-candidate win-prob** is the squashed **rollout value** the search
  optimized. **Visits** are the real MCTS visit counts. **Rails** is the number
  of cushions before the pot, measured from the **actual simulated event trace**.
- The chosen shot is the **most-visited** candidate. A bank or combo only appears
  as the pick when the search *values* it — trick shots are selected, never
  canned.
- The shot caption ("cue → rail → 3-ball → corner") is generated from the real
  physics event trace (`src/ai/trace.ts`), the same mechanism as CueTip.

## Metrics (PLAN.md §6)

Instrumented as real search output: physics rollouts per decision and candidate
count are shown live in the overlay panel; win-rate vs. the baseline and average
shots-to-win are the training-time metrics recorded during the Colab run.
