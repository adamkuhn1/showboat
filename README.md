# Showboat

2D bar-pool game with a **real trained-ML** opponent, a from-scratch event-based
physics engine, and a **live reasoning overlay** driven by actual search data.

> **Which brain is playing?** The shipped app currently runs the **pure-search
> MCTS baseline** and says so in the UI ("search baseline"). It upgrades to the
> **trained net** automatically once a trained ONNX model is dropped at
> `public/model/showboat.onnx` (the label flips to "trained net (ONNX)"). We
> never present the baseline as the trained AI — the multi-hour self-play run
> happens on Colab (see `training/README.md`); the pipeline is complete and the
> plumbing is proven, the trained weights are the one thing pending.

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

- **Train (Python, offline):** `pooltool` simulates; `LightZero` (Sampled
  EfficientZero → EfficientZero V2) runs self-play over the 4-D continuous action
  (`phi, V0, a, b`, **no theta**); reward is **outcome-only** so banks/combos are
  **emergent, never scripted**. The policy/value MLP exports to **ONNX**. See
  `training/README.md` for the Colab recipe and the honest dev-box blocker
  (pooltool needs Python 3.10+; the dev box is 3.9).
- **Play (browser):** `onnxruntime-web` loads the ONNX net (WebGPU / WASM), a
  **client-side TS-MCTS** searches over the candidate-shot set using the Rust
  physics for rollouts and the net for priors/value. Fully client-side and free;
  no server.

Until the trained model lands, the same MCTS runs with **uniform priors + a
physics rollout value** — the **pure-search baseline**, which is also the
win-rate benchmark from `PLAN.md §6`.

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
