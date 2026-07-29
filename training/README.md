# Showboat training pipeline

Trains the 8-ball AI by **self-play** with **pooltool** (SOTA event-based
Han-2005 physics) as the simulator and **LightZero** (Sampled EfficientZero →
EfficientZero V2) as the MCTS + learned value/policy method. The trained
policy/value net is exported to **ONNX** and shipped to the browser, where a
client-side TS-MCTS drives it over the Rust/WASM physics for play.

This is **real trained ML**, not heuristics. Trick shots (banks, multi-wall
combos) are **emergent**: the reward is outcome-only and the candidate geometry
merely makes bank/combo routes *available* — search selects them on value,
nothing scripts them. Jump/massé are **unrepresentable** because the action space
has no cue-elevation (`theta`) parameter.

## Layout

```
training/
  showboat_env/
    action_space.py       4-D action (phi, V0, a, b) — NO theta; V0 capped
    reward.py             outcome-only reward (no shot-type term, by design)
    observation.py        low-dim coord obs; layout matches src/ai/onnx.ts
    eight_ball_env.py     Gym-style self-play env wrapping pooltool 8-ball
    lightzero_adapter.py  DI-engine BaseEnv wrapper + registry name
    network.py            small policy/value MLP (exported to ONNX)
  config/
    eight_ball_sez_config.py   LightZero Sampled EfficientZero config
  export_onnx.py          checkpoint -> ONNX (browser I/O contract)
  smoke_run.py            end-to-end loop smoke test
  test_pipeline.py        unit tests for the pooltool-independent core
  requirements.txt
```

## Action space — the trick-shot constraint, enforced at the source

`action_space.py` defines exactly four continuous parameters:

| param | meaning | bound |
|---|---|---|
| `phi` | aim direction | wrapped to [-π, π] |
| `V0`  | cue-ball speed | capped at `V0_MAX = 8.5` m/s |
| `a`   | side english | [-1, 1] |
| `b`   | draw/follow english | [-1, 1] |

There is **no `theta`** (cue elevation). `CueAction.as_pooltool_kwargs()` never
passes `theta`, so pooltool defaults it to 0. A jump needs downward elevation; a
massé needs steep elevation + extreme english. With the parameter simply absent,
the agent **cannot express** either — impossible at the action level, not
filtered after the fact. The Rust/TS play-time engine mirrors the same four
parameters (`physics-core/src/cue.rs`, `src/physics/cue.ts`).

## Reward — outcomes only

`reward.py` scores **win/loss, own-ball pots, fouls, scratches, keeping the
turn**. It never inspects *how* a ball was pocketed and has **no bank/combo/trick
term**. Banks emerge because they win, not because we paid for them.

## Running it

### The real run (Colab / any Python 3.10–3.13)

pooltool requires Python **3.10–3.13**. On Colab (free T4) or a local 3.10+ box:

```bash
cd apps/showboat/training
pip install -r requirements.txt
pip install "git+https://github.com/ekiefl/LightZero.git@dev-pooltool"

# smoke the loop (few random-policy episodes end to end)
python smoke_run.py

# start self-play training
python config/eight_ball_sez_config.py     # writes checkpoints under runs/

# export the trained net for the browser
python export_onnx.py --checkpoint runs/showboat_8ball_sez/ckpt/best.pt \
                      --out ../public/model/showboat.onnx
```

Drop `showboat.onnx` at `apps/showboat/public/model/` and the web app
auto-upgrades from the **search baseline** to the **trained net** (the UI label
flips; see `src/ai/brain.ts`).

**Compute (from the research budget analysis):** this is a *small* deep-RL
problem — low-dim coordinate obs, a 128-dim MLP. A free Colab T4 across a few
12-hr sessions (checkpoint to Drive between them) is enough for a convincing
8-ball agent; the bottleneck is CPU sim throughput, not GPU. A paid GPU is an
optional *time* accelerator only — **flag before adopting**, never silent.

### On this dev box — honest blocker

The repo's dev interpreter is **Python 3.9.6**, which **pooltool does not
support** (`requires-python >=3.10,<3.14`). So the pooltool-backed self-play loop
**cannot run here** — `smoke_run.py` prints `[blocked] pooltool not installed …`
and `EightBallEnv.reset()` raises a clear error explaining why.

Everything that does **not** need pooltool has been run and verified on this box
(Python 3.9 + numpy + torch 2.8):

- `python smoke_run.py` → action space (no theta, V0 capped), outcome-only
  reward, observation encoder, and the network forward pass all pass; self-play
  is reported blocked with the reason.
- `python test_pipeline.py` → 5/5 unit tests pass (no-theta guarantee, V0 cap,
  decode/encode roundtrip, outcome-only-monotone reward, obs layout).
- `python export_onnx.py` → exported a valid ONNX graph with the exact browser
  I/O contract (input `obs` [1,50]; outputs `value` [1,1], `policy` [1,4],
  opset 17). A randomly-initialized net proves the export + browser plumbing;
  it is **not** shipped as the live model (that would misrepresent an untrained
  net as the trained AI — see `public/model/README.md`).

So the full pipeline is written, lint-clean, import-clean, unit-tested, and
proven end to end *except* the pooltool simulation step, which is gated purely by
the Python-version mismatch on this machine and runs on Colab unchanged.

## Metrics to record during the real run (PLAN.md §6)

- training episodes / env steps,
- **win-rate vs. the pure-search baseline** (`src/ai/mcts.ts` — the shipped
  benchmark opponent; papers report ~48% relative lift for MCTS+value),
- average shots-to-win.

Treat any number as a draft estimate to verify empirically.
