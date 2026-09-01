# Showboat

Browser-based 8-ball against a computer opponent that specifically hunts for
trick shots — banks, kicks, combos — verified by real physics rather than
scripted, with its reasoning shown live as it plays.

## What it does

- 2D bar pool (7-foot table, standard 8-ball rules), fully in the browser,
  no backend.
- Every turn, the opponent generates candidate shots (direct, bank, kick,
  combo), scores them with a trained ranker, verifies the top candidates
  with full physics simulation, and plays the best one a simulated rollout
  actually pots legally.
- A small neural network does the ranking by default; a classical heuristic
  scorer is the fallback and can be forced on for comparison.
- The reasoning panel is built from the same decision data the agent used
  to pick its shot, not separately-written flavor text.

## How it works

```
game state
  → generate candidates    pure geometry: direct / bank / kick / combo
  → rank                   neural MLP (or classical fallback) scores each
  → verify top 10          full physics simulation, jittered rollouts
  → select                 legal + trick-preferred + robust under jitter
  → play + overlay         recorded frames drive playback and the panel
```

- `src/physics/` — event-based simulation (time-of-next-event, not fixed
  timestep), closed-form per-phase trajectories, in the Han-2005/pooltool
  lineage.
- `src/game/` — 8-ball rules as a pure function of simulation result: fouls,
  ball-in-hand, win/loss.
- `src/ai/` — candidate generation, the neural ranker, and measured-shot
  classification: a shot only counts as a "bank" if the simulation log
  actually shows a rail contact, never by generator intent.
- `src/render/` — canvas rendering; playback replays recorded simulation
  frames, it never re-simulates.

## Technical highlights

- **No jump shots or massé, structurally.** `CueAction` has no cue-elevation
  axis, so those shots are unrepresentable, not just discouraged.
- **Trick-shot bias is one constant.** `DIRECT_ORDER_DISCOUNT` in
  `src/ai/agent.ts` halves the ranking score of direct shots; everything
  else about scoring and selection is unchanged.
- **Physics correctness details:** sliding-phase spin-up uses the `(5/2)/R`
  torque factor from `I = 2/5·mR²`; cushions absorb the roll component along
  their normal (otherwise balls pin against rails in repeated
  micro-collisions); rolling resistance is calibrated so a medium-power shot
  settles in a few seconds, matching a real table.
- **Held-out evaluation is split by table position, not by row** — every
  candidate from one layout shares geometry, so a row-level split would
  leak between train and test.

## Results

The ranker is a 13→20→12→1 MLP (~545 parameters), trained on 9,390 labeled
shot candidates sampled from 260 table positions (208 positions/7,555 rows
train, 26/927 validation, 26/908 held out and never touched until final
evaluation):

| | Neural | Classical |
|---|---|---|
| Held-out BCE | 0.261 | 0.310 |
| Held-out AUC | 0.824 | 0.755 |
| Mean per-position Spearman | 0.280 | 0.130 |

Neural beats classical on all three gate metrics — the model only ships as
the default if it does, and `weights.json` records the pass/fail itself so
the app's default can't drift from what evaluation found. In 30 self-play
games (identical everything except which ranker each side used), the
neural-ranked agent won 21 (70%).

## Run locally

```bash
npm install
npm run dev          # http://localhost:5175
npm run build
npm test
npm run typecheck
```

Training scripts (`npm run train:generate|fit|evaluate|selfplay`) regenerate
the dataset and model from scratch; the shipped `src/ai/weights.json` is
already trained. Force the classical ranker with `?ranker=classical`.

## Limitations

- Ball-in-hand is "anywhere" after any foul; the behind-the-head-string
  restriction after an opening scratch isn't modeled.
- Draw/follow uses a roll-vector approximation, not full 3D rigid-body spin.
- Pockets use a jaw-radius capture test — no liners/knuckles, so very fast
  balls never rattle out.
- Safety play is a single heuristic roll-up, not a searched strategy.
- Shot power is a heuristic function of path length; the ranker doesn't
  optimize power per shot.
