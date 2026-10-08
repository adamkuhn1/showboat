# Showboat

[![CI](https://github.com/adamkuhn1/showboat/actions/workflows/ci.yml/badge.svg)](https://github.com/adamkuhn1/showboat/actions/workflows/ci.yml)

Browser-based 8-ball against a computer opponent that specifically hunts for
trick shots — banks, kicks, combos — verified by real physics rather than
scripted, with its reasoning shown live as it plays.

## What it does

- 2D bar pool (7-foot table, standard 8-ball rules), fully in the browser,
  no backend.
- Every turn, the opponent generates candidate shots (direct, bank, kick,
  combo), scores them with a trained ranker, verifies the top candidates
  with full physics simulation, and plays the best one a simulated rollout
  actually pots legally. A trick shot is only preferred when it also pots in
  at least 2 of 3 jittered re-executions.
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
  → select                 legal; trick-preferred only if robust (≥ 2/3 under jitter)
  → or play safe           simulated roll-ups, illegal first contacts rejected
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
- **Trick-shot bias is bounded by robustness.** `DIRECT_ORDER_DISCOUNT` in
  `src/ai/agent.ts` halves the ranking score of direct shots, which decides
  what gets verified. Selection (`selectShot`) then prefers a measured trick
  shot only among shots that pot in at least `ROBUST_FRACTION` (2/3) of their
  jittered re-executions; below that bar it takes the most robust shot.
- **Safeties are simulated too.** When nothing pots, `chooseSafety` simulates
  low-power roll-ups to each legal ball and discards any that foul (wrong
  first contact, no rail, scratch).
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
the app's default can't drift from what evaluation found. Self-play results
are under Evaluation below.

## Evaluation

Held-out test set: **26 table positions, 908 candidate rows** (25 positions
have enough label variance for a per-position Spearman; only 25 rows, from
15 positions, are positives at label > 0.5, which is what AUC counts). The
classical baseline is `classicalScore` in `src/ai/ranker.ts`: a hand-written
multiplicative pot-probability heuristic over the same 13 features, with no
learned parameters.

95% intervals come from a position-level bootstrap (5,000 resamples of the 26
held-out positions, seed 20261001; neural and classical scored on the same
resample, so the differences are paired). Predictions are regenerated from
the shipped weights, and the point estimates match `training/metrics.json`
exactly.

| | Neural | Classical | Neural − classical |
|---|---|---|---|
| AUC | 0.824 (0.749–0.888) | 0.755 (0.653–0.845) | +0.069 (−0.006 to +0.156) |
| BCE | 0.261 (0.216–0.306) | 0.310 (0.248–0.373) | −0.049 (−0.073 to −0.028) |
| Mean Spearman | 0.280 (0.212–0.349) | 0.130 (0.062–0.202) | +0.150 (+0.056 to +0.243) |

The BCE and Spearman gains have intervals that exclude zero. The AUC gain
does not quite (96% of resamples favour neural); with 26 positions it is
suggestive, not established.

**Top-10 recall** (does a candidate that pots in most of its jittered
rollouts land in the 10 the agent verifies, using the agent's real order):
neural 13/15 positions (0.87, 0.67–1.00), classical 11/15 (0.73,
0.50–0.94), a random 10 0.45. This is ranked among the ≤ 40 candidates
sampled per position, not every candidate the game generates, and 15
positions is a small sample.

**Self-play** (neural-ranked vs classical-ranked agent, everything else
identical, seed 42), re-run after the robustness-gated selection and
simulated safeties:

- 30 games: neural won 19/30 (63%, Wilson 95% 46–78%). The previous selection
  rule, re-run at the same seed, won 21/30.
- 300 games: neural won 171/300 (57%, Wilson 95% 51–63%), about 22 minutes
  on a laptop.

Reproduce with `npm run train:selfplay` (`GAMES=300` for the long run) and
then `npm run train:bootstrap`; everything lands in `training/ci.json`.

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
already trained. `npm run train:bootstrap` recomputes the confidence
intervals below into `training/ci.json` without touching the weights. Force
the classical ranker with `?ranker=classical`.

`?embed=1` is the portfolio-iframe mode: no title, the host's palette, the
table scaled to the frame width, and the content height posted to the parent
as `{ source: "portfolio-embed", type: "resize", id: "showboat", height }`.

## Limitations

- Ball-in-hand is "anywhere" after any foul; the behind-the-head-string
  restriction after an opening scratch isn't modeled.
- Draw/follow uses a roll-vector approximation, not full 3D rigid-body spin.
- Pockets use a jaw-radius capture test — no liners/knuckles, so very fast
  balls never rattle out.
- Safety play is a small simulated search over roll-ups (legality plus the
  opponent's best ranked shot afterwards), not a real safety strategy.
- Shot power is a heuristic function of path length; the ranker doesn't
  optimize power per shot.
