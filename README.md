# Showboat

2D bar pool (7-foot bar box, standard 8-ball rules) played against an AI that
strongly prefers trick shots — banks, kicks off the rail, multi-wall banks and
combos — and shows you its actual reasoning while it decides.

Human vs AI. You break. Everything runs in the browser; no backend.

## Run

```bash
npm install
npm run dev          # -> http://localhost:5175
npm run build        # production build (dist/)
npm test             # physics, rules and AI tests (29)
npm run typecheck
```

## Architecture

One pipeline, one representation of every shot:

```
game state
  └─ generateCandidates()        pure geometry: direct / bank(1-2 rail) / kick / combo
      └─ ranker.score(features)  neural MLP (or classical fallback) -> P(success)
          └─ verify top 10       full physics simulation of each, frames recorded
              └─ select          legal + measured-trick-preferred + robustness
                  └─ ONE ShotReport: its outcome is committed, its frames are
                     the playback, its event log is the overlay's data source
```

- `src/physics/` — event-based (time-of-next-event, not fixed timestep) engine
  in the Han-2005 / pooltool lineage. SI units, standard literature
  coefficients, closed-form per-phase trajectories with analytic + guarded
  numeric event solvers. Ported from this repo's earlier pure-TS engine, plus
  three reviewed fixes found while porting: sliding-phase spin-up rate was
  missing the `(5/2)/R` torque factor from `I = 2/5·mR²`; cushions now
  absorb the roll component along their normal (both prevented balls from
  pinning against rails in endless micro-collisions); and the event loop
  carries a hard iteration cap independent of simulated time, since a
  same-instant repeated collision could in principle hold `t` still and
  defeat the existing time-based cap.
- `src/game/` — 8-ball rules as a pure function of (pre-state, sim result):
  groups, fouls (wrong first contact, scratch, no-rail), ball-in-hand, 8-ball
  win/loss. Fully unit tested.
- `src/ai/` — candidate generation, feature extraction, ranker, measured shot
  classification, and the turn state machine described above.
- `src/render/` — canvas renderer + playback. Playback interpolates the
  position keyframes recorded by the authoritative simulation; it never
  re-simulates. One global precomputed time-warp per shot slows smoothly
  around first contact / rail hits / pockets and is fixed before the first
  frame — no per-ball rates, nothing coupled to live contact state.

## The AI is trained ML, and here is exactly what that means

The ranker that orders candidate shots is a small MLP (13 → 20 → 12 → 1,
~545 parameters, `src/ai/weights.json`) trained by `training/`:

1. `npm run train:generate` — seeded random mid-game positions; every
   candidate the generator proposes is labelled by jittered physics rollouts
   (σ ≈ 0.46° aim, σ = 0.03 power) of this exact engine. Label = fraction of
   rollouts that legally pot the intended ball. Current dataset: **8,726
   labelled rows from 240 sampled table positions** (up from an earlier,
   much smaller run — more positions is what actually made the gate below
   trustworthy).
2. `npm run train:fit` — hand-written Adam/backprop loop (no framework; the
   network is small enough that auditable beats convenient). The split is by
   whole table position, three ways, never by row: candidates from one
   layout share geometry, so a row-level split would leak. 192 positions
   (6,962 rows) train, 24 positions (904 rows) validation for early
   stopping — best epoch 97, validation BCE 0.311. The remaining 24
   positions (860 rows) are a **test** set this script never reads.
3. `npm run train:evaluate` — evaluates the exported weights through the
   same `neuralScore()` the app bundles, against the classical baseline, on
   that untouched test set, and **writes the gate result into the shipped
   `weights.json` itself** (`meta.gatePassed`) so the app's own default can't
   drift from what this script found. The gate was fixed before this run:
   neural must beat classical on all three of held-out BCE, AUC and mean
   per-position Spearman, no partial credit. **Current result: GATE PASS.**
   BCE 0.329 vs classical's 0.404; AUC 0.749 vs 0.720; mean per-position
   Spearman 0.303 vs 0.117. The two rankers disagree on the top-ranked
   candidate 71% of the time, and when they disagree neural's pick has a
   higher true success rate on average (0.250 vs classical's 0.215). Full
   numbers: `training/metrics.json`.
   **The app ships neural as the default ranker because of this result** —
   `?ranker=classical` forces the heuristic on instead, for comparison.
4. `npm run train:selfplay` — neural-ranked agent vs classical-ranked agent,
   full racks, identical everything else. 30 games, seed 42: **classical
   wins 16/30 (53%)**, neural wins 14/30 (47%) — statistically a wash at this
   sample size, not a second win for either side. Avg shots-to-win is close
   too (10.4 vs 11.0). Neural's potted balls are trick shots somewhat more
   often — 86% (97/113) vs classical's 78% (91/117).

What the model is NOT: it does not choose the shot alone. It orders
candidates; the physics engine then verifies the top 10 and selection
requires a simulated legal pot. If `weights.json` fails shape validation, or
the ranker's own held-out gate, the app falls back to the interpretable
classical scorer and the thinking panel says "classical ranker" — it never
labels a shot "neural" unless a neural model actually scored it. The honest
summary of this experiment: the ranker beats the classical heuristic on the
metric it was trained on (predicting shot success), but that edge is real
without yet being large enough to show up as a clear game-level win-rate
advantage over 30 racks. Both facts are worth knowing, so both are reported.

## Trick-shot preference (and its honesty)

- `DIRECT_ORDER_DISCOUNT = 0.5` in `src/ai/agent.ts` halves direct shots'
  ordering score. That single constant is the entire style bias.
- Selection prefers candidates whose **measured** result is a trick shot:
  `src/ai/classify.ts` reads the simulation event log (cue rails before
  contact, target-ball rails before dropping, chain length) — never the
  generator's intent. A "bank" that never touched a rail cannot be labelled a
  bank.
- Nothing is scripted: banks/kicks/combos emerge from mirror-image candidate
  geometry surviving physics verification and jittered robustness rollouts.

## No jump shots, no massé — structurally

`CueAction` (`src/physics/cue.ts`) is `{phi, power, sideSpin, topSpin}`.
There is no cue-elevation axis anywhere in the state or action space, so jump
and massé shots are unrepresentable, not merely discouraged. Speed is capped
at 8.5 m/s. The physics is strictly planar.

## The overlay never claims more than the mechanism

Every string in the thinking panel is formatted from the same
`Decision`/`EvaluatedCandidate` objects the agent selected with: candidate
counts by kind, the candidate currently being verified with its measured
outcome ("pots it (3/3 under jitter)" / "misses in simulation"), and the
selected shot with a reason composed from measured fields. Dashed lines on
the table are hypothetical candidate geometry; solid lines are the verified
route extracted from the frames of the exact simulation that is then played
back.

## Known limitations

- Ball-in-hand is "anywhere" after any foul (bar rules); the behind-the-head-
  string restriction after an opening scratch is not modelled.
- The 2D engine approximates draw/follow with a roll-vector model rather than
  full 3D rigid-body spin; draw shots check the cue ball rather than pulling
  it back dramatically.
- Pocket capture is a jaw-radius test; there are no pocket liners/knuckles,
  so very fast balls never rattle out.
- The AI's safety play is a single heuristic roll-up, not a searched safety.
- Candidate power is a heuristic function of path length; the ranker learns
  around it rather than optimising power per shot.

## File budget

27 production files in `src/` (3,495 lines), plus 3 test files (510 lines)
and 5 training scripts (713 lines) — 35 files / 4,718 lines total.
