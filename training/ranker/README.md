# Showboat candidate ranker — Phase 2A feasibility spike

Status: **feasibility spike, not production**. See
`docs/repair/showboat-ml/ARCHITECTURE_DECISION.md` for the full architecture
decision and `docs/repair/showboat-ml/EVALUATION_SPEC.md` for the evaluation
framework this spike deliberately does not yet meet in full. This is proof
that the pipeline works end to end (authoritative physics → labels → real
training → ONNX → browser inference → measurable ranking change), not a
claim that the AI is "trained" in any user-facing sense yet — no such claim
ships from this phase.

## What this is

A flat MLP that scores one pool-shot candidate (from `apps/showboat/src/ai/candidates.ts`)
at a time: given the board state plus that candidate's action/geometry, predict
the probability it legally pots its target under small execution noise. See
`apps/showboat/src/ai/ranker/schema.json` for the exact, versioned input
contract — both this training code and the browser inference code
(`apps/showboat/src/ai/onnx.ts`'s `evaluateCandidateRows`) consume it, and
`apps/showboat/src/ai/ranker/encode.ts` is the ONE feature encoder both the
dataset generator and the browser use — there is no second (Python)
re-implementation of it to drift out of sync.

## Reproduction

All commands from `apps/showboat/`.

1. **Generate the dataset** (Node/tsx, loads the exact WASM physics binary the
   browser ships, reuses `candidates.ts`/`rules.ts`/`trace.ts` unmodified):
   ```
   npm run ranker:gen-dataset --workspace=apps/showboat
   # or, to control scope:
   RANKER_N_STATES=15 RANKER_N_PERTURB=8 npx tsx training/ranker/gen_dataset.ts
   ```
   Writes `training/ranker/dataset/<schema_version>.ndjson` (currently
   `showboat-ranker-v2.ndjson`) + a `.meta.json` sidecar (seed, row counts,
   label statistics) — the filename tracks `encode.ts`'s `SCHEMA_VERSION`
   automatically so a stale-named file can't silently disagree with its own
   contents. The committed dataset in this repo was generated with
   `RANKER_N_STATES=15 RANKER_N_PERTURB=8`, seed `20260803` (hardcoded in
   `gen_dataset.ts`) — 712 rows from 15 randomized open-table states. A full
   60-state run works but takes materially longer (some perturbed shots are
   legitimately expensive physics simulations — see "Known limitations"
   below); 15 states was chosen for a spike, not because more wasn't possible.

2. **Train** (Python 3.10+, needs `torch`, `onnx`; both installable without
   `pooltool`/LightZero — this track does not touch the parked RL pipeline's
   dependencies):
   ```
   cd training/ranker
   python train.py --dataset dataset/showboat-ranker-v2.ndjson --out artifacts/run1
   ```
   Splits by `state_id` (not by row) to prevent leakage between candidates
   drawn from the same board. Prints train/val loss per 20 epochs and final
   held-out test metrics (BCE loss, Brier score, Spearman correlation between
   predicted score and empirical label). Writes `artifacts/run1/checkpoint.pt`
   + `artifacts/run1/manifest.json`.

3. **Export to ONNX** (requires a real checkpoint by default):
   ```
   python export_onnx.py --run artifacts/run1 --out artifacts/run1/showboat-ranker.onnx
   ```
   Refuses to run without `--run` unless `--allow-random-init` is passed
   explicitly (which marks the manifest `random_init: true` and prints a
   warning) — the old (parked) `training/export_onnx.py` silently exported
   random weights by default; this one does not repeat that. Uses the legacy
   TorchScript exporter (`dynamo=False`) deliberately: PyTorch's newer
   dynamo-based exporter defaults to writing weights into a separate
   `.onnx.data` file for this model, which `onnxruntime-web` cannot load from
   an in-memory buffer (only via browser-only `fetch`/`MountedFiles`
   plumbing this app doesn't use) — a single self-contained `.onnx` file is
   both simpler to ship and the only form the browser loader supports.
   Writes `artifacts/run1/showboat-ranker.manifest.json` (schema version,
   dataset hash, training config, seed, git commit, eval metrics, ONNX file
   hash, export timestamp, `calibration_applied: false`).

4. **Verify the model actually changes candidate ranking** (the Phase 2A
   acceptance test, real `onnxruntime-web` inference on the real exported
   artifact, no mocks):
   ```
   npx vitest run src/ai/ranker/rankerIntegration.test.ts --workspace=apps/showboat
   ```

## Current run's results (`artifacts/run1`, schema `showboat-ranker-v2`)

712 rows (15 states × up to 48 candidates, now including `rail-combo`, × 8
perturbations), 523/96/93 train/val/test split by state. Test set: BCE loss
0.461, Brier score 0.051, Spearman correlation between predicted score and
empirical label **0.117** — weak measured signal on this particular
held-out split. Mean predicted probability (0.196) still tracks the mean
empirical label (0.169) reasonably.

**Three numbers were produced across this schema's development, in this
order, each from a real fix, not a re-roll for a better result: 0.437
(original v1 schema) → 0.054 (after adding `rail-combo`, v2 schema) → 0.117
(after fixing a real labeling bug — `candidates.ts`'s combo/rail-combo
candidates set `target` to the first-contact ball, not the ball that
actually drops into the pocket; `gen_dataset.ts` and `shotSearch.ts` were
both checking the wrong ball id for these two kinds, via the same bug — see
`Candidate.potId` in `candidates.ts` for the fix).** Report all three rather
than the best one: with `n_test` around 2-3 *states* (not rows — the split
is by state, per the leakage-prevention rule), a handful of candidates from
a couple of unlucky/lucky states can swing a rank correlation enormously.
That volatility is itself evidence for the "small dataset" limitation below,
not a reason to retry seeds until a better
number appears — Phase 2A's actual acceptance bar is the deterministic
fixture proving the model changes candidate ordering
(`rankerIntegration.test.ts`), which this run still passes; it does not
require a good Spearman score. Full rigor (larger dataset, calibration,
baseline comparisons, confidence intervals over many states) is
Phase 2D/2G's job per `EVALUATION_SPEC.md` — this number is not
citable as a finished result in either direction.

## Known limitations (Phase 2A, by design — not hidden)

- **State sampling is a placeholder, not Phase 2C's pipeline.** There is no
  general legal-mid-game-state sampler in the repo (confirmed absent in
  `docs/repair/showboat-ml/02-physics-data-path.md`). `gen_dataset.ts`
  instead randomly removes a subset of object balls from the standard rack
  and jitters positions with overlap rejection, then treats each result as
  an open-table single-shot decision for `applyShotRules`. Real, but not a
  full game-history-derived state distribution — that's Phase 2C.
- **Some perturbed shots are genuinely slow to simulate.** A handful of
  candidates (chaotic bank/combo geometries with jittered aim/power) can take
  several hundred milliseconds to over a second in the Rust physics engine —
  this is real simulation cost, not a bug in the generator (confirmed via
  per-call timing instrumentation in `gen_dataset.ts`, not assumed). This is
  why the committed dataset uses 15 states, not the full 60 originally
  attempted; a production-scale Phase 2C run should budget for this and/or
  investigate whether the Rust engine's iteration/time caps need tuning for
  batch generation specifically (a full interactive game never hits this
  because production candidates are heuristically generated to be
  well-behaved, not randomly jittered at generation scale).
- **Small dataset.** 712 rows across 15 states is enough to prove the
  pipeline end to end, not enough for a rigorous held-out evaluation — the
  Spearman swing documented above (0.437 → 0.054 across two schema versions
  of essentially the same generation process) is direct evidence of this,
  not a contradiction to explain away. Do not cite either Spearman/Brier
  number as a finished result.
- **No calibration.** The predicted values are useful for *ranking*
  candidates against each other; they are not yet validated as calibrated
  probabilities. UI copy must not call them a percentage until Phase 2G.
- **Not wired into live gameplay.** `evaluateCandidateRows`/
  `tryLoadRankerModel` exist in `onnx.ts` and `mcts.ts` accepts
  `netSeedScores`, but `App.tsx` does not call `tryLoadRankerModel` and no
  `.onnx` file ships to `public/model/` — the live game is unaffected. Full
  gameplay integration (PUCT-style search integration, `brainLabel()`
  depending on manifest validation) is Phase 2F.

## Relationship to the parked RL track

`apps/showboat/training/showboat_env/` (PoolTool + LightZero self-play) is a
separate, parked initiative — not replaced or deleted by this directory. See
`docs/repair/showboat-ml/ARCHITECTURE_DECISION.md`'s "Disposition of existing
ML code" section for why it was kept rather than deleted, and why its
`OBS_DIM=50` contract mismatch against the browser's `OBS_DIM=48` is left
unfixed (that schema belongs to a track that has never been executed and
isn't the active path — fixing it now would be polishing code nobody is
currently building on).
