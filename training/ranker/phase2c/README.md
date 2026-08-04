# Showboat candidate ranker — Phase 2C training-data pipeline

Status: **dataset pipeline, not a trained model.** This directory builds and
validates the authoritative training-data pipeline for Phase 2D's candidate
evaluator. It does not train or ship a production model — see
`docs/repair/showboat-ml/phase-2c/DATASET_DESIGN.md` for the full design and
`docs/repair/showboat-ml/phase-2c/DIAGNOSTIC_RESULTS.md` for the (explicitly
non-final) diagnostic training runs this phase does perform, to validate the
dataset is structurally learnable before Phase 2D invests in a real model.

## What this is

Real self-play games (`selfplay.ts`, reusing 100% of production game logic —
`makeGame`/`takeShot`/`applyShotRules`/`planTurn`/`legalTargets`) plus
controlled synthetic states (`controlled.ts`), labeled by running every
candidate `generateCandidates()` produces through `n_perturbations` real,
jittered executions of the authoritative WASM physics + `rules.ts` pipeline
(`gen_dataset_v3.ts`'s `processState`). Rows are candidate-conditioned
(schema at `dataset_schema.json`, `showboat-dataset-v1`) and split by
`family_id` (game/trajectory/controlled-state lineage), never by row or
raw `state_id`, to prevent near-duplicate leakage across train/val/test
(`split.ts`).

## Reproduction

All commands from `apps/showboat/`.

1. **Generate a dataset** (Node/tsx, real WASM physics):
   ```
   npm run ranker:gen-smoke      # ~25-30s, tiny, CI/contract-verification scale
   npm run ranker:gen-pilot      # data-quality/throughput/diagnostic scale
   npm run ranker:gen-baseline   # Phase 2D training target — see THROUGHPUT_REPORT.md
   npm run ranker:gen-full       # the brief's original 10k-state nominal target; not run this phase
   ```
   Writes one NDJSON shard per work unit (one self-play game, or one
   controlled state) + an append-only `manifest.jsonl` + a `progress.json`
   + a `dataset_manifest.json` (provenance, per-kind/per-split counts,
   shard hashes, whole-dataset hash) to `data/<profile>/` — gitignored
   entirely (fully reproducible from code + `RANKER_SEED`, see
   `phase2c/.gitignore`). Re-running the same profile+seed reproduces an
   identical `dataset_hash` (timestamps excluded — they're expected to
   differ run to run).

   **Resumable**: killing a run at any point and re-running the same
   command skips already-complete units and continues — verified to
   produce the identical `dataset_hash` as an uninterrupted run (see
   `PHASE_2C_CLOSURE.md`). Pass `--restart-clean` to wipe a profile's
   output and start fresh instead of resuming (required if you change
   `RANKER_SEED`, the schema, or the physics binary for an existing
   profile dir — resuming across an incompatible change is refused, not
   silently allowed).

   **Parallel** (worker_threads, since units are independently seeded):
   ```
   RANKER_PROFILE=pilot RANKER_WORKERS=4 npm run ranker:gen-parallel
   ```
   Same resumability semantics; see `THROUGHPUT_REPORT.md` for measured
   scaling and the chosen worker count.

2. **Validate** (schema/NaN/range/duplicate/leakage checks against the
   acceptance-criteria failure-mode list):
   ```
   RANKER_PROFILE=pilot npm run ranker:validate
   ```

3. **Check for split leakage independently** (re-reads the written shards,
   not the in-process check `gen_dataset_v3.ts` already runs before writing):
   ```
   RANKER_PROFILE=pilot npm run ranker:split-check
   ```

4. **Generate a human-readable report** (from the manifest + shards, not
   hand-duplicated stats — writes `data/<profile>/REPORT.md`):
   ```
   RANKER_PROFILE=pilot npm run ranker:report
   ```

5. **Run diagnostic training checks** (Python 3.10+, `torch`+`numpy`; a
   throwaway MLP, NOT the Phase 2D model):
   ```
   python3 -m venv /tmp/showboat-venv && /tmp/showboat-venv/bin/pip install torch numpy
   cd training/ranker/phase2c
   /tmp/showboat-venv/bin/python diagnostic.py --dataset-dir data/pilot --out data/pilot/diagnostics
   ```
   Runs: tiny-subset overfit, shuffled-label control, candidate-conditioned
   vs. state-only comparison, constant-mean baseline comparison. Writes
   `diagnostic_results.json`. See `DIAGNOSTIC_RESULTS.md` for the pilot
   run's actual numbers, reported honestly including any unfavorable ones.

6. **Semantic/integration tests** (real WASM physics, not mocks):
   ```
   npx vitest run training/ranker/phase2c/gen_dataset_v3.test.ts src/ai/shotSearch.test.ts
   ```
   Most rule-semantics fixtures (illegal first contact, scratch, no-rail
   foul, legal/illegal 8-ball win/loss, combo `potId != target`, and the
   endpoint-only-labeling fix's event-sequence fixtures) already exist in
   `src/game/rules.test.ts` and `src/ai/shotSearch.test.ts` and are not
   duplicated here; `gen_dataset_v3.test.ts` tests what's specific to
   Phase 2C — `classifyPocketed` and `processState`'s real-physics
   integration.

7. **Rigorous tiny-subset memorization test** (fixes the original
   diagnostic's wrong-metric bug — see `PHASE_2C_CLOSURE.md` item #2):
   ```
   python memorization_test.py --dataset-dir data/pilot
   ```
   Evaluates memorization via prediction-to-target MAE, not raw BCE
   (which has a nonzero floor for soft/non-binary targets by construction).
   Includes negative controls (zero optimization steps, zeroed features,
   post-hoc-permuted evaluation) that must fail, proving the test itself
   has real discriminating power.

8. **Perturbation rollout benchmark** (8 vs. 16 vs. 32, chose the adaptive
   policy `processState` now uses):
   ```
   npx tsx training/ranker/phase2c/perturbation_benchmark.ts
   ```
   See `PERTURBATION_BENCHMARK.md`.

9. **Per-stage throughput profile**:
   ```
   npx tsx training/ranker/phase2c/throughput_profile.ts
   ```
   See `THROUGHPUT_REPORT.md`.

## Relationship to the Phase 2A/2B spike

`training/ranker/gen_dataset.ts`/`train.py`/`export_onnx.py` (one directory
up) are the Phase 2A/2B feasibility-spike pipeline and its shipped
`artifacts/run1` model — kept as-is, not replaced. This phase's dataset
format (`showboat-dataset-v1`) is a new, separate schema for a
candidate-conditioned dataset with real lineage/splitting; it does not
replace `src/ai/ranker/schema.json` (`showboat-ranker-v2`, the per-candidate
*feature* encoding both pipelines still share via `encode.ts`).
