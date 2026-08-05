# Phase 2E — relational (Deep Sets) candidate ranker

Research track comparing Phase 2D's candidate-conditioned MLP against an
entity/Deep-Sets relational model on the same frozen Phase 2C baseline dataset.
Nothing here is wired into the live game; no file under `apps/showboat/src/` is
modified by this directory.

## What each file is

| file | role |
|---|---|
| `data.py` | Caches the read-only Phase 2C baseline NDJSON shards to one `.npz`. Never re-splits or re-encodes. |
| `metrics.py` | Independently reimplemented metrics. `python metrics.py` self-verifies the tie-aware Spearman against `scipy.stats.spearmanr` and `phase2c/diagnostic.py`, and demonstrates the pre-fix ordinal-rank bug would disagree (so the check has power). |
| `entities.py` | The relational representation: 16 ball entities + shot context, derived from the SAME 68 floats `encode.ts` already produces, as pure ONNX-exportable torch ops. Full feature documentation in the module docstring. |
| `models.py` | `TinyNet` (Phase 2D's, state-dict compatible), `FlatEntityMLP` (attribution ablation), `DeepSetsRanker` (the proposal), `DeepSetsAttn` (the GNN arm). |
| `search.py` | Bounded validation-only config search, 9 points, 1 seed. The MLP gets the same courtesy search as the relational arm. Every point logged, including losers. |
| `train.py` | 5-seed training/eval driver for every arm, ablation and control. Never reads the test split. |
| `controls.py` | Permutation invariance, same-group relabelling invariance, pocketed-ball masking, live-ball removal. |
| `zero_control_check.py` | Forensics on the all-zero-input control's float-precision artefact. |
| `budget.py` | Physics-budget proxy: recall-of-best@k. |
| `sample_efficiency.py` | Learning curves, train sub-sampled by state. |
| `export_onnx.py` | ONNX export + Python `onnxruntime` parity + dynamic-batch check + manifest. |
| `parity_web.mjs` / `latency_runs.sh` | Real `onnxruntime-web` (wasm) parity and repeated latency measurement. |
| `frozen_config.json` | The frozen configuration AND the recommendation, committed before the test split was read. `evaluate_test.py` refuses to run without it. |
| `evaluate_test.py` | The only file that reads the test split. One pass, frozen checkpoints, validation-fit Platt parameters. |
| `summarize.py` / `summarize_test.py` | Print the tables from the results JSONs. Compute nothing new. |

## Reproduction

Environment: Python 3.9+, `torch`, `numpy`, `scipy`, `onnx`, `onnxruntime`
(all in `training/requirements.txt`), plus Node with `onnxruntime-web` for the
browser parity hop.

```bash
python data.py --dataset-dir ../phase2c/data/baseline --cache results/dataset.npz
python metrics.py
python search.py
python train.py --variants mlp_matched mlp_wide deepsets deepsets_small \
                           flat_entity_mlp zero_input_mlp zero_input_deepsets
python train.py --variants deepsets_no_relations deepsets_no_kind \
                           deepsets_state_only deepsets_candidate_only \
                           deepsets_pot_id deepsets_attn
python controls.py
python budget.py
python zero_control_check.py
python sample_efficiency.py
python export_onnx.py --variant deepsets
python export_onnx.py --variant mlp_matched
bash latency_runs.sh /path/to/node_modules/onnxruntime-web/dist/ort.node.min.mjs 8
python evaluate_test.py          # one-shot; needs frozen_config.json
python summarize.py ; python summarize_test.py
```

Seeds are Phase 2D's: `20260804 20260805 20260806 20260807 20260808`.

## Artifact policy

Same as Phase 2C/2D (`training/ranker/.gitignore`): checkpoints, `.onnx` files,
parity fixtures and `results/dataset.npz` are **not** committed — regenerate
with the commands above. The small metric-summary JSONs under `results/` **are**
committed; they are the evidence behind the phase's conclusions.

## Headline result

The relational model wins on every held-out metric with no seed overlap, needs
~4–10x less data for equal quality, roughly doubles ranking quality on the
`double-bank` candidates Phase 2D flagged as its weak point, and exports to a
single ONNX graph with the **identical `[N,68] -> [N,1]` contract** the live
game already speaks. It costs ~30–45x more browser inference time
(0.04 ms → ~1.1–1.9 ms per 24-candidate batch), which is still less than one
~23 ms physics rollout.
