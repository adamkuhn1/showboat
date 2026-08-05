# Model directory

**The model the game actually uses lives in [`ranker/`](./ranker/README.md).**
That is the trained Phase 2D candidate ranker: committed, hash-verified, and
wired into live shot selection through `src/ai/neural/evaluator.ts`.

## `showboat.onnx` — the parked whole-board policy/value slot

This directory's original purpose was a drop point for a whole-board
policy/value network from the LightZero/PoolTool self-play pipeline
(`training/`). That network **was never trained**, and the architecture review
(`docs/repair/showboat-ml/ARCHITECTURE_DECISION.md`) parked that track in
favour of candidate-conditioned ranking, which is what the audit found actually
missing: a whole-board scalar cannot tell two candidate shots at the same board
apart, which is the decision the AI has to make every turn.

The loader for it (`tryLoadModel`/`evaluate` in `src/ai/onnx.ts`) is kept, not
deleted, per that decision's disposition — but **no live code path calls it**.
`src/App.tsx` loads the ranker in `model/ranker/`, not this slot. If the parked
RL track is ever revived, note that its Python observation layout is 50 floats
while `onnx.ts`'s `OBS_DIM` is 48; that mismatch is documented and unfixed in
`docs/ACTIVE_PLAN.md` and would have to be resolved first.

A randomly-initialized net exported as `showboat.onnx` would be a plumbing
smoke test only, and must never be presented as a trained AI.
