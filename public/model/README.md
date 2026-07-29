# Trained model drop point

Place the trained ONNX network here as `showboat.onnx`.

When this file is present and loads, the app switches from the **search baseline**
to the **trained net (ONNX)** brain, and the UI label updates to say so
(`src/ai/brain.ts`). Until then the app runs the pure-search MCTS baseline and
says "search baseline" — we never present the baseline as the trained AI.

Produce this file from the training pipeline:

```
cd apps/showboat/training
python export_onnx.py --checkpoint runs/showboat_8ball_sez/ckpt/best.pt \
                      --out ../public/model/showboat.onnx
```

The I/O contract the browser expects (see `src/ai/onnx.ts`):
- input  `obs`    float32[1, 50]
- output `value`  float32[1, 1]  (win expectation, tanh)
- output `policy` float32[1, 4]  (action mean over phi/V0/a/b, tanh)

A randomly-initialized net exported the same way is a valid *plumbing* smoke
test, but must NOT be shipped as `showboat.onnx` — that would misrepresent an
untrained net as the trained AI.
