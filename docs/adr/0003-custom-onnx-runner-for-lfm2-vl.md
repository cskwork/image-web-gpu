# 0003. Run LFM2-VL with a hand-written ONNX Runtime pipeline

Status: accepted (initial commit `394c867` and `aa624b4`, 2026-03-23)

## Context

LiquidAI publishes LFM2-VL as separate ONNX graphs: token embeddings, a vision encoder, and a decoder with conv-state and KV caches. Their names and cache dtypes differ between the 450M and 1.6B releases. The app needs WebGPU execution, its own caching (see ADR 0002), and control over memory.

## Decision

- `src/vl-model.js` creates three `ort.InferenceSession`s directly. It replaces each `<image>` token embedding 1:1 with an image embedding and runs greedy decoding with explicit cache tensors (`initializeCache` and `updateCache`).
- `src/vl-processor.js` ports the Python `Lfm2VlImageProcessorFast`: smart resize, tiling, 16 px patches, normalisation, and padding masks.
- `src/config.js` gives each model an `arch` block (`embedTokens`, `visionEncoder`, `decoder`, `cacheType`). This let the 450M model be added in `aa624b4` without forking the runner.
- Transformers.js is used only for `AutoTokenizer`. Its `fetch` is temporarily pointed at tokenizer files that were already downloaded and cached.

## Consequences

- The app has full control over providers, external-data loading, and the caches, which is what makes the OPFS and IndexedDB paths possible.
- About 1,550 lines of model-specific code must track upstream changes to the processor and graphs. The pure preprocessing part is now covered by `tests/vl-processor.test.js`.
- Transformers.js still ships its own ONNX Runtime build, so the production bundle contains two WebGPU `.wasm` files.

## Alternatives considered

- The Transformers.js model classes, as used for SmolVLM in `ab56c58`. The repository has no record of why LFM2-VL was not run that way. Unknown.
