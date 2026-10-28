# Architecture

The app is a static Vite site with no backend. Everything below runs in the visitor's browser tab.

## Components

| Module | Role |
|--------|------|
| `src/app.js` | Orchestrator. It probes WebGPU, picks the desktop or mobile path, loads the runtimes on demand, runs the capture loop, applies verdicts to the signal head, title, favicon, and history, and shows inline error messages. |
| `src/webcam.js` | Starts `getUserMedia` (ideal 1280x720, front camera). `captureFrame(resolution)` draws the current frame into a square canvas and returns a JPEG data URL (quality 0.85). |
| `src/config.js` | The model registry: Hugging Face base URL, size label, quantization per component, and an `arch` block that maps component file names and the cache dtype for the 450M and 1.6B layouts. |
| `src/infer.js` | A thin facade that the UI imports dynamically. It holds the engine singleton. |
| `src/webgpu-inference.js` | `WebGPUInference`: prevents concurrent loads, disposes the old model before switching, and converts the app's message format into text, image list, and image positions. |
| `src/vl-model.js` | `VLModel`: downloads and caches the weights, creates the ONNX Runtime sessions, builds the prompt with the tokenizer's chat template, merges image embeddings into the token embeddings, and runs greedy decoding. |
| `src/vl-processor.js` | A port of the LFM2-VL image processor: smart resize to multiples of 32, tiling for large images, 16x16 patch extraction, normalisation to [-1, 1], and padding plus attention mask to 1024 patches per tile. |
| `src/focus-analyzer.js` | Post-processing. It maps the model text to `focused`, `distracted`, `absent`, or `unknown`: an exact one-word match first, then Korean and English keyword scoring. |
| `src/face-detector.js` | The mobile path. MediaPipe Face Landmarker (GPU delegate, VIDEO mode, one face) with blendshapes and a transformation matrix, followed by rules. |
| `src/session-store.js` | The tally, focus rate, and last 20 verdicts in `localStorage`. The storage object is injected so the logic can be tested in Node. |
| `src/smolvlm.js` | Unused. Left over from the SmolVLM mobile experiment. |

## Desktop data flow (VLM)

1. **Load.** `app.js` calls `infer.loadModel(id)`. `VLModel.load` fetches `tokenizer.json` and `tokenizer_config.json`, and then the three ONNX graphs in order: `embed_tokens`, the vision encoder (`vision_encoder` or `embed_images`), and the decoder (`decoder_model_merged` or `decoder`). Each graph may have external data files (`.onnx_data`, `.onnx_data_1`, and so on), found with `HEAD` requests. Progress is reported at 20, 40, and 60 percent per graph, plus the MB downloaded per file.
2. **Capture.** Each loop iteration captures one square frame at the chosen size (256, 384, 448, or 512 px). All of these sizes stay on the single-tile path of the processor.
3. **Prompt.** One user message holds the image and an English prompt that asks for one word: "focused", "distracted", or "absent". The image embedding cache is cleared before each frame.
4. **Vision encoder.** `processImage` produces `pixel_values [tiles, 1024, 768]`, `pixel_attention_mask`, and `spatial_shapes`. A 384 px frame gives 24x24 patches and 144 image tokens after the 2x downsample.
5. **Decode.** The `<image>` token positions in the text embeddings are replaced 1:1 with image embeddings. The decoder then runs autoregressively. The conv-state and KV caches are float16 for 450M and float32 for 1.6B. Decoding picks the argmax token and stops at EOS or after 128 new tokens.
6. **Verdict.** `classifyFocus` maps the text to a status. `session-store.recordVerdict` updates the tally and history, and the next frame is captured right away.

## Mobile data flow (face landmarks)

`IS_MOBILE_DEVICE` is set from the user-agent string, or from a touch screen narrower than 1024 px. On mobile, `face-detector.js` loads the MediaPipe WASM from jsDelivr and the float16 `.task` model from Google Storage. It then runs `detectForVideo` on each animation frame:

- No face: **absent**.
- The 14 eye blendshape scores sum to less than 0.03 (for example, only hair is visible): **distracted** ("얼굴 미확인", face not confirmed).
- Both eye-blink scores are above 0.45 for more than 800 ms: **distracted**. Shorter blinks are ignored.
- The smoothed head yaw is above 40° or the pitch above 35°, or the eye-look-out score is above 0.6 or the eye-look-down score above 0.7: **distracted**.
- Otherwise: **focused**.

The signal head updates every frame. The history records one verdict every 3 seconds.

## Weight cache and memory

| Situation | Where the bytes go | Code |
|-----------|-------------------|------|
| A file at or under the threshold (2 GB on desktop, 50 MB on mobile) | Streamed into one pre-allocated `ArrayBuffer`, passed to ONNX Runtime, and written to IndexedDB `onnx-model-cache/files` without waiting | `downloadAsBuffer`, `fetchWithCache` |
| A file over the threshold, with OPFS available | Streamed chunk by chunk into an OPFS file named `onnx_<base64 url>`, then passed to ONNX Runtime as a blob URL | `streamToOPFS` |
| A file over the threshold, without OPFS | The remote URL is passed to ONNX Runtime as external data | `loadOnnxWithExternalData` |

Incomplete entries are detected on read. For IndexedDB, the stored `expectedSize` is compared with the buffer length. For OPFS, the file size is compared with the `HEAD` `content-length`. Incomplete entries are downloaded again. "캐시 삭제" (clear cache) empties IndexedDB, the `onnx_*` OPFS files, and a legacy Cache API bucket. Cache usage is read from `navigator.storage.estimate()`. Switching models calls `session.release()` on all three sessions before the new ones are created.

## External dependencies

| Dependency | Used for |
|------------|----------|
| `onnxruntime-web` 1.24.3 | Model execution with the WebGPU provider and a WASM fallback |
| `@huggingface/transformers` 3.8.1 | `AutoTokenizer` only. Its fetch is pointed at files that were already downloaded. |
| `@mediapipe/tasks-vision` ^0.10.33 | Mobile face landmarks |
| Hugging Face, Google Storage, jsDelivr | Model weights and the MediaPipe WASM at runtime |
| Vite 5 | Dev server and build. `optimizeDeps` excludes ONNX Runtime and Transformers.js. |
| Vercel | Hosting. `vercel.json` sets the COOP `same-origin` and COEP `require-corp` headers. |

## Failure handling

- **No WebGPU.** The badge shows "CPU 모드 (느림)". Sessions list `wasm` after `webgpu`, so ONNX Runtime can fall back to the single-threaded WASM backend.
- **Model load fails.** The load is wrapped. A numeric ONNX Runtime error code is turned into a message that mentions a possible WebGPU memory or compatibility problem. The UI resets the progress bar and shows a notice that suggests checking the network, clearing the cache, or picking a smaller model. A second load while one is running is rejected.
- **Camera errors.** Denied permission, a missing camera, and a camera in use each get their own message (`cameraErrorMessage` in `app.js`).
- **Analysis errors.** A failed frame waits 1.5 s (`ANALYSIS_RETRY_MS`) before the next try, so the loop does not keep the GPU busy on errors.
- **Storage.** A corrupt or invalid saved session is ignored. A storage quota error on save returns `false` and does not throw. IndexedDB read and write errors fall back to the network.
