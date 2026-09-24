# AI Focus Monitor (WebGPU)

Browser-based employee focus monitoring powered by on-device AI. All inference runs locally -- no data leaves the browser.

**Live Demo:** https://image-web-gpu.vercel.app

## Overview

Real-time webcam analysis that classifies user state as **focused**, **distracted**, or **absent**. The system uses two different analysis paths depending on device:

- **Desktop:** LiquidAI vision-language models (LFM2-VL) analyze captured frames via ONNX Runtime WebGPU, generating natural language descriptions that are classified into focus states.
- **Mobile:** MediaPipe Face Landmarker (3.6 MB) runs at 30 FPS, tracking 478 facial landmarks and 52 blendshapes for real-time gaze, head pose, and eye-closure detection.

## Features

- **Dual Analysis Pipeline** -- VLM-based scene understanding on desktop, lightweight landmark tracking on mobile.
- **On-Device Inference** -- Zero server communication. All models execute directly in the browser.
- **Focus Classification** -- Detects focused (eyes open, facing screen), distracted (eyes closed 800ms+, looking away, hair-only), and absent (no person) states.
- **Focus Analytics** -- Live focus rate percentage, per-session statistics, and timestamped analysis history.
- **OPFS Caching** -- Model weights cached in Origin Private File System for instant subsequent loads.
- **Face Quality Validation** -- Eye blendshape sum check prevents false "focused" when only hair/back of head is visible.
- **Session Persistence** -- Tally, focus rate, and the last 20 verdicts are kept in `localStorage`, so a reload resumes the session. A reset control starts a new one.
- **Glanceable State** -- The tab title and favicon show the current verdict, so you can check it from another tab.
- **Fast First Load** -- The AI runtimes (ONNX Runtime, Transformers.js, MediaPipe) load on demand, so the first page load is about 16 KB of JS.
- **Inline Recovery** -- Camera-permission, missing-camera, busy-camera, and model-load failures show in-page messages that say how to fix them.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| VLM Models (Desktop) | LiquidAI LFM2-VL-450M, LFM2.5-VL-1.6B (ONNX) |
| VLM Inference | ONNX Runtime Web (WebGPU backend) |
| Face Detection (Mobile) | MediaPipe Face Landmarker (float16, 3.6 MB) |
| Build | Vite |
| Styling | Vanilla CSS, signal-head design system (see `DESIGN.md`); self-hosted Pretendard, Barlow Semi Condensed, Doto |
| Deploy | Vercel |

## Models

### Desktop -- Vision-Language Models

| Model | Size | Use Case |
|-------|------|----------|
| LFM2-VL-450M Q4F16 | ~316 MB | Default (recommended) |
| LFM2.5-VL-1.6B Q4-Q4 | ~1.8 GB | High-performance |
| LFM2.5-VL-1.6B Q4-FP16 | ~2.3 GB | High-precision |

### Mobile -- Face Landmarker

| Model | Size | FPS |
|-------|------|-----|
| MediaPipe Face Landmarker (float16) | ~3.6 MB | 30 |

## Requirements

- WebGPU-enabled browser (Chrome 113+, Edge 113+)
- Webcam access
- Desktop: ~1-4 GB memory depending on VLM model choice
- Mobile: ~50 MB memory

## Local Development

```bash
npm install
npm run dev
```

Run unit tests (Node's built-in test runner):

```bash
npm test
```

Build for production:

```bash
npm run build
npm run preview
```

## Architecture

```
src/
  app.js              # Main orchestrator (routes desktop/mobile paths)
  config.js           # VLM model configuration and registry
  webcam.js           # Camera stream management
  face-detector.js    # MediaPipe Face Landmarker (mobile path)
                      #   - 478 landmarks + 52 blendshapes
                      #   - Head pose estimation (yaw/pitch)
                      #   - Eye closure detection (800ms sustain)
                      #   - Face quality validation (blendshape sum)
  session-store.js    # Session tally/history persistence (localStorage)
  focus-analyzer.js   # VLM response classifier (desktop path)
                      #   - Keyword-based scoring
                      #   - focused/distracted/absent classification
  infer.js            # VLM inference pipeline coordinator
  vl-model.js         # Vision-language model loader (ONNX)
  vl-processor.js     # Image/text preprocessing
  smolvlm.js          # SmolVLM inference utilities
  webgpu-inference.js # WebGPU/ONNX session management
```

## Privacy

All processing happens on-device. Camera frames are analyzed locally and never transmitted to any server. Model weights are downloaded once from Hugging Face (VLM) or Google Storage (MediaPipe) and cached in the browser.

## License

MIT
