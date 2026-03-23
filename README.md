# AI Focus Monitor (WebGPU)

Browser-based employee focus monitoring powered by on-device AI vision models. All inference runs locally via WebGPU -- no data leaves the browser.

**Live Demo:** https://image-web-gpu.vercel.app

## Overview

Real-time webcam analysis that classifies user state as **focused**, **distracted**, or **absent** using LiquidAI vision-language models running entirely in the browser with ONNX Runtime WebGPU acceleration.

## Features

- **On-Device AI Inference** -- Vision-language models (LFM2-VL) execute directly in the browser via WebGPU. Zero server communication.
- **Real-Time Face Detection** -- MediaPipe Face Landmarker tracks facial landmarks at 30 FPS for gaze and presence analysis.
- **Multiple Model Options** -- Choose between lightweight (450M, ~316 MB) or high-performance (1.6B, ~1.8-2.3 GB) models.
- **Focus Analytics** -- Live focus rate, per-session statistics, and timestamped analysis history.
- **OPFS Caching** -- Models are cached in Origin Private File System for instant subsequent loads.
- **Mobile Support** -- Optimized memory management with fallback to smaller models on mobile devices.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Models | LiquidAI LFM2-VL-450M, LFM2.5-VL-1.6B (ONNX) |
| Inference | ONNX Runtime Web (WebGPU backend) |
| Face Detection | MediaPipe Face Landmarker |
| Build | Vite |
| Styling | Vanilla CSS (Pretendard, taste-skill design system) |
| Deploy | Vercel |

## Models

| Model | Size | Use Case |
|-------|------|----------|
| LFM2-VL-450M Q4F16 | ~316 MB | Mobile / default (recommended) |
| LFM2.5-VL-1.6B Q4-Q4 | ~1.8 GB | Desktop high-performance |
| LFM2.5-VL-1.6B Q4-FP16 | ~2.3 GB | Desktop high-precision |

## Requirements

- WebGPU-enabled browser (Chrome 113+, Edge 113+)
- Webcam access
- ~1-4 GB available memory depending on model choice

## Local Development

```bash
npm install
npm run dev
```

Build for production:

```bash
npm run build
npm run preview
```

## Architecture

```
src/
  app.js              # Main application orchestrator
  config.js           # Model configuration and registry
  webcam.js           # Camera stream management
  face-detector.js    # MediaPipe face landmark detection
  focus-analyzer.js   # Focus state classification logic
  infer.js            # Inference pipeline coordinator
  vl-model.js         # Vision-language model loader
  vl-processor.js     # Image/text preprocessing
  smolvlm.js          # SmolVLM inference utilities
  webgpu-inference.js # WebGPU session management
```

## Privacy

All processing happens on-device. Camera frames are analyzed locally and never transmitted to any server. Model weights are downloaded once from Hugging Face and cached in the browser's Origin Private File System.

## License

MIT
