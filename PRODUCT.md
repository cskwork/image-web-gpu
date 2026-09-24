# Product

<!-- impeccable:product-schema 1 -->

> Context note: written during an unattended portfolio revamp (2026-09-24). No
> interview channel was available, so facts marked **[inferred]** come from the
> code, README, and live site rather than from the owner. Everything else is read
> directly from the repository.

## Platform

web

## Users

- **[inferred]** Primary: developers, AI practitioners, and portfolio reviewers who
  want to see a vision model run fully inside the browser on their own webcam.
  They arrive curious, stay for one or two minutes, and judge whether the demo
  works and feels trustworthy.
- **[inferred]** Secondary: a knowledge worker who wants to keep an eye on their own
  attention during a desk session (self-monitoring, not supervising others).

## Product Purpose

A browser app that watches the webcam and classifies the person in frame as
focused (집중), distracted (주의 산만), or away (자리 비움), then keeps a running tally,
a focus rate, and a timestamped history. Success is: the model loads, the camera
starts, and the verdicts update live without anything leaving the device.

## Positioning

All inference runs on the visitor's own device. Desktop uses LiquidAI LFM2-VL
vision-language models through ONNX Runtime Web on WebGPU; mobile uses MediaPipe
Face Landmarker (about 3.6 MB, real time). No frames are sent to a server.

## Operating Context

- Desktop Chrome or Edge with WebGPU is the recommended environment; mobile works
  on a lighter face-landmark path.
- First use downloads model weights (about 316 MB to 2.3 GB on desktop, about
  3.6 MB on mobile) from Hugging Face or Google Storage and caches them in the
  browser (OPFS/IndexedDB). A clear-cache control exists.
- Requires camera permission. The site is served with COOP/COEP headers
  (cross-origin isolation), so every asset must be same-origin or CORS/CORP-clean.

## Capabilities and Constraints

- Model choice (desktop): 기본 (~316 MB), 고성능 (~1.8 GB), 고성능 고정밀 (~2.3 GB).
- Capture resolution choice (desktop): 256, 384 (recommended), 448, 512 px.
- Live states: 대기 중, 분석 중, 집중 중, 주의 산만, 자리 비움.
- Session stats: total, focused, distracted, absent, focus rate, last 20 history rows.
- Frontend only: no backend, no accounts, no analytics.
- Korean UI (lang="ko").

## Brand Commitments

- Name in UI: "AI 근무 집중도 모니터" / "AI 집중도 분석".
- Voice: plain, factual Korean. Technical model names stay out of primary UI labels
  (existing decision recorded in config.js).

## Evidence on Hand

- Working demo at https://image-web-gpu.vercel.app.
- No testimonials, benchmarks, accuracy numbers, or customer claims exist. Do not
  invent accuracy percentages or speed figures.

## Product Principles

1. **Local by construction.** Never imply or introduce server processing of frames.
2. **Honest about cost.** Model size, download time, and device limits are shown before
   the visitor commits.
3. **The verdict is the product.** Current state must be readable at a glance, from a
   distance, while the person is working in another window.
4. **Self-observation, not surveillance.** **[inferred]** Framing stays on the user
   watching their own focus.

## Accessibility & Inclusion

- **[inferred]** Status must not rely on color alone (label + shape), and live verdict
  changes should be announced politely to assistive technology.
- Respect reduced motion.
