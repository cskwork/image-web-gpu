# 0001. Mobile uses MediaPipe face landmarks instead of a vision-language model

Status: accepted (commit `bf8e125`, 2026-03-23)

## Context

The desktop path runs LFM2-VL through ONNX Runtime Web. On phones, the same path crashed the browser. Three fixes were tried, in this order:

1. `175426d`: force 256 px input and at most 40 new tokens on mobile. The model loaded, but the browser crashed once the camera and inference ran together.
2. `ab56c58`: switch mobile to SmolVLM-256M (189 MB) through Transformers.js. The commit message reports the decoder shrinking from 221 MB to 77 MB.
3. `bf8e125`: the commit message says VLMs of 189 to 316 MB were still too large for mobile WASM.

## Decision

On devices that `app.js` detects as mobile, load MediaPipe Face Landmarker (float16, about 3.6 MB) instead of any VLM. Judge the frame with rules in `src/face-detector.js`: face present, eye-blendshape quality, head yaw and pitch, gaze blendshapes, and eye closure that lasts more than 800 ms.

## Consequences

- Phones get a live verdict on every animation frame with a download of a few MB.
- The two paths use different methods. Desktop asks a VLM, and mobile applies hand-tuned thresholds, so their verdicts are not comparable. Two later fixes (`8ed592b` for the face-quality gate, `2fd4cf0` for the blink threshold of 0.45 and the 800 ms sustain) show that the thresholds need manual tuning.
- `src/smolvlm.js` from step 2 is still in the tree but is no longer imported.

## Alternatives considered

- A smaller input and a lower token cap on the VLM (`175426d`): not enough.
- A smaller VLM, SmolVLM-256M (`ab56c58`): still too large.
