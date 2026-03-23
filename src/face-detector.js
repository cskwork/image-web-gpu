/**
 * MediaPipe Face Landmarker 기반 집중도 감지
 * 모델 ~3.6MB, 모바일 30FPS 실시간 처리
 */

import { FaceLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';

let landmarker = null;
let drawingUtils = null;
let overlayCanvas = null;
let overlayCtx = null;
let running = false;
let lastVideoTime = -1;
let onStatusChange = null;

// 상태
let currentStatus = 'unknown';
let smoothedYaw = 0;
let smoothedPitch = 0;
const SMOOTH_FACTOR = 0.15;

// 임계값 (완화: 자연스러운 움직임 허용)
const YAW_THRESHOLD = 40;   // 좌우 회전 (기존 25 -> 40)
const PITCH_THRESHOLD = 35;  // 상하 회전 (기존 20 -> 35)
const BLINK_THRESHOLD = 0.8; // 눈 감김 (기존 0.6 -> 0.8, 깜빡임 무시)
const FACE_QUALITY_THRESHOLD = 0.03; // 눈 blendshape 합산 최소값 (이하 = 얼굴 미확인)

/**
 * 초기화
 * @param {HTMLCanvasElement} canvas - 비디오 위 오버레이 캔버스
 * @param {Function} statusCallback - (status, details) => void
 */
export async function initFaceDetector(canvas, statusCallback, progressCallback) {
  overlayCanvas = canvas;
  overlayCtx = canvas.getContext('2d');
  onStatusChange = statusCallback;

  if (progressCallback) progressCallback({ status: 'loading', progress: 20, file: '얼굴 감지 모델 (3.6MB)' });

  const vision = await FilesetResolver.forVisionTasks(WASM_URL);

  if (progressCallback) progressCallback({ status: 'loading', progress: 60, file: '얼굴 감지 모델 초기화' });

  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });

  drawingUtils = new DrawingUtils(overlayCtx);

  if (progressCallback) progressCallback({ status: 'done', progress: 100, file: '' });
}

/**
 * 실시간 감지 루프 시작
 * @param {HTMLVideoElement} video
 */
export function startDetection(video) {
  if (!landmarker) return;
  running = true;
  lastVideoTime = -1;
  detectLoop(video);
}

/**
 * 감지 루프 중지
 */
export function stopDetection() {
  running = false;
  if (overlayCtx && overlayCanvas) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }
}

/**
 * 초기화 여부
 */
export function isFaceDetectorReady() {
  return landmarker !== null;
}

function detectLoop(video) {
  if (!running || !landmarker) return;

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    // 캔버스 크기 동기화
    if (overlayCanvas.width !== video.videoWidth || overlayCanvas.height !== video.videoHeight) {
      overlayCanvas.width = video.videoWidth;
      overlayCanvas.height = video.videoHeight;
    }

    const result = landmarker.detectForVideo(video, performance.now());
    processResult(result);
  }

  requestAnimationFrame(() => detectLoop(video));
}

function processResult(result) {
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  // 얼굴 미감지 -> 부재
  if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
    updateStatus('absent', '자리 비움');
    drawStatusBanner('자리 비움', '#ef4444');
    return;
  }

  const landmarks = result.faceLandmarks[0];
  const blendshapes = result.faceBlendshapes?.[0]?.categories || [];
  const matrix = result.facialTransformationMatrixes?.[0]?.data;

  // 얼굴 품질 검증: 눈 관련 blendshape 합산으로 실제 얼굴 감지 여부 확인
  // 머리카락/뒷통수만 보이면 blendshape 값이 전부 0에 가까움
  const faceQuality = computeFaceQuality(blendshapes);
  if (faceQuality < FACE_QUALITY_THRESHOLD) {
    updateStatus('distracted', '얼굴 미확인');
    drawFaceMesh(landmarks, 'distracted');
    drawStatusBanner('얼굴 미확인', statusColor('distracted'));
    return;
  }

  // 머리 회전 각도 계산
  const { yaw, pitch } = estimateHeadPose(landmarks, matrix);
  smoothedYaw = smoothedYaw * (1 - SMOOTH_FACTOR) + Math.abs(yaw) * SMOOTH_FACTOR;
  smoothedPitch = smoothedPitch * (1 - SMOOTH_FACTOR) + Math.abs(pitch) * SMOOTH_FACTOR;

  // 눈 감김 확인
  const eyeBlinkL = getBlendshape(blendshapes, 'eyeBlinkLeft');
  const eyeBlinkR = getBlendshape(blendshapes, 'eyeBlinkRight');
  const eyesClosed = eyeBlinkL > BLINK_THRESHOLD && eyeBlinkR > BLINK_THRESHOLD;

  // 시선 방향 (눈)
  const lookOutL = getBlendshape(blendshapes, 'eyeLookOutLeft');
  const lookOutR = getBlendshape(blendshapes, 'eyeLookOutRight');
  const lookDown = (getBlendshape(blendshapes, 'eyeLookDownLeft') + getBlendshape(blendshapes, 'eyeLookDownRight')) / 2;
  const gazeAway = lookOutL > 0.6 || lookOutR > 0.6 || lookDown > 0.7;

  // 판정
  let status, label;
  if (eyesClosed) {
    status = 'distracted';
    label = '눈 감음';
  } else if (smoothedYaw > YAW_THRESHOLD || smoothedPitch > PITCH_THRESHOLD) {
    status = 'distracted';
    label = '다른 곳 응시';
  } else if (gazeAway) {
    status = 'distracted';
    label = '시선 이탈';
  } else {
    status = 'focused';
    label = '집중 중';
  }

  updateStatus(status, label);

  // 오버레이 그리기
  drawFaceMesh(landmarks, status);
  drawStatusBanner(label, statusColor(status));
}

function estimateHeadPose(landmarks, matrix) {
  if (matrix && matrix.length >= 16) {
    // 변환 행렬에서 회전 추출
    const yaw = Math.atan2(matrix[8], matrix[10]) * (180 / Math.PI);
    const pitch = Math.asin(-matrix[9]) * (180 / Math.PI);
    return { yaw, pitch };
  }

  // 폴백: 코-귀 랜드마크 거리로 추정
  const nose = landmarks[1];
  const leftEar = landmarks[234];
  const rightEar = landmarks[454];

  const leftDist = Math.sqrt((nose.x - leftEar.x) ** 2 + (nose.y - leftEar.y) ** 2);
  const rightDist = Math.sqrt((nose.x - rightEar.x) ** 2 + (nose.y - rightEar.y) ** 2);

  const ratio = leftDist / (rightDist + 0.001);
  const yaw = (ratio - 1) * 60;
  const pitch = (nose.y - 0.5) * 40;

  return { yaw, pitch };
}

/**
 * 얼굴 품질 점수 계산: 눈 관련 blendshape 활성도 합산
 * 머리카락/뒷통수만 보이면 값이 전부 0에 가까움
 */
function computeFaceQuality(blendshapes) {
  const eyeKeys = [
    'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight',
    'eyeLookOutLeft', 'eyeLookOutRight',
    'eyeLookUpLeft', 'eyeLookUpRight',
    'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight',
  ];
  let sum = 0;
  for (const key of eyeKeys) {
    sum += getBlendshape(blendshapes, key);
  }
  return sum;
}

function getBlendshape(categories, name) {
  const item = categories.find((c) => c.categoryName === name);
  return item ? item.score : 0;
}

function updateStatus(status, label) {
  if (status !== currentStatus) {
    currentStatus = status;
    if (onStatusChange) onStatusChange(status, label);
  }
}

function statusColor(status) {
  switch (status) {
    case 'focused': return '#22c55e';
    case 'distracted': return '#f59e0b';
    case 'absent': return '#ef4444';
    default: return '#94a3b8';
  }
}

function drawFaceMesh(landmarks, status) {
  const color = statusColor(status);
  const w = overlayCanvas.width;
  const h = overlayCanvas.height;

  // 얼굴 윤곽선
  overlayCtx.save();
  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = 2;
  overlayCtx.globalAlpha = 0.6;

  // 바운딩 박스 계산
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const pt of landmarks) {
    if (pt.x < minX) minX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y > maxY) maxY = pt.y;
  }

  const pad = 0.03;
  const bx = (minX - pad) * w;
  const by = (minY - pad) * h;
  const bw = (maxX - minX + pad * 2) * w;
  const bh = (maxY - minY + pad * 2) * h;

  // 둥근 바운딩 박스
  const r = 12;
  overlayCtx.beginPath();
  overlayCtx.roundRect(bx, by, bw, bh, r);
  overlayCtx.stroke();

  // 모서리 강조
  overlayCtx.lineWidth = 3;
  overlayCtx.globalAlpha = 0.9;
  const cornerLen = 20;

  // 좌상
  overlayCtx.beginPath();
  overlayCtx.moveTo(bx, by + cornerLen);
  overlayCtx.lineTo(bx, by);
  overlayCtx.lineTo(bx + cornerLen, by);
  overlayCtx.stroke();
  // 우상
  overlayCtx.beginPath();
  overlayCtx.moveTo(bx + bw - cornerLen, by);
  overlayCtx.lineTo(bx + bw, by);
  overlayCtx.lineTo(bx + bw, by + cornerLen);
  overlayCtx.stroke();
  // 좌하
  overlayCtx.beginPath();
  overlayCtx.moveTo(bx, by + bh - cornerLen);
  overlayCtx.lineTo(bx, by + bh);
  overlayCtx.lineTo(bx + cornerLen, by + bh);
  overlayCtx.stroke();
  // 우하
  overlayCtx.beginPath();
  overlayCtx.moveTo(bx + bw - cornerLen, by + bh);
  overlayCtx.lineTo(bx + bw, by + bh);
  overlayCtx.lineTo(bx + bw, by + bh - cornerLen);
  overlayCtx.stroke();

  overlayCtx.restore();
}

function drawStatusBanner(label, color) {
  const w = overlayCanvas.width;
  const h = overlayCanvas.height;

  // 하단 배너
  const bannerH = 36;
  const bannerY = h - bannerH - 12;

  overlayCtx.save();
  overlayCtx.fillStyle = color;
  overlayCtx.globalAlpha = 0.85;
  overlayCtx.beginPath();
  overlayCtx.roundRect(12, bannerY, w - 24, bannerH, 8);
  overlayCtx.fill();

  overlayCtx.globalAlpha = 1;
  overlayCtx.fillStyle = '#fff';
  overlayCtx.font = 'bold 15px "Noto Sans KR", sans-serif';
  overlayCtx.textAlign = 'center';
  overlayCtx.textBaseline = 'middle';
  overlayCtx.fillText(label, w / 2, bannerY + bannerH / 2);
  overlayCtx.restore();
}

/**
 * 리소스 해제
 */
export function disposeFaceDetector() {
  running = false;
  if (landmarker) {
    landmarker.close();
    landmarker = null;
  }
}
