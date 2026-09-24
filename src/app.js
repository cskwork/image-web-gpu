/**
 * AI 근무 집중도 모니터 - 메인 애플리케이션
 *
 * 무거운 AI 런타임(ONNX Runtime, Transformers.js, MediaPipe)은 모델을 로드할 때
 * 동적 import로 불러온다. 랜딩과 앱 셸은 가벼운 번들만으로 즉시 뜬다.
 */

import { getAvailableModels, getConfig } from './config.js';
import { startWebcam, stopWebcam, captureFrame, isActive } from './webcam.js';
import { classifyFocus, getFocusDisplayInfo, FocusStatus } from './focus-analyzer.js';
import {
  HISTORY_LIMIT,
  createSession,
  recordVerdict,
  focusRate,
  loadSession,
  saveSession,
  clearSession,
} from './session-store.js';

// ========================================
// 상태
// ========================================

const IS_MOBILE_DEVICE =
  /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) ||
  (navigator.maxTouchPoints > 0 && screen.width < 1024);

const APP_TITLE = 'AI 근무 집중도 모니터';
const ANALYSIS_RETRY_MS = 1500;

let monitoring = false;
let captureLoopRunning = false;
let session = createSession();
let signalState = 'idle';

// 지연 로드되는 AI 엔진 모듈
let inferModule = null;
let faceModule = null;

const loadInfer = async () => (inferModule ??= await import('./infer.js'));
const loadFace = async () => (faceModule ??= await import('./face-detector.js'));

const storage = (() => {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
})();

// ========================================
// DOM 요소
// ========================================

const $ = (id) => document.getElementById(id);

// ========================================
// 랜딩 화면
// ========================================

function initLanding() {
  $('landing-enter-btn').addEventListener('click', () => {
    $('landing').classList.add('hidden');
    $('app').classList.remove('hidden');
    window.scrollTo(0, 0);
    $('btn-load-model').focus({ preventScroll: true });
    refreshCacheInfo();
  });
}

// ========================================
// 안내 메시지 (alert 대신 화면 안에서 표시)
// ========================================

function showNotice(message, { error = false } = {}) {
  const box = $('app-notice');
  $('app-notice-text').textContent = message;
  box.classList.toggle('is-error', error);
  box.classList.remove('hidden');
}

function hideNotice() {
  $('app-notice').classList.add('hidden');
}

function cameraErrorMessage(err) {
  switch (err?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return '카메라 권한이 거부되었습니다. 주소창의 카메라 아이콘에서 권한을 허용한 뒤 다시 시작하세요.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return '연결된 카메라를 찾을 수 없습니다. 웹캠 연결을 확인한 뒤 다시 시작하세요.';
    case 'NotReadableError':
    case 'AbortError':
      return '다른 앱이 카메라를 사용 중입니다. 화상회의 앱 등을 종료한 뒤 다시 시작하세요.';
    default:
      if (!navigator.mediaDevices?.getUserMedia) {
        return '이 브라우저는 카메라 접근을 지원하지 않습니다. 최신 Chrome 또는 Edge에서 열어주세요.';
      }
      return '웹캠에 접근할 수 없습니다. 권한과 연결 상태를 확인해주세요.';
  }
}

// ========================================
// WebGPU 확인
// ========================================

async function checkWebGPU() {
  const badge = $('webgpu-badge');
  const setCpu = () => {
    badge.textContent = 'CPU 모드 (느림)';
    badge.className = 'badge badge-neutral';
    return false;
  };

  if (!navigator.gpu) return setCpu();

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return setCpu();
    const info = adapter.info || {};
    const desc = info.description || info.vendor || 'GPU';
    badge.textContent = `GPU: ${desc}`;
    badge.className = 'badge badge-success';
    return true;
  } catch {
    return setCpu();
  }
}

// ========================================
// 모델 관리
// ========================================

function populateModels() {
  const select = $('model-select');
  const models = getAvailableModels();
  const config = getConfig();
  select.innerHTML = '';
  models.forEach((model) => {
    const opt = document.createElement('option');
    opt.value = model.id;
    opt.textContent = `${model.label} (${model.size})`;
    if (model.id === config.defaultModel) opt.selected = true;
    select.appendChild(opt);
  });
}

function isEngineReady() {
  return IS_MOBILE_DEVICE
    ? Boolean(faceModule?.isFaceDetectorReady())
    : Boolean(inferModule?.isModelLoaded());
}

async function handleLoadModel() {
  const loadBtn = $('btn-load-model');
  const progressContainer = $('progress-container');
  const progressBar = $('progress-bar');
  const progressFill = $('progress-fill');
  const progressText = $('progress-text');
  const modelBadge = $('model-badge');

  const setProgress = (pct) => {
    progressFill.style.transform = `scaleX(${pct / 100})`;
    progressBar.setAttribute('aria-valuenow', String(pct));
  };

  loadBtn.disabled = true;
  hideNotice();
  progressContainer.classList.remove('hidden');
  setProgress(0);
  progressText.textContent = 'AI 엔진 준비 중...';
  modelBadge.textContent = '로딩 중...';
  modelBadge.className = 'badge badge-neutral';

  if (monitoring) await toggleMonitoring();

  const onProgress = (progress) => {
    if (progress.status === 'loading') {
      setProgress(Math.round(progress.progress || 0));
      progressText.textContent = progress.file ? `다운로드: ${progress.file}` : '모델 로딩 중...';
    } else if (progress.status === 'done') {
      setProgress(100);
    }
  };

  try {
    if (IS_MOBILE_DEVICE) {
      // 모바일: MediaPipe Face Landmarker (3.6MB, 실시간 30FPS)
      const face = await loadFace();
      await face.initFaceDetector($('face-overlay-canvas'), onFaceStatus, onProgress);
    } else {
      // 데스크톱: LFM2-VL (선택된 모델)
      const modelId = $('model-select').value;
      if (!modelId) return;
      const infer = await loadInfer();
      infer.clearImageCache();
      await infer.loadModel(modelId, { progressCallback: onProgress });
    }

    progressContainer.classList.add('hidden');
    modelBadge.textContent = 'AI 모델 준비 완료';
    modelBadge.className = 'badge badge-success';
    $('btn-start').disabled = false;
    $('btn-start').querySelector('span').textContent = '모니터링 시작';
    $('camera-idle-text').textContent = '카메라 꺼짐 — 모니터링 시작을 누르면 카메라가 켜집니다.';
    await refreshCacheInfo();
  } catch (err) {
    console.error('모델 로딩 실패:', err);
    setProgress(0);
    modelBadge.textContent = '로드 실패';
    modelBadge.className = 'badge badge-error';
    progressText.textContent = `오류: ${err?.message || err}`;
    showNotice('모델을 불러오지 못했습니다. 네트워크 연결을 확인하고 다시 시도하세요. 계속 실패하면 캐시를 삭제한 뒤 더 작은 모델을 선택해보세요.', { error: true });
  } finally {
    loadBtn.disabled = false;
  }
}

async function refreshCacheInfo() {
  const el = $('cache-info');
  const btn = $('btn-clear-cache');
  let info = null;
  try {
    info = await (await loadInfer()).getCacheInfo();
  } catch (err) {
    console.warn('캐시 정보 조회 실패:', err);
  }
  if (info && info.used > 1024 * 1024) {
    const mb = info.used / 1024 / 1024;
    el.textContent = mb >= 1000 ? `${(mb / 1024).toFixed(1)} GB 캐시됨` : `${mb.toFixed(0)} MB 캐시됨`;
    btn.disabled = false;
  } else {
    el.textContent = '캐시 없음';
    btn.disabled = true;
  }
}

async function handleClearCache() {
  if (!confirm('다운로드된 모델 파일을 삭제하시겠습니까?\n다음 사용 시 다시 다운로드됩니다.')) return;
  await (await loadInfer()).clearModelCache();
  await refreshCacheInfo();
}

// ========================================
// 모니터링 (웹캠 + 추론)
// ========================================

function setStartButton(active) {
  const btn = $('btn-start');
  btn.classList.toggle('active', active);
  btn.querySelector('span').textContent = active ? '모니터링 중지' : '모니터링 시작';
  btn.querySelector('svg').innerHTML = active
    ? '<rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/>'
    : '<polygon points="6,4 20,12 6,20"/>';
}

async function toggleMonitoring() {
  if (monitoring) {
    // 중지
    monitoring = false;
    faceModule?.stopDetection();
    stopWebcam();
    $('video-wrapper').dataset.camera = 'off';
    setStartButton(false);
    updateFocusOverlay('idle');
    setSignal('idle');
    return;
  }

  // 시작
  if (!isEngineReady()) {
    showNotice('먼저 아래 "AI 모델 준비"에서 모델을 로드해주세요.');
    $('btn-load-model').focus();
    return;
  }

  hideNotice();
  const btn = $('btn-start');
  btn.disabled = true;
  try {
    const video = $('webcam-video');
    await startWebcam(video);
    monitoring = true;
    $('video-wrapper').dataset.camera = 'on';
    setStartButton(true);

    if (IS_MOBILE_DEVICE) {
      // 모바일: MediaPipe 실시간 감지 (requestAnimationFrame 루프)
      faceModule.startDetection(video);
    } else {
      // 데스크톱: VLM 추론 루프
      updateFocusOverlay('analyzing');
      captureLoop();
    }
  } catch (err) {
    console.error('웹캠 시작 실패:', err);
    stopWebcam();
    showNotice(cameraErrorMessage(err), { error: true });
  } finally {
    btn.disabled = false;
  }
}

async function captureLoop() {
  if (captureLoopRunning) return;
  captureLoopRunning = true;

  while (monitoring && isActive()) {
    const ok = await analyzeFrame();
    // 연속 오류 시 CPU/GPU를 태우지 않도록 잠시 대기
    if (!ok && monitoring) await new Promise((r) => setTimeout(r, ANALYSIS_RETRY_MS));
  }

  captureLoopRunning = false;
}

/** @returns {Promise<boolean>} 분석 성공 여부 */
async function analyzeFrame() {
  if (!monitoring || !inferModule) return false;

  const resolution = parseInt($('resolution-select').value, 10);
  const maxTokens = 128;

  const dataURL = captureFrame(resolution);
  if (!dataURL) return false;

  // 캡처 플래시
  const flash = $('capture-flash');
  flash.classList.add('active');
  setTimeout(() => flash.classList.remove('active'), 150);

  updateFocusOverlay('analyzing');

  try {
    const prompt = 'Look at this image. Answer with one word: "focused", "distracted", or "absent".\n- focused: face visible, eyes OPEN, looking at screen/camera\n- distracted: eyes closed, looking away, only hair/back of head visible, or using phone\n- absent: no person in frame';
    const messages = [
      {
        role: 'user',
        content: [
          { type: 'image', value: dataURL },
          { type: 'text', value: prompt },
        ],
      },
    ];
    inferModule.clearImageCache();
    const response = await inferModule.generate(messages, { maxNewTokens: maxTokens });

    if (!monitoring) return true;

    const { status } = classifyFocus(response);
    applyVerdict(status, response);
    return true;
  } catch (err) {
    console.error('분석 오류:', err);
    $('signal-head').dataset.analyzing = 'false';
    return false;
  }
}

// ========================================
// MediaPipe 상태 콜백 (모바일)
// ========================================

let lastFaceStatusTime = 0;
const FACE_LOG_INTERVAL = 3000; // 3초마다 히스토리 기록

function onFaceStatus(status, label) {
  updateFocusOverlay(status);
  setSignal(status);

  const now = Date.now();
  if (now - lastFaceStatusTime > FACE_LOG_INTERVAL) {
    lastFaceStatusTime = now;
    recordAndRender(status, label);
  }
}

// ========================================
// 판정 반영
// ========================================

function applyVerdict(status, description) {
  updateFocusOverlay(status);
  setSignal(status);
  recordAndRender(status, description);
}

function recordAndRender(status, description) {
  session = recordVerdict(session, status, description);
  saveSession(storage, session);
  updateCurrentStatus(description, status);
  renderSession();
}

// ========================================
// UI 업데이트
// ========================================

const OVERLAY_LABELS = {
  [FocusStatus.FOCUSED]: '집중 중',
  [FocusStatus.DISTRACTED]: '주의 산만',
  [FocusStatus.ABSENT]: '자리 비움',
  analyzing: '분석 중...',
};

function updateFocusOverlay(status) {
  const icon = $('focus-icon');
  const label = $('focus-label');

  icon.className = 'focus-icon';
  if (status in OVERLAY_LABELS) {
    icon.classList.add(status);
    label.textContent = OVERLAY_LABELS[status];
  } else {
    label.textContent = '대기 중';
  }

  $('signal-head').dataset.analyzing = status === 'analyzing' ? 'true' : 'false';
}

const LAMP_STATES = [FocusStatus.FOCUSED, FocusStatus.DISTRACTED, FocusStatus.ABSENT];
const LAMP_COLORS = { focused: '#1fd1ad', distracted: '#ffb020', absent: '#ff4d3f' };
const LAMP_Y = { absent: 8, distracted: 16, focused: 24 };
const DEFAULT_FAVICON = '/favicon.svg';

/** 신호등 점등: 즉시 켜지고, 이전 등은 한 박자 잔광 */
function setSignal(status) {
  const next = LAMP_STATES.includes(status) ? status : 'idle';
  if (next === signalState) return;

  const head = $('signal-head');
  const prevLamp = head.querySelector(`[data-lamp="${signalState}"]`);
  if (prevLamp) {
    prevLamp.classList.remove('is-decaying');
    void prevLamp.offsetWidth; // 애니메이션 재시작
    prevLamp.classList.add('is-decaying');
    setTimeout(() => prevLamp.classList.remove('is-decaying'), 600);
  }

  signalState = next;
  head.dataset.state = next;
  const label = next === 'idle' ? '대기 중' : OVERLAY_LABELS[next];
  head.setAttribute('aria-label', `판정 신호: ${label}`);

  // 다른 탭에서도 보이도록 탭 제목과 아이콘에 반영
  document.title = next === 'idle' ? APP_TITLE : `[${label}] ${APP_TITLE}`;
  setFavicon(next);

  if (next !== 'idle') $('live-announcer').textContent = `현재 상태: ${label}`;
}

function setFavicon(state) {
  let link = document.querySelector('link[rel="icon"]');
  if (!link) {
    link = document.createElement('link');
    link.rel = 'icon';
    document.head.appendChild(link);
  }
  if (state === 'idle') {
    link.href = DEFAULT_FAVICON;
    return;
  }
  const lamps = Object.entries(LAMP_Y)
    .map(([k, y]) => `<circle cx='16' cy='${y}' r='4' fill='${k === state ? LAMP_COLORS[k] : '#2a3038'}'/>`)
    .join('');
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect x='8' y='1' width='16' height='30' rx='4' fill='#0d0f12'/>${lamps}</svg>`;
  link.href = `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

function updateCurrentStatus(description, status) {
  const card = $('current-status');
  card.className = 'verdict';
  if (status) card.classList.add(status);

  $('verdict-word').textContent = getFocusDisplayInfo(status).label;
  $('current-description').textContent = description || '분석 결과 없음';
  $('status-time').textContent = new Date().toLocaleTimeString('ko-KR');
}

function renderSession() {
  const { stats, history } = session;

  $('stat-total').textContent = stats.total;
  $('stat-focused').textContent = stats.focused;
  $('stat-distracted').textContent = stats.distracted;
  $('stat-absent').textContent = stats.absent;

  const rate = focusRate(stats);
  $('focus-rate-value').textContent = rate === null ? '--%' : `${rate}%`;
  $('focus-rate-fill').style.transform = `scaleX(${(rate ?? 0) / 100})`;
  $('signal-rate').textContent = rate === null ? '--' : String(rate);

  renderSteps(history);
  renderHistory(history);
  $('btn-reset-session').disabled = stats.total === 0;
}

function renderSteps(history) {
  const row = $('step-row');
  const ordered = history.slice(0, HISTORY_LIMIT).reverse(); // 오래된 → 최근
  const cells = [];
  for (let i = 0; i < HISTORY_LIMIT; i++) {
    const entry = ordered[i - (HISTORY_LIMIT - ordered.length)];
    const li = document.createElement('li');
    li.className = 'step';
    if (entry) {
      li.dataset.status = entry.status;
      const time = new Date(entry.at).toLocaleTimeString('ko-KR');
      li.title = `${time} ${getFocusDisplayInfo(entry.status).label}`;
      li.setAttribute('aria-label', li.title);
      if (i === HISTORY_LIMIT - 1) li.classList.add('is-now');
    } else {
      li.setAttribute('aria-hidden', 'true');
    }
    cells.push(li);
  }
  row.replaceChildren(...cells);
}

function renderHistory(history) {
  const list = $('history-list');
  if (history.length === 0) {
    list.innerHTML = '<div class="history-empty">아직 분석 기록이 없습니다.</div>';
    return;
  }

  const items = history.map((entry) => {
    const info = getFocusDisplayInfo(entry.status);
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `
      <div class="history-dot ${entry.status}"></div>
      <div class="history-body">
        <div class="history-text">${escapeHtml(entry.text)}</div>
        <div class="history-meta">
          <span class="history-time">${new Date(entry.at).toLocaleTimeString('ko-KR')}</span>
          <span class="history-status ${entry.status}">${info.label}</span>
        </div>
      </div>
    `;
    return item;
  });
  list.replaceChildren(...items);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ========================================
// 세션 저장/복원
// ========================================

function restoreSession() {
  const saved = loadSession(storage);
  if (saved && saved.stats.total > 0) {
    session = saved;
    const started = new Date(saved.startedAt).toLocaleString('ko-KR', {
      month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
    const note = $('session-note');
    note.textContent = `${started}부터 이어진 세션입니다. 새로 시작하려면 세션 초기화를 누르세요.`;
    note.classList.remove('hidden');
  }
  renderSession();
}

function handleResetSession() {
  if (!confirm('집계와 분석 기록을 모두 지우고 새 세션을 시작할까요?')) return;
  clearSession(storage);
  session = createSession();
  $('session-note').classList.add('hidden');
  $('current-status').className = 'verdict';
  $('verdict-word').textContent = monitoring ? '분석 중...' : '대기 중';
  $('current-description').textContent = '모니터링을 시작하면 AI가 실시간으로 상태를 분석합니다.';
  $('status-time').textContent = '--:--:--';
  renderSession();
}

// ========================================
// 초기화
// ========================================

function checkMobile() {
  if (!IS_MOBILE_DEVICE) return;

  const warning = $('mobile-warning');
  if (warning) warning.classList.remove('hidden');

  // 모바일: 해상도/모델 선택기 숨기기 (얼굴 감지 모델 사용)
  const resGroup = $('resolution-select')?.closest('.control-group');
  if (resGroup) resGroup.style.display = 'none';
  const modelRow = $('model-select')?.closest('.model-row');
  if (modelRow) {
    const label = modelRow.querySelector('.control-label');
    if (label) label.textContent = 'AI 모델 (모바일 · 얼굴 감지 약 3.6MB)';
    $('model-select').style.display = 'none';
  }
}

async function init() {
  initLanding();
  checkMobile();
  populateModels();
  restoreSession();

  // 이벤트 바인딩
  $('btn-load-model').addEventListener('click', handleLoadModel);
  $('btn-start').addEventListener('click', toggleMonitoring);
  $('btn-clear-cache').addEventListener('click', handleClearCache);
  $('btn-reset-session').addEventListener('click', handleResetSession);
  $('app-notice-close').addEventListener('click', hideNotice);

  await checkWebGPU();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
