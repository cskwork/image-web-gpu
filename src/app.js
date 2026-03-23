/**
 * AI 근무 집중도 모니터 - 메인 애플리케이션
 */

import { loadModel, isModelLoaded, generate, clearImageCache, clearModelCache, getCacheInfo } from './infer.js';
import { getAvailableModels, getModelConfig, getConfig } from './config.js';
import { startWebcam, stopWebcam, captureFrame, isActive } from './webcam.js';
import { classifyFocus, getFocusDisplayInfo, FocusStatus } from './focus-analyzer.js';

// ========================================
// 상태
// ========================================

const IS_MOBILE_DEVICE =
  /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) ||
  (navigator.maxTouchPoints > 0 && screen.width < 1024);

let monitoring = false;
let captureLoopRunning = false;

const stats = {
  total: 0,
  focused: 0,
  distracted: 0,
  absent: 0,
};

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
  });
}

// ========================================
// WebGPU 확인
// ========================================

async function checkWebGPU() {
  const badge = $('webgpu-badge');
  if (!navigator.gpu) {
    badge.textContent = 'CPU 모드 (느림)';
    badge.className = 'badge badge-neutral';
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      badge.textContent = 'CPU 모드 (느림)';
      badge.className = 'badge badge-neutral';
      return false;
    }
    const info = adapter.info || {};
    const desc = info.description || info.vendor || 'GPU';
    badge.textContent = `GPU: ${desc}`;
    badge.className = 'badge badge-success';
    return true;
  } catch (e) {
    badge.textContent = 'CPU 모드 (느림)';
    badge.className = 'badge badge-neutral';
    return false;
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

async function handleLoadModel() {
  const modelId = $('model-select').value;
  if (!modelId) return;

  const loadBtn = $('btn-load-model');
  const progressContainer = $('progress-container');
  const progressFill = $('progress-fill');
  const progressText = $('progress-text');
  const modelBadge = $('model-badge');

  loadBtn.disabled = true;
  progressContainer.classList.remove('hidden');
  modelBadge.textContent = '로딩 중...';
  modelBadge.className = 'badge badge-neutral';

  // 기존 캡처 중지
  if (monitoring) toggleMonitoring();
  clearImageCache();

  try {
    await loadModel(modelId, {
      progressCallback: (progress) => {
        if (progress.status === 'loading') {
          const pct = Math.round(progress.progress || 0);
          progressFill.style.width = `${pct}%`;
          // 파일명에서 기술 정보 제거
          const file = progress.file || '';
          const cleanFile = file
            .replace(/embed_tokens[_\w]*/g, '텍스트 임베딩')
            .replace(/vision_encoder[_\w]*/g, '이미지 인코더')
            .replace(/embed_images[_\w]*/g, '이미지 인코더')
            .replace(/decoder_model_merged[_\w]*/g, '디코더')
            .replace(/decoder[_\w]*/g, '디코더')
            .replace(/\.onnx_?d?a?t?a?_?\d*/g, '')
            .replace(/fp16|q4f16|q4|q8/gi, '')
            .trim();
          progressText.textContent = cleanFile ? `다운로드: ${cleanFile}` : '모델 로딩 중...';
        } else if (progress.status === 'done') {
          progressFill.style.width = '100%';
        }
      },
    });

    progressContainer.classList.add('hidden');
    modelBadge.textContent = 'AI 모델 준비 완료';
    modelBadge.className = 'badge badge-success';
    $('btn-start').disabled = false;
    $('btn-start').querySelector('span').textContent = '모니터링 시작';
    await refreshCacheInfo();
  } catch (err) {
    console.error('모델 로딩 실패:', err);
    progressFill.style.width = '0%';
    modelBadge.textContent = '로드 실패';
    modelBadge.className = 'badge badge-error';

    // 손상된 캐시 자동 정리 후 재시도 안내
    const msg = err.message || '';
    const isCorrupt = msg.includes('Cache') || msg.includes('fetch') || msg.includes('network') || msg.includes('aborted');
    if (isCorrupt) {
      progressText.textContent = '캐시 손상 감지 - 정리 중...';
      await clearModelCache();
      await refreshCacheInfo();
      progressText.textContent = '캐시를 정리했습니다. "모델 로드"를 다시 눌러주세요.';
    } else {
      progressText.textContent = `오류: ${msg}`;
    }
  } finally {
    loadBtn.disabled = false;
  }
}

async function refreshCacheInfo() {
  const info = await getCacheInfo();
  const el = $('cache-info');
  const btn = $('btn-clear-cache');
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
  await clearModelCache();
  await refreshCacheInfo();
}

// ========================================
// 모니터링 (웹캠 + 추론)
// ========================================

async function toggleMonitoring() {
  const btn = $('btn-start');

  if (monitoring) {
    // 중지
    monitoring = false;
    stopWebcam();
    btn.classList.remove('active');
    btn.querySelector('span').textContent = '모니터링 시작';
    btn.querySelector('svg').innerHTML = '<polygon points="5,3 19,12 5,21"/>';
    updateFocusOverlay('idle');
    return;
  }

  // 시작
  if (!isModelLoaded()) {
    alert('먼저 AI 모델을 로드해주세요.');
    return;
  }

  try {
    const video = $('webcam-video');
    await startWebcam(video);
    monitoring = true;
    btn.classList.add('active');
    btn.querySelector('span').textContent = '모니터링 중지';
    btn.querySelector('svg').innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
    updateFocusOverlay('analyzing');
    captureLoop();
  } catch (err) {
    console.error('웹캠 시작 실패:', err);
    alert('웹캠에 접근할 수 없습니다. 권한을 확인해주세요.');
  }
}

async function captureLoop() {
  if (captureLoopRunning) return;
  captureLoopRunning = true;

  while (monitoring && isActive()) {
    await analyzeFrame();
  }

  captureLoopRunning = false;
}

async function analyzeFrame() {
  if (!monitoring) return;

  // 모바일: 메모리 절약을 위해 해상도/토큰 제한
  const isMobile = IS_MOBILE_DEVICE;
  const resolution = isMobile ? 256 : parseInt($('resolution-select').value, 10);
  const maxTokens = isMobile ? 40 : 128;

  const dataURL = captureFrame(resolution);
  if (!dataURL) return;

  // 캡처 플래시
  const flash = $('capture-flash');
  flash.classList.add('active');
  setTimeout(() => flash.classList.remove('active'), 150);

  updateFocusOverlay('analyzing');

  try {
    // 영어 프롬프트 (짧고 효율적, 모든 모델에서 안정적)
    // 응답은 focus-analyzer에서 한국어로 분류
    const messages = [
      {
        role: 'user',
        content: [
          { type: 'image', value: dataURL },
          {
            type: 'text',
            value: 'Is the person looking at the camera or screen? Answer briefly: focused, distracted, or absent.',
          },
        ],
      },
    ];

    // 이미지 캐시 클리어 (매 프레임 새 이미지)
    clearImageCache();

    const response = await generate(messages, { maxNewTokens: maxTokens });

    if (!monitoring) return;

    // 집중도 분류
    const { status } = classifyFocus(response);

    // UI 업데이트
    updateFocusOverlay(status);
    updateCurrentStatus(response, status);
    updateStats(status);
    addHistoryItem(response, status);
  } catch (err) {
    console.error('분석 오류:', err);
  }
}

// ========================================
// UI 업데이트
// ========================================

function updateFocusOverlay(status) {
  const icon = $('focus-icon');
  const label = $('focus-label');

  icon.className = 'focus-icon';
  switch (status) {
    case FocusStatus.FOCUSED:
    case 'focused':
      icon.classList.add('focused');
      label.textContent = '집중 중';
      break;
    case FocusStatus.DISTRACTED:
    case 'distracted':
      icon.classList.add('distracted');
      label.textContent = '주의 산만';
      break;
    case FocusStatus.ABSENT:
    case 'absent':
      icon.classList.add('absent');
      label.textContent = '자리 비움';
      break;
    case 'analyzing':
      icon.classList.add('analyzing');
      label.textContent = '분석 중...';
      break;
    default:
      label.textContent = '대기 중';
      break;
  }
}

function updateCurrentStatus(description, status) {
  const card = $('current-status');
  const descEl = $('current-description');
  const timeEl = $('status-time');

  card.className = 'status-card';
  if (status) card.classList.add(status);

  descEl.textContent = description || '분석 결과 없음';
  timeEl.textContent = new Date().toLocaleTimeString('ko-KR');
}

function updateStats(status) {
  stats.total++;
  if (status === FocusStatus.FOCUSED) stats.focused++;
  else if (status === FocusStatus.DISTRACTED) stats.distracted++;
  else if (status === FocusStatus.ABSENT) stats.absent++;

  $('stat-total').textContent = stats.total;
  $('stat-focused').textContent = stats.focused;
  $('stat-distracted').textContent = stats.distracted;
  $('stat-absent').textContent = stats.absent;

  // 집중률 계산
  const rate = stats.total > 0 ? Math.round((stats.focused / stats.total) * 100) : 0;
  $('focus-rate-value').textContent = `${rate}%`;
  $('focus-rate-fill').style.width = `${rate}%`;
}

function addHistoryItem(description, status) {
  const list = $('history-list');

  // 빈 상태 메시지 제거
  const empty = list.querySelector('.history-empty');
  if (empty) empty.remove();

  const info = getFocusDisplayInfo(status);
  const time = new Date().toLocaleTimeString('ko-KR');

  const item = document.createElement('div');
  item.className = 'history-item';
  item.innerHTML = `
    <div class="history-dot ${status}"></div>
    <div class="history-body">
      <div class="history-text">${escapeHtml(description)}</div>
      <div class="history-meta">
        <span class="history-time">${time}</span>
        <span class="history-status ${status}">${info.label}</span>
      </div>
    </div>
  `;

  list.prepend(item);

  // 최대 20개 유지
  while (list.children.length > 20) {
    list.removeChild(list.lastChild);
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ========================================
// 초기화
// ========================================

function checkMobile() {
  if (!IS_MOBILE_DEVICE) return;

  const warning = $('mobile-warning');
  if (warning) warning.classList.remove('hidden');

  // 모바일: 해상도 선택기 숨기고 256px 강제 (메모리 절약)
  const resGroup = $('resolution-select')?.closest('.control-group');
  if (resGroup) resGroup.style.display = 'none';
}

async function init() {
  initLanding();
  checkMobile();
  populateModels();
  await checkWebGPU();

  // 이벤트 바인딩
  $('btn-load-model').addEventListener('click', handleLoadModel);
  $('btn-start').addEventListener('click', toggleMonitoring);
  $('btn-clear-cache').addEventListener('click', handleClearCache);

  await refreshCacheInfo();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
