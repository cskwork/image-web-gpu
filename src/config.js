/**
 * 모델 구성
 * 450M (모바일/기본) + 1.6B (데스크톱/고성능)
 */

const HF_450M = 'https://huggingface.co/onnx-community/LFM2-VL-450M-ONNX/resolve/main';
const HF_1_6B = 'https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main';

// 분할된 외부 데이터 파일 개수 (2GB 초과 시)
export const EXTERNAL_DATA_FILE_COUNTS = {
  'decoder_fp16': 2,
};

// 모델 아키텍처별 컴포넌트 이름 매핑
const ARCH_450M = {
  embedTokens: 'embed_tokens',
  visionEncoder: 'vision_encoder',
  decoder: 'decoder_model_merged',
  cacheType: 'float16',
};

const ARCH_1_6B = {
  embedTokens: 'embed_tokens',
  visionEncoder: 'embed_images',
  decoder: 'decoder',
  cacheType: 'float32',
};

// 모델 구성 (UI에는 기술명을 숨기고 용도/크기만 표시)
export const MODELS = {
  // === 450M 모델 (모바일/기본) ===
  'LFM2-VL-450M-Q4F16': {
    id: 'LFM2-VL-450M-Q4F16',
    path: HF_450M,
    label: '기본 (권장)',
    size: '~316 MB',
    arch: ARCH_450M,
    quantization: { decoder: 'q4f16', visionEncoder: 'q4f16' },
  },
  // === 1.6B 모델 (데스크톱/고성능) ===
  'LFM2.5-VL-1.6B-merge-linear-Q4-Q4': {
    id: 'LFM2.5-VL-1.6B-merge-linear-Q4-Q4',
    path: HF_1_6B,
    label: '고성능',
    size: '~1.8 GB',
    arch: ARCH_1_6B,
    quantization: { decoder: 'q4', visionEncoder: 'q4' },
  },
  'LFM2.5-VL-1.6B-merge-linear-Q4-FP16': {
    id: 'LFM2.5-VL-1.6B-merge-linear-Q4-FP16',
    path: HF_1_6B,
    label: '고성능 고정밀',
    size: '~2.3 GB',
    arch: ARCH_1_6B,
    quantization: { decoder: 'q4', visionEncoder: 'fp16' },
  },
};

// 기본 설정
export const DEFAULT_CONFIG = {
  defaultModel: 'LFM2-VL-450M-Q4F16',
  maxNewTokens: 512,
  temperature: 0.0,
};

export function getConfig() {
  const config = { ...DEFAULT_CONFIG };
  if (typeof import.meta !== 'undefined' && import.meta.env) {
    if (import.meta.env.VITE_DEFAULT_MODEL) {
      config.defaultModel = import.meta.env.VITE_DEFAULT_MODEL;
    }
  }
  return config;
}

export function getModelConfig(modelId) {
  return MODELS[modelId];
}

export function getAvailableModels() {
  return Object.values(MODELS);
}
