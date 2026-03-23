/**
 * Configuration for LFM2.5-VL Demo
 * WebGPU inference with ONNX models from HuggingFace Hub
 */

const HF_BASE = 'https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main';

// External data file counts for split exports (used when primary file is ≤2GB)
// Components not listed default to 1 file
export const EXTERNAL_DATA_FILE_COUNTS = {
  'decoder_fp16': 2,  // .onnx_data + .onnx_data_1
};

// 모델 구성 (UI에는 기술명을 숨기고 크기/속도만 표시)
export const MODELS = {
  'LFM2.5-VL-1.6B-merge-linear-Q4-Q4': {
    id: 'LFM2.5-VL-1.6B-merge-linear-Q4-Q4',
    path: HF_BASE,
    label: '경량 (빠른 속도)',
    size: '~1.8 GB',
    quantization: { decoder: 'q4', embedImages: 'q4' }
  },
  'LFM2.5-VL-1.6B-merge-linear-Q4-FP16': {
    id: 'LFM2.5-VL-1.6B-merge-linear-Q4-FP16',
    path: HF_BASE,
    label: '균형 (권장)',
    size: '~2.3 GB',
    quantization: { decoder: 'q4', embedImages: 'fp16' }
  },
  'LFM2.5-VL-1.6B-merge-linear-FP16-FP16': {
    id: 'LFM2.5-VL-1.6B-merge-linear-FP16-FP16',
    path: HF_BASE,
    label: '고정밀 (느림)',
    size: '~3.5 GB',
    quantization: { decoder: 'fp16', embedImages: 'fp16' }
  }
};

// Default settings
export const DEFAULT_CONFIG = {
  defaultModel: 'LFM2.5-VL-1.6B-merge-linear-Q4-FP16',
  maxNewTokens: 512,
  temperature: 0.0
};

// Get config with optional env overrides
export function getConfig() {
  const config = { ...DEFAULT_CONFIG };

  if (typeof import.meta !== 'undefined' && import.meta.env) {
    if (import.meta.env.VITE_DEFAULT_MODEL) {
      config.defaultModel = import.meta.env.VITE_DEFAULT_MODEL;
    }
  }

  return config;
}

// Get model configuration by ID
export function getModelConfig(modelId) {
  return MODELS[modelId];
}

// Get all available models
export function getAvailableModels() {
  return Object.values(MODELS);
}
