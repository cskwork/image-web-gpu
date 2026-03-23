/**
 * SmolVLM-256M 모바일 추론 모듈
 * Transformers.js 파이프라인으로 간단하게 로드/실행
 * 총 ~189MB (Q4F16) - 모바일 WASM에서 실행 가능
 */

import {
  AutoProcessor,
  AutoModelForVision2Seq,
  RawImage,
  TextStreamer,
} from '@huggingface/transformers';

const MODEL_ID = 'HuggingFaceTB/SmolVLM-256M-Instruct';
const DTYPE = {
  embed_tokens: 'q4f16',
  vision_encoder: 'q4f16',
  decoder_model_merged: 'q4f16',
};

let processor = null;
let model = null;
let loading = false;

/**
 * 모델 로드
 */
export async function loadSmolVLM(progressCallback) {
  if (loading) throw new Error('Already loading');
  if (model && processor) return;

  loading = true;
  const report = (status, progress, file) => {
    if (progressCallback) progressCallback({ status, progress, file });
  };

  try {
    report('loading', 10, 'SmolVLM 프로세서');
    processor = await AutoProcessor.from_pretrained(MODEL_ID);

    report('loading', 30, 'SmolVLM 모델 (189MB)');
    model = await AutoModelForVision2Seq.from_pretrained(MODEL_ID, {
      dtype: DTYPE,
      device: 'wasm',
      progress_callback: (p) => {
        if (p.status === 'progress' && p.total) {
          const pct = Math.round(30 + (p.loaded / p.total) * 60);
          report('loading', pct, `${(p.loaded / 1e6).toFixed(0)} / ${(p.total / 1e6).toFixed(0)} MB`);
        }
      },
    });

    report('done', 100, '');
  } catch (e) {
    processor = null;
    model = null;
    throw e;
  } finally {
    loading = false;
  }
}

/**
 * 모델 로드 여부
 */
export function isSmolVLMLoaded() {
  return model !== null && processor !== null;
}

/**
 * 이미지 분석
 * @param {string} imageDataURL - 이미지 data URL
 * @param {string} prompt - 텍스트 프롬프트
 * @param {object} options - { maxNewTokens }
 * @returns {Promise<string>}
 */
export async function analyzeWithSmolVLM(imageDataURL, prompt, options = {}) {
  if (!model || !processor) throw new Error('Model not loaded');

  const { maxNewTokens = 30 } = options;

  // 이미지 로드
  const image = await RawImage.fromURL(imageDataURL);

  // 채팅 템플릿 적용
  const messages = [
    {
      role: 'user',
      content: [
        { type: 'image' },
        { type: 'text', text: prompt },
      ],
    },
  ];

  const text = processor.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
  });

  const inputs = await processor(text, [image], { do_image_splitting: false });

  // 생성
  const output = await model.generate({
    ...inputs,
    max_new_tokens: maxNewTokens,
    do_sample: false,
  });

  // 입력 토큰 제거 후 디코딩
  const promptLength = inputs.input_ids.dims[1];
  const generated = output.slice(null, [promptLength, null]);
  const decoded = processor.batch_decode(generated, { skip_special_tokens: true });

  return decoded[0] || '';
}

/**
 * 리소스 해제
 */
export async function disposeSmolVLM() {
  if (model) {
    try { await model.dispose(); } catch {}
    model = null;
  }
  processor = null;
}
