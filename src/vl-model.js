/**
 * LFM2-VL Model Runner for ONNX Runtime Web
 *
 * Runs VL model inference using three ONNX models:
 * 1. embed_tokens.onnx - Text token embeddings
 * 2. embed_images.onnx - Image embeddings from patches
 * 3. decoder.onnx - Autoregressive decoder with conv state cache
 */

import * as ort from 'onnxruntime-web';
import { AutoTokenizer, env } from '@huggingface/transformers';
import { processImage, loadImage } from './vl-processor.js';
import { EXTERNAL_DATA_FILE_COUNTS } from './config.js';

// WASM 스레딩 비활성화 (worker에서 document 접근 오류 방지)
ort.env.wasm.numThreads = 1;

// Debug logging - set to false for production, toggle via setDebug(true) in console
let DEBUG = false;
export function setDebug(value) { DEBUG = value; console.log(`Debug logging ${value ? 'enabled' : 'disabled'}`); }
const log = (...args) => { if (DEBUG) console.log(...args); };

// IndexedDB 캐시 (Safari 호환 - Cache API는 Safari에서 대용량 파일 저장 실패)
const DB_NAME = 'onnx-model-cache';
const DB_VERSION = 1;
const STORE_NAME = 'files';

// 모바일: 50MB 이상 파일은 ONNX Runtime이 URL로 직접 스트리밍 (JS 메모리 절약)
// 데스크톱: 2GB까지 JS에서 버퍼링 (IDB 캐시 가능)
const IS_MOBILE = typeof navigator !== 'undefined' && (
  /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) ||
  (navigator.maxTouchPoints > 0 && (typeof screen !== 'undefined' && screen.width < 1024))
);
const LARGE_FILE_THRESHOLD = IS_MOBILE ? 50 * 1024 * 1024 : 2 * 1024 * 1024 * 1024;

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbGet(key) {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  } catch { return null; }
}

async function idbPut(key, value) {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (e) {
    console.warn('[IDB WRITE ERROR]', e);
  }
}

async function idbGetAllKeys() {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).getAllKeys();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  } catch { return []; }
}

async function idbClear() {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).clear();
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
    });
  } catch { return false; }
}

/**
 * 다운로드 -> ArrayBuffer 직접 반환 (메모리 최적화)
 * chunks 배열 없이 사전 할당 버퍼에 바로 쓴다.
 * 피크 메모리: contentLength 1배 (이전: 3-5배)
 */
async function downloadAsBuffer(url, options = {}, onProgress) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`Fetch failed: ${response.status}`);

  const contentLength = parseInt(response.headers.get('content-length') || '0', 10);

  // 진행률 불필요하거나 크기 모르면 한번에 읽기
  if (!contentLength || !onProgress) {
    return { buffer: await response.arrayBuffer(), contentLength };
  }

  // 사전 할당 버퍼에 직접 스트리밍 (chunk 배열 없이)
  const reader = response.body.getReader();
  const buffer = new Uint8Array(contentLength);
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer.set(value, received);
    received += value.length;
    onProgress(received, contentLength);
  }

  return { buffer: buffer.buffer, contentLength };
}

/**
 * IndexedDB 캐시 기반 fetch (Safari/Chrome 모두 호환)
 * 메모리 최적화: ArrayBuffer를 한 번만 할당, clone/copy 없음
 */
async function fetchWithCache(url, options = {}, onProgress = null) {
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return fetch(url, options);
  }

  const fileName = url.split('/').pop();

  // 1. IndexedDB 캐시 확인
  try {
    const cached = await idbGet(url);
    if (cached && cached.buffer) {
      const expectedSize = cached.expectedSize || 0;
      if (expectedSize > 0 && cached.buffer.byteLength < expectedSize) {
        log(`[IDB INCOMPLETE] ${fileName}: ${(cached.buffer.byteLength / 1e6).toFixed(1)} / ${(expectedSize / 1e6).toFixed(1)} MB`);
      } else {
        log(`[IDB HIT] ${fileName} (${(cached.buffer.byteLength / 1e6).toFixed(1)} MB)`);
        return new Response(cached.buffer, { status: 200 });
      }
    }
  } catch (e) {
    log(`[IDB READ ERROR] ${e.message}`);
  }

  // 2. 다운로드 (단일 ArrayBuffer로 직접 수신)
  log(`[Network] ${fileName}...`);
  const { buffer, contentLength } = await downloadAsBuffer(url, options, onProgress);

  // 3. IDB 저장 (fire-and-forget, structured clone은 IDB 내부에서 처리)
  idbPut(url, { buffer, expectedSize: contentLength, savedAt: Date.now() })
    .then(() => log(`[IDB SAVED] ${fileName} (${(buffer.byteLength / 1e6).toFixed(1)} MB)`))
    .catch((e) => log(`[IDB SAVE FAIL] ${fileName}: ${e.message}`));

  // 4. Response 래핑 (buffer 참조만 전달, 복사 없음)
  return new Response(buffer, { status: 200 });
}

/**
 * 모델 캐시 삭제
 */
export async function clearModelCache() {
  // IndexedDB 삭제
  const idbCleared = await idbClear();
  // 기존 Cache API도 정리 (이전 버전 호환)
  try { await caches.delete('onnx-models-v1'); } catch {}
  log(idbCleared ? 'Model cache cleared' : 'No cache to clear');
  return idbCleared;
}

/**
 * 캐시 사용량 조회
 */
export async function getCacheInfo() {
  try {
    // navigator.storage.estimate로 전체 사용량 조회 (버퍼 로드 없이)
    let used = 0;
    let available = 0;

    if (navigator.storage?.estimate) {
      const est = await navigator.storage.estimate();
      used = est.usage || 0;
      available = est.quota || 0;
    } else {
      // 폴백: IDB에서 expectedSize 합산 (버퍼 로드 없이)
      const keys = await idbGetAllKeys();
      for (const key of keys) {
        const entry = await idbGet(key);
        if (entry) used += entry.expectedSize || 0;
      }
    }

    return { used, available };
  } catch (e) {
    log('Error getting cache info:', e);
    return null;
  }
}

/**
 * Load tokenizer from model path (local or S3)
 * @param {string} modelPath - Path to model directory (local or S3 URL)
 * @returns {Promise<{tokenizer: object, specialTokens: object}>} - Tokenizer instance and special token IDs
 */
async function loadTokenizerFromPath(modelPath) {
  const isRemote = modelPath.startsWith('http://') || modelPath.startsWith('https://');
  log(`Loading tokenizer from ${isRemote ? 'remote' : 'local'}: ${modelPath}`);

  const fetchOptions = isRemote ? { mode: 'cors', credentials: 'omit' } : {};

  // Fetch tokenizer files (with caching)
  const [tokenizerResponse, configResponse] = await Promise.all([
    fetchWithCache(`${modelPath}/tokenizer.json`, fetchOptions),
    fetchWithCache(`${modelPath}/tokenizer_config.json`, fetchOptions),
  ]);

  if (!tokenizerResponse.ok) {
    throw new Error(`Failed to fetch tokenizer.json: ${tokenizerResponse.status}`);
  }
  if (!configResponse.ok) {
    throw new Error(`Failed to fetch tokenizer_config.json: ${configResponse.status}`);
  }

  const tokenizerJSON = await tokenizerResponse.text();
  const configJSON = await configResponse.text();

  log('Tokenizer files fetched, creating tokenizer...');

  // Parse tokenizer.json to extract special token IDs from added_tokens
  const tokenizerData = JSON.parse(tokenizerJSON);
  const specialTokens = {};

  if (tokenizerData.added_tokens) {
    for (const token of tokenizerData.added_tokens) {
      specialTokens[token.content] = token.id;
    }
    log('Found special tokens:', Object.keys(specialTokens).length);
  }

  // Create a unique fake model ID
  const fakeModelId = `tokenizer-${Date.now()}`;

  // Cache of files to serve
  const fileCache = {
    'tokenizer.json': tokenizerJSON,
    'tokenizer_config.json': configJSON,
  };

  // Intercept fetch to serve our cached files
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : input.url;

    // Check if this is a request for our fake model
    if (url.includes(fakeModelId)) {
      for (const [filename, content] of Object.entries(fileCache)) {
        if (url.includes(filename)) {
          log(`Serving cached ${filename}`);
          return new Response(content, {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        }
      }
      // Return 404 for other files (like config.json which tokenizer doesn't need)
      return new Response('Not found', { status: 404 });
    }

    return originalFetch(input, init);
  };

  // Disable local model check
  const originalAllowLocal = env.allowLocalModels;
  env.allowLocalModels = false;

  try {
    const tokenizer = await AutoTokenizer.from_pretrained(fakeModelId);
    log('Tokenizer created successfully');
    return { tokenizer, specialTokens };
  } finally {
    // Restore original state
    globalThis.fetch = originalFetch;
    env.allowLocalModels = originalAllowLocal;
  }
}

export class VLModel {
  constructor() {
    this.tokenizer = null;
    this.embedTokensSession = null;
    this.visionEncoderSession = null;
    this.decoderSession = null;
    this.config = null;
    this.imageTokenId = null;
    this.eosTokenId = null;
    this.hiddenSize = 1024;

    // 아키텍처별 설정 (load 시 주입)
    this.arch = null;

    // Image embedding cache (persists between turns)
    this.imageCache = new Map();
  }

  /**
   * Clear the image embedding cache (call when starting a new conversation)
   */
  clearImageCache() {
    this.imageCache.clear();
  }

  /**
   * Load the VL model from a directory
   * @param {string} modelPath - Path to model directory (S3 URL)
   * @param {object} options - Loading options
   * @param {function} options.progressCallback - Progress callback
   * @param {string} options.device - Device to use ('webgpu' or 'wasm')
   * @param {string} options.quantization - Quantization type ('q4', 'q8', or null for fp32)
   */
  async load(modelPath, options = {}) {
    const { progressCallback, device = 'webgpu', quantization = null, arch = null } = options;

    // 아키텍처 설정 저장 (컴포넌트 파일명, 캐시 타입)
    this.arch = arch || {
      embedTokens: 'embed_tokens',
      visionEncoder: 'embed_images',
      decoder: 'decoder',
      cacheType: 'float32',
    };

    const report = (status, progress = 0, file = '') => {
      if (progressCallback) {
        progressCallback({ status, progress, file });
      }
    };

    // Determine execution provider
    const executionProviders = device === 'webgpu'
      ? ['webgpu', 'wasm']
      : ['wasm'];

    try {
      // Load tokenizer and extract special token IDs
      report('loading', 0, 'tokenizer');
      const { tokenizer, specialTokens } = await loadTokenizerFromPath(modelPath);
      this.tokenizer = tokenizer;

      // Load chat template from S3 if not already set in tokenizer
      if (!this.tokenizer.chat_template) {
        try {
          const templateResponse = await fetch(`${modelPath}/chat_template.jinja`, {
            mode: 'cors',
            credentials: 'omit',
          });
          if (templateResponse.ok) {
            const template = await templateResponse.text();
            this.tokenizer.chat_template = template;
            log('Loaded chat template from model path');
          }
        } catch (e) {
          console.warn('Could not load chat template:', e);
        }
      }

      // Get special token IDs from parsed tokenizer.json
      this.imageTokenId = specialTokens['<image>'] ?? null;
      this.imageStartTokenId = specialTokens['<|image_start|>'] ?? null;
      this.imageEndTokenId = specialTokens['<|image_end|>'] ?? null;
      this.imageSplitTokenId = specialTokens['<|image_split|>'] ?? null;
      this.eosTokenId = this.tokenizer.eos_token_id;

      log('Image token ID:', this.imageTokenId);
      log('Image start token ID:', this.imageStartTokenId);
      log('Image end token ID:', this.imageEndTokenId);
      log('EOS token ID:', this.eosTokenId);

      if (this.imageTokenId === null) {
        console.warn('Warning: <image> token not found in tokenizer');
      }

      // Load config
      report('loading', 10, 'config');
      const configResponse = await fetch(`${modelPath}/config.json`, {
        mode: 'cors',
        credentials: 'omit',
      });
      this.config = await configResponse.json();
      // VL models have config in text_config
      const textConfig = this.config.text_config || this.config;
      this.hiddenSize = textConfig.hidden_size || 1024;
      this.numKVHeads = textConfig.num_key_value_heads || 8;
      this.headDim = Math.floor(this.hiddenSize / (textConfig.num_attention_heads || 16));
      log('Model config:', { hiddenSize: this.hiddenSize, numKVHeads: this.numKVHeads, headDim: this.headDim });

      // Get external data files using hardcoded file counts (no probing)
      const getExternalDataFiles = async (basePath, fileName, fetchOptions) => {
        const fileCount = EXTERNAL_DATA_FILE_COUNTS[fileName] || 1;
        const files = [];

        // Get primary file
        const primaryUrl = `${basePath}/onnx/${fileName}.onnx_data`;
        try {
          const headResp = await fetch(primaryUrl, { method: 'HEAD', ...fetchOptions });
          if (!headResp.ok) return []; // No external data
          files.push({
            path: `${fileName}.onnx_data`,
            url: primaryUrl,
            size: parseInt(headResp.headers.get('content-length') || '0', 10)
          });
        } catch (e) {
          return []; // No external data
        }

        // Get additional numbered files based on hardcoded count
        for (let i = 1; i < fileCount; i++) {
          const url = `${basePath}/onnx/${fileName}.onnx_data_${i}`;
          try {
            const resp = await fetch(url, { method: 'HEAD', ...fetchOptions });
            if (resp.ok) {
              files.push({
                path: `${fileName}.onnx_data_${i}`,
                url,
                size: parseInt(resp.headers.get('content-length') || '0', 10)
              });
            }
          } catch (e) {
            log(`Warning: Expected file ${fileName}.onnx_data_${i} not found`);
          }
        }

        return files;
      };

      // Helper to load ONNX model with external data (with caching and progress)
      // customProviders allows overriding execution providers for specific sessions
      const loadOnnxWithExternalData = async (name, progress, quantSuffix = quantization, customProviders = null) => {
        // Build filename with optional quantization suffix
        const suffix = quantSuffix ? `_${quantSuffix}` : '';
        const fileName = `${name}${suffix}`;
        report('loading', progress, `${fileName}.onnx`);

        const onnxPath = `${modelPath}/onnx/${fileName}.onnx`;
        const fetchOptions = { mode: 'cors', credentials: 'omit' };

        log(`Loading ${fileName}...`);

        // Progress callback for download progress
        const makeProgressCallback = (file) => (received, total) => {
          const mb = (received / 1024 / 1024).toFixed(0);
          const totalMb = (total / 1024 / 1024).toFixed(0);
          report('loading', progress, `${file}: ${mb} / ${totalMb} MB`);
        };

        // Get external data files (uses size-based format detection)
        const dataFiles = await getExternalDataFiles(modelPath, fileName, fetchOptions);
        const totalDataSize = dataFiles.reduce((sum, f) => sum + f.size, 0);
        log(`Found ${dataFiles.length} external data file(s) for ${fileName}, total: ${(totalDataSize / 1024 / 1024).toFixed(1)} MB`);

        // Use custom providers if specified, otherwise use default
        const providers = customProviders || executionProviders;
        const sessionOptions = {
          executionProviders: providers,
        };

        // Fetch ONNX file (with caching and progress)
        const onnxResponse = await fetchWithCache(onnxPath, fetchOptions, makeProgressCallback(`${fileName}.onnx`));
        if (!onnxResponse.ok) {
          throw new Error(`Failed to fetch ${fileName}.onnx: ${onnxResponse.status}`);
        }
        const onnxBuffer = await onnxResponse.arrayBuffer();
        log(`Loaded ${fileName}.onnx: ${(onnxBuffer.byteLength / 1024 / 1024).toFixed(1)} MB`);

        if (dataFiles.length > 0) {
          // Load each file individually - use memory for cacheable files, URL for oversized
          sessionOptions.externalData = [];
          for (const f of dataFiles) {
            if (f.size > LARGE_FILE_THRESHOLD) {
              // JS 메모리 절약: ONNX Runtime이 URL에서 직접 스트리밍
              const sizeMB = (f.size / 1024 / 1024).toFixed(0);
              log(`Streaming ${f.path} (${sizeMB} MB) via ONNX Runtime URL loader`);
              report('loading', progress, `${fileName}: ${sizeMB}MB 로딩 중 (잠시 기다려 주세요)...`);
              sessionOptions.externalData.push({
                path: f.path,
                data: f.url,
              });
            } else {
              // File fits in memory - fetch with caching and progress
              const dataResponse = await fetchWithCache(f.url, fetchOptions, makeProgressCallback(f.path));
              if (!dataResponse.ok) {
                throw new Error(`Failed to fetch ${f.path}: ${dataResponse.status}`);
              }
              const dataBuffer = await dataResponse.arrayBuffer();
              log(`Loaded ${f.path}: ${(dataBuffer.byteLength / 1024 / 1024).toFixed(1)} MB`);
              sessionOptions.externalData.push({
                path: f.path,
                data: new Uint8Array(dataBuffer),
              });
            }
          }
          report('loading', progress, `${fileName} (initializing)`);
        } else {
          report('loading', progress, `${fileName} (initializing)`);
        }

        const session = await ort.InferenceSession.create(new Uint8Array(onnxBuffer), sessionOptions);
        log(`Session created for ${fileName}`);
        return session;
      };

      // Parse quantization config (can be string for legacy or object for new format)
      const quantConfig = typeof quantization === 'object' ? quantization : {
        decoder: quantization,
        visionEncoder: quantization,
      };

      // Load embed_tokens
      const embedTokensQuant = quantConfig.decoder ? (quantConfig.decoder.includes('f16') ? quantConfig.decoder : 'fp16') : null;
      this.embedTokensSession = await loadOnnxWithExternalData(this.arch.embedTokens, 20, embedTokensQuant);

      // Load vision encoder (embed_images or vision_encoder)
      const visionEncoderQuant = quantConfig.visionEncoder || quantConfig.embedImages || null;
      this.visionEncoderSession = await loadOnnxWithExternalData(this.arch.visionEncoder, 40, visionEncoderQuant);

      // Load decoder (decoder or decoder_model_merged)
      const decoderQuant = quantConfig.decoder || null;
      this.decoderSession = await loadOnnxWithExternalData(this.arch.decoder, 60, decoderQuant);

      report('done', 100, '');
      return true;

    } catch (error) {
      // Better error reporting for ORT errors
      let errorMessage = error;
      if (typeof error === 'number') {
        errorMessage = `ONNX Runtime error code: ${error}. This may indicate a WebGPU memory or compatibility issue.`;
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }
      console.error('Failed to load VL model:', errorMessage);
      throw new Error(errorMessage);
    }
  }

  /**
   * Process images and get embeddings (with caching)
   * @param {string[]} imageInputs - Array of image URLs or data URLs
   * @returns {Promise<{embeddings: Float32Array, numTokens: number, tokensPerImage: number[]}>}
   */
  async getImageEmbeddings(imageInputs) {
    const allEmbeddings = [];
    const tokensPerImage = [];
    let totalTokens = 0;
    let cacheHits = 0;
    let cacheMisses = 0;

    for (const input of imageInputs) {
      // Check cache first
      if (this.imageCache.has(input)) {
        const cached = this.imageCache.get(input);
        allEmbeddings.push(cached.embeddings);
        tokensPerImage.push(cached.numTokens);
        totalTokens += cached.numTokens;
        cacheHits++;
        continue;
      }

      // Cache miss - load and process the image
      cacheMisses++;
      const img = await loadImage(input);
      const processed = await processImage(img);

      log(`Image processed: ${processed.numTiles} tiles, shape [${processed.shape.join(', ')}]`);

      // Create tensors - use shape from processed output
      const patchesPerTile = processed.shape[1];  // 1024

      const pixelValuesTensor = new ort.Tensor(
        'float32',
        processed.pixelValues,
        processed.shape  // [num_tiles, patches_per_tile, 768]
      );

      const attentionMaskTensor = new ort.Tensor(
        'int64',
        processed.attentionMask,  // BigInt64Array
        [processed.numTiles, patchesPerTile]  // [num_tiles, patches_per_tile]
      );

      const spatialShapesTensor = new ort.Tensor(
        'int64',
        processed.spatialShapes,  // BigInt64Array
        [processed.numTiles, 2]  // [num_tiles, 2]
      );

      // Run embed_images
      let outputs = await this.visionEncoderSession.run({
        pixel_values: pixelValuesTensor,
        pixel_attention_mask: attentionMaskTensor,
        spatial_shapes: spatialShapesTensor,
      });

      // Output shape: [num_image_tokens, hidden_dim] (already flattened)
      let embeddings = outputs.image_features;
      log('Image embeddings shape:', embeddings.dims);

      // Output is 2D: [num_tokens, hidden_dim]
      const numTokens = embeddings.dims[0];

      // Store in cache (copy the data since tensor might be reused)
      const embeddingsCopy = new Float32Array(embeddings.data);
      this.imageCache.set(input, { embeddings: embeddingsCopy, numTokens });

      tokensPerImage.push(numTokens);
      totalTokens += numTokens;
      allEmbeddings.push(embeddingsCopy);
    }

    if (DEBUG && (cacheHits > 0 || cacheMisses > 1)) {
      log(`Image embeddings: ${cacheHits} cached, ${cacheMisses} computed, ${totalTokens} total tokens`);
    }

    // Concatenate all image embeddings
    const totalLength = allEmbeddings.reduce((sum, e) => sum + e.length, 0);
    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const emb of allEmbeddings) {
      combined.set(emb, offset);
      offset += emb.length;
    }

    return { embeddings: combined, numTokens: totalTokens, tokensPerImage };
  }

  /**
   * Get text embeddings from token IDs
   * @param {number[]} inputIds - Token IDs as regular numbers
   * @returns {Promise<ort.Tensor>} - Text embeddings tensor
   */
  async getTextEmbeddings(inputIds) {
    const inputTensor = new ort.Tensor(
      'int64',
      new BigInt64Array(inputIds.map(id => BigInt(id))),
      [1, inputIds.length]
    );
    const outputs = await this.embedTokensSession.run({ input_ids: inputTensor });
    return outputs.inputs_embeds;
  }

  /**
   * Build combined embeddings by replacing image tokens with image embeddings (1:1)
   * Each <image> token position gets replaced with exactly one image embedding.
   * The sequence length remains the same.
   *
   * @param {number[]} inputIds - Token IDs
   * @param {ort.Tensor} textEmbeddings - Text embeddings tensor
   * @param {Float32Array} imageEmbeddings - Concatenated image embeddings
   */
  buildCombinedEmbeddings1to1(inputIds, textEmbeddings, imageEmbeddings) {
    const [, seqLen, hiddenDim] = textEmbeddings.dims;
    const textEmb = textEmbeddings.data;
    const imgEmb = imageEmbeddings;

    // Find all image token positions
    const imagePositions = [];
    for (let i = 0; i < inputIds.length; i++) {
      if (inputIds[i] === this.imageTokenId) {
        imagePositions.push(i);
      }
    }

    const numImageEmbeddings = imgEmb.length / hiddenDim;
    if (imagePositions.length !== numImageEmbeddings) {
      console.warn(`Image token mismatch: ${imagePositions.length} <image> tokens vs ${numImageEmbeddings} embeddings`);
    }

    // Copy text embeddings and replace image token positions
    const result = new Float32Array(textEmb);

    for (let i = 0; i < Math.min(imagePositions.length, numImageEmbeddings); i++) {
      const pos = imagePositions[i];
      const embStart = i * hiddenDim;
      const dstStart = pos * hiddenDim;
      result.set(imgEmb.slice(embStart, embStart + hiddenDim), dstStart);
    }

    return new ort.Tensor('float32', result, [1, seqLen, hiddenDim]);
  }

  /**
   * Initialize cache for decoder (both conv states and KV cache)
   */
  initializeCache() {
    const cache = {};
    const useFloat16 = this.arch && this.arch.cacheType === 'float16';
    const dtype = useFloat16 ? 'float16' : 'float32';
    const TypedArray = useFloat16 ? Uint16Array : Float32Array;

    for (const name of this.decoderSession.inputNames) {
      if (name.startsWith('past_conv')) {
        cache[name] = new ort.Tensor(
          dtype,
          new TypedArray(1 * this.hiddenSize * 3),
          [1, this.hiddenSize, 3]
        );
      } else if (name.startsWith('past_key_values')) {
        cache[name] = new ort.Tensor(
          dtype,
          new TypedArray(0),
          [1, this.numKVHeads, 0, this.headDim]
        );
      }
    }

    return cache;
  }

  /**
   * Update cache from decoder outputs
   */
  updateCache(cache, outputs) {
    for (const name of Object.keys(outputs)) {
      if (name.startsWith('present_conv')) {
        // Conv states: present_conv.X -> past_conv.X
        const cacheName = name.replace('present_conv', 'past_conv');
        if (cacheName in cache) {
          cache[cacheName] = outputs[name];
        }
      } else if (name.startsWith('present.')) {
        // KV cache: present.X.key -> past_key_values.X.key
        const cacheName = name.replace('present.', 'past_key_values.');
        if (cacheName in cache) {
          cache[cacheName] = outputs[name];
        }
      }
    }
  }

  /**
   * Generate text given messages with optional images
   * @param {Array} messages - Chat messages
   * @param {object} options - Generation options
   */
  async generate(messages, options = {}) {
    const { maxNewTokens = 256, onToken, images = [], messageImageMap = new Map() } = options;

    log(`=== VL Generate: ${messages.length} messages, ${images.length} images ===`);

    // Process images FIRST to get patch counts
    let imageEmbeddings = null;
    let tokensPerImage = [];
    let totalImageTokens = 0;

    if (images.length > 0) {
      const result = await this.getImageEmbeddings(images);
      imageEmbeddings = result.embeddings;
      tokensPerImage = result.tokensPerImage;
      totalImageTokens = result.numTokens;
      log(`Image tokens: ${totalImageTokens} (per-image: [${tokensPerImage.join(', ')}])`);
    }

    // Build prompt with <image> tokens placed in EACH message that has images
    // This is critical: each user message that sent an image needs its <image> token(s)
    let promptMessages = messages;
    if (images.length > 0) {
      promptMessages = messages.map((msg, idx) => {
        // Check if this message has images via messageImageMap
        if (msg.role === 'user' && messageImageMap.has(idx)) {
          const messageImages = messageImageMap.get(idx);
          const imageTokens = messageImages.map(() => '<image>').join('');
          return { ...msg, content: imageTokens + msg.content };
        }
        return msg;
      });
    }

    // Apply chat template
    const prompt = this.tokenizer.apply_chat_template(promptMessages, {
      add_generation_prompt: true,
      tokenize: false,
    });

    // Tokenize
    const encoded = this.tokenizer.encode(prompt);
    let inputIds = [...encoded];

    // Expand each <image> token to the correct count for that image
    // Add boundary tokens if available: <image_start> [tokens] <image_end>
    if (images.length > 0) {
      const expandedIds = [];
      let imageIdx = 0;

      for (const id of inputIds) {
        if (id === this.imageTokenId && imageIdx < tokensPerImage.length) {
          // Add start boundary if available
          if (this.imageStartTokenId) {
            expandedIds.push(this.imageStartTokenId);
          }

          // Replace single <image> with N copies
          const count = tokensPerImage[imageIdx];
          for (let i = 0; i < count; i++) {
            expandedIds.push(this.imageTokenId);
          }

          // Add end boundary if available
          if (this.imageEndTokenId) {
            expandedIds.push(this.imageEndTokenId);
          }

          imageIdx++;
        } else {
          expandedIds.push(id);
        }
      }
      inputIds = expandedIds;
    }

    // Get text embeddings for expanded sequence
    const textEmbeddings = await this.getTextEmbeddings(inputIds);

    // Replace image token embeddings with actual image embeddings (1:1)
    let inputsEmbeds;
    if (images.length > 0) {
      inputsEmbeds = this.buildCombinedEmbeddings1to1(inputIds, textEmbeddings, imageEmbeddings);
    } else {
      inputsEmbeds = textEmbeddings;
    }

    log(`Input sequence: ${inputsEmbeds.dims[1]} tokens, ${(inputsEmbeds.data.length * 4 / 1024 / 1024).toFixed(1)} MB`);

    // Initialize fresh cache for this generation
    // (KV cache is used within generation for autoregressive decoding)
    const cache = this.initializeCache();

    // Generation loop
    const seqLen = inputsEmbeds.dims[1];
    let curLen = seqLen;
    let currentEmbeds = inputsEmbeds;
    const generatedTokens = [];

    for (let step = 0; step < maxNewTokens; step++) {
      // Prepare attention mask
      const attentionMask = new ort.Tensor(
        'int64',
        new BigInt64Array(curLen).fill(1n),
        [1, curLen]
      );

      // Run decoder (LFM2 models don't use position_ids - position is implicit from attention)
      const feeds = {
        inputs_embeds: currentEmbeds,
        attention_mask: attentionMask,
        ...cache,
      };

      const outputs = await this.decoderSession.run(feeds);

      // Get logits - shape is [batch, seq_len, vocab_size]
      const logits = outputs.logits;
      const vocabSize = logits.dims[2];
      const logitsData = logits.data;

      // Get last token logits
      const lastLogitStart = (logits.dims[1] - 1) * vocabSize;
      const lastLogits = logitsData.slice(lastLogitStart, lastLogitStart + vocabSize);

      // Greedy decoding - find max
      let maxIdx = 0;
      let maxVal = lastLogits[0];
      for (let i = 1; i < vocabSize; i++) {
        if (lastLogits[i] > maxVal) {
          maxVal = lastLogits[i];
          maxIdx = i;
        }
      }

      generatedTokens.push(maxIdx);

      // Callback with token
      if (onToken) {
        const tokenText = this.tokenizer.decode([maxIdx]);
        const shouldStop = onToken(tokenText, maxIdx);
        if (shouldStop) break;
      }

      // Check for EOS
      if (maxIdx === this.eosTokenId) {
        break;
      }

      // Update cache for next token
      this.updateCache(cache, outputs);

      // Get embedding for next token
      const nextEmbeds = await this.getTextEmbeddings([maxIdx]);
      currentEmbeds = nextEmbeds;
      curLen++;
    }

    return this.tokenizer.decode(generatedTokens, { skip_special_tokens: true });
  }

  /**
   * Free resources
   */
  async dispose() {
    this.clearImageCache();
    this.tokenizer = null;

    // Properly release ONNX sessions to free GPU resources
    if (this.embedTokensSession) {
      try {
        await this.embedTokensSession.release();
      } catch (e) {
        console.warn('Error releasing embedTokensSession:', e);
      }
      this.embedTokensSession = null;
    }
    if (this.visionEncoderSession) {
      try {
        await this.visionEncoderSession.release();
      } catch (e) {
        console.warn('Error releasing embedImagesSession:', e);
      }
      this.visionEncoderSession = null;
    }
    if (this.decoderSession) {
      try {
        await this.decoderSession.release();
      } catch (e) {
        console.warn('Error releasing decoderSession:', e);
      }
      this.decoderSession = null;
    }
  }
}

export default VLModel;
