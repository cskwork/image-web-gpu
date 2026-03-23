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

// Debug logging - set to false for production, toggle via setDebug(true) in console
let DEBUG = false;
export function setDebug(value) { DEBUG = value; console.log(`Debug logging ${value ? 'enabled' : 'disabled'}`); }
const log = (...args) => { if (DEBUG) console.log(...args); };

// Cache configuration
const CACHE_NAME = 'onnx-models-v1';

// Threshold for URL-based ONNX loading (files too large for JS memory)
// Set to 2GB - files larger than this will stream instead of loading into memory
const LARGE_FILE_THRESHOLD = 2 * 1024 * 1024 * 1024; // 2GB

/**
 * Fetch with streaming progress tracking
 * @param {string} url - URL to fetch
 * @param {object} options - Fetch options
 * @param {function} onProgress - Progress callback (received, total) => void
 * @returns {Promise<Response>} - Response with complete body
 */
async function fetchWithProgress(url, options = {}, onProgress) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Fetch failed: ${response.status}`);
  }

  const contentLength = parseInt(response.headers.get('content-length') || '0', 10);
  if (!contentLength || !onProgress) {
    // No size info or no callback - return as-is
    return response;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, contentLength);
  }

  // Combine chunks into single buffer
  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  // Create new Response with fresh Headers for Cache API compatibility
  // Using the original headers object from a consumed response can cause issues
  return new Response(new Blob([buffer]), {
    status: response.status,
    headers: new Headers(response.headers),
  });
}

/**
 * Fetch with caching support using Cache API
 * @param {string} url - URL to fetch
 * @param {object} options - Fetch options
 * @param {function} onProgress - Optional progress callback (received, total) => void
 * @returns {Promise<Response>} - Response (from cache or network)
 */
async function fetchWithCache(url, options = {}, onProgress = null) {
  // Skip caching for local files
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return fetch(url, options);
  }

  const fileName = url.split('/').pop();

  // 1. Try cache read with validation
  try {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(url);
    if (cached) {
      // Validate by reading body - catches corrupted entries from failed cache.put()
      try {
        const buffer = await cached.clone().arrayBuffer();
        log(`[Cache HIT] ${fileName} (${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB)`);
        // Return a new Response with the validated buffer
        return new Response(buffer, {
          status: cached.status,
          statusText: cached.statusText,
          headers: cached.headers,
        });
      } catch (bodyError) {
        // Corrupted cache entry - delete it and re-fetch
        log(`[Cache CORRUPT] ${fileName} - deleting and re-fetching`);
        await cache.delete(url);
      }
    }
  } catch (e) {
    log(`[Cache ERROR] ${e.message}`);
  }

  // 2. Fetch from network with progress tracking
  log(`[Network] Fetching ${fileName}...`);
  const response = await fetchWithProgress(url, options, onProgress);

  // 3. Try to cache successful response (fire-and-forget)
  if (response.ok) {
    tryCacheResponse(url, response.clone());
  }

  return response;
}

/**
 * Try to cache a response (non-blocking, best-effort)
 * @param {string} url - URL to cache
 * @param {Response} response - Response to cache
 */
async function tryCacheResponse(url, response) {
  try {
    // Check available space before caching
    if (navigator.storage?.estimate) {
      const { usage = 0, quota = 0 } = await navigator.storage.estimate();
      const available = quota - usage;
      const responseSize = parseInt(response.headers.get('content-length') || '0', 10);

      // Skip if we don't have space for this file + 100MB buffer
      const BUFFER = 100 * 1024 * 1024;
      if (responseSize > 0 && available < responseSize + BUFFER) {
        log(`[Cache SKIP] Not enough space (need ${((responseSize + BUFFER) / 1e9).toFixed(2)} GB, have ${(available / 1e9).toFixed(2)} GB)`);
        return;
      }
    }

    const cache = await caches.open(CACHE_NAME);
    await cache.put(url, response);
    log(`[Cached] ${url.split('/').pop()}`);
  } catch (e) {
    // Caching failed, but download succeeded - that's fine
    console.warn(`[Cache WRITE ERROR] ${url.split('/').pop()}:`, e.name, e.message, e);
  }
}

/**
 * Clear the model cache
 * @returns {Promise<boolean>} - True if cache was deleted
 */
export async function clearModelCache() {
  const deleted = await caches.delete(CACHE_NAME);
  log(deleted ? 'Model cache cleared' : 'No cache to clear');
  return deleted;
}

/**
 * Get cache storage usage info (specifically for model cache)
 * @returns {Promise<{used: number, available: number}|null>}
 */
export async function getCacheInfo() {
  try {
    // Calculate actual size of just the model cache
    const cache = await caches.open(CACHE_NAME);
    const keys = await cache.keys();

    let totalSize = 0;
    for (const request of keys) {
      const response = await cache.match(request);
      if (response) {
        // Get the response body as blob to measure size
        const blob = await response.clone().blob();
        totalSize += blob.size;
      }
    }

    // Get quota info for available space
    let available = 0;
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      available = estimate.quota || 0;
    }

    return {
      used: totalSize,
      available: available,
    };
  } catch (e) {
    console.warn('Error getting cache info:', e);
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
    this.embedImagesSession = null;
    this.decoderSession = null;
    this.config = null;
    this.imageTokenId = null;
    this.eosTokenId = null;
    this.hiddenSize = 1024;  // Default for 450M

    // Image embedding cache (persists between turns)
    this.imageCache = new Map();  // URL -> { embeddings, numTokens }
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
    const { progressCallback, device = 'webgpu', quantization = null } = options;

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
              // File too large for JS memory - let ONNX Runtime stream it
              log(`Large file ${f.path} (${(f.size / 1024 / 1024 / 1024).toFixed(2)} GB), using URL-based loading`);
              report('loading', progress, `${fileName} (streaming ${f.path}...)`);
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
        embedImages: quantization === 'q4' ? 'q8' : quantization,
      };

      // Load embed_tokens (always fp16 when quantized, no q4/q8 version exists)
      const embedTokensQuant = quantConfig.decoder ? 'fp16' : null;
      this.embedTokensSession = await loadOnnxWithExternalData('embed_tokens', 20, embedTokensQuant);

      // Load embed_images (use specified quantization)
      const embedImagesQuant = quantConfig.embedImages || null;
      this.embedImagesSession = await loadOnnxWithExternalData('embed_images', 40, embedImagesQuant);

      // Load decoder (use specified quantization)
      const decoderQuant = quantConfig.decoder || null;
      this.decoderSession = await loadOnnxWithExternalData('decoder', 60, decoderQuant);

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
      let outputs = await this.embedImagesSession.run({
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

    for (const name of this.decoderSession.inputNames) {
      if (name.startsWith('past_conv')) {
        // Conv states: [batch, hidden_size, kernel_size-1]
        // Kernel size is 4, so we need 3 states
        cache[name] = new ort.Tensor(
          'float32',
          new Float32Array(1 * this.hiddenSize * 3),
          [1, this.hiddenSize, 3]
        );
      } else if (name.startsWith('past_key_values')) {
        // KV cache: [batch, num_kv_heads, past_seq_len, head_dim]
        // Initialize with 0 length sequence
        cache[name] = new ort.Tensor(
          'float32',
          new Float32Array(0),  // Empty cache initially
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
    if (this.embedImagesSession) {
      try {
        await this.embedImagesSession.release();
      } catch (e) {
        console.warn('Error releasing embedImagesSession:', e);
      }
      this.embedImagesSession = null;
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
