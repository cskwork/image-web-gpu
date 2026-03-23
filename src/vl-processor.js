/**
 * LFM2-VL Image Processor for WebGPU/ONNX Runtime Web
 *
 * Implements the image preprocessing logic from Lfm2VlImageProcessorFast:
 * 1. Split image into tiles (512x512)
 * 2. Extract 16x16 patches from each tile (32x32 = 1024 patches per tile)
 * 3. Flatten each patch to 768 values (16*16*3)
 * 4. Normalize: (pixel / 255 - 0.5) / 0.5 = pixel / 127.5 - 1
 *
 * Output shapes match Python processor:
 * - pixel_values: [num_tiles, 1024, 768]
 * - pixel_attention_mask: [num_tiles, 1024]
 */

// Configuration from preprocessor_config.json
const CONFIG = {
  tileSize: 512,
  maxTiles: 10,
  minTiles: 2,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
  rescaleFactor: 1 / 255,
  useThumbnail: true,
  patchSize: 16,  // Each patch is 16x16 pixels
  patchesPerTile: 32,  // 512 / 16 = 32 patches per side = 1024 per tile
  downsampleFactor: 2,
  minImageTokens: 64,
  maxImageTokens: 256,
  maxPixelsTolerance: 2.0,
};

// Pre-computed normalization constants for faster patch extraction
// Formula: (pixel / 255 - 0.5) / 0.5 = pixel / 127.5 - 1.0
const NORM_SCALE = 1 / 127.5;
const NORM_OFFSET = -1.0;

// Pre-computed patch info for common live-caption resolutions (all 32-aligned)
const PRECOMPUTED_SIZES = {
  256: { width: 256, height: 256, patchesH: 16, patchesW: 16 },  // 256/16 = 16
  384: { width: 384, height: 384, patchesH: 24, patchesW: 24 },  // 384/16 = 24
  448: { width: 448, height: 448, patchesH: 28, patchesW: 28 },  // 448/16 = 28
  512: { width: 512, height: 512, patchesH: 32, patchesW: 32 },  // 512/16 = 32
};

/**
 * Round number to closest value divisible by factor
 */
function roundByFactor(number, factor) {
  return Math.round(number / factor) * factor;
}

/**
 * Ceil number to smallest value >= number divisible by factor
 */
function ceilByFactor(number, factor) {
  return Math.ceil(number / factor) * factor;
}

/**
 * Floor number to largest value <= number divisible by factor
 */
function floorByFactor(number, factor) {
  return Math.floor(number / factor) * factor;
}

/**
 * Find the closest aspect ratio from target ratios to match input aspect ratio
 * Matches Python's find_closest_aspect_ratio()
 */
function findClosestAspectRatio(aspectRatio, targetRatios, width, height, imageSize) {
  let bestRatioDiff = Infinity;
  let bestRatio = [1, 1];
  const area = width * height;

  for (const ratio of targetRatios) {
    const targetAspectRatio = ratio[0] / ratio[1];
    const ratioDiff = Math.abs(aspectRatio - targetAspectRatio);

    if (ratioDiff < bestRatioDiff) {
      bestRatioDiff = ratioDiff;
      bestRatio = ratio;
    } else if (ratioDiff === bestRatioDiff) {
      // If equally close, prefer ratio that better matches original image area
      const targetArea = imageSize * imageSize * ratio[0] * ratio[1];
      if (area > 0.5 * targetArea) {
        bestRatio = ratio;
      }
    }
  }

  return bestRatio;
}

/**
 * Check if image is too large to process as one tile
 * Matches Python's _is_img_too_large()
 */
function isImageTooLarge(width, height) {
  const { patchSize, maxImageTokens, downsampleFactor, maxPixelsTolerance } = CONFIG;
  const hBar = Math.max(patchSize, roundByFactor(height, patchSize));
  const wBar = Math.max(patchSize, roundByFactor(width, patchSize));
  const maxPixels = maxImageTokens * (patchSize ** 2) * (downsampleFactor ** 2) * maxPixelsTolerance;
  return hBar * wBar > maxPixels;
}

/**
 * Smart resize to ensure dimensions divisible by patchSize * downsampleFactor
 * and total pixels within [minPixels, maxPixels]
 * Matches Python's _smart_resize()
 * @returns {{width: number, height: number}}
 */
function smartResize(width, height) {
  const { patchSize, downsampleFactor, minImageTokens, maxImageTokens } = CONFIG;
  const totalFactor = patchSize * downsampleFactor;  // 32
  const minPixels = minImageTokens * (patchSize ** 2) * (downsampleFactor ** 2);
  const maxPixels = maxImageTokens * (patchSize ** 2) * (downsampleFactor ** 2);

  let hBar = Math.max(totalFactor, roundByFactor(height, totalFactor));
  let wBar = Math.max(totalFactor, roundByFactor(width, totalFactor));

  if (hBar * wBar > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels);
    hBar = Math.max(totalFactor, floorByFactor(height / beta, totalFactor));
    wBar = Math.max(totalFactor, floorByFactor(width / beta, totalFactor));
  } else if (hBar * wBar < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width));
    hBar = ceilByFactor(height * beta, totalFactor);
    wBar = ceilByFactor(width * beta, totalFactor);
  }

  return { width: wBar, height: hBar };
}

/**
 * Get number of tokens for an image of given dimensions
 * Matches Python's _get_tokens_num()
 */
function getTokensNum(height, width) {
  const { patchSize, downsampleFactor } = CONFIG;
  const numPatchesHeight = Math.floor(height / patchSize);
  const numPatchesWidth = Math.floor(width / patchSize);
  const dwnNumPatchesHeight = Math.ceil(numPatchesHeight / downsampleFactor);
  const dwnNumPatchesWidth = Math.ceil(numPatchesWidth / downsampleFactor);
  return dwnNumPatchesHeight * dwnNumPatchesWidth;
}

/**
 * Calculate optimal tile grid for an image
 * Matches Python's _high_res_preprocessor() grid selection
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @returns {{rows: number, cols: number}} - Tile grid dimensions
 */
function calculateTileGrid(width, height) {
  const { tileSize, minTiles, maxTiles } = CONFIG;
  const aspectRatio = width / height;

  // Generate valid patch grid configurations (width, height)
  // Matches Python: [(w, h) for n in range(min_tiles, max_tiles + 1) for w in range(1, n + 1) for h in range(1, n + 1) if min_tiles <= w * h <= max_tiles]
  const targetRatios = [];
  for (let n = minTiles; n <= maxTiles; n++) {
    for (let w = 1; w <= n; w++) {
      for (let h = 1; h <= n; h++) {
        if (w * h >= minTiles && w * h <= maxTiles) {
          // Check if already exists
          if (!targetRatios.some(r => r[0] === w && r[1] === h)) {
            targetRatios.push([w, h]);
          }
        }
      }
    }
  }
  // Sort by total tiles
  targetRatios.sort((a, b) => (a[0] * a[1]) - (b[0] * b[1]));

  if (targetRatios.length === 0) {
    return { rows: 1, cols: 1 };
  }

  // Find best matching grid configuration
  const [gridWidth, gridHeight] = findClosestAspectRatio(
    aspectRatio, targetRatios, width, height, tileSize
  );

  return { rows: gridHeight, cols: gridWidth };
}

/**
 * Process an image into flattened patches for VL model
 * Matches Python's _resize_and_maybe_split() logic
 * @param {HTMLImageElement|HTMLCanvasElement|ImageData} image - Input image or raw ImageData
 * @returns {Promise<{pixelValues: Float32Array, attentionMask: BigInt64Array, numTiles: number, shape: number[]}>}
 */
export async function processImage(image) {
  let width, height;
  let inputImageData = null;  // For direct ImageData input

  if (image instanceof ImageData) {
    // Direct ImageData input - skip canvas creation entirely
    width = image.width;
    height = image.height;
    inputImageData = image;
  } else if (image instanceof HTMLImageElement) {
    width = image.naturalWidth;
    height = image.naturalHeight;
  } else {
    width = image.width;
    height = image.height;
  }

  const { tileSize, patchSize, useThumbnail } = CONFIG;
  const patchesPerSide = CONFIG.patchesPerTile;  // 32
  const maxPatchesPerTile = patchesPerSide * patchesPerSide;  // 1024
  const patchDim = patchSize * patchSize * 3;  // 768

  // Check if image needs splitting (matches Python's _resize_and_maybe_split)
  const needsSplitting = isImageTooLarge(width, height);

  if (needsSplitting) {
    // HIGH-RES PATH: Split into tiles + optional thumbnail
    // Matches Python's _high_res_preprocessor()
    const { rows, cols } = calculateTileGrid(width, height);
    const totalGridTiles = rows * cols;

    // Only use tiling if we get more than 1 tile
    if (totalGridTiles > 1) {
      const numTiles = totalGridTiles + (useThumbnail ? 1 : 0);

      // Output arrays - use max patches per tile for uniform shape
      const pixelValues = new Float32Array(numTiles * maxPatchesPerTile * patchDim);
      const attentionMask = new BigInt64Array(numTiles * maxPatchesPerTile);
      const spatialShapes = new BigInt64Array(numTiles * 2);

      // STEP 1: Resize ENTIRE image to target grid dimensions (matches Python)
      const targetWidth = tileSize * cols;
      const targetHeight = tileSize * rows;

      const resizedCanvas = document.createElement('canvas');
      resizedCanvas.width = targetWidth;
      resizedCanvas.height = targetHeight;
      const resizedCtx = resizedCanvas.getContext('2d');
      resizedCtx.drawImage(image, 0, 0, targetWidth, targetHeight);

      // STEP 2: Extract tiles by CROPPING from resized image (matches Python)
      let tileIdx = 0;
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const tileCanvas = document.createElement('canvas');
          tileCanvas.width = tileSize;
          tileCanvas.height = tileSize;
          const tileCtx = tileCanvas.getContext('2d');

          // Crop tile from resized image
          tileCtx.drawImage(
            resizedCanvas,
            col * tileSize, row * tileSize, tileSize, tileSize,  // source crop
            0, 0, tileSize, tileSize  // dest (same size, no scaling)
          );

          const tileData = tileCtx.getImageData(0, 0, tileSize, tileSize);
          extractPatchesFromFullTile(tileData, pixelValues, attentionMask, tileIdx, patchesPerSide, maxPatchesPerTile);

          // Spatial shape for this tile
          spatialShapes[tileIdx * 2] = BigInt(patchesPerSide);      // height in patches
          spatialShapes[tileIdx * 2 + 1] = BigInt(patchesPerSide);  // width in patches

          tileIdx++;
        }
      }

      // STEP 3: Add thumbnail LAST (matches Python - thumbnail is appended)
      // Thumbnail uses smart resize to variable dimensions (like single-tile path)
      if (useThumbnail) {
        const thumbResized = smartResize(width, height);
        const thumbWidth = thumbResized.width;
        const thumbHeight = thumbResized.height;

        const thumbCanvas = document.createElement('canvas');
        thumbCanvas.width = thumbWidth;
        thumbCanvas.height = thumbHeight;
        const thumbCtx = thumbCanvas.getContext('2d');
        thumbCtx.drawImage(image, 0, 0, thumbWidth, thumbHeight);

        const thumbData = thumbCtx.getImageData(0, 0, thumbWidth, thumbHeight);
        const thumbPatchesH = thumbHeight / patchSize;
        const thumbPatchesW = thumbWidth / patchSize;

        extractPatchesFromVariableSize(thumbData, pixelValues, attentionMask, tileIdx, thumbPatchesH, thumbPatchesW, maxPatchesPerTile);

        // Spatial shape for thumbnail (variable based on smart resize)
        spatialShapes[tileIdx * 2] = BigInt(thumbPatchesH);
        spatialShapes[tileIdx * 2 + 1] = BigInt(thumbPatchesW);

        tileIdx++;
      }

      return {
        pixelValues,
        attentionMask,
        spatialShapes,
        numTiles,
        shape: [numTiles, maxPatchesPerTile, patchDim],
      };
    }
  }

  // SINGLE-TILE PATH: Smart resize only (no splitting)
  // Matches Python's else branch in _resize_and_maybe_split()

  let resizedWidth, resizedHeight, actualPatchesH, actualPatchesW;
  let imageData;

  // OPTIMIZATION: Check if dimensions are pre-computed (32-aligned live caption sizes)
  const precomputed = PRECOMPUTED_SIZES[width];
  const isAlreadyAligned = precomputed && width === height;

  if (inputImageData && isAlreadyAligned) {
    // FAST PATH: Direct ImageData with known dimensions - skip all resizing
    resizedWidth = width;
    resizedHeight = height;
    actualPatchesH = precomputed.patchesH;
    actualPatchesW = precomputed.patchesW;
    imageData = inputImageData;
  } else if (isAlreadyAligned) {
    // Dimensions already 32-aligned, skip smartResize computation
    resizedWidth = width;
    resizedHeight = height;
    actualPatchesH = precomputed.patchesH;
    actualPatchesW = precomputed.patchesW;

    // Still need to get ImageData from the image
    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = resizedWidth;
    resizedCanvas.height = resizedHeight;
    const resizedCtx = resizedCanvas.getContext('2d');
    resizedCtx.drawImage(image, 0, 0, resizedWidth, resizedHeight);
    imageData = resizedCtx.getImageData(0, 0, resizedWidth, resizedHeight);
  } else {
    // Standard path: compute smart resize
    const resized = smartResize(width, height);
    resizedWidth = resized.width;
    resizedHeight = resized.height;
    actualPatchesH = resizedHeight / patchSize;
    actualPatchesW = resizedWidth / patchSize;

    // Create canvas at actual resized dimensions
    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = resizedWidth;
    resizedCanvas.height = resizedHeight;
    const resizedCtx = resizedCanvas.getContext('2d');
    resizedCtx.drawImage(image, 0, 0, resizedWidth, resizedHeight);
    imageData = resizedCtx.getImageData(0, 0, resizedWidth, resizedHeight);
  }

  const numTiles = 1;
  const pixelValues = new Float32Array(numTiles * maxPatchesPerTile * patchDim);
  const attentionMask = new BigInt64Array(numTiles * maxPatchesPerTile);
  const spatialShapes = new BigInt64Array(numTiles * 2);

  extractPatchesFromVariableSize(imageData, pixelValues, attentionMask, 0, actualPatchesH, actualPatchesW, maxPatchesPerTile);

  spatialShapes[0] = BigInt(actualPatchesH);
  spatialShapes[1] = BigInt(actualPatchesW);

  return {
    pixelValues,
    attentionMask,
    spatialShapes,
    numTiles,
    shape: [numTiles, maxPatchesPerTile, patchDim],
  };
}

/**
 * Extract patches from a full 512x512 tile (all patches are valid)
 * @param {ImageData} tileData - Tile image data (512x512)
 * @param {Float32Array} pixelValues - Output pixel values array
 * @param {BigInt64Array} attentionMask - Output attention mask array
 * @param {number} tileIdx - Index of this tile
 * @param {number} patchesPerSide - Number of patches per side (32 for 512x512)
 * @param {number} maxPatchesPerTile - Max patches per tile for array indexing (1024)
 */
function extractPatchesFromFullTile(tileData, pixelValues, attentionMask, tileIdx, patchesPerSide, maxPatchesPerTile) {
  const patchSize = CONFIG.patchSize;
  const patchDim = patchSize * patchSize * 3;
  const tileWidth = tileData.width;

  const pixels = tileData.data;
  const tileOffset = tileIdx * maxPatchesPerTile * patchDim;
  const maskOffset = tileIdx * maxPatchesPerTile;

  let patchIdx = 0;

  for (let py = 0; py < patchesPerSide; py++) {
    for (let px = 0; px < patchesPerSide; px++) {
      const patchStartX = px * patchSize;
      const patchStartY = py * patchSize;

      // All patches in full tile are valid
      attentionMask[maskOffset + patchIdx] = 1n;

      // Extract and normalize patch pixels using pre-computed constants
      const patchOffset = tileOffset + patchIdx * patchDim;
      let outIdx = 0;

      // Flatten patch: iterate over pixels in patch, then channels
      // Optimized: srcIdx calculated once per pixel, use pre-computed normalization
      for (let dy = 0; dy < patchSize; dy++) {
        const rowOffset = (patchStartY + dy) * tileWidth;
        for (let dx = 0; dx < patchSize; dx++) {
          const srcIdx = (rowOffset + patchStartX + dx) * 4;
          // Optimized normalization: pixel * (1/127.5) - 1.0
          pixelValues[patchOffset + outIdx++] = pixels[srcIdx] * NORM_SCALE + NORM_OFFSET;
          pixelValues[patchOffset + outIdx++] = pixels[srcIdx + 1] * NORM_SCALE + NORM_OFFSET;
          pixelValues[patchOffset + outIdx++] = pixels[srcIdx + 2] * NORM_SCALE + NORM_OFFSET;
        }
      }

      patchIdx++;
    }
  }
}

/**
 * Extract patches from variable-sized image and pad to maxPatchesPerTile
 * Matches Python's convert_image_to_patches + pad_along_first_dim
 * @param {ImageData} imageData - Image data at actual dimensions
 * @param {Float32Array} pixelValues - Output pixel values array
 * @param {BigInt64Array} attentionMask - Output attention mask array
 * @param {number} tileIdx - Index of this tile
 * @param {number} patchesH - Number of patches in height
 * @param {number} patchesW - Number of patches in width
 * @param {number} maxPatchesPerTile - Max patches per tile for padding (1024)
 */
function extractPatchesFromVariableSize(imageData, pixelValues, attentionMask, tileIdx, patchesH, patchesW, maxPatchesPerTile) {
  const patchSize = CONFIG.patchSize;
  const patchDim = patchSize * patchSize * 3;
  const imageWidth = imageData.width;

  const pixels = imageData.data;
  const tileOffset = tileIdx * maxPatchesPerTile * patchDim;
  const maskOffset = tileIdx * maxPatchesPerTile;

  const actualPatches = patchesH * patchesW;

  // Extract actual patches
  let patchIdx = 0;
  for (let py = 0; py < patchesH; py++) {
    for (let px = 0; px < patchesW; px++) {
      const patchStartX = px * patchSize;
      const patchStartY = py * patchSize;

      // Mark as valid
      attentionMask[maskOffset + patchIdx] = 1n;

      // Extract and normalize patch pixels using pre-computed constants
      const patchOffset = tileOffset + patchIdx * patchDim;
      let outIdx = 0;

      // Flatten patch: iterate over pixels in patch, then channels
      // Optimized: srcIdx calculated once per pixel, use pre-computed normalization
      for (let dy = 0; dy < patchSize; dy++) {
        const rowOffset = (patchStartY + dy) * imageWidth;
        for (let dx = 0; dx < patchSize; dx++) {
          const srcIdx = (rowOffset + patchStartX + dx) * 4;
          // Optimized normalization: pixel * (1/127.5) - 1.0
          pixelValues[patchOffset + outIdx++] = pixels[srcIdx] * NORM_SCALE + NORM_OFFSET;
          pixelValues[patchOffset + outIdx++] = pixels[srcIdx + 1] * NORM_SCALE + NORM_OFFSET;
          pixelValues[patchOffset + outIdx++] = pixels[srcIdx + 2] * NORM_SCALE + NORM_OFFSET;
        }
      }

      patchIdx++;
    }
  }

  // Pad remaining patches with zeros and mask = 0
  for (let i = actualPatches; i < maxPatchesPerTile; i++) {
    attentionMask[maskOffset + i] = 0n;
    // pixelValues already initialized to 0
  }
}

/**
 * Load an image from URL or data URL
 * @param {string} src - Image URL or data URL
 * @returns {Promise<HTMLImageElement>}
 */
export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}
