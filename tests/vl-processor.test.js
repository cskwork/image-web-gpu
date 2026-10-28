import { test } from 'node:test';
import assert from 'node:assert/strict';

// processImage checks `image instanceof ImageData`; Node has no ImageData, so provide a minimal one.
// Only the 32-aligned square fast path (the live webcam capture sizes) avoids <canvas> and runs here.
globalThis.ImageData ??= class ImageData {
  constructor(data, width, height) {
    this.data = data;
    this.width = width;
    this.height = height;
  }
};

const { isImageTooLarge, smartResize, getTokensNum, calculateTileGrid, processImage } =
  await import('../src/vl-processor.js');

const PATCH = 16;
const MAX_PATCHES = 1024; // 32 x 32 patches per 512 px tile
const PATCH_DIM = PATCH * PATCH * 3;

test('webcam capture sizes (256-512 px squares) stay on the single-tile path', () => {
  for (const size of [256, 384, 448, 512]) {
    assert.equal(isImageTooLarge(size, size), false, `${size}px`);
  }
  assert.equal(isImageTooLarge(1280, 720), true);
});

test('smartResize returns 32-aligned sizes within the 64-256 image-token budget', () => {
  for (const [w, h] of [[640, 480], [1280, 720], [100, 50], [2000, 500], [384, 384]]) {
    const r = smartResize(w, h);
    assert.equal(r.width % 32, 0, `${w}x${h} width`);
    assert.equal(r.height % 32, 0, `${w}x${h} height`);
    const tokens = getTokensNum(r.height, r.width);
    assert.ok(tokens >= 64 && tokens <= 256, `${w}x${h} -> ${tokens} tokens`);
  }
  assert.deepEqual(smartResize(640, 480), { width: 576, height: 416 });
  assert.deepEqual(smartResize(384, 384), { width: 384, height: 384 });
});

test('getTokensNum applies the 2x downsample to the 16 px patch grid', () => {
  assert.equal(getTokensNum(384, 384), 144); // 24 x 24 patches -> 12 x 12 tokens
  assert.equal(getTokensNum(512, 512), 256);
});

test('calculateTileGrid picks a grid matching the aspect ratio', () => {
  assert.deepEqual(calculateTileGrid(1280, 720), { rows: 1, cols: 2 });
  assert.deepEqual(calculateTileGrid(2000, 500), { rows: 1, cols: 4 });
});

test('processImage normalises pixels to [-1, 1] and masks padding patches', async () => {
  const size = 384;
  const data = new Uint8ClampedArray(size * size * 4);
  // First pixel white, everything else black.
  data[0] = data[1] = data[2] = 255;
  const out = await processImage(new ImageData(data, size, size));

  assert.equal(out.numTiles, 1);
  assert.deepEqual(out.shape, [1, MAX_PATCHES, PATCH_DIM]);
  assert.deepEqual(Array.from(out.spatialShapes), [24n, 24n]);

  const valid = 24 * 24;
  assert.equal(out.attentionMask.filter((v) => v === 1n).length, valid);
  assert.equal(out.attentionMask[valid - 1], 1n);
  assert.equal(out.attentionMask[valid], 0n);

  assert.equal(out.pixelValues[0], 1); // 255 -> 1
  assert.equal(out.pixelValues[3], -1); // 0 -> -1
  assert.equal(out.pixelValues[valid * PATCH_DIM], 0); // padding stays zero
});
