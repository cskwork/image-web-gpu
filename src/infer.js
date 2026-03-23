/**
 * Inference Router
 * Routes inference requests to WebGPU
 */

import { getWebGPUInference, clearModelCache, getCacheInfo } from './webgpu-inference.js';

// Re-export cache utilities for UI
export { clearModelCache, getCacheInfo };

// Engine instance (lazy initialized)
let webgpuEngine = null;

/**
 * Load a model
 * @param {string} modelId - Model ID from config
 * @param {object} options - Loading options
 */
export async function loadModel(modelId, options = {}) {
    if (!webgpuEngine) {
        webgpuEngine = getWebGPUInference();
    }

    await webgpuEngine.loadModel(modelId, options);
}

/**
 * Check if a model is loaded
 * @returns {boolean}
 */
export function isModelLoaded() {
    return webgpuEngine && webgpuEngine.isModelLoaded();
}

/**
 * Get current model ID
 * @returns {string|null}
 */
export function getCurrentModelId() {
    return webgpuEngine ? webgpuEngine.getCurrentModelId() : null;
}

/**
 * Generate a response from messages
 * @param {Array<Object>} messages - Array of message objects with role and content
 * @param {object} options - Generation options
 * @param {function} options.onToken - Token callback for streaming
 * @returns {Promise<string>} Generated response
 */
export async function generate(messages, options = {}) {
    if (!webgpuEngine) {
        webgpuEngine = getWebGPUInference();
    }

    if (!webgpuEngine.isModelLoaded()) {
        throw new Error('Model not loaded. Please load a model first.');
    }

    return await webgpuEngine.generate(messages, options);
}

/**
 * Clear the image embedding cache (call when starting a new conversation)
 */
export function clearImageCache() {
    if (webgpuEngine) {
        webgpuEngine.clearImageCache();
    }
}

/**
 * Dispose resources
 */
export function dispose() {
    if (webgpuEngine) {
        webgpuEngine.dispose();
        webgpuEngine = null;
    }
}
