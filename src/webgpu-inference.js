/**
 * WebGPU Inference Wrapper
 * Provides a clean interface between the app and the VL model
 */

import { VLModel, clearModelCache, getCacheInfo, setDebug } from './vl-model.js';
import { getModelConfig } from './config.js';

// Expose debug toggle on window for browser console access
window.setDebug = setDebug;

// Re-export cache utilities
export { clearModelCache, getCacheInfo, setDebug };

export class WebGPUInference {
  constructor() {
    this.model = null;
    this.currentModelId = null;
    this.isLoading = false;
    this.isReady = false;
  }

  /**
   * Load a model
   * @param {string} modelId - Model ID from config
   * @param {object} options - Loading options
   * @param {function} options.progressCallback - Progress callback
   */
  async loadModel(modelId, options = {}) {
    if (this.isLoading) {
      throw new Error('Model is already loading');
    }

    if (this.currentModelId === modelId && this.isReady) {
      return;
    }

    this.isLoading = true;
    this.isReady = false;

    try {
      const modelConfig = getModelConfig(modelId);
      if (!modelConfig) {
        throw new Error(`Model configuration not found: ${modelId}`);
      }

      // Dispose old model if exists
      if (this.model) {
        this.model.dispose();
        this.model = null;
      }

      // Create new model instance
      this.model = new VLModel();

      // Load the model with quantization from config
      await this.model.load(modelConfig.path, {
        device: 'webgpu',
        quantization: modelConfig.quantization || { decoder: 'q4', embedImages: 'q4' },
        progressCallback: options.progressCallback,
      });

      this.currentModelId = modelId;
      this.isReady = true;

    } catch (error) {
      this.model = null;
      this.currentModelId = null;
      this.isReady = false;
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Generate a response from messages
   * @param {Array<Object>} messages - Array of message objects with role and content
   * @param {object} options - Generation options
   * @param {function} options.onToken - Token callback for streaming
   * @returns {Promise<string>} Generated response
   */
  async generate(messages, options = {}) {
    if (!this.isReady || !this.model) {
      throw new Error('Model not loaded. Please load a model first.');
    }

    // Convert app message format to VL model format
    const { vlMessages, images, messageImageMap } = this.convertMessages(messages);

    // Generate response
    return await this.model.generate(vlMessages, {
      maxNewTokens: options.maxNewTokens || 512,
      images: images,
      messageImageMap: messageImageMap,
      onToken: options.onToken,
    });
  }

  /**
   * Convert app message format to VL model format
   * @param {Array<Object>} messages - App messages
   * @returns {{vlMessages: Array, images: Array<string>, messageImageMap: Map}}
   */
  convertMessages(messages) {
    const vlMessages = [];
    const images = [];
    const messageImageMap = new Map();

    for (const message of messages) {
      const { role, content } = message;

      if (typeof content === 'string') {
        vlMessages.push({ role, content });
      } else if (Array.isArray(content)) {
        let textContent = '';
        const messageImages = [];

        for (const item of content) {
          if (item.type === 'text') {
            textContent += item.value;
          } else if (item.type === 'image') {
            messageImages.push(item.value);
            images.push(item.value);
          }
        }

        if (textContent.trim() || messageImages.length > 0) {
          if (messageImages.length > 0) {
            messageImageMap.set(vlMessages.length, messageImages);
          }
          vlMessages.push({ role, content: textContent || '' });
        }
      } else {
        vlMessages.push({ role, content: String(content || '') });
      }
    }

    return { vlMessages, images, messageImageMap };
  }

  /**
   * Check if a model is loaded
   * @returns {boolean}
   */
  isModelLoaded() {
    return this.isReady;
  }

  /**
   * Get current model ID
   * @returns {string|null}
   */
  getCurrentModelId() {
    return this.currentModelId;
  }

  /**
   * Clear the image embedding cache
   */
  clearImageCache() {
    if (this.model) {
      this.model.clearImageCache();
    }
  }

  /**
   * Dispose the model and free resources
   */
  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.currentModelId = null;
    this.isReady = false;
  }
}

// Singleton instance
let webgpuInstance = null;

/**
 * Get the WebGPU inference singleton
 * @returns {WebGPUInference}
 */
export function getWebGPUInference() {
  if (!webgpuInstance) {
    webgpuInstance = new WebGPUInference();
  }
  return webgpuInstance;
}
