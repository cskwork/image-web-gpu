import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  publicDir: 'public',

  server: {
    port: 3000,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },

  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },

  optimizeDeps: {
    exclude: ['@huggingface/transformers', 'onnxruntime-web'],
  },

  build: {
    target: 'esnext',
    outDir: 'dist',
  },

  resolve: {
    alias: {
      '@': '/src',
    },
  },
});
