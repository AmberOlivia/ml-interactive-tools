import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// Set VITE_BASE when deploying under a subpath (e.g. GitHub Pages: /repo-name/)
const base = process.env.VITE_BASE ?? '/';

export default defineConfig({
  base,
  plugins: [react()],
  resolve: {
    alias: {
      '@shared': resolve(__dirname, 'src/shared'),
    },
  },
  build: {
    rollupOptions: {
      input: {
        index: resolve(__dirname, 'index.html'),
        'forward-prop': resolve(__dirname, 'forward-prop.html'),
        backprop: resolve(__dirname, 'backprop.html'),
      },
    },
  },
});
