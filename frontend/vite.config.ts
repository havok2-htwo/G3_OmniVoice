import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import { resolve } from 'path';

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        admin: resolve(__dirname, 'admin.html')
      }
    }
  },
  server: {
    host: '0.0.0.0',
    port: 5181,
    strictPort: true,
    proxy: {
      // Keep the browser's Host so the backend's same-origin check on admin writes matches the Origin header.
      '/api': {
        target: 'http://127.0.0.1:8091',
        changeOrigin: false
      },
      '/v1': {
        target: 'http://127.0.0.1:8091',
        changeOrigin: true
      },
      '/docs': {
        target: 'http://127.0.0.1:8091',
        changeOrigin: true
      },
      '/openapi.json': {
        target: 'http://127.0.0.1:8091',
        changeOrigin: true
      },
      '/redoc': {
        target: 'http://127.0.0.1:8091',
        changeOrigin: true
      }
    }
  }
});
