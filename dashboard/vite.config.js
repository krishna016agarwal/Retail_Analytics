import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const targetApiUrl       = process.env.VITE_CENTRAL_API_URL       || 'http://127.0.0.1:8000';
const inventoryApiUrl    = process.env.VITE_INVENTORY_API_URL      || 'http://127.0.0.1:8001';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    proxy: {
      // Crowd / Queue analytics (existing backend, port 8000)
      '/api': {
        target: targetApiUrl,
        changeOrigin: true,
        secure: false,
      },
      '/health': {
        target: targetApiUrl,
        changeOrigin: true,
        secure: false,
      },
      // Inventory analytics (new backend, port 8001)
      '/inventory': {
        target: inventoryApiUrl,
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
