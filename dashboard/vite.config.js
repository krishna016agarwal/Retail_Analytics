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
        configure: (proxy) => {
          proxy.on('error', (err, req, res) => {
            if (res && !res.headersSent && typeof res.writeHead === 'function') {
              res.writeHead(502, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ status: 'offline', message: 'Edge service offline' }));
            }
          });
        },
      },
      '/health': {
        target: targetApiUrl,
        changeOrigin: true,
        secure: false,
        configure: (proxy) => {
          proxy.on('error', (err, req, res) => {
            if (res && !res.headersSent && typeof res.writeHead === 'function') {
              res.writeHead(502, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ status: 'offline' }));
            }
          });
        },
      },
      // Inventory analytics (new backend, port 8001)
      '/inventory/': {
        target: inventoryApiUrl,
        changeOrigin: true,
        secure: false,
        configure: (proxy) => {
          proxy.on('error', (err, req, res) => {
            if (res && !res.headersSent && typeof res.writeHead === 'function') {
              res.writeHead(502, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ status: 'offline', message: 'Inventory backend offline' }));
            }
          });
        },
      },
    },
  },
});
