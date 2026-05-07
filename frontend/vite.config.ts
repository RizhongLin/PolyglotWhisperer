import path from 'node:path';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { TanStackRouterVite } from '@tanstack/router-plugin/vite';
import { defineConfig } from 'vite';

// Vite builds to ../src/pgw/templates/dist so the Python wheel ships the SPA
// as static assets — FastAPI serves index.html for any non-API route.

const BACKEND = process.env['PGW_DEV_BACKEND'] ?? 'http://127.0.0.1:8321';
const PROXY_PATHS = [
  '/api',
  '/jobs',
  '/uploads',
  '/ws',
  '/openapi.json',
  '/icon.png',
  '/logo.png',
];

export default defineConfig({
  plugins: [
    TanStackRouterVite({
      routesDirectory: 'src/routes',
      generatedRouteTree: 'src/routeTree.gen.ts',
      autoCodeSplitting: true,
    }),
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    outDir: path.resolve(__dirname, '../src/pgw/templates/dist'),
    emptyOutDir: true,
    sourcemap: false,
    target: 'es2022',
  },
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      PROXY_PATHS.map((p) => [p, BACKEND]),
    ),
  },
});
