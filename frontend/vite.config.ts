import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    // Dev-only: Vite's import analysis appends `?import` to onnxruntime's
    // dynamic import of its /vad/ .mjs loader. public/ serving tolerates the
    // query, but strip it anyway so the asset can never fall through to the
    // SPA-fallback HTML. Prod builds keep the import native — no query.
    // (The /vad/ assets themselves are copied into public/ on postinstall —
    // see scripts/copy-vad-assets.mjs.)
    {
      name: 'vad-assets-ignore-query',
      configureServer(server) {
        server.middlewares.use((req, _res, next) => {
          if (req.url?.startsWith('/vad/')) req.url = req.url.replace(/\?.*$/, '')
          next()
        })
      },
    },
  ],
  // In dev the app is served by Vite (5173) but the backend lives on 8000;
  // proxy the API and the WebSocket state channel through so the browser
  // talks to same-origin relative URLs.
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
})
