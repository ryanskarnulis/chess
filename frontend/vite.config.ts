import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
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
