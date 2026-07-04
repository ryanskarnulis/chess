import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// Kept separate from vite.config.ts so the production build's `tsc -b` never
// has to reconcile Vitest's bundled Vite types with the app's Vite.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.ts',
  },
})
