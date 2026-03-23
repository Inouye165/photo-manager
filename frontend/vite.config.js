import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      '/api': 'http://127.0.0.1:5000',
      '/image': 'http://127.0.0.1:5000',
      '/working_dir': 'http://127.0.0.1:5000',
      '/mirror': 'http://127.0.0.1:5000',
    },
  },
})
