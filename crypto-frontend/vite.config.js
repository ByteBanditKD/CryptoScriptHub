import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 12003,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://localhost:5005',
        changeOrigin: true,
      }
    }
  }
})
