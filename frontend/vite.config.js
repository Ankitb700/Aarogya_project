import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Backend origin the dev proxy forwards to (same machine by default).
const BACKEND = process.env.BACKEND_ORIGIN || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // host: true exposes the dev server on the LAN so phones/tablets can test it.
    host: true,
    // Allow any host header (needed for ngrok / LAN IP / tunnels).
    allowedHosts: true,
    // Single-origin proxy: the page and the WebSocket share one origin, so it
    // works over HTTPS tunnels (wss) with no mixed-content/CORS problems and
    // only one tunnel (to :5173) is needed.
    proxy: {
      '/ws': { target: BACKEND, ws: true, changeOrigin: true },
      '/api': { target: BACKEND, changeOrigin: true },
      '/health': { target: BACKEND, changeOrigin: true },
    },
  },
})
