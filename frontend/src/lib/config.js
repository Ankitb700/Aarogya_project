// Same-origin endpoints. The Vite dev proxy (vite.config.js) forwards /ws, /api
// and /health to the backend, so the page and WebSocket share one origin. This
// works over HTTPS tunnels (ngrok) as wss:// with no mixed-content or CORS issues.
//
// Override with VITE_WS_ANALYZE / VITE_API_BASE only if you intentionally point
// the frontend at a different backend host.
const loc = typeof window !== 'undefined'
  ? window.location
  : { protocol: 'http:', host: 'localhost:5173' }

const wsProto = loc.protocol === 'https:' ? 'wss' : 'ws'

export const API_BASE = import.meta.env.VITE_API_BASE || '' // same origin
export const WS_ANALYZE = import.meta.env.VITE_WS_ANALYZE || `${wsProto}://${loc.host}/ws/analyze`
