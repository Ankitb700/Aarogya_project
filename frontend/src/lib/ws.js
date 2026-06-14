import { WS_ANALYZE } from './config.js'

// Thin wrapper around the analyze WebSocket. Sends JPEG frame blobs (binary),
// parses incoming status/metrics/final JSON messages, and exposes callbacks.
export class AnalyzeSocket {
  constructor({ onMessage, onOpen, onClose, onError } = {}) {
    this.onMessage = onMessage
    this.onOpen = onOpen
    this.onClose = onClose
    this.onError = onError
    this.ws = null
  }

  connect() {
    const ws = new WebSocket(WS_ANALYZE)
    ws.binaryType = 'arraybuffer'
    ws.onopen = () => this.onOpen?.()
    ws.onclose = () => this.onClose?.()
    ws.onerror = (e) => this.onError?.(e)
    ws.onmessage = (ev) => {
      try {
        this.onMessage?.(JSON.parse(ev.data))
      } catch {
        /* ignore non-JSON */
      }
    }
    this.ws = ws
  }

  get isOpen() {
    return this.ws && this.ws.readyState === WebSocket.OPEN
  }

  sendFrame(blob) {
    if (this.isOpen) this.ws.send(blob)
  }

  stop() {
    if (this.isOpen) this.ws.send(JSON.stringify({ action: 'stop' }))
  }

  close() {
    try { this.ws?.close() } catch { /* noop */ }
    this.ws = null
  }
}
