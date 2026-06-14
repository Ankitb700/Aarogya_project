// Webcam capture: stream -> <video> -> offscreen <canvas> -> JPEG blob, on an
// interval. Calls onFrame(blob) for each grabbed frame.
export class FrameCapturer {
  constructor({ fps = 18, quality = 0.6, width = 480 } = {}) {
    this.fps = fps
    this.quality = quality
    this.width = width
    this.stream = null
    this.video = null
    this.canvas = null
    this.timer = null
  }

  async start(videoEl, onFrame) {
    this.video = videoEl
    // On insecure origins (e.g. http://<LAN-IP> on a phone) the camera API is
    // not exposed at all. Surface a clear, specific error.
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const err = new Error('Camera unavailable: this page must be served over HTTPS (or localhost).')
      err.code = 'INSECURE_CONTEXT'
      throw err
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    })
    videoEl.srcObject = this.stream
    await videoEl.play()

    this.canvas = document.createElement('canvas')
    const interval = 1000 / this.fps
    this.timer = setInterval(() => this._grab(onFrame), interval)
  }

  _grab(onFrame) {
    const v = this.video
    if (!v || v.videoWidth === 0) return
    const scale = this.width / v.videoWidth
    const w = this.width
    const h = Math.round(v.videoHeight * scale)
    this.canvas.width = w
    this.canvas.height = h
    const ctx = this.canvas.getContext('2d')
    ctx.drawImage(v, 0, 0, w, h)
    this.canvas.toBlob(
      (blob) => { if (blob) onFrame(blob) },
      'image/jpeg',
      this.quality,
    )
  }

  stop() {
    if (this.timer) clearInterval(this.timer)
    this.timer = null
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop())
    this.stream = null
    if (this.video) this.video.srcObject = null
  }
}
