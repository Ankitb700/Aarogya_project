FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV runtime: libGL.so.1, libglib-2.0.so.0
    libgl1-mesa-glx \
    libglib2.0-0 \
    # OpenCV GUI (loaded by some cv2 internals even with headless)
    libxcb1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # ONNX Runtime backend
    libgomp1 \
    # NumPy / SciPy build time (wheel fallback)
    g++ \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────
WORKDIR /app

# Copy only requirements first for Docker layer caching
COPY backend/requirements.txt /app/backend/requirements.txt

WORKDIR /app/backend
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────
WORKDIR /app
COPY . .

# ── Runtime ──────────────────────────────────────────────────────────
WORKDIR /app/backend
ENV KERAS_BACKEND=jax
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV OPENCV_OPENCL_RUNTIME=""

EXPOSE 8080

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
