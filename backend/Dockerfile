FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────
COPY . .

# ── Runtime ──────────────────────────────────────────────────────────
ENV KERAS_BACKEND=jax
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV OPENCV_OPENCL_RUNTIME=""

EXPOSE 8080

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
