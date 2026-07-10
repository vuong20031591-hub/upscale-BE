# Upscale-BE Docker image
# Real-ESRGAN + CodeFormer inference API (FastAPI + Uvicorn)
# Base: CUDA 12.4.1 runtime (matches torch==2.4.1+cu124 in requirements.txt)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Singapore

# System deps: python3.10, ffmpeg (opencv codecs), git (some deps pull from git)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential git curl ca-certificates \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CUDA 12.4 wheels first (cache layer)
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu124

# App requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy app source
COPY app ./app
COPY codeformer_minimal ./codeformer_minimal
COPY facelib ./facelib
COPY run.py ./

# Runtime dirs
RUN mkdir -p weights tmp logs output

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
