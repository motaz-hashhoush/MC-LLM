# syntax=docker/dockerfile:1
# ── MC-LLM Inference Gateway (GPU Optimized) ────────────────────────────────
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base

# Install Python 3.12 and essential OS deps
# Using deadsnakes PPA since Ubuntu 22.04 doesn't have 3.12 in main repo
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    && add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommending \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    curl \
    gcc \
    libc6-dev \
    libpq-dev \
    libsndfile1 \
    ffmpeg && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    rm -rf /var/lib/apt/lists/*

# Fix python symlinks to ensure 'python' and 'python3' point to 3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"

WORKDIR /opt/mc-llm

# ── Dependencies ─────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY app/ app/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
