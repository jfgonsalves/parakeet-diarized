FROM docker.io/pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    build-essential \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install dependencies with uv
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0 torchvision torchaudio \
    && uv pip install --system --no-cache -r requirements.txt

RUN uv pip install --system --no-cache git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5


# Install specific transformers commit for MedASR compatibility
COPY . /app

RUN useradd -m -u 1000 appuser && chown -R appuser /app

USER appuser

EXPOSE 8000

# Environment variables
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DEBUG=0 \
    ASR_BACKEND=parakeet \
    MODEL_ID= \
    HUGGINGFACE_ACCESS_TOKEN=

ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

VOLUME ["/app/.cache/huggingface"]

CMD ["sh", "-c", "uvicorn main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
