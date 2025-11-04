FROM docker.io/pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt uvicorn[standard]

COPY . /app

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
ENV HOST=0.0.0.0 PORT=8000 DEBUG=0 HUGGINGFACE_ACCESS_TOKEN=
ENV HF_HOME=/app/.cache/huggingface TRANSFORMERS_CACHE=/app/.cache/huggingface
VOLUME ["/app/.cache/huggingface"]

CMD ["sh", "-c", "uvicorn main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
