# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        tar \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Download llama.cpp binary ─────────────────────────────────────────────────
ARG LLAMA_VERSION=b8457
ENV LD_LIBRARY_PATH=/app/llama-bin/llama-b8457:$LD_LIBRARY_PATH

RUN mkdir -p /app/llama-bin && \
    curl -fsSL \
    "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_VERSION}/llama-${LLAMA_VERSION}-bin-ubuntu-x64.tar.gz" \
    -o /tmp/llama.tar.gz && \
    tar -xzf /tmp/llama.tar.gz -C /app/llama-bin && \
    find /app/llama-bin -name "llama-server" -exec chmod +x {} \; && \
    rm /tmp/llama.tar.gz && \
    # Symlink all backends into standard lib path
    find /app/llama-bin -name "*.so" | xargs -I{} ln -sf {} /usr/local/lib/ && \
    ldconfig

ENV LLAMA_SERVER_BIN=/app/llama-bin/llama-b8457/llama-server
ENV GGML_BACKEND_PATH=/app/llama-bin/llama-b8457

# Print where llama-server landed so we can confirm the path
RUN find /app/llama-bin -name "llama-server"

# ── App code ──────────────────────────────────────────────────────────────────
COPY app_llamacpp.py .

# ── Directories ───────────────────────────────────────────────────────────────
ENV HF_HOME=/app/model_cache
VOLUME ["/app/model_cache"]

# ── Default config (all overridable with -e at runtime) ──────────────────────
ENV MODEL_PATH=/app/model_cache/model.gguf
ENV N_GPU_LAYERS=0
ENV N_CTX=8192
ENV N_THREADS=16
ENV LLAMA_PORT=8080
ENV API_PORT=8000
ENV REPO_ID=mradermacher/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-i1-GGUF
ENV GGUF_FILE=Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated.i1-Q4_K_M.gguf

# ── Expose API port ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "app_llamacpp:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]