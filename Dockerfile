# syntax=docker/dockerfile:1.7
#
# G3 OmniVoice TTS Server — GPU text-to-speech / voice-cloning service.
# Target host: Linux + NVIDIA driver + nvidia-container-toolkit (RTX 5090 / sm_120 OK).
# The image ships NO models: k2-fsa/OmniVoice (~3 GB) is downloaded from Hugging Face
# into the mounted /app/models volume on first boot (OMNIVOICE_TTS_ALLOW_MODEL_DOWNLOADS=true).

############################  1) Frontend build  ############################
FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend

# Install deps from the lockfile first (better layer caching). Vite 7 needs Node >=20.19.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Build the multi-page dashboard (index/admin/demo) -> /build/frontend/dist
COPY frontend/ ./
RUN npm run build


############################  2) Python runtime  ############################
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models/.hf \
    # Server config (env_prefix OMNIVOICE_TTS_). Paths match the editable-install
    # PROJECT_ROOT (=/app), so the frontend/dist default resolves correctly too.
    OMNIVOICE_TTS_HOST=0.0.0.0 \
    OMNIVOICE_TTS_PORT=8091 \
    OMNIVOICE_TTS_RUNTIME_BACKEND=omnivoice \
    OMNIVOICE_TTS_MODELS_ROOT_DIR=/app/models \
    OMNIVOICE_TTS_DATA_DIR=/app/data \
    OMNIVOICE_TTS_ALLOW_MODEL_DOWNLOADS=true \
    # Trim the CUDA caching-allocator reserved pool (lower idle VRAM; shares the GPU
    # with the Whisper container). expandable_segments is fine on Linux.
    PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256

# System libraries:
#   ffmpeg      -> audio container decode (imageio-ffmpeg also bundles a binary)
#   libsndfile1 -> soundfile
#   curl        -> container HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CUDA 13.0 wheels — the exact known-good stack from requirements-lock.txt.
# (Linux Triton for torch.compile is pulled in automatically as a torch dependency;
#  compile_model defaults to off so it is not required at runtime.)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

# Backend package. Editable install keeps config.py's PROJECT_ROOT resolving to /app
# (parents[3] of backend/src/omnivoice_tts_server/config.py) — matching start_server.bat —
# so DEFAULT_FRONTEND_DIST = /app/frontend/dist works out of the box.
#
# This pulls omnivoice==0.1.5, kernels, transformers, etc. Without pins those float to the
# newest PyPI release, which does NOT reproduce the machine-verified stack (the fp8/kernels/
# transformers combo is version-sensitive). So constrain the whole dependency set with
# requirements-lock.txt via -c (NOT -r: the lock is a Windows conda snapshot pinning
# triton-windows and +cu130 torch, which -r would try to install and fail on). As a
# constraints file it only caps versions of packages actually pulled — forcing
# transformers==5.9.0, huggingface_hub==1.16.1, accelerate==1.13.0, tokenizers==0.22.2,
# kernels==0.14.1, numpy==2.4.4, ... — while the Windows-only lines stay inert and the
# already-installed +cu130 torch trio (above) satisfies its pins untouched.
COPY backend/ ./backend/
COPY requirements-lock.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ./backend \
        -c requirements-lock.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130

# Built frontend + helper tools.
COPY --from=frontend /build/frontend/dist ./frontend/dist
COPY tools/ ./tools/

# Run as non-root; pre-own the volume mountpoints so named volumes inherit ownership.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/models /app/data /app/frontend \
    && chown -R app:app /app
USER app

EXPOSE 8091
VOLUME ["/app/models", "/app/data"]

# First boot downloads ~3 GB + warms up the model, so allow a long startup grace period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=5 \
    CMD curl -fsS http://localhost:8091/openapi.json >/dev/null || exit 1

CMD ["python", "-u", "-m", "omnivoice_tts_server.main"]
