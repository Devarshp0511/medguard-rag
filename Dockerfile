# MedGuard RAG API — backend service
#
# Notes on choices:
# - python:3.11-slim keeps the image reasonably small while matching the
#   Python version this project was developed against.
# - The BGE embedding model (~130MB) downloads on first run inside the
#   container unless baked in; we accept the one-time startup delay rather
#   than complicating the build with a pre-download step, since this is a
#   portfolio project, not a latency-critical production service.
# - Only requirements.txt is copied before the full source, so Docker's
#   layer cache can skip the (slow) pip install step on rebuilds that only
#   change application code, not dependencies.

FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers / torch wheels to build cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only PyTorch explicitly before the rest of requirements.txt.
# Without this, pip's default resolution for sentence-transformers' torch
# dependency pulls the full CUDA-enabled build on Linux -- multiple GBs of
# NVIDIA libraries that are useless here (this runs on CPU only, and Docker
# Desktop on Mac doesn't pass through GPU access anyway). This single line
# cuts build time and image size dramatically.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
