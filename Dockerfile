# MedGuard RAG API — backend service
#
# Notes on choices:
# - python:3.11-slim keeps the image reasonably small while matching the
#   Python version this project was developed against.
# - Uses requirements-api.txt (NOT the full requirements.txt) -- a minimal
#   dependency set excluding sentence-transformers/torch/ragas/langchain,
#   none of which the deployed API actually needs. This is what fixes the
#   "out of memory" crash on Render's free tier: PyTorch alone used
#   400-600MB+ just importing, before serving a single request. Query-time
#   embedding now runs through fastembed instead (ONNX-based, much lighter
#   -- see app/core/embeddings.py). No CUDA-avoidance workaround needed
#   anymore either, since there's no torch install at all.
# - Only requirements-api.txt is copied before the full source, so Docker's
#   layer cache can skip the pip install step on rebuilds that only change
#   application code, not dependencies.

FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
