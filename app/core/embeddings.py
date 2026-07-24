"""
Shared embedding utility, backed by fastembed instead of sentence-transformers.

Why the switch: sentence-transformers pulls in PyTorch, which alone uses
400-600MB+ of memory just importing + loading a model -- comfortably
exceeding Render's free-tier 512MB cap, which is what actually crashed our
first deployment (see Render deploy logs: "Ran out of memory (used over
512MB)"). fastembed is maintained by Qdrant specifically for this kind of
constrained, inference-only use case: it uses ONNX Runtime instead of
PyTorch, with a dramatically smaller memory footprint, no GPU dependencies,
and no CUDA downloads to worry about (the problem we already hit once
during local Docker builds).

Correctness note: fastembed's BAAI/bge-small-en-v1.5 is a quantized ONNX
port of the same model we originally used via sentence-transformers. Mixing
precisions between query-time and document embeddings can introduce small
numerical drift, so rather than accept that risk, the whole corpus was
re-embedded with fastembed (see re-run instructions in engineering-notes.md)
-- document and query vectors are now produced by the exact same backend,
guaranteeing consistency.

This module is the ONLY place that imports fastembed -- every other module
that needs embeddings (ingestion, retrieval, generation, the API) imports
from here, so there is exactly one place to change if the embedding backend
or model ever needs to change again.
"""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # unchanged from the original sentence-transformers model

_model: TextEmbedding | None = None


def get_embedding_model() -> TextEmbedding:
    """Loads the model once and reuses it -- fastembed's own internal
    caching handles the ONNX session, but we still avoid re-constructing
    the TextEmbedding wrapper object on every call."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds a batch of texts, L2-normalized to unit length -- matching the
    normalize_embeddings=True behavior our original sentence-transformers
    code relied on for correct cosine-similarity search in Qdrant.
    """
    model = get_embedding_model()
    vectors = list(model.embed(texts))
    normalized = []
    for v in vectors:
        norm = np.linalg.norm(v)
        normalized.append((v / norm).tolist() if norm > 0 else v.tolist())
    return normalized


def embed_text(text: str) -> list[float]:
    """Convenience wrapper for embedding a single string (e.g. one query)."""
    return embed_texts([text])[0]
