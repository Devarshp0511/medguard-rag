"""
Embeds every LabelChunk's text into a vector using fastembed (a lightweight,
ONNX-based embedding runtime maintained by Qdrant), and loads those vectors
into Qdrant.

Why fastembed instead of sentence-transformers/PyTorch: PyTorch alone uses
400-600MB+ of memory just importing + loading a model, which crashed our
first Render deployment (free tier caps at 512MB). fastembed uses ONNX
Runtime instead, with a dramatically smaller memory footprint -- see
app/core/embeddings.py for the full reasoning and the shared embedding
functions this script now uses.

Pipeline position: this is the bridge between our two storage layers --
LabelChunk rows in Postgres/SQLite already have a stable vector_id (a UUID
assigned at chunk-creation time in chunk_labels.py). This script:
  1. Reads chunk text + vector_id from the SQL database
  2. Embeds the text into a vector
  3. Upserts {id: vector_id, vector: [...], payload: {metadata}} into Qdrant

Idempotent: Qdrant upserts are keyed by point ID (our vector_id), so
re-running this script simply overwrites existing points rather than
duplicating them -- safe to re-run after adding more chunks, or after
switching embedding backends (as we did here).

Run as: python -m app.ingestion.embed_chunks
"""

from __future__ import annotations

import argparse
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.embeddings import EMBEDDING_DIM, embed_texts
from app.models.db_models import LabelChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64  # chunks per embedding batch -- balances speed vs memory


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        https=False,  # local Docker Qdrant serves plain HTTP, not TLS --
        # without this, the client can default to HTTPS and fail with
        # "SSL: WRONG_VERSION_NUMBER" against a non-TLS local instance.
        # Set to True (or drop this line) when pointed at Qdrant Cloud.
    )


def ensure_collection(client: QdrantClient) -> None:
    """
    Creates the Qdrant collection if it doesn't already exist. Safe to call
    every run -- checks existence first rather than blindly recreating,
    which would wipe previously embedded data.
    """
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection_name in existing:
        logger.info("Collection %r already exists", settings.qdrant_collection_name)
        return

    client.create_collection(
        collection_name=settings.qdrant_collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    logger.info("Created collection %r (dim=%d, cosine distance)", settings.qdrant_collection_name, EMBEDDING_DIM)


def embed_and_upsert(session: Session, client: QdrantClient, batch_size: int = EMBED_BATCH_SIZE) -> dict[str, int]:
    """
    Streams through every LabelChunk in batches: embed the batch's text,
    upsert the vectors + metadata into Qdrant. Batching (rather than
    one-chunk-at-a-time) is significantly faster for both the embedding
    model and the Qdrant client.
    """
    chunks = session.query(LabelChunk).all()
    total = len(chunks)
    logger.info("Embedding %d chunks in batches of %d", total, batch_size)

    embedded_count = 0

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.chunk_text for c in batch]

        # embed_texts() L2-normalizes internally, matching what cosine
        # similarity search in Qdrant expects.
        vectors = embed_texts(texts)

        points = [
            PointStruct(
                id=chunk.vector_id,
                vector=vector,
                payload={
                    "drug_id": chunk.drug_id,
                    "drug_name": chunk.drug.name,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "chunk_text": chunk.chunk_text,
                },
            )
            for chunk, vector in zip(batch, vectors)
        ]

        client.upsert(collection_name=settings.qdrant_collection_name, points=points)
        embedded_count += len(points)

        if (i // batch_size) % 5 == 0:
            logger.info("Progress: %d / %d chunks embedded", embedded_count, total)

    return {"total_chunks": total, "embedded": embedded_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed label chunks and load into Qdrant")
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)
    args = parser.parse_args()

    engine = create_engine(settings.database_url)
    client = get_qdrant_client()
    ensure_collection(client)

    with Session(engine) as session:
        summary = embed_and_upsert(session, client, batch_size=args.batch_size)

    logger.info("Embedding complete: %s", summary)

    # Quick sanity check: confirm Qdrant actually has points now.
    info = client.get_collection(settings.qdrant_collection_name)
    logger.info("Qdrant collection now has %d points", info.points_count)


if __name__ == "__main__":
    main()
