"""
Embeds every LabelChunk's text into a vector using a local, open-source
embedding model (BAAI/bge-small-en-v1.5 via sentence-transformers), and
loads those vectors into Qdrant.

Why this model: purpose-built for retrieval (unlike older baseline models
like all-MiniLM), runs free on CPU with no API key, and produces small
384-dimensional vectors -- fast enough to embed our full chunk set on a
laptop in minutes rather than requiring GPU infrastructure or per-token
API billing. See project docs for the fuller quality-vs-cost tradeoff
discussion (cloud options like Voyage AI score higher on raw benchmarks).

Pipeline position: this is the bridge between our two storage layers --
LabelChunk rows in Postgres/SQLite already have a stable vector_id (a UUID
assigned at chunk-creation time in chunk_labels.py). This script:
  1. Reads chunk text + vector_id from the SQL database
  2. Embeds the text into a vector
  3. Upserts {id: vector_id, vector: [...], payload: {metadata}} into Qdrant

Idempotent: Qdrant upserts are keyed by point ID (our vector_id), so
re-running this script simply overwrites existing points rather than
duplicating them -- safe to re-run after adding more chunks.

Run as: python -m app.ingestion.embed_chunks
"""

from __future__ import annotations

import argparse
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.db_models import LabelChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # fixed by the model above; must match Qdrant collection config
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
    Loads the embedding model once, then streams through every LabelChunk
    in batches: embed the batch's text, upsert the vectors + metadata into
    Qdrant. Batching (rather than one-chunk-at-a-time) is significantly
    faster for both the embedding model and the Qdrant client.
    """
    logger.info("Loading embedding model %s (first run downloads ~130MB)...", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chunks = session.query(LabelChunk).all()
    total = len(chunks)
    logger.info("Embedding %d chunks in batches of %d", total, batch_size)

    embedded_count = 0

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.chunk_text for c in batch]

        # normalize_embeddings=True makes cosine similarity search behave
        # correctly and consistently -- BGE models expect this.
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        points = [
            PointStruct(
                id=chunk.vector_id,
                vector=vector.tolist(),
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
