"""
One-off fix: creates the payload index on `drug_id` that Qdrant Cloud
requires before filtering on that field (local self-hosted Qdrant allowed
unindexed filtering via a slower full scan; Cloud enforces an explicit
index). This doesn't touch or re-embed any existing data -- indexes are a
separate structure Qdrant builds over already-stored payloads.

Run as: python create_qdrant_index.py
"""

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

from app.core.config import settings

url = input("Paste your Qdrant Cloud cluster URL: ").strip().rstrip("/")
client = QdrantClient(url=url, api_key=settings.qdrant_api_key)

client.create_payload_index(
    collection_name=settings.qdrant_collection_name,
    field_name="drug_id",
    field_schema=PayloadSchemaType.INTEGER,
)

print(f"Payload index created on 'drug_id' for collection {settings.qdrant_collection_name!r}.")
