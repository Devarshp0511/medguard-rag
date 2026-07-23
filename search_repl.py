"""
Interactive semantic search tester. Loads the embedding model once, then
lets you type any number of questions and see what Qdrant retrieves --
no file editing needed between queries.

Run as: python3 search_repl.py
Type 'quit' or press Ctrl+C to exit.
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.core.config import settings

print("Loading embedding model (one-time, ~130MB on first run)...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, https=False)

print(f"Connected to Qdrant collection: {settings.qdrant_collection_name}")
print("Type a question and press Enter. Type 'quit' to exit.\n")

while True:
    try:
        query = input("Query> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

    if not query:
        continue
    if query.lower() in ("quit", "exit", "q"):
        break

    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    results = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=query_vector,
        limit=5,
    ).points

    print()
    for r in results:
        print(f"  [{r.score:.3f}] {r.payload['drug_name']} | {r.payload['section']}")
        print(f"    {r.payload['chunk_text'][:180]}...")
    print()
