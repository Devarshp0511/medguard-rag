"""
Hybrid retrieval: combines exact structured lookups (DDInter interactions)
with drug-filtered semantic search (Qdrant) -- this is the piece that fixes
the gap we found by hand-testing search_repl.py: raw vector search alone
has no idea *which drug* a question is about, so it searches everything
and returns whatever text happens to score highest across the whole corpus.

Pipeline:
  1. extract_drug_names(): scan the question for any drug name present in
     our database. Deterministic substring matching, not an LLM call --
     good enough for v1, and keeps this module testable without an API key.
     Longest-name-first matching avoids "Ibuprofen" partial-matching inside
     "Ibuprofen (topical)" and picking the wrong one.
  2. structured_lookup(): if exactly two drugs were mentioned, check DDInter
     for a direct interaction record (severity, if documented).
  3. semantic_search(): run the vector search, but *filtered* to only the
     chunks belonging to the drug(s) identified in step 1 -- this is the
     actual fix. Qdrant supports payload filtering alongside vector
     similarity, so we're not searching the whole 1,535-chunk corpus,
     only the handful of chunks that belong to the relevant drug(s).
  4. retrieve(): orchestrates all three and returns a single structured
     result the generation layer (week 2) will consume.

Run as a quick manual test: python -m app.retrieval.hybrid_search
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.embeddings import embed_text
from app.models.db_models import Drug, Interaction


@dataclass
class RetrievalResult:
    matched_drugs: list[str]
    structured_interaction: dict | None
    retrieved_chunks: list[dict] = field(default_factory=list)


def extract_drug_names(query: str, session: Session) -> list[Drug]:
    """
    Finds every drug from our `drugs` table that's mentioned by name in the
    query. Case-insensitive, longest-name-first so a more specific match
    (e.g. "Ketorolac (ophthalmic)") wins over a shorter substring match
    (e.g. "Ketorolac") when both would otherwise match.
    """
    query_lower = query.lower()
    all_drugs = session.query(Drug).all()
    # Longest name first -- see docstring above for why this ordering matters.
    all_drugs.sort(key=lambda d: len(d.name), reverse=True)

    matched: list[Drug] = []
    remaining = query_lower
    for drug in all_drugs:
        name_lower = drug.name.lower()
        if name_lower in remaining:
            matched.append(drug)
            # Remove the matched span so a shorter drug name contained
            # within it can't also match separately (e.g. prevents
            # "Ketorolac" matching again after "Ketorolac (ophthalmic)" did).
            remaining = remaining.replace(name_lower, "")

    return matched


def structured_lookup(drug_a: Drug, drug_b: Drug, session: Session) -> dict | None:
    """Exact structured DDInter lookup for a specific drug pair, if one exists."""
    lo, hi = sorted([drug_a, drug_b], key=lambda d: d.id)
    interaction = (
        session.query(Interaction).filter_by(drug_a_id=lo.id, drug_b_id=hi.id).one_or_none()
    )
    if interaction is None:
        return None

    return {
        "drug_a": lo.name,
        "drug_b": hi.name,
        "severity": interaction.severity.value,
        "mechanism": interaction.mechanism,  # usually None -- see load_ddinter.py notes
        "management": interaction.management,
        "source": interaction.source,
    }


def semantic_search(
    query: str,
    client: QdrantClient,
    drug_ids: list[int] | None,
    top_k: int = 5,
) -> list[dict]:
    """
    Vector search, optionally filtered to only chunks belonging to specific
    drug_ids. Without a filter, falls back to searching the whole corpus --
    which is the old, less precise behavior, kept as a deliberate fallback
    for cases where no known drug was mentioned in the query at all.
    """
    query_vector = embed_text(query)

    query_filter = None
    if drug_ids:
        query_filter = Filter(
            should=[FieldCondition(key="drug_id", match=MatchValue(value=did)) for did in drug_ids]
        )

    results = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    ).points

    return [
        {
            "score": r.score,
            "drug_name": r.payload["drug_name"],
            "section": r.payload["section"],
            "text": r.payload["chunk_text"],
        }
        for r in results
    ]


def retrieve(
    query: str,
    session: Session,
    client: QdrantClient,
    top_k: int = 5,
) -> RetrievalResult:
    """Main entry point: orchestrates drug extraction, structured lookup, and filtered semantic search."""
    matched_drugs = extract_drug_names(query, session)

    structured = None
    if len(matched_drugs) == 2:
        structured = structured_lookup(matched_drugs[0], matched_drugs[1], session)

    drug_ids = [d.id for d in matched_drugs] if matched_drugs else None
    chunks = semantic_search(query, client, drug_ids=drug_ids, top_k=top_k)

    return RetrievalResult(
        matched_drugs=[d.name for d in matched_drugs],
        structured_interaction=structured,
        retrieved_chunks=chunks,
    )


def _manual_test() -> None:
    """Quick sanity check with a couple of real queries."""
    engine = create_engine(settings.database_url)
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, https=False)

    test_queries = [
        "Is it safe to take Warfarin while pregnant?",
        "Does Warfarin interact with Ibuprofen?",
        "What are the kidney warnings for Indomethacin?",
    ]

    with Session(engine) as session:
        for q in test_queries:
            print(f"\n{'=' * 70}\nQuery: {q}")
            result = retrieve(q, session, client)
            print(f"Matched drugs: {result.matched_drugs}")
            if result.structured_interaction:
                print(f"Structured interaction: {result.structured_interaction}")
            print(f"Retrieved {len(result.retrieved_chunks)} chunks:")
            for c in result.retrieved_chunks:
                print(f"  [{c['score']:.3f}] {c['drug_name']} | {c['section']}")
                print(f"    {c['text'][:150]}...")


if __name__ == "__main__":
    _manual_test()
