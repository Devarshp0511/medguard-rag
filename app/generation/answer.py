"""
Generation layer: turns retrieval results into cited, grounded natural-language
answers using the Claude API.

Design principles enforced through the system prompt (not left to model
whim), each with a specific reason:

  1. Grounded only in retrieved evidence -- if a claim isn't supported by
     the passages we hand it, the model must not make it. This is what
     stops it from filling gaps with training-memory guesses, which is the
     specific failure mode we built RAG to prevent.

  2. Mandatory citations -- every factual claim must reference a chunk by
     drug name + section. If it can't cite, it can't claim. This is what
     makes the answer verifiable by the reader.

  3. Refuses gracefully when evidence is thin -- if retrieval returned
     nothing relevant, the correct answer is an explicit "insufficient
     information," not a confident guess. Requires the prompt to name this
     option explicitly, or the model tends to try harder rather than admit.

  4. Never a definitive verdict -- framing is "here is what the sources
     say" not "here is what you should do." This is decision support, not
     prescription; legal and ethical framing that matches how real clinical
     tools describe themselves.

Structured interaction data (DDInter severity, when a pair is matched) is
provided separately from the unstructured chunk evidence, and the prompt
tells Claude to lead with it -- because a Major severity flag from a
curated database is more reliable than a paragraph of prose text.

Run as: python -m app.generation.answer
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.retrieval.hybrid_search import RetrievalResult, retrieve

SYSTEM_PROMPT = """You are a clinical decision-support assistant for licensed
healthcare professionals. You answer drug interaction and dosage questions
using ONLY the evidence provided in each query. You do not use general
medical knowledge from your training to fill gaps.

Rules you must follow, without exception:

1. GROUND EVERY CLAIM IN THE PROVIDED EVIDENCE. If the evidence doesn't
   support a claim, do not make it. If the user asks something the evidence
   doesn't cover, say so plainly: "The provided sources do not address X."

2. CITE EVERY FACTUAL CLAIM. Use inline citations in the format
   [Drug Name, section_name] immediately after each claim. Multiple sources
   for one claim get multiple citations.

3. LEAD WITH STRUCTURED SEVERITY IF PROVIDED. When a structured interaction
   record is present (severity: Major/Moderate/Minor), state that severity
   first, before discussing the label evidence. A curated severity rating
   is more reliable than paragraph text.

4. WHEN EVIDENCE IS THIN OR ABSENT, REFUSE CLEANLY. Do not try to be
   helpful by guessing. Say: "The provided sources do not contain enough
   information to answer this reliably." Then list what sources WOULD be
   needed.

5. NEVER PRESCRIBE. Frame everything as "the sources indicate..." or
   "documented in the label..." -- never "you should" or "it is safe to."
   This is decision support, not a prescription. Add a brief closing note
   that clinical judgment and current authoritative sources should always
   be verified.

6. BE CONCISE. Real clinicians read fast. Aim for 3-6 short paragraphs.
"""


@dataclass
class GeneratedAnswer:
    answer_text: str
    citations_used: list[str]  # e.g. ["Warfarin/drug_interactions", ...]
    evidence_sufficient: bool  # crude heuristic based on retrieval size + score
    retrieval: RetrievalResult


def format_evidence_for_prompt(retrieval: RetrievalResult) -> str:
    """Formats the retrieval result into a compact evidence block for the
    system to reason over. Structured interaction record first (if any),
    then numbered label-text chunks."""
    parts = []

    if retrieval.structured_interaction:
        s = retrieval.structured_interaction
        parts.append(
            f"STRUCTURED INTERACTION RECORD (from DDInter):\n"
            f"  Drug A: {s['drug_a']}\n"
            f"  Drug B: {s['drug_b']}\n"
            f"  Severity: {s['severity']}\n"
            f"  Source: {s['source']}"
        )

    if retrieval.retrieved_chunks:
        parts.append("LABEL TEXT EVIDENCE (from openFDA):")
        for i, c in enumerate(retrieval.retrieved_chunks, start=1):
            parts.append(
                f"  [{i}] {c['drug_name']} | section: {c['section']} | score: {c['score']:.3f}\n"
                f"      {c['text']}"
            )
    else:
        parts.append("LABEL TEXT EVIDENCE: (none retrieved)")

    return "\n\n".join(parts)


def evidence_is_sufficient(retrieval: RetrievalResult) -> bool:
    """
    Rough heuristic for whether we retrieved enough usable evidence to
    warrant attempting a real answer vs. refusing outright. Deliberately
    generous -- the LLM's own refusal logic (rule 4 in SYSTEM_PROMPT) is
    the primary safeguard; this is just a cheap pre-check.
    """
    if not retrieval.retrieved_chunks and not retrieval.structured_interaction:
        return False
    # If every chunk scored below ~0.4, similarity is weak enough that the
    # retrieval is probably off-topic. BGE-normalized cosine scores in
    # this project's data cluster >0.5 for genuine relevance.
    if retrieval.retrieved_chunks:
        top_score = max(c["score"] for c in retrieval.retrieved_chunks)
        if top_score < 0.4:
            return False
    return True


def answer_query(
    query: str,
    session: Session,
    qdrant_client: QdrantClient,
    anthropic_client: anthropic.Anthropic,
    top_k: int = 5,
) -> GeneratedAnswer:
    retrieval = retrieve(query, session, qdrant_client, top_k=top_k)
    sufficient = evidence_is_sufficient(retrieval)

    user_content = (
        f"Clinician question:\n{query}\n\n"
        f"Evidence retrieved:\n{format_evidence_for_prompt(retrieval)}"
    )

    response = anthropic_client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    # Concatenate any text blocks in the response (usually just one).
    answer_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )

    citations_used = [
        f"{c['drug_name']}/{c['section']}" for c in retrieval.retrieved_chunks
    ]

    return GeneratedAnswer(
        answer_text=answer_text,
        citations_used=citations_used,
        evidence_sufficient=sufficient,
        retrieval=retrieval,
    )


def _manual_test() -> None:
    """A quick end-to-end sanity check with three real questions -- covers
    a drug pair with structured data, a single-drug question, and a
    deliberately underspecified question that should refuse."""
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set in .env -- see .env.example for the required keys."
        )

    engine = create_engine(settings.database_url)
    qdrant_client = QdrantClient(
        host=settings.qdrant_host, port=settings.qdrant_port, https=False
    )
    anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    test_queries = [
        "Does Warfarin interact with Ibuprofen?",
        "What are the kidney warnings for Indomethacin?",
        "Is it safe to take a drug I did not name?",
    ]

    with Session(engine) as session:
        for q in test_queries:
            print(f"\n{'=' * 70}\nQ: {q}\n")
            result = answer_query(q, session, qdrant_client, anthropic_client)
            print(f"Matched drugs: {result.retrieval.matched_drugs}")
            print(f"Evidence sufficient (pre-check): {result.evidence_sufficient}")
            print(f"\n--- ANSWER ---\n{result.answer_text}\n")


if __name__ == "__main__":
    _manual_test()
