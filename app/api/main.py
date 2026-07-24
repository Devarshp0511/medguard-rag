"""
FastAPI backend for MedGuard RAG.

Exposes three endpoints:
  POST /query         -- full RAG pipeline: question → retrieval → generation → cited answer
  POST /search        -- retrieval only: question → matched drugs + retrieved chunks (no LLM)
  GET  /interactions   -- direct structured lookup: drug_a + drug_b → severity

Why three separate endpoints instead of just /query:
  - /search is useful for debugging retrieval without spending API credits
  - /interactions is a fast, free, structured-only lookup
  - /query is the full pipeline (costs Claude API credits per call)

Startup: loads the embedding model and connects to Qdrant once at app start,
not per-request. The model is ~130MB in memory -- acceptable for a single-
instance service, worth noting if you ever scale horizontally.

Run as: uvicorn app.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.generation.answer import EMBEDDING_MODEL_NAME, answer_query
from app.models.db_models import Drug, Interaction
from app.retrieval.hybrid_search import (
    extract_drug_names,
    retrieve,
    semantic_search,
    structured_lookup,
)

# --- Shared resources (loaded once at startup) ---

_resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup, clean up on shutdown."""
    _resources["engine"] = create_engine(settings.database_url)
    _resources["embed_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # Cloud deployment sets QDRANT_URL (full HTTPS cluster URL) + QDRANT_API_KEY.
    # Local dev leaves QDRANT_URL unset and connects to Docker/localhost instead.
    if settings.qdrant_url:
        _resources["qdrant"] = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
    else:
        _resources["qdrant"] = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port, https=False
        )
    _resources["anthropic"] = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    yield
    _resources.clear()


app = FastAPI(
    title="MedGuard RAG",
    description=(
        "Drug interaction & dosage clinical decision support, powered by "
        "retrieval-augmented generation over DDInter + FDA label data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Allow Streamlit (or any frontend) running on a different port to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response schemas ---


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class ChunkResult(BaseModel):
    score: float
    drug_name: str
    section: str
    text: str


class StructuredInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: str
    mechanism: str | None
    management: str | None
    source: str


class QueryResponse(BaseModel):
    answer: str
    matched_drugs: list[str]
    structured_interaction: StructuredInteraction | None
    retrieved_chunks: list[ChunkResult]
    evidence_sufficient: bool


class SearchResponse(BaseModel):
    matched_drugs: list[str]
    structured_interaction: StructuredInteraction | None
    retrieved_chunks: list[ChunkResult]


class InteractionRequest(BaseModel):
    drug_a: str
    drug_b: str


# --- Endpoints ---


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Guards cost-incurring endpoints with an optional shared-secret header.
    If settings.app_api_key is unset (the default for local dev), this is a
    no-op -- the check only activates once a real key is configured, e.g.
    in a public deployment's environment variables.
    """
    if settings.app_api_key is None:
        return
    if x_api_key != settings.app_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
def full_query(req: QueryRequest):
    """Full RAG pipeline: retrieval + Claude generation with citations."""
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured")

    with Session(_resources["engine"]) as session:
        result = answer_query(
            req.question,
            session,
            _resources["embed_model"],
            _resources["qdrant"],
            _resources["anthropic"],
            top_k=req.top_k,
        )

    structured = None
    if result.retrieval.structured_interaction:
        structured = StructuredInteraction(**result.retrieval.structured_interaction)

    return QueryResponse(
        answer=result.answer_text,
        matched_drugs=result.retrieval.matched_drugs,
        structured_interaction=structured,
        retrieved_chunks=[ChunkResult(**c) for c in result.retrieval.retrieved_chunks],
        evidence_sufficient=result.evidence_sufficient,
    )


@app.post("/search", response_model=SearchResponse)
def search_only(req: QueryRequest):
    """Retrieval only — no LLM call, no API cost. Useful for debugging."""
    with Session(_resources["engine"]) as session:
        result = retrieve(
            req.question,
            session,
            _resources["embed_model"],
            _resources["qdrant"],
            top_k=req.top_k,
        )

    structured = None
    if result.structured_interaction:
        structured = StructuredInteraction(**result.structured_interaction)

    return SearchResponse(
        matched_drugs=result.matched_drugs,
        structured_interaction=structured,
        retrieved_chunks=[ChunkResult(**c) for c in result.retrieved_chunks],
    )


@app.post("/interactions", response_model=StructuredInteraction | None)
def interaction_lookup(req: InteractionRequest):
    """Direct structured DDInter lookup — no vector search, no LLM. Free and instant."""
    with Session(_resources["engine"]) as session:
        drug_a = session.query(Drug).filter(Drug.name.ilike(req.drug_a)).first()
        drug_b = session.query(Drug).filter(Drug.name.ilike(req.drug_b)).first()

        if not drug_a:
            raise HTTPException(status_code=404, detail=f"Drug not found: {req.drug_a}")
        if not drug_b:
            raise HTTPException(status_code=404, detail=f"Drug not found: {req.drug_b}")

        result = structured_lookup(drug_a, drug_b, session)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No interaction record found for {req.drug_a} + {req.drug_b}",
            )
        return StructuredInteraction(**result)


@app.get("/health")
def health_check():
    """Quick liveness check — confirms API is up and Qdrant is reachable."""
    try:
        info = _resources["qdrant"].get_collection(settings.qdrant_collection_name)
        return {
            "status": "ok",
            "qdrant_points": info.points_count,
            "database": settings.database_url.split("///")[-1],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")
