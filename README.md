# MedGuard RAG — Drug Interaction & Dosage Clinical Decision Support

A Retrieval-Augmented Generation (RAG) system that helps answer drug interaction
and dosage questions by combining a structured drug-interaction database with
real FDA drug label text — every answer grounded in cited evidence, never a
bare model guess.

> **Intended use:** clinical reference / decision-support tool for licensed
> professionals. This is **not** a substitute for professional medical judgment,
> and is not intended for direct patient use.

## Why RAG, not a plain chatbot

A language model asked "does warfarin interact with ibuprofen?" from memory
alone can hallucinate — sound confident and still be wrong. That's an
unacceptable failure mode for a healthcare tool. This system instead:

1. Retrieves real evidence first (structured interaction records + real FDA
   label text)
2. Only then generates an answer, grounded in and citing what was retrieved
3. Surfaces its own uncertainty rather than guessing when evidence is thin

## Architecture

```
Data sources                Ingestion & indexing            Storage
─────────────               ──────────────────              ───────
DDInter 2.0        ──►      Load into SQL table    ──►      Postgres/SQLite
(structured DDI)            (drug pairs, severity)           `interactions` table

openFDA API         ──►     Fetch → clean → chunk   ──►      Qdrant (vectors)
(unstructured labels)       (sentence-aware,                 + SQLite `label_chunks`
                             markup-stripped)                 (text + metadata)

                                    │
                                    ▼
                     Hybrid retrieval (query time)
              ┌─────────────────────────────────────┐
              │ 1. Extract drug name(s) from question │
              │ 2. Exact structured lookup (severity) │
              │ 3. Vector search, filtered to those   │
              │    drug(s)' chunks only                │
              └─────────────────────────────────────┘
                                    │
                                    ▼
                    Claude API generation (cited answer)
                                    │
                                    ▼
                 FastAPI + MCP server + Streamlit UI
```

## Data sources

| Source | What it provides | License / access |
|---|---|---|
| [DDInter 2.0](https://ddinter2.scbdd.com) | 15,140 structured drug-drug interaction records (severity: Minor/Moderate/Major/Unknown) across 1,503 drugs, category B (blood/blood-forming organs) | Free, academic; original DDInter release is CC-BY-NC-SA — **non-commercial use only** |
| [openFDA](https://open.fda.gov) | Official FDA drug label text — boxed warnings, contraindications, dosage & administration, drug interactions | Free, public domain, no key required (optional free key raises rate limits) |

**Why category B specifically:** blood/anticoagulant interactions (e.g.
warfarin) are among the most clinically significant and well-documented DDI
categories, giving the project real-world relevance without requiring the
full 8-category, 300K+ row dataset for a v1.

## Key engineering decisions

These are the deliberate, documented tradeoffs made in this project — the
kind of judgment calls worth being able to explain in an interview.

- **Combination products are excluded from the FDA label corpus.** A
  multi-ingredient product's label (e.g. Triumeq = abacavir + dolutegravir +
  lamivudine) documents warnings for the *combination*, not any single
  ingredient — attributing that text to just one drug would misattribute
  clinical reasoning. Detected via `openfda.substance_name` count > 1.
- **Structured pair normalization.** Every interaction pair is stored with
  the smaller internal drug ID as `drug_a_id`, so a `UniqueConstraint`
  reliably catches (A,B)/(B,A) duplicates regardless of source ordering.
- **Chunking never crosses label sections.** A `drug_interactions` chunk
  never blends into `dosage_and_administration` — they answer different
  clinical questions and mixing them blurs retrieval precision.
- **Sentence-aware chunking with overlap**, not fixed character splitting —
  avoids severing a clause from its subject mid-sentence.
- **Local, open-source embeddings** (`BAAI/bge-small-en-v1.5` via
  `sentence-transformers`) rather than a cloud embedding API — zero cost,
  reproducible by anyone who clones the repo, no API key dependency for the
  retrieval layer. (Tradeoff: cloud options like Voyage AI benchmark higher
  on raw quality — a reasonable v2 extension.)
- **Hybrid retrieval, not pure vector search.** Drug interactions are
  look-up-table facts as much as narrative text; pure semantic search alone
  would risk hallucinating on "does X interact with Y" when it should be a
  deterministic lookup. Structured DDInter lookup + drug-filtered semantic
  search over FDA text run together.
- **Idempotent ingestion scripts throughout** — fetch, chunk, and embed
  steps all skip already-processed records, so any step can be safely
  re-run or resumed after an interruption without duplicating data or
  re-spending API quota.

## Setup

```bash
# 1. Environment
cp .env.example .env   # fill in ANTHROPIC_API_KEY, optionally QDRANT_API_KEY

# 2. Dependencies
pip3 install -r requirements.txt   # (see note below if this file doesn't exist yet)

# 3. Vector store
docker run -d --name medguard-qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

# 4. Ingestion pipeline (run once, in order)
python3 -m app.ingestion.load_ddinter data/raw/ddinter_downloads_code_B.csv
python3 -m app.ingestion.fetch_openfda
python3 -m app.ingestion.chunk_labels
python3 -m app.ingestion.embed_chunks

# 5. Try it
python3 search_repl.py                    # raw semantic search
python3 -m app.retrieval.hybrid_search     # hybrid structured + semantic retrieval
```

## Project status

- [x] Structured data: DDInter loaded (15,140 interactions, 1,503 drugs)
- [x] Unstructured data: openFDA fetch pipeline built and validated
- [x] Chunking pipeline: sentence-aware, markup-stripped, section-scoped
- [x] Vector store: Qdrant running, embeddings validated end-to-end
- [x] Hybrid retrieval: structured + drug-filtered semantic search working
- [ ] Full-scale openFDA fetch (in progress — 1,503 drugs)
- [ ] Claude API generation layer with mandatory citations
- [ ] Ragas evaluation suite
- [ ] FastAPI + MCP server
- [ ] Streamlit UI
- [ ] Docker Compose + CI

## Disclaimer

This is a portfolio/educational project. It is not FDA-approved, not a
medical device, and should never be used as the sole basis for a clinical
decision. Always verify against current, authoritative sources.
