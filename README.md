# MedGuard RAG — Drug Interaction & Dosage Clinical Decision Support

A Retrieval-Augmented Generation (RAG) system that helps answer drug interaction
and dosage questions by combining a structured drug-interaction database with
real FDA drug label text — every answer grounded in cited evidence, never a
bare model guess.

> **Intended use:** clinical reference / decision-support tool for licensed
> professionals. This is **not** a substitute for professional medical judgment,
> and is not intended for direct patient use.

**🔗 Live demo:** [medguard-rag.vercel.app](https://medguard-rag.vercel.app)
(free-tier backend — first request after inactivity may take up to ~50s to wake)

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
              FastAPI backend (Render) + custom clinical frontend (Vercel)
              Vector store: Qdrant Cloud in production, local Docker in dev
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
- **fastembed over sentence-transformers/PyTorch for embeddings.** The
  project originally used `sentence-transformers` locally — reproducible and
  free, but PyTorch alone uses 400-600MB+ just importing, which crashed the
  first production deployment on Render's free tier (512MB cap). Switched to
  `fastembed` (Qdrant's own ONNX-based runtime, same `BAAI/bge-small-en-v1.5`
  model) for a dramatically smaller memory footprint, then re-embedded the
  full corpus so query-time and stored vectors are generated by the same
  backend — avoiding any precision-mismatch risk from mixing embedding
  libraries. Still local/free/no API key required, just lighter.
- **Qdrant Cloud requires an explicit payload index for filtering** that
  local self-hosted Qdrant doesn't strictly enforce (it falls back to a
  slower full scan instead). Hybrid retrieval's drug-scoped filtering worked
  locally but failed in production until a payload index was added on
  `drug_id` — now created automatically by the ingestion pipeline for any
  fresh setup, local or cloud.
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

**Quickest path — one command via Docker Compose** (recommended):

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY
docker compose up --build
```

This builds and starts all three services — Qdrant, the FastAPI backend, and
the frontend — wired together automatically. Open `http://localhost:8080`
once it's running.

Note: the ingestion pipeline (fetching ~800 real FDA labels, chunking, and
embedding ~22,000 chunks) takes significant time on first run due to
external API rate limits. A populated `data/processed/medguard.db` and
`qdrant_storage/` are not committed to this repo (see `.gitignore`) since
they're large, binary, and fully reproducible — expect roughly 1-2 hours
for a from-scratch ingestion run if you're rebuilding the dataset yourself.

**Manual / local dev path** (running each service directly, without Docker):

```bash
# 1. Environment
cp .env.example .env   # fill in ANTHROPIC_API_KEY, optionally QDRANT_API_KEY

# 2. Dependencies
pip3 install -r requirements.txt

# 3. Vector store
docker run -d --name medguard-qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

# 4. Ingestion pipeline (run once, in order)
python3 -m app.ingestion.load_ddinter data/raw/ddinter_downloads_code_B.csv
python3 -m app.ingestion.fetch_openfda
python3 -m app.ingestion.chunk_labels
python3 -m app.ingestion.embed_chunks

# 5. Run the backend
uvicorn app.api.main:app --reload --port 8000

# 6. Open the frontend
open frontend/index.html   # or serve it however you prefer

# Optional: raw retrieval testing without the API
python3 search_repl.py                    # raw semantic search
python3 -m app.retrieval.hybrid_search     # hybrid structured + semantic retrieval
```

## Deployment

The live demo runs on three separate free-tier services:

| Layer | Host | Notes |
|---|---|---|
| Frontend | [Vercel](https://vercel.com) | Static `frontend/index.html`, no build step |
| API | [Render](https://render.com) | Deployed from `Dockerfile` using `requirements-api.txt` (minimal deps — no PyTorch) |
| Vector store | [Qdrant Cloud](https://cloud.qdrant.io) | Free tier (1GB), migrated from local via `migrate_to_cloud.py` snapshot export/import |

To migrate a local Qdrant collection to a cloud cluster yourself:
```bash
python3 migrate_to_cloud.py   # prompts for your cluster URL, reads QDRANT_API_KEY from .env
```

Deployed API environment variables (set in Render's dashboard, never committed):
`ANTHROPIC_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `DATABASE_URL`, `CLAUDE_MODEL`, `APP_API_KEY`.

`APP_API_KEY` is an optional shared-secret header required on `/query` — a
deterrent against casual bots burning API credits on a public demo, not real
authentication (the frontend is static HTML, so the key is visible in
client-side source to anyone who looks).

## Project status

- [x] Structured data: DDInter loaded (15,140 interactions, 1,503 drugs)
- [x] Unstructured data: openFDA fetch complete (822 real FDA labels, 116
      combination products correctly excluded, 512 not found — mostly OTC
      minerals/electrolytes without prescription-style labels)
- [x] Chunking pipeline: sentence-aware, markup-stripped, section-scoped
      (21,885 chunks total)
- [x] Vector store: Qdrant populated and validated (21,885 embedded points)
- [x] Hybrid retrieval: structured + drug-filtered semantic search working
- [x] Claude API generation layer with mandatory citations and graceful
      refusal on insufficient evidence
- [x] Ragas evaluation suite (faithfulness, answer relevancy, context
      precision/recall — see `docs/eval_report*.csv` and
      `docs/engineering-notes.md` for findings)
- [x] FastAPI backend (`/query`, `/search`, `/interactions`, `/health`) with
      optional shared-secret protection on cost-incurring endpoints
- [x] Custom clinical-grade frontend (static HTML/JS, no framework)
- [x] Docker Compose bundling Qdrant + API + frontend, one-command startup
- [x] Live public deployment (Vercel + Render + Qdrant Cloud)
- [ ] MCP server wrapper (FastAPI only so far — a natural v2 extension)
- [ ] CI pipeline (lint/test/eval-regression gate)

## Disclaimer

This is a portfolio/educational project. It is not FDA-approved, not a
medical device, and should never be used as the sole basis for a clinical
decision. Always verify against current, authoritative sources.
