# Engineering Notes & Decision Log

This documents real decisions, bugs, and tradeoffs encountered while building
MedGuard RAG — kept as a running log rather than reconstructed after the fact,
so the reasoning is honest rather than tidied up in hindsight.

## Data acquisition

**DDInter CSV structure.** The bulk CSV export
(`ddinter_downloads_code_{class}.csv`) contains only
`DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level` — no mechanism or
management text. That richer text lives on individual drug-pair detail pages
on DDInter's site, out of scope for bulk ingestion. This directly motivated
using openFDA as the second data source: DDInter supplies the reliable
severity fact, openFDA supplies the clinical reasoning behind it.

**Data quality check before writing the loader.** Verified on the real
15,140-row file before writing any ingestion code: zero exact duplicates,
zero reversed-pair (A,B)/(B,A) collisions, zero drug-ID naming
inconsistencies, zero empty fields. Severity distribution: Moderate 7,359 /
Unknown 4,242 / Major 2,739 / Minor 800 — the ~28% Unknown rate is a real
signal worth surfacing in the eventual eval (does the generation layer
hedge appropriately when severity is undocumented?).

## openFDA fetch

**404 vs empty results.** openFDA returns an actual HTTP 404 when a search
matches nothing, rather than a 200 with an empty result list. Handled
explicitly — an unhandled 404 would have crashed the fetch loop on the
first unmatched drug.

**Coverage gaps are real and explainable, not bugs.** Common OTC
electrolytes/minerals (ferrous fumarate, magnesium citrate, sodium
bicarbonate) frequently have no openFDA-indexed label — they're often sold
as generic salts/supplements rather than under an FDA-approved prescription
label filed under that exact generic name. Measured hit rate on a
representative 100-drug sample: 52% found, 38% not found, 9% correctly
skipped as combination products.

**Naming mismatches.** Some DDInter chemical names don't match openFDA's
indexing (e.g. "Acetylsalicylic acid" vs "Aspirin"). Not fixed in v1 —
documented as a known gap; a v2 could add a name-normalization/alias table.

**Combination product false positives.** openFDA's `generic_name`/
`brand_name` search matched "Iron" into a multivitamin ("Integra F"),
demonstrating why we validate `openfda.substance_name` count == 1 rather
than trusting the initial search match at face value.

## Chunking

**Embedded SPL markup.** Raw openFDA section text contains XML/HTML-like
tags from the source SPL format (`<paragraph>`, `<content
styleCode="bold">`, `<list>`, `<item>`) that must be stripped before
embedding — otherwise markup tokens pollute the embedding and waste tokens.

**Sentence-splitter edge case.** Converting bulleted list markers
(`&#x2022;`) into a `"- "` prefix broke the initial sentence-boundary
regex, which only recognized a capital letter or digit after `". "` — a
dash blocked the split. Fixed by widening the lookahead pattern and
stripping the resulting duplicated leading dash from split sentences.
Caught by testing against real markup-laden text before running at scale,
not discovered later against silently-wrong production data.

## Vector store

**SSL wrong-version error against local Qdrant.** `qdrant-client` can
default to assuming HTTPS; our local Docker Qdrant instance serves plain
HTTP (no TLS cert configured for local dev). Fixed with an explicit
`https=False` on the client — otherwise fails with
`[SSL: WRONG_VERSION_NUMBER]`.

**Embedding model choice: `BAAI/bge-small-en-v1.5`.** Chosen over a cloud
embedding API for zero cost, no API key dependency, and reproducibility —
anyone cloning the repo can run the full pipeline without needing project
credentials. Runs CPU-only in a few minutes for our chunk volume. Tradeoff
acknowledged: cloud options (e.g. Voyage AI) benchmark higher on raw
retrieval quality; a reasonable "v2" swap once cost/quality tradeoffs
matter more than reproducibility.

## Retrieval

**The gap discovered through manual testing, not anticipated in advance.**
Early hand-testing of raw semantic search (no drug filter) surfaced a real
design flaw: a query like "is it safe to take this while pregnant?" has no
mechanism to know *which* drug "this" refers to, so it searched the entire
1,535-chunk corpus and returned whichever drug's pregnancy-related text
happened to score highest — not necessarily the drug the user meant. Fixed
by building `extract_drug_names()` (deterministic substring matching
against the `drugs` table, longest-name-first to avoid partial-match
collisions) and filtering Qdrant search to only the identified drug(s)'
chunks via payload filtering. This is the "hybrid" half of hybrid
retrieval: structured lookup + scoped semantic search, not just vector
search alone.

## Known limitations / v2 candidates

- Drug name extraction is substring matching, not NLP-based — won't catch
  misspellings, abbreviations, or brand names not in our `drugs` table.
- No handling yet for 3+ drug interaction questions (polypharmacy).
- Chunking uses a simple regex sentence splitter — can mis-split on medical
  abbreviations (e.g. "Fig. 2", "e.g.").
- Only category B (blood/blood-forming organs) drugs are loaded from
  DDInter; other categories (A, D, H, L, P, R, V) are not yet ingested.
