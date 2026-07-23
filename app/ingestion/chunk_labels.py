"""
Chunks raw openFDA label JSON (from fetch_openfda.py) into clean, retrievable
text pieces, and loads them into the label_chunks table.

Design decisions:

1. Chunk boundaries never cross a label section. A drug_interactions chunk
   and a dosage_and_administration chunk are answering different clinical
   questions -- mixing them would blur the embedding and hurt retrieval
   precision for both.

2. HTML/XML markup is stripped before chunking. Raw openFDA text contains
   embedded SPL markup (<paragraph>, <content styleCode="bold">, <list>,
   <item>, tables) -- verified in our own fetched Dolutegravir/Triumeq data.
   This markup is noise for a semantic embedding model and wastes tokens.

3. Long sections are split on sentence boundaries with overlap, not fixed
   character counts. Splitting mid-sentence can sever a clause from its
   subject ("...should be avoided" with no antecedent), destroying meaning.
   A small overlap (last ~2 sentences of chunk N repeated at the start of
   chunk N+1) means a fact sitting near a boundary isn't orphaned entirely
   in one chunk.

4. Each chunk gets a stable UUID as its vector_id at chunk-creation time
   (not at embedding time). This makes the pipeline resumable: if the
   embedding step fails partway through, we know exactly which vector_ids
   still need embedding, rather than re-deriving IDs and risking mismatches.

Run as: python -m app.ingestion.chunk_labels
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import uuid
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.db_models import Base, Drug, LabelChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Target chunk size in characters. ~1200 chars is roughly 250-300 tokens --
# small enough for precise retrieval, large enough to hold a complete thought.
TARGET_CHUNK_CHARS = 1200
OVERLAP_SENTENCES = 2

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
# Simple sentence splitter: splits on '. ' / '? ' / '! ' followed by a
# capital letter or digit -- not perfect (medical text has abbreviations
# like "e.g." and "Fig. 2" that can cause false splits), but adequate for
# chunk boundaries where an occasional early split just means a slightly
# smaller chunk, not a correctness bug.
# Simple sentence splitter: splits on '. ' / '? ' / '! ' followed by a
# capital letter, digit, or a bullet dash (list items converted from
# "&#x2022;" become "- " in clean_text, and should always start a new
# chunk-eligible unit even though they don't start with a capital letter).
# Not a perfect splitter -- medical text has abbreviations like "e.g." and
# "Fig. 2" that can cause false splits -- but adequate for chunk boundaries,
# where an occasional early split just means a slightly smaller chunk.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9\-])")


def clean_text(raw: str) -> str:
    """Strip embedded SPL/XML markup and normalize whitespace."""
    no_tags = _TAG_RE.sub(" ", raw)
    # Decode the most common HTML entities we've actually seen in the data
    no_tags = no_tags.replace("&#x2022;", "-").replace("&lt;", "<").replace("&gt;", ">")
    return _WHITESPACE_RE.sub(" ", no_tags).strip()


def split_sentences(text: str) -> list[str]:
    raw_sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    # Bullet markers ("- ") survive as a leading token on list-item sentences
    # after the split above; strip a single leading "- " so list items read
    # as clean sentences rather than "- - Digoxin toxicity...".
    return [re.sub(r"^-\s+", "", s) for s in raw_sentences]


def chunk_section_text(text: str) -> list[str]:
    """
    Splits one section's cleaned text into chunks of roughly
    TARGET_CHUNK_CHARS, on sentence boundaries, with a small sentence
    overlap between consecutive chunks.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > TARGET_CHUNK_CHARS and current:
            chunks.append(" ".join(current))
            # Start next chunk with the last few sentences of this one,
            # for continuity across the boundary.
            current = current[-OVERLAP_SENTENCES:]
            current_len = sum(len(s) for s in current)

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_drug_file(json_path: Path, session: Session) -> int:
    """
    Reads one drug's raw openFDA JSON file, chunks each section, and
    inserts LabelChunk rows. Returns the number of chunks created.
    Idempotent: skips if this drug already has chunks in the DB.
    """
    payload = json.loads(json_path.read_text())
    drug_name = payload["drug_name"]

    drug = session.query(Drug).filter_by(name=drug_name).one_or_none()
    if drug is None:
        logger.warning("Drug %r not found in DB, skipping file %s", drug_name, json_path.name)
        return 0

    existing_count = session.query(LabelChunk).filter_by(drug_id=drug.id).count()
    if existing_count > 0:
        logger.debug("Drug %s already has %d chunks, skipping", drug_name, existing_count)
        return 0

    chunks_created = 0
    for section, raw_text in payload["sections"].items():
        cleaned = clean_text(raw_text)
        pieces = chunk_section_text(cleaned)

        for idx, piece in enumerate(pieces):
            chunk = LabelChunk(
                drug_id=drug.id,
                chunk_text=piece,
                section=section,
                chunk_index=idx,
                vector_id=str(uuid.uuid4()),
                source_url=f"https://api.fda.gov/drug/label.json (drug={drug_name})",
            )
            session.add(chunk)
            chunks_created += 1

    return chunks_created


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk raw openFDA JSON into label_chunks")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/openfda"),
    )
    args = parser.parse_args()

    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)

    json_files = sorted(args.input_dir.glob("*.json"))
    logger.info("Found %d raw openFDA files to process", len(json_files))

    total_chunks = 0
    drugs_processed = 0

    with Session(engine) as session:
        for path in json_files:
            created = chunk_drug_file(path, session)
            if created > 0:
                total_chunks += created
                drugs_processed += 1
                if drugs_processed % 25 == 0:
                    session.commit()
                    logger.info("Committed batch: %d drugs, %d chunks so far", drugs_processed, total_chunks)

        session.commit()

    logger.info(
        "Chunking complete: %d drugs processed, %d total chunks created",
        drugs_processed,
        total_chunks,
    )


if __name__ == "__main__":
    main()
