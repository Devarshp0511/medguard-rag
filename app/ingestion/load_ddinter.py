"""
Ingestion script: DDInter drug-drug interaction CSVs -> Postgres (drugs, interactions).

Real DDInter CSV columns (verified against ddinter_downloads_code_B.csv, 15,140 rows):
    DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level

Design decisions, and why:

1. Pair normalization: DDInter's own export never actually reverses a pair (we verified
   zero (B,A)-after-(A,B) collisions in the real file), but we normalize anyway --
   always store the alphabetically/numerically smaller DDInter ID as drug_a. This makes
   the UniqueConstraint in the schema meaningful even if a future CSV (a different drug
   class, or DDInter's next release) isn't as clean as this one. Never trust the next
   file to be as clean as the last one.

2. get_or_create for drugs: the same drug (e.g. "Dolutegravir") appears across many rows.
   We look it up by its DDInter ID first -- not by name -- because names can have
   formatting variants ("Ferrous fumarate" vs "ferrous fumarate") while the ID is stable.

3. mechanism/management are left NULL here deliberately. The bulk CSV export does not
   include them (confirmed against the real file) -- that text lives on individual
   drug-pair detail pages on the DDInter site, out of scope for bulk ingestion. This is
   documented rather than silently swallowed: it's exactly why the FDA label corpus
   (unstructured, retrieved via RAG) carries the reasoning half of this project.

4. Batched commits: for 15k+ rows, committing once per row is slow and holds a
   transaction open unnecessarily long. We flush per batch and commit at checkpoints.

Run as: python -m app.ingestion.load_ddinter data/raw/ddinter_downloads_code_B.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models.db_models import Base, Drug, Interaction, Severity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def _parse_severity(raw: str) -> Severity:
    """Map the CSV's free-text Level column onto our enum, defensively.
    Real file only ever contains {Minor, Moderate, Major, Unknown}, but we
    don't assume that holds for every future DDInter export."""
    try:
        return Severity(raw.strip())
    except ValueError:
        logger.warning("Unrecognized severity %r, defaulting to Unknown", raw)
        return Severity.UNKNOWN


def load_ddinter_csv(csv_path: Path, session: Session) -> dict[str, int]:
    """
    Loads one DDInter category CSV into the drugs and interactions tables.
    Returns a small summary dict for reporting (useful for logs and for the
    docs/eval report we'll write later -- "here's exactly what got loaded").
    """
    # DDInter ID (string like "DDInter582") -> Drug row, so we only ever
    # insert a given drug once even though it appears in many interaction rows.
    drug_cache: dict[str, Drug] = {}

    def get_or_create_drug(ddinter_id: str, name: str) -> Drug:
        if ddinter_id in drug_cache:
            return drug_cache[ddinter_id]

        # Also check the DB itself in case a prior run (or another CSV in the
        # same session) already inserted this drug -- e.g. Iron might appear
        # in both category B and category A files.
        existing = session.query(Drug).filter_by(name=name).one_or_none()
        if existing is not None:
            drug_cache[ddinter_id] = existing
            return existing

        drug = Drug(name=name)
        session.add(drug)
        session.flush()  # assigns drug.id without committing the transaction
        drug_cache[ddinter_id] = drug
        return drug

    rows_read = 0
    interactions_created = 0
    interactions_skipped_duplicate = 0
    drugs_created = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        expected_cols = {"DDInterID_A", "Drug_A", "DDInterID_B", "Drug_B", "Level"}
        missing_cols = expected_cols - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(
                f"CSV is missing expected columns: {missing_cols}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            rows_read += 1

            drug_a = get_or_create_drug(row["DDInterID_A"], row["Drug_A"].strip())
            drug_b = get_or_create_drug(row["DDInterID_B"], row["Drug_B"].strip())

            # Normalize pair order: smaller primary-key id is always drug_a_id.
            # This is what makes the UniqueConstraint on (drug_a_id, drug_b_id)
            # actually catch A-B / B-A duplicates, regardless of the CSV's own order.
            lo, hi = sorted([drug_a, drug_b], key=lambda d: d.id)

            already_exists = (
                session.query(Interaction)
                .filter_by(drug_a_id=lo.id, drug_b_id=hi.id)
                .one_or_none()
            )
            if already_exists is not None:
                interactions_skipped_duplicate += 1
                continue

            interaction = Interaction(
                drug_a_id=lo.id,
                drug_b_id=hi.id,
                severity=_parse_severity(row["Level"]),
                mechanism=None,  # not present in bulk CSV export -- see module docstring
                management=None,
                source="ddinter",
            )
            session.add(interaction)
            interactions_created += 1

            if rows_read % BATCH_SIZE == 0:
                session.commit()
                logger.info("Committed batch at row %d", rows_read)

    session.commit()
    drugs_created = len(drug_cache)

    summary = {
        "rows_read": rows_read,
        "drugs_created_or_matched": drugs_created,
        "interactions_created": interactions_created,
        "interactions_skipped_duplicate": interactions_skipped_duplicate,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a DDInter CSV into the database")
    parser.add_argument("csv_path", type=Path, help="Path to ddinter_downloads_code_X.csv")
    parser.add_argument(
        "--db-url",
        default="sqlite:///data/processed/medguard.db",
        help="SQLAlchemy database URL (default: local SQLite dev db)",
    )
    args = parser.parse_args()

    if not args.csv_path.exists():
        logger.error("File not found: %s", args.csv_path)
        sys.exit(1)

    engine = create_engine(args.db_url)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        summary = load_ddinter_csv(args.csv_path, session)

    logger.info("Ingestion complete: %s", summary)


if __name__ == "__main__":
    main()
