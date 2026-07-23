"""
Fetches unstructured FDA drug label text from the openFDA API for drugs
already loaded into our `drugs` table (via load_ddinter.py), and saves the
raw JSON responses to data/raw/openfda/ -- one file per drug.

Why save raw responses rather than going straight to chunking:
Same principle as keeping the original DDInter CSV around -- if our chunking
strategy changes later (and it will, once we evaluate retrieval quality),
we re-chunk from the saved raw JSON instead of re-hitting a rate-limited
external API. API responses are "raw data" now, same status as the CSV.

Endpoint reference (confirmed against open.fda.gov/apis/drug/label/):
    https://api.fda.gov/drug/label.json?search=openfda.generic_name:"warfarin"&limit=1

Key label sections we extract (standard SPL fields, present on most
prescription drug labels -- coverage varies per drug, hence the .get()
fallbacks throughout):
    boxed_warning              -- the most serious FDA warning, if any
    warnings / warnings_and_cautions
    drug_interactions           -- directly relevant to our use case
    contraindications
    dosage_and_administration
    indications_and_usage

Rate limiting: openFDA allows 240 requests/minute, 1000/day without a key;
40,000/day with a free key. We throttle conservatively and support an
optional API key via settings.openfda_api_key.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.db_models import Drug

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OPENFDA_BASE_URL = "https://api.fda.gov/drug/label.json"

# Sections we care about for a drug-interaction / dosage use case.
# Not every label has every section -- openFDA simply omits missing fields,
# it does not return null placeholders, so downstream code must not assume
# presence.
#
# Verified against a real response (warfarin, fetched 2026-07-14):
# top-level keys include boxed_warning, indications_and_usage,
# dosage_and_administration, dosage_and_administration_table,
# contraindications, warnings_and_cautions, drug_interactions,
# drug_interactions_table, adverse_reactions, and more. Note: there is
# no plain "warnings" field on modern labels -- it was replaced by
# "warnings_and_cautions" under the current SPL format. We keep "warnings"
# in this list anyway as a defensive fallback for older/legacy label
# records that may still use it.
RELEVANT_SECTIONS = [
    "boxed_warning",
    "warnings",
    "warnings_and_cautions",
    "drug_interactions",
    "drug_interactions_table",
    "contraindications",
    "dosage_and_administration",
    "dosage_and_administration_table",
    "indications_and_usage",
]

# Being polite to a free public API: small delay between requests even
# though we're well under the rate limit for a few hundred drugs.
REQUEST_DELAY_SECONDS = 0.3


def fetch_label_for_drug(drug_name: str, api_key: str | None = None) -> dict | None:
    """
    Queries openFDA for a single drug's label by generic name.
    Returns the first matching result dict, or None if no label is found
    (very common for less common drugs, combination products, or drugs
    where the brand name differs significantly from DDInter's naming).
    """
    # Try generic_name first (most reliable match to DDInter's naming),
    # fall back to brand_name if nothing matches.
    for field in ("openfda.generic_name", "openfda.brand_name"):
        query = f'{field}:"{quote(drug_name)}"'
        params = {"search": query, "limit": 1}
        if api_key:
            params["api_key"] = api_key

        try:
            resp = requests.get(OPENFDA_BASE_URL, params=params, timeout=10)
        except requests.RequestException as e:
            logger.warning("Request failed for %s (%s): %s", drug_name, field, e)
            continue

        if resp.status_code == 404:
            # openFDA returns 404 (not empty results) when nothing matches --
            # this is expected and common, not an error condition.
            continue
        if resp.status_code == 429:
            logger.warning("Rate limited on %s -- backing off 5s", drug_name)
            time.sleep(5)
            continue
        if resp.status_code != 200:
            logger.warning("Unexpected status %d for %s", resp.status_code, drug_name)
            continue

        data = resp.json()
        results = data.get("results", [])
        if results:
            return results[0]

    return None


def is_combination_product(label: dict) -> bool:
    """
    Detects multi-ingredient combination products (e.g. Triumeq = abacavir +
    dolutegravir + lamivudine) so we can exclude them from v1.

    Why this matters: a combination product's warnings/interactions section
    describes the combined regimen, not any single ingredient in isolation.
    Attributing that text to just one drug (e.g. "dolutegravir") would be a
    real correctness bug in a clinical tool -- the boxed warning on Triumeq
    is largely driven by abacavir hypersensitivity risk, not dolutegravir.

    openFDA's openfda.substance_name is a list of every active ingredient in
    the product. More than one entry means it's a combination product.
    """
    substances = label.get("openfda", {}).get("substance_name", [])
    return len(substances) > 1


def extract_relevant_text(label: dict) -> dict[str, str]:
    """
    Pulls just the sections we care about out of a full openFDA label record.
    Each section field in the raw response is a list of strings (openFDA's
    SPL-derived format almost always wraps text in a single-element list) --
    we join defensively in case a label has multiple paragraphs per section.
    """
    extracted = {}
    for section in RELEVANT_SECTIONS:
        value = label.get(section)
        if value:
            extracted[section] = " ".join(value) if isinstance(value, list) else str(value)
    return extracted


def fetch_all_labels(session: Session, output_dir: Path, limit: int | None = None) -> dict[str, int]:
    """
    Iterates every drug currently in our `drugs` table, fetches its openFDA
    label if one exists, and writes {drug_name}.json to output_dir.
    Skips drugs we've already fetched (based on file existing) so this
    script is safely re-runnable without burning API quota on repeats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    query = session.query(Drug)
    if limit:
        query = query.limit(limit)
    drugs = query.all()

    found = 0
    not_found = 0
    skipped_existing = 0
    combination_skipped = 0

    for drug in drugs:
        safe_filename = drug.name.replace("/", "_").replace(" ", "_").lower()
        out_path = output_dir / f"{safe_filename}.json"

        if out_path.exists():
            skipped_existing += 1
            continue

        label = fetch_label_for_drug(drug.name, api_key=settings.openfda_api_key)
        time.sleep(REQUEST_DELAY_SECONDS)

        if label is None:
            not_found += 1
            logger.info("No FDA label found for: %s", drug.name)
            continue

        if is_combination_product(label):
            combination_skipped += 1
            logger.info(
                "Skipping %s -- matched a combination product (%s)",
                drug.name,
                label.get("openfda", {}).get("brand_name", ["unknown"])[0],
            )
            continue

        extracted = extract_relevant_text(label)
        if not extracted:
            # Label exists but has none of our target sections (e.g. an
            # OTC monograph drug with a very different label structure)
            not_found += 1
            continue

        payload = {
            "drug_name": drug.name,
            "drug_id": drug.id,
            "sections": extracted,
            "openfda_meta": label.get("openfda", {}),
        }
        out_path.write_text(json.dumps(payload, indent=2))
        found += 1
        logger.info("Saved label for %s (%d sections)", drug.name, len(extracted))

    return {
        "total_drugs_checked": len(drugs),
        "labels_found": found,
        "labels_not_found": not_found,
        "combination_products_skipped": combination_skipped,
        "skipped_already_fetched": skipped_existing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch openFDA labels for drugs in our database")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only fetch labels for the first N drugs (useful for testing before a full run)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/openfda"),
    )
    args = parser.parse_args()

    engine = create_engine(settings.database_url)
    with Session(engine) as session:
        summary = fetch_all_labels(session, args.output_dir, limit=args.limit)

    logger.info("Fetch complete: %s", summary)


if __name__ == "__main__":
    main()
