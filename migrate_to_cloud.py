"""
Migrates the local Qdrant collection to Qdrant Cloud using Qdrant's native
snapshot export/import mechanism -- the correct, production-supported way
to move a collection between instances (not manual file copying, which is
fragile and version-sensitive).

Steps:
  1. Ask local Qdrant to create a snapshot of the collection (a point-in-time
     export bundling vectors + payloads + config into one file).
  2. Download that snapshot file to disk.
  3. Upload it to the cloud cluster's snapshot-recovery endpoint, which
     recreates the collection (including vector config) from the snapshot
     automatically -- no need to pre-create the collection on the cloud side.
  4. Verify the point counts match on both sides.

Run as: python migrate_to_cloud.py
Requires QDRANT_CLOUD_URL and QDRANT_API_KEY to be set (reads from .env via
the project's existing config, or set them directly as env vars below).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

from app.core.config import settings

LOCAL_URL = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
COLLECTION = settings.qdrant_collection_name

# Set this to your cluster's real URL (from the Qdrant Cloud dashboard),
# e.g. "https://a3be8920-....aws.cloud.qdrant.io:6333"
CLOUD_URL = input("Paste your Qdrant Cloud cluster URL: ").strip().rstrip("/")
CLOUD_API_KEY = settings.qdrant_api_key
if not CLOUD_API_KEY:
    print("ERROR: QDRANT_API_KEY not set in your .env -- required for cloud auth.")
    sys.exit(1)


def create_local_snapshot() -> str:
    print(f"Creating snapshot of local collection {COLLECTION!r}...")
    resp = requests.post(f"{LOCAL_URL}/collections/{COLLECTION}/snapshots")
    resp.raise_for_status()
    snapshot_name = resp.json()["result"]["name"]
    print(f"Snapshot created: {snapshot_name}")
    return snapshot_name


def download_snapshot(snapshot_name: str) -> Path:
    print("Downloading snapshot to local disk...")
    url = f"{LOCAL_URL}/collections/{COLLECTION}/snapshots/{snapshot_name}"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    out_path = Path(snapshot_name)
    with out_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Downloaded {out_path} ({size_mb:.1f} MB)")
    return out_path


def upload_to_cloud(snapshot_path: Path) -> None:
    print(f"Uploading snapshot to cloud cluster ({CLOUD_URL})...")
    url = f"{CLOUD_URL}/collections/{COLLECTION}/snapshots/upload?priority=snapshot"
    headers = {"api-key": CLOUD_API_KEY}

    with snapshot_path.open("rb") as f:
        files = {"snapshot": (snapshot_path.name, f)}
        resp = requests.post(url, headers=headers, files=files, timeout=300)

    if resp.status_code != 200:
        print(f"Upload failed: {resp.status_code} -- {resp.text}")
        sys.exit(1)
    print("Snapshot uploaded and collection recovered on cloud cluster.")


def verify(local_count: int) -> None:
    print("Verifying point counts match...")
    headers = {"api-key": CLOUD_API_KEY}
    resp = requests.get(f"{CLOUD_URL}/collections/{COLLECTION}", headers=headers)
    resp.raise_for_status()
    cloud_count = resp.json()["result"]["points_count"]

    print(f"  Local:  {local_count} points")
    print(f"  Cloud:  {cloud_count} points")
    if cloud_count == local_count:
        print("MATCH -- migration successful.")
    else:
        print("MISMATCH -- something may have gone wrong, investigate before deploying.")


def get_local_point_count() -> int:
    resp = requests.get(f"{LOCAL_URL}/collections/{COLLECTION}")
    resp.raise_for_status()
    return resp.json()["result"]["points_count"]


def main() -> None:
    local_count = get_local_point_count()
    print(f"Local collection has {local_count} points.\n")

    snapshot_name = create_local_snapshot()
    # Give Qdrant a moment to finish writing the snapshot file before we
    # try to download it -- snapshot creation is usually near-instant for
    # our data size, but this avoids a race on slower disks.
    time.sleep(2)

    snapshot_path = download_snapshot(snapshot_name)
    upload_to_cloud(snapshot_path)
    verify(local_count)

    print(f"\nLocal snapshot file kept at: {snapshot_path} (safe to delete once verified)")


if __name__ == "__main__":
    main()
