from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    """Calculate the SHA-256 hash of a source file."""

    path = Path(path)
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_provenance(
    path: str | Path,
    records_read: int,
    valid_records: int,
    invalid_records: int,
    duplicates_removed: int,
) -> dict[str, Any]:
    """Build an auditable provenance record for one ingestion run."""

    path = Path(path)

    return {
        "source_file": str(path.resolve()),
        "source_format": path.suffix.lower().lstrip("."),
        "source_sha256": file_sha256(path),
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_read": records_read,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "duplicate_records_removed": duplicates_removed,
    }