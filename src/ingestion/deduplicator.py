from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def record_fingerprint(record: dict[str, Any]) -> str:
    """Return a deterministic fingerprint for a canonical record."""

    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deduplicate_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Remove exact duplicate canonical records.

    Returns:
        (unique_records, duplicates_removed)
    """

    unique_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates_removed = 0

    for record in records:
        fingerprint = record_fingerprint(record)

        if fingerprint in seen:
            duplicates_removed += 1
            continue

        seen.add(fingerprint)
        unique_records.append(record)

    return unique_records, duplicates_removed