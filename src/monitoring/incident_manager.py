from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INCIDENT_DIR = (
    ROOT / "reports" / "monitoring" / "incidents"
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def create_incident(
    title: str,
    description: str,
    severity: str = "MEDIUM",
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:

    normalized_severity = (
        str(severity)
        .strip()
        .upper()
    )

    allowed = {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }

    if normalized_severity not in allowed:
        raise ValueError(
            "Invalid incident severity."
        )

    incident_id = (
        "INC-"
        + uuid.uuid4().hex[:12].upper()
    )

    incident = {
        "incident_id": incident_id,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "OPEN",
        "severity": normalized_severity,
        "title": str(title).strip(),
        "description": str(description).strip(),
        "evidence": evidence or {},
    }

    INCIDENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        INCIDENT_DIR
        / f"{incident_id}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            incident,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return incident


def close_incident(
    incident_id: str,
    resolution: str,
) -> dict[str, Any]:

    incident_path = (
        INCIDENT_DIR
        / f"{incident_id}.json"
    )

    if not incident_path.exists():
        raise FileNotFoundError(
            f"Incident not found: {incident_id}"
        )

    with incident_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        incident = json.load(handle)

    incident["status"] = "CLOSED"
    incident["updated_at_utc"] = utc_now()
    incident["resolution"] = str(
        resolution
    ).strip()

    with incident_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            incident,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return incident