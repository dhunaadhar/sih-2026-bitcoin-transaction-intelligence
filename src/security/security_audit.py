from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "security"
AUDIT_LOG = REPORT_DIR / "security_audit.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_security_event(
    event: str,
    actor: str = "system",
    action: str = "INFO",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one auditable security event without secrets."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_utc": utc_now(),
        "event": event,
        "actor": actor,
        "action": action,
        "details": details or {},
        "secrets_included": False,
        "mode": "offline",
    }

    with AUDIT_LOG.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

    return record


def run_security_audit_test() -> dict[str, Any]:
    return record_security_event(
        event="M18 security audit validation",
        actor="system",
        action="TEST",
        details={
            "component": "security",
            "validation": True,
        },
    )


if __name__ == "__main__":
    print(
        json.dumps(
            run_security_audit_test(),
            indent=2,
        )
    )