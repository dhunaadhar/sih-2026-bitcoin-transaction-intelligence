"""
External Heartbeat Monitoring
-----------------------------
M16.5 reliability component.

The SIH system periodically writes a heartbeat record.
An independent monitoring process/machine can check whether
the heartbeat is still fresh.

This module does NOT send email by itself.
It provides a deterministic heartbeat mechanism that an
independent monitor can use to detect host/application failure.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "monitoring"
HEARTBEAT_PATH = REPORT_DIR / "heartbeat.json"

DEFAULT_MAX_AGE_SECONDS = 120


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


def write_heartbeat(
    status: str = "alive",
    component: str = "sih-system",
) -> dict:
    """
    Write the current system heartbeat.

    Parameters
    ----------
    status:
        Current heartbeat status.
    component:
        Name of the monitored system/component.

    Returns
    -------
    dict
        The heartbeat record.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = utc_now()

    heartbeat = {
        "component": component,
        "status": status,
        "timestamp_utc": timestamp.isoformat(),
        "epoch_seconds": timestamp.timestamp(),
        "mode": "offline",
    }

    HEARTBEAT_PATH.write_text(
        json.dumps(heartbeat, indent=2),
        encoding="utf-8",
    )

    return heartbeat


def read_heartbeat() -> dict:
    """Read the latest heartbeat record."""
    if not HEARTBEAT_PATH.exists():
        raise FileNotFoundError(
            f"Heartbeat file not found: {HEARTBEAT_PATH}"
        )

    return json.loads(
        HEARTBEAT_PATH.read_text(encoding="utf-8")
    )


def check_heartbeat(
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
) -> dict:
    """
    Check whether the latest heartbeat is valid and fresh.

    This function is intended to be executed independently from
    the monitored application in a production deployment.
    """
    try:
        heartbeat = read_heartbeat()
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {
            "status": "FAILED",
            "reason": "HEARTBEAT_UNAVAILABLE",
            "detail": str(exc),
            "mode": "offline",
        }

    required_fields = {
        "component",
        "status",
        "timestamp_utc",
        "epoch_seconds",
        "mode",
    }

    missing_fields = sorted(
        required_fields - set(heartbeat.keys())
    )

    if missing_fields:
        return {
            "status": "FAILED",
            "reason": "INVALID_HEARTBEAT",
            "missing_fields": missing_fields,
            "mode": "offline",
        }

    if heartbeat["status"] != "alive":
        return {
            "status": "FAILED",
            "reason": "SYSTEM_NOT_ALIVE",
            "heartbeat_status": heartbeat["status"],
            "component": heartbeat["component"],
            "mode": "offline",
        }

    try:
        heartbeat_time = float(heartbeat["epoch_seconds"])
    except (TypeError, ValueError):
        return {
            "status": "FAILED",
            "reason": "INVALID_HEARTBEAT_TIMESTAMP",
            "mode": "offline",
        }

    age_seconds = utc_now().timestamp() - heartbeat_time

    if age_seconds < 0:
        return {
            "status": "FAILED",
            "reason": "HEARTBEAT_FROM_FUTURE",
            "age_seconds": age_seconds,
            "mode": "offline",
        }

    if age_seconds > max_age_seconds:
        return {
            "status": "FAILED",
            "reason": "HEARTBEAT_TIMEOUT",
            "age_seconds": round(age_seconds, 3),
            "max_age_seconds": max_age_seconds,
            "component": heartbeat["component"],
            "mode": "offline",
        }

    return {
        "status": "HEALTHY",
        "reason": "HEARTBEAT_CURRENT",
        "age_seconds": round(age_seconds, 3),
        "max_age_seconds": max_age_seconds,
        "component": heartbeat["component"],
        "heartbeat_timestamp_utc": heartbeat["timestamp_utc"],
        "mode": "offline",
    }


def write_heartbeat_report(result: dict) -> None:
    """Write the heartbeat check result for auditability."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "heartbeat_check.json"

    report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    """
    CLI behavior.

    Running the module normally writes a fresh heartbeat.
    Passing --check performs an independent heartbeat check.
    """
    if "--check" in sys.argv:
        result = check_heartbeat()
        write_heartbeat_report(result)

        print(json.dumps(result, indent=2))

        return 0 if result["status"] == "HEALTHY" else 1

    heartbeat = write_heartbeat()

    print(json.dumps(heartbeat, indent=2))
    print(f"Heartbeat written to: {HEARTBEAT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())