from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.monitoring.health_monitor import (
    run_health_check,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "monitoring"

WATCHDOG_REPORT = (
    REPORT_DIR / "watchdog_status.json"
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def run_watchdog() -> dict[str, Any]:
    health = run_health_check()

    status = (
        "healthy"
        if health["status"] == "healthy"
        else "degraded"
    )

    result = {
        "status": status,
        "timestamp_utc": utc_now(),
        "mode": "offline",
        "health_status": health["status"],
        "failed_checks": health[
            "failed_required_checks"
        ],
        "watchdog_action": (
            "NO_ACTION_REQUIRED"
            if status == "healthy"
            else "INVESTIGATION_REQUIRED"
        ),
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with WATCHDOG_REPORT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return result


if __name__ == "__main__":
    result = run_watchdog()

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    raise SystemExit(
        0
        if result["status"] == "healthy"
        else 1
    )