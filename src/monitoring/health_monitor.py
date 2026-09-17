from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DERIVED_DIR = ROOT / "data" / "derived"
REPORT_DIR = ROOT / "reports" / "monitoring"

RANKED_ALERTS_PATH = DERIVED_DIR / "ranked_alerts.parquet"
RISK_SCORES_PATH = DERIVED_DIR / "unified_risk_scores.parquet"
TEMPORAL_GRAPH_PATH = (
    DERIVED_DIR
    / "temporal_transaction_entity_evidence.parquet"
)

GRAPH_NODES_PATH = (
    DERIVED_DIR
    / "investigation_graph"
    / "investigation_graph_nodes.parquet"
)

GRAPH_EDGES_PATH = (
    DERIVED_DIR
    / "investigation_graph"
    / "investigation_graph_edges.parquet"
)

DASHBOARD_PATH = ROOT / "src" / "dashboard" / "index.html"
API_PATH = ROOT / "src" / "api" / "app.py"


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def check_file(
    name: str,
    path: Path,
    required: bool = True,
) -> dict[str, Any]:

    exists = path.exists()
    size_bytes = (
        path.stat().st_size
        if exists and path.is_file()
        else 0
    )

    healthy = (
        exists
        and (
            not path.is_file()
            or size_bytes > 0
        )
    )

    return {
        "name": name,
        "path": str(path),
        "required": required,
        "exists": exists,
        "size_bytes": size_bytes,
        "healthy": healthy,
    }


def run_health_check() -> dict[str, Any]:

    checks = [
        check_file(
            "ranked_alerts",
            RANKED_ALERTS_PATH,
        ),
        check_file(
            "risk_scores",
            RISK_SCORES_PATH,
        ),
        check_file(
            "temporal_graph_features",
            TEMPORAL_GRAPH_PATH,
        ),
        check_file(
            "investigation_graph_nodes",
            GRAPH_NODES_PATH,
        ),
        check_file(
            "investigation_graph_edges",
            GRAPH_EDGES_PATH,
        ),
        check_file(
            "dashboard",
            DASHBOARD_PATH,
        ),
        check_file(
            "api",
            API_PATH,
        ),
    ]

    required_checks = [
        item
        for item in checks
        if item["required"]
    ]

    failed = [
        item["name"]
        for item in required_checks
        if not item["healthy"]
    ]

    status = (
        "healthy"
        if not failed
        else "degraded"
    )

    result = {
        "status": status,
        "mode": "offline",
        "timestamp_utc": utc_now(),
        "checks": checks,
        "failed_required_checks": failed,
        "summary": {
            "total_checks": len(checks),
            "healthy_checks": sum(
                bool(item["healthy"])
                for item in checks
            ),
            "failed_checks": len(failed),
        },
    }

    return result


def write_health_report(
    result: dict[str, Any],
) -> Path:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_DIR
        / "health_check.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            result,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def main() -> int:

    result = run_health_check()

    output_path = write_health_report(
        result
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        f"\nREPORT: {output_path}"
    )

    return (
        0
        if result["status"] == "healthy"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )