from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "security"


ROLES = {
    "ANALYST": {
        "read_alerts": True,
        "read_graph": True,
        "read_risk": True,
        "export_reports": True,
        "manage_incidents": True,
        "manage_system": False,
    },
    "AUTHORITY": {
        "read_alerts": True,
        "read_graph": True,
        "read_risk": True,
        "export_reports": True,
        "manage_incidents": False,
        "manage_system": False,
    },
    "ADMIN": {
        "read_alerts": True,
        "read_graph": True,
        "read_risk": True,
        "export_reports": True,
        "manage_incidents": True,
        "manage_system": True,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_permission(role: str, permission: str) -> bool:
    """Check whether a defined role has a specific permission."""
    return bool(
        ROLES.get(role, {}).get(permission, False)
    )


def run_access_control_validation() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    checks = {
        "analyst_can_read_alerts": check_permission(
            "ANALYST", "read_alerts"
        ),
        "analyst_can_export_reports": check_permission(
            "ANALYST", "export_reports"
        ),
        "analyst_cannot_manage_system": not check_permission(
            "ANALYST", "manage_system"
        ),
        "authority_can_read_alerts": check_permission(
            "AUTHORITY", "read_alerts"
        ),
        "authority_can_export_reports": check_permission(
            "AUTHORITY", "export_reports"
        ),
        "authority_cannot_manage_system": not check_permission(
            "AUTHORITY", "manage_system"
        ),
        "admin_can_manage_system": check_permission(
            "ADMIN", "manage_system"
        ),
        "unknown_role_denied": not check_permission(
            "UNKNOWN", "read_alerts"
        ),
        "unknown_permission_denied": not check_permission(
            "ANALYST", "unknown_permission"
        ),
    }

    result = {
        "status": (
            "PASS"
            if all(checks.values())
            else "FAIL"
        ),
        "timestamp_utc": utc_now(),
        "mode": "offline",
        "roles": ROLES,
        "checks": checks,
        "default_deny": True,
    }

    path = REPORT_DIR / "access_control_validation.json"

    path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


if __name__ == "__main__":
    print(
        json.dumps(
            run_access_control_validation(),
            indent=2,
        )
    )