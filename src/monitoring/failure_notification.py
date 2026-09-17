"""
System Failure Notification
---------------------------
M17.3 controlled system-failure notification workflow.

Handles:
- Local watchdog/health failures
- Heartbeat timeout failures
- Incident creation
- Notification attempts
- Auditable notification records

This module does not claim guaranteed delivery.
For complete host failure, an independent external monitor must
invoke the failure notification mechanism.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.monitoring.incident_manager import create_incident
from src.monitoring.notification_service import (
    send_local_test_notification,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_failure_message(
    component: str,
    failure_reason: str,
    severity: str,
    incident_id: str,
) -> tuple[str, str]:
    """Build a controlled system-failure notification."""

    subject = (
        f"SIH System Failure Alert | "
        f"{component} | {severity}"
    )

    message = (
        "A system reliability failure has been detected.\n\n"
        f"Component: {component}\n"
        f"Failure reason: {failure_reason}\n"
        f"Severity: {severity}\n"
        f"Incident ID: {incident_id}\n"
        f"Detected at UTC: {utc_now()}\n\n"
        "This notification reports a system reliability event. "
        "It does not represent an investigative or blockchain-risk "
        "finding.\n\n"
        "Notification delivery is best-effort and is not guaranteed."
    )

    return subject, message


def notify_system_failure(
    component: str,
    failure_reason: str,
    severity: str = "HIGH",
    evidence: dict | None = None,
) -> dict:
    """
    Create a system-failure incident and perform a controlled
    notification attempt.
    """

    incident = create_incident(
        title=f"System failure: {component}",
        description=failure_reason,
        severity=severity,
        evidence=evidence or {},
    )

    subject, message = build_failure_message(
        component=component,
        failure_reason=failure_reason,
        severity=severity,
        incident_id=incident["incident_id"],
    )

    notification_records = send_local_test_notification(
        subject=subject,
        message=message,
    )

    return {
        "workflow": "system_failure_notification",
        "timestamp_utc": utc_now(),
        "component": component,
        "failure_reason": failure_reason,
        "severity": severity,
        "incident": incident,
        "notification_attempts": notification_records,
        "delivery_guaranteed": False,
        "mode": "offline",
    }


def main() -> int:
    """
    Run a synthetic failure notification test.

    No real email is sent unless the notification service is
    explicitly extended/configured for live delivery.
    """

    result = notify_system_failure(
        component="SIH-API",
        failure_reason="Synthetic M17 system-failure test",
        severity="HIGH",
        evidence={
            "test": True,
            "source": "manual_validation",
        },
    )

    print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())