"""
Notification Service
--------------------
M17 authority notification component.

Supports:
- Controlled recipient loading
- Local/offline notification simulation
- Notification attempt logging
- Explicit delivery status
- No claim of guaranteed delivery
- No credentials stored in source code
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.monitoring.notification_config import (
    get_notification_recipients,
    get_smtp_config,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "monitoring"
NOTIFICATION_DIR = REPORT_DIR / "notifications"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_notification_record(
    subject: str,
    message: str,
    recipient: str,
    channel: str,
    status: str,
    error: str | None = None,
) -> dict:
    """
    Create an auditable notification-attempt record.
    """
    return {
        "notification_id": f"NTF-{uuid.uuid4()}",
        "timestamp_utc": utc_now(),
        "recipient": recipient,
        "channel": channel,
        "subject": subject,
        "status": status,
        "error": error,
        "delivery_guaranteed": False,
        "mode": "offline",
    }


def write_notification_record(record: dict) -> Path:
    """
    Persist a notification attempt for auditability.
    """
    NOTIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    path = (
        NOTIFICATION_DIR
        / f"{record['notification_id']}.json"
    )

    path.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    return path


def send_local_test_notification(
    subject: str,
    message: str,
) -> list[dict]:
    """
    Perform an offline notification simulation.

    No network connection is made and no email is sent.
    This validates recipient handling and audit logging.
    """
    recipients = get_notification_recipients()

    if not recipients:
        record = create_notification_record(
            subject=subject,
            message=message,
            recipient="NO_RECIPIENT_CONFIGURED",
            channel="offline_test",
            status="NOT_SENT",
            error="No notification recipients configured.",
        )

        write_notification_record(record)

        return [record]

    records = []

    for recipient in recipients:
        record = create_notification_record(
            subject=subject,
            message=message,
            recipient=recipient,
            channel="offline_test",
            status="SIMULATED",
        )

        write_notification_record(record)
        records.append(record)

    return records


def validate_smtp_configuration() -> dict:
    """
    Validate whether the required SMTP configuration exists.

    This function does not connect to an SMTP server.
    """
    smtp = get_smtp_config()

    required = {
        "host": bool(smtp["host"]),
        "port": bool(smtp["port"]),
        "username": bool(smtp["username"]),
        "password": bool(smtp["password"]),
        "sender": bool(smtp["sender"]),
    }

    return {
        "configured": all(required.values()),
        "fields": required,
        "credentials_exposed": False,
    }


if __name__ == "__main__":
    result = send_local_test_notification(
        subject="M17 notification service test",
        message="Offline notification workflow validation.",
    )

    print(json.dumps(result, indent=2))

    print(
        json.dumps(
            {
                "smtp_configuration": validate_smtp_configuration()
            },
            indent=2,
        )
    )