"""
Notification Configuration
---------------------------
M17 authority notification configuration.

Recipients and SMTP credentials are supplied externally through
environment variables and are never hard-coded into application
source code.
"""

from __future__ import annotations

import os


def get_notification_recipients() -> list[str]:
    """Return configured notification recipients."""
    raw = os.getenv("NOTIFICATION_RECIPIENTS", "")

    return [
        address.strip()
        for address in raw.split(",")
        if address.strip()
    ]


def get_smtp_config() -> dict[str, str]:
    """Return SMTP configuration without exposing secrets."""
    return {
        "host": os.getenv("SMTP_HOST", ""),
        "port": os.getenv("SMTP_PORT", "587"),
        "username": os.getenv("SMTP_USERNAME", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "sender": os.getenv("SMTP_SENDER", ""),
    }


def notification_configuration_status() -> dict:
    """Return safe notification configuration status."""
    recipients = get_notification_recipients()
    smtp = get_smtp_config()

    return {
        "recipients_configured": len(recipients) > 0,
        "recipient_count": len(recipients),
        "smtp_host_configured": bool(smtp["host"]),
        "smtp_port_configured": bool(smtp["port"]),
        "smtp_username_configured": bool(smtp["username"]),
        "smtp_password_configured": bool(smtp["password"]),
        "smtp_sender_configured": bool(smtp["sender"]),
        "credentials_exposed": False,
        "mode": "offline",
    }


if __name__ == "__main__":
    import json

    print(
        json.dumps(
            notification_configuration_status(),
            indent=2,
        )
    )