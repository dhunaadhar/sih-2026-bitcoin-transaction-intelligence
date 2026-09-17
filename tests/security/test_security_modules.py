from __future__ import annotations

import json
from pathlib import Path

from src.monitoring.external_heartbeat import (
    check_heartbeat,
    write_heartbeat,
)
from src.monitoring.health_monitor import run_health_check
from src.monitoring.incident_manager import (
    close_incident,
    create_incident,
)
from src.monitoring.notification_service import (
    validate_smtp_configuration,
)
from src.security.access_control import (
    check_permission,
    run_access_control_validation,
)
from src.security.integrity_verifier import (
    run_integrity_verification,
)
from src.security.security_audit import (
    record_security_event,
)
from src.security.security_baseline import (
    run_security_baseline,
)


def test_health_monitor_is_healthy():
    result = run_health_check()

    assert result["status"] == "healthy"
    assert result["summary"]["failed_checks"] == 0


def test_heartbeat_write_and_check():
    heartbeat = write_heartbeat()

    assert heartbeat["status"] == "alive"
    assert heartbeat["mode"] == "offline"

    result = check_heartbeat()

    assert result["status"] == "HEALTHY"
    assert result["reason"] == "HEARTBEAT_CURRENT"


def test_incident_lifecycle():
    incident = create_incident(
        title="Automated test incident",
        description="M19 incident lifecycle validation",
        severity="LOW",
        evidence={"test": True},
    )

    assert incident["status"] == "OPEN"
    assert incident["severity"] == "LOW"

    closed = close_incident(
        incident["incident_id"],
        "Automated validation completed",
    )

    assert closed["status"] == "CLOSED"
    assert closed["resolution"] == (
        "Automated validation completed"
    )


def test_notification_configuration_does_not_expose_secrets():
    result = validate_smtp_configuration()

    assert result["credentials_exposed"] is False
    assert "password" not in result


def test_access_control_default_deny():
    assert check_permission(
        "ANALYST",
        "read_alerts",
    ) is True

    assert check_permission(
        "ANALYST",
        "manage_system",
    ) is False

    assert check_permission(
        "UNKNOWN",
        "read_alerts",
    ) is False

    result = run_access_control_validation()

    assert result["status"] == "PASS"
    assert result["default_deny"] is True


def test_security_baseline():
    result = run_security_baseline()

    assert result["status"] == "PASS"
    assert (
        result["credential_protection"]["secrets_returned"]
        is False
    )


def test_integrity_verification():
    result = run_integrity_verification()

    assert result["status"] == "PASS"
    assert (
        result["source_provenance"]["status"]
        == "PASS"
    )


def test_security_audit_logging(tmp_path, monkeypatch):
    import src.security.security_audit as audit

    audit_path = tmp_path / "security_audit.jsonl"

    monkeypatch.setattr(
        audit,
        "AUDIT_LOG",
        audit_path,
    )

    monkeypatch.setattr(
        audit,
        "REPORT_DIR",
        tmp_path,
    )

    result = record_security_event(
        event="M19 automated security test",
        actor="pytest",
        action="TEST",
        details={"validation": True},
    )

    assert result["event"] == (
        "M19 automated security test"
    )
    assert result["secrets_included"] is False

    assert audit_path.exists()

    records = [
        json.loads(line)
        for line in audit_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(records) == 1
    assert records[0]["actor"] == "pytest"