from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MAX_RESTARTS = 3
DEFAULT_RESTART_DELAY_SECONDS = 5.0
DEFAULT_HEALTH_TIMEOUT_SECONDS = 10.0
DEFAULT_HEALTH_POLL_SECONDS = 5.0

REPORT_DIR = Path(
    os.environ.get(
        "SIH_RUNTIME_REPORT_DIR",
        str(ROOT / "reports" / "monitoring"),
    )
)

SUPERVISOR_REPORT = REPORT_DIR / "process_supervisor.json"


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
        return value if value >= 0 else default
    except (TypeError, ValueError):
        return default


def _write_report(result: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUPERVISOR_REPORT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _build_command(
    command: list[str] | None = None,
) -> list[str]:
    """Build the supervised child command."""
    if command:
        return list(command)

    return [
        sys.executable,
        "-m",
        "src.desktop",
    ]


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return

    try:
        if os.name == "nt":
            process.terminate()
        else:
            process.send_signal(signal.SIGTERM)

        process.wait(timeout=10)

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_supervisor(
    max_restarts: int | None = None,
    restart_delay_seconds: float | None = None,
    health_poll_seconds: float | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run the SIH desktop application under bounded process supervision.

    The supervisor owns the desktop.py process. An unexpected child
    termination triggers an automatic restart until the configured
    restart limit is reached.

    A clean child exit is treated as an intentional shutdown and does
    not trigger an automatic restart.
    """
    configured_max_restarts = (
        DEFAULT_MAX_RESTARTS
        if max_restarts is None
        else max(0, int(max_restarts))
    )
    configured_restart_delay = (
        DEFAULT_RESTART_DELAY_SECONDS
        if restart_delay_seconds is None
        else max(0.0, float(restart_delay_seconds))
    )
    configured_poll_seconds = (
        DEFAULT_HEALTH_POLL_SECONDS
        if health_poll_seconds is None
        else max(0.25, float(health_poll_seconds))
    )

    command = _build_command(command)

    restart_count = 0
    launch_count = 0
    events: list[dict[str, Any]] = []
    started_at = utc_now()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        launch_count += 1

        environment = os.environ.copy()
        environment.setdefault("SIH_MODE", "offline")
        environment.setdefault("SIH_ENV", "production")
        environment.setdefault("SIH_HOST", "127.0.0.1")
        environment.setdefault("SIH_PORT", "8000")

        events.append(
            {
                "timestamp_utc": utc_now(),
                "event": "PROCESS_START",
                "launch_count": launch_count,
                "restart_count": restart_count,
            }
        )

        _write_report(
            {
                "status": "RUNNING",
                "timestamp_utc": utc_now(),
                "started_at_utc": started_at,
                "mode": "offline",
                "command": command,
                "restart_policy": {
                    "max_restarts": configured_max_restarts,
                    "restart_delay_seconds": configured_restart_delay,
                },
                "launch_count": launch_count,
                "restart_count": restart_count,
                "events": events[-20:],
            }
        )

        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=environment,
        )

        while process.poll() is None:
            time.sleep(configured_poll_seconds)

        return_code = process.returncode

        if return_code == 0:
            events.append(
                {
                    "timestamp_utc": utc_now(),
                    "event": "PROCESS_EXIT",
                    "exit_code": return_code,
                    "reason": "CLEAN_EXIT",
                    "restart_count": restart_count,
                }
            )

            result = {
                "status": "STOPPED",
                "timestamp_utc": utc_now(),
                "started_at_utc": started_at,
                "mode": "offline",
                "command": command,
                "restart_policy": {
                    "max_restarts": configured_max_restarts,
                    "restart_delay_seconds": configured_restart_delay,
                },
                "launch_count": launch_count,
                "restart_count": restart_count,
                "last_exit_code": return_code,
                "automatic_restart": False,
                "events": events[-20:],
            }

            _write_report(result)
            return result

        events.append(
            {
                "timestamp_utc": utc_now(),
                "event": "PROCESS_CRASH",
                "exit_code": return_code,
                "restart_count": restart_count,
            }
        )

        if restart_count >= configured_max_restarts:
            result = {
                "status": "FAILED",
                "timestamp_utc": utc_now(),
                "started_at_utc": started_at,
                "mode": "offline",
                "command": command,
                "restart_policy": {
                    "max_restarts": configured_max_restarts,
                    "restart_delay_seconds": configured_restart_delay,
                },
                "launch_count": launch_count,
                "restart_count": restart_count,
                "last_exit_code": return_code,
                "automatic_restart": False,
                "failure_reason": "RESTART_LIMIT_REACHED",
                "events": events[-20:],
            }

            _write_report(result)
            return result

        restart_count += 1

        events.append(
            {
                "timestamp_utc": utc_now(),
                "event": "AUTOMATIC_RESTART_SCHEDULED",
                "restart_number": restart_count,
                "delay_seconds": configured_restart_delay,
            }
        )

        _write_report(
            {
                "status": "RESTARTING",
                "timestamp_utc": utc_now(),
                "started_at_utc": started_at,
                "mode": "offline",
                "command": command,
                "restart_policy": {
                    "max_restarts": configured_max_restarts,
                    "restart_delay_seconds": configured_restart_delay,
                },
                "launch_count": launch_count,
                "restart_count": restart_count,
                "last_exit_code": return_code,
                "automatic_restart": True,
                "events": events[-20:],
            }
        )

        time.sleep(configured_restart_delay)


def main() -> int:
    result = run_supervisor(
        max_restarts=_env_int(
            "SIH_MAX_RESTARTS",
            DEFAULT_MAX_RESTARTS,
        ),
        restart_delay_seconds=_env_float(
            "SIH_RESTART_DELAY_SECONDS",
            DEFAULT_RESTART_DELAY_SECONDS,
        ),
        health_poll_seconds=_env_float(
            "SIH_SUPERVISOR_POLL_SECONDS",
            DEFAULT_HEALTH_POLL_SECONDS,
        ),
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0 if result["status"] == "STOPPED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
