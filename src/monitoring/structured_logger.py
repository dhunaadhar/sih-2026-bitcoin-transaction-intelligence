from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "reports" / "monitoring"
LOG_PATH = LOG_DIR / "application.jsonl"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).replace(microsecond=0).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(
            payload,
            ensure_ascii=False,
        )


def get_logger(
    name: str = "sih",
) -> logging.Logger:
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(
        LOG_PATH,
        encoding="utf-8",
    )

    handler.setFormatter(
        JsonFormatter()
    )

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def log_event(
    message: str,
    level: int = logging.INFO,
    **context: Any,
) -> None:
    logger = get_logger()

    if context:
        message = (
            f"{message} | "
            + json.dumps(
                context,
                ensure_ascii=False,
                default=str,
            )
        )

    logger.log(
        level,
        message,
    )