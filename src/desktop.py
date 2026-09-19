from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn
import webview


APP_ROOT = Path(__file__).resolve().parents[1]

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from src.api.app import app


HOST = os.environ.get("SIH_HOST", "127.0.0.1")
PORT = int(os.environ.get("SIH_PORT", "8000"))
URL = f"http://{HOST}:{PORT}/dashboard/"
HEALTH_URL = f"http://{HOST}:{PORT}/api/health"


def run_server():
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=os.environ.get("SIH_LOG_LEVEL", "warning"),
    )


def wait_for_server(timeout=30.0):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                HEALTH_URL,
                timeout=1.0,
            ):
                return
        except Exception:
            time.sleep(0.25)

    raise RuntimeError(
        f"SIH backend did not become ready at {URL}"
    )


def main():
    server_thread = threading.Thread(
        target=run_server,
        daemon=True,
    )
    server_thread.start()

    wait_for_server()

    webview.create_window(
        "SIH 2026 - Bitcoin Transaction Intelligence",
        URL,
        width=1440,
        height=900,
        min_size=(1100, 700),
        resizable=True,
    )

    webview.start()


if __name__ == "__main__":
    main()