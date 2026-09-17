#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-linux"

echo "[1/6] Project root: ${ROOT_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required."
    exit 1
fi

PYTHON_VERSION="$(python3 --version)"
echo "[2/6] ${PYTHON_VERSION}"

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'; then
    echo "ERROR: Python 3.11 is required for the validated deployment environment."
    exit 1
fi

echo "[3/6] Creating isolated virtual environment..."
python3 -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[4/6] Upgrading local packaging tools..."
python -m pip install --upgrade pip setuptools wheel

echo "[5/6] Installing project dependencies..."
python -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "[6/6] Running deployment import validation..."
cd "${ROOT_DIR}"

python -B - <<'PY'
from src.api.app import app
from src.monitoring.health_monitor import run_health_check
from src.security.security_baseline import run_security_baseline_check

print(f"API routes: {len(app.routes)}")

health = run_health_check()
print(f"Health status: {health['status']}")

security = run_security_baseline_check()
print(f"Security status: {security['status']}")

if health["status"] != "healthy":
    raise SystemExit("ERROR: health check failed.")

if security["status"] != "PASS":
    raise SystemExit("ERROR: security baseline failed.")

print("DEPLOYMENT IMPORT VALIDATION: PASS")
PY

echo
echo "Offline Linux installation completed successfully."