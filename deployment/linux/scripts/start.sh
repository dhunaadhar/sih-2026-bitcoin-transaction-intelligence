#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-linux"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: Linux virtual environment not found."
    echo "Run deployment/linux/scripts/install_offline.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${ROOT_DIR}"

export SIH_MODE="${SIH_MODE:-offline}"
export SIH_ENV="${SIH_ENV:-production}"
export SIH_HOST="${SIH_HOST:-127.0.0.1}"
export SIH_PORT="${SIH_PORT:-8000}"

echo "Starting SIH 2026 Bitcoin Transaction Intelligence"
echo "Mode: ${SIH_MODE}"
echo "Environment: ${SIH_ENV}"
echo "Address: http://${SIH_HOST}:${SIH_PORT}"

exec python -m uvicorn src.api.app:app \
    --host "${SIH_HOST}" \
    --port "${SIH_PORT}"