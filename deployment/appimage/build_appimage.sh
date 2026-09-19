#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APPIMAGE_DIR="${ROOT_DIR}/deployment/appimage"
APPDIR="${APPIMAGE_DIR}/AppDir"
DIST_DIR="${APPIMAGE_DIR}/dist"
VENV_DIR="${ROOT_DIR}/.venv-linux"

APP_NAME="SIH-2026-Bitcoin-Transaction-Intelligence"
APPIMAGETOOL="${HOME}/appimage-tools/appimagetool-x86_64.AppImage"

echo "=== SIH 2026 AppImage Builder ==="
echo "ROOT: ${ROOT_DIR}"

# ================================================================
# VALIDATION
# ================================================================

[[ -x "${VENV_DIR}/bin/python" ]] || {
    echo "ERROR: Python 3.11 virtual environment not found."
    exit 1
}

[[ -x "${APPIMAGETOOL}" ]] || {
    echo "ERROR: AppImageTool not found."
    exit 1
}

GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse --short HEAD)"
GIT_BRANCH="$(git -C "${ROOT_DIR}" branch --show-current)"

echo "Git branch: ${GIT_BRANCH}"
echo "Git commit: ${GIT_COMMIT}"

"${VENV_DIR}/bin/python" --version

# ================================================================
# CLEAN BUILD
# ================================================================

echo "[1/7] Cleaning previous build..."

rm -rf "${APPDIR}"

mkdir -p "${DIST_DIR}"

rm -f \
    "${DIST_DIR}/${APP_NAME}"-*.AppImage

mkdir -p \
    "${APPDIR}/usr/share/${APP_NAME}" \
    "${APPDIR}/usr/share/applications" \
    "${APPDIR}/usr/share/icons/hicolor/256x256/apps" \
    "${APPDIR}/opt"

# ================================================================
# COPY APPLICATION SOURCE
# ================================================================

echo "[2/7] Copying application source..."

mkdir -p \
    "${APPDIR}/usr/share/${APP_NAME}/src"

cp -a \
    "${ROOT_DIR}/src/." \
    "${APPDIR}/usr/share/${APP_NAME}/src/"

cp -a \
    "${ROOT_DIR}/requirements.txt" \
    "${APPDIR}/usr/share/${APP_NAME}/"

if [[ -d "${ROOT_DIR}/deployment/linux/config" ]]; then
    cp -a \
        "${ROOT_DIR}/deployment/linux/config" \
        "${APPDIR}/usr/share/${APP_NAME}/"
fi

# ================================================================
# VERIFY PACKAGED SOURCE
# ================================================================

echo "[3/7] Verifying packaged source..."

SOURCE_APP_JS="${ROOT_DIR}/src/dashboard/app.js"
PACKAGED_APP_JS="${APPDIR}/usr/share/${APP_NAME}/src/dashboard/app.js"

SOURCE_WALLET_JS="${ROOT_DIR}/src/dashboard/wallet.js"
PACKAGED_WALLET_JS="${APPDIR}/usr/share/${APP_NAME}/src/dashboard/wallet.js"

SOURCE_IMPORT="${ROOT_DIR}/src/ingestion/investigator_import.py"
PACKAGED_IMPORT="${APPDIR}/usr/share/${APP_NAME}/src/ingestion/investigator_import.py"

cmp -s \
    "${SOURCE_APP_JS}" \
    "${PACKAGED_APP_JS}" || {
        echo "ERROR: packaged app.js differs from source."
        exit 1
    }

cmp -s \
    "${SOURCE_WALLET_JS}" \
    "${PACKAGED_WALLET_JS}" || {
        echo "ERROR: packaged wallet.js differs from source."
        exit 1
    }

cmp -s \
    "${SOURCE_IMPORT}" \
    "${PACKAGED_IMPORT}" || {
        echo "ERROR: packaged investigator_import.py differs from source."
        exit 1
    }

grep -q \
    'window.inspectWalletFromGraph' \
    "${PACKAGED_WALLET_JS}" || {
        echo "ERROR: packaged wallet.js does not contain graph wallet inspection."
        exit 1
    }

if grep -q \
    'await window.inspectWalletFromGraph' \
    "${PACKAGED_APP_JS}"; then

    echo "ERROR: packaged app.js still contains obsolete awaited wallet bridge."
    exit 1
fi

echo "Source/package verification: PASSED"

# ================================================================
# COPY MODELS
# ================================================================

echo "[4/7] Copying models..."

cp -a \
    "${ROOT_DIR}/models" \
    "${APPDIR}/usr/share/${APP_NAME}/"

# ================================================================
# COPY DATA
# ================================================================

echo "[5/7] Copying data..."

#
# The generated intelligence data is not necessarily tracked by Git.
#
# When building from WSL, prefer the authoritative Windows-generated
# dataset so the AppImage does not accidentally package an older
# Linux clone of data/graph or data/derived.
#

DATA_SOURCE="${ROOT_DIR}/data"

WINDOWS_DATA_SOURCE="/mnt/d/Hackathons/SIH 2026/final/data"

if [[ -d "${WINDOWS_DATA_SOURCE}" ]]; then
    DATA_SOURCE="${WINDOWS_DATA_SOURCE}"

    echo "Using authoritative Windows data source:"
    echo "  ${DATA_SOURCE}"
else
    echo "Windows data source not available."
    echo "Using repository data source:"
    echo "  ${DATA_SOURCE}"
fi

[[ -d "${DATA_SOURCE}" ]] || {
    echo "ERROR: Data directory not found:"
    echo "  ${DATA_SOURCE}"
    exit 1
}

rm -rf \
    "${APPDIR}/usr/share/${APP_NAME}/data"

cp -a \
    "${DATA_SOURCE}" \
    "${APPDIR}/usr/share/${APP_NAME}/"

echo "Data source:"
echo "  ${DATA_SOURCE}"

# ================================================================
# COPY PYTHON RUNTIME
# ================================================================

echo "[6/7] Copying Python runtime..."

rm -rf \
    "${APPDIR}/opt/venv-linux"

cp -a \
    "${VENV_DIR}" \
    "${APPDIR}/opt/venv-linux"

# ================================================================
# BUILD METADATA
# ================================================================

echo "${GIT_COMMIT}" > \
    "${APPDIR}/usr/share/${APP_NAME}/BUILD_COMMIT"

echo "${GIT_BRANCH}" > \
    "${APPDIR}/usr/share/${APP_NAME}/BUILD_BRANCH"

# ================================================================
# APPRUN
# ================================================================

cat > "${APPDIR}/AppRun" <<'APPRUN'
#!/usr/bin/env bash

set -euo pipefail

APPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

APP_ROOT="${APPDIR}/usr/share/SIH-2026-Bitcoin-Transaction-Intelligence"
VENV="${APPDIR}/opt/venv-linux"

RUNTIME_DATA="${XDG_DATA_HOME:-${HOME}/.local/share}/SIH-2026-Bitcoin-Transaction-Intelligence/data"

mkdir -p \
    "${RUNTIME_DATA}/derived/investigator_imports"

export SIH_RUNTIME_DATA_DIR="${RUNTIME_DATA}"

export SIH_MODE="${SIH_MODE:-offline}"
export SIH_ENV="${SIH_ENV:-production}"
export SIH_HOST="${SIH_HOST:-127.0.0.1}"
export SIH_PORT="${SIH_PORT:-8000}"

export PYTHONPATH="${APP_ROOT}:${PYTHONPATH:-}"

cd "${APP_ROOT}"

exec "${VENV}/bin/python" "${APP_ROOT}/src/desktop.py"
APPRUN

chmod +x \
    "${APPDIR}/AppRun"

# ================================================================
# DESKTOP ENTRY
# ================================================================

cat > "${APPDIR}/usr/share/applications/${APP_NAME}.desktop" <<DESKTOP
[Desktop Entry]
Name=SIH 2026 Bitcoin Transaction Intelligence
Comment=Offline AI-powered Bitcoin transaction monitoring and analysis
Exec=${APP_NAME}
Icon=sih-2026
Terminal=true
Type=Application
Categories=Utility;Security;
DESKTOP

cp \
    "${APPDIR}/usr/share/applications/${APP_NAME}.desktop" \
    "${APPDIR}/${APP_NAME}.desktop"

# ================================================================
# ICON
# ================================================================

cat > "${APPDIR}/usr/share/icons/hicolor/256x256/apps/sih-2026.svg" <<'ICON'
<svg xmlns="http://www.w3.org/2000/svg"
     width="256"
     height="256"
     viewBox="0 0 256 256">

<rect
    width="256"
    height="256"
    rx="32"/>

<circle
    cx="128"
    cy="128"
    r="78"
    fill="none"
    stroke="#f59e0b"
    stroke-width="12"/>

<path
    d="M128 70v116
       M105 82h38
       c22 0 35 10 35 27
       0 17-13 27-35 27h-38
       m0-54h43
       c19 0 30 9 30 23
       0 15-11 24-30 24h-43
       m0 0h45
       c22 0 35 10 35 27
       0 18-13 28-35 28h-45"
    fill="none"
    stroke="white"
    stroke-width="10"
    stroke-linecap="round"/>

</svg>
ICON

cp \
    "${APPDIR}/usr/share/icons/hicolor/256x256/apps/sih-2026.svg" \
    "${APPDIR}/sih-2026.svg"

# ================================================================
# FINAL APPIMAGE
# ================================================================

echo "[7/7] Building AppImage..."

OUTPUT="${DIST_DIR}/${APP_NAME}-${GIT_COMMIT}-x86_64.AppImage"

rm -f \
    "${OUTPUT}"

"${APPIMAGETOOL}" \
    "${APPDIR}" \
    "${OUTPUT}"

chmod +x \
    "${OUTPUT}"

# ================================================================
# FINAL VERIFICATION
# ================================================================

echo
echo "=== APPIMAGE BUILD COMPLETE ==="
echo "Application: ${APP_NAME}"
echo "Git branch:  ${GIT_BRANCH}"
echo "Git commit:  ${GIT_COMMIT}"
echo "Data source: ${DATA_SOURCE}"
echo "Output:      ${OUTPUT}"
echo

ls -lh \
    "${OUTPUT}"