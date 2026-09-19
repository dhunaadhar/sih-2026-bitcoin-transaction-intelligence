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

[[ -x "${VENV_DIR}/bin/python" ]] || {
    echo "ERROR: Python 3.11 virtual environment not found."
    exit 1
}

[[ -x "${APPIMAGETOOL}" ]] || {
    echo "ERROR: AppImageTool not found."
    exit 1
}

"${VENV_DIR}/bin/python" --version

rm -rf "${APPDIR}"

mkdir -p \
    "${APPDIR}/usr/share/${APP_NAME}" \
    "${APPDIR}/usr/share/applications" \
    "${APPDIR}/usr/share/icons/hicolor/256x256/apps" \
    "${APPDIR}/opt"

echo "[1/6] Copying application..."
cp -a "${ROOT_DIR}/src" \
    "${APPDIR}/usr/share/${APP_NAME}/"

cp -a "${ROOT_DIR}/requirements.txt" \
    "${APPDIR}/usr/share/${APP_NAME}/"

if [[ -d "${ROOT_DIR}/deployment/linux/config" ]]; then
    cp -a "${ROOT_DIR}/deployment/linux/config" \
        "${APPDIR}/usr/share/${APP_NAME}/"
fi

echo "[2/6] Copying models..."
cp -a "${ROOT_DIR}/models" \
    "${APPDIR}/usr/share/${APP_NAME}/"

echo "[3/6] Copying data..."
cp -a "${ROOT_DIR}/data" \
    "${APPDIR}/usr/share/${APP_NAME}/"

echo "[4/6] Copying Python runtime..."
cp -a "${VENV_DIR}" \
    "${APPDIR}/opt/venv-linux"

echo "[5/6] Creating launcher..."

cat > "${APPDIR}/AppRun" <<'APPRUN'
#!/usr/bin/env bash

set -euo pipefail

APPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="${APPDIR}/usr/share/SIH-2026-Bitcoin-Transaction-Intelligence"
VENV="${APPDIR}/opt/venv-linux"
RUNTIME_DATA="${XDG_DATA_HOME:-${HOME}/.local/share}/SIH-2026-Bitcoin-Transaction-Intelligence/data"
mkdir -p "${RUNTIME_DATA}/derived/investigator_imports"
export SIH_RUNTIME_DATA_DIR="${RUNTIME_DATA}"

export SIH_MODE="${SIH_MODE:-offline}"
export SIH_ENV="${SIH_ENV:-production}"
export SIH_HOST="${SIH_HOST:-127.0.0.1}"
export SIH_PORT="${SIH_PORT:-8000}"
export PYTHONPATH="${APP_ROOT}:${PYTHONPATH:-}"

cd "${APP_ROOT}"

exec "${VENV}/bin/python" -m uvicorn \
    src.api.app:app \
    --host "${SIH_HOST}" \
    --port "${SIH_PORT}"
APPRUN

chmod +x "${APPDIR}/AppRun"

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

cp "${APPDIR}/usr/share/applications/${APP_NAME}.desktop" \
   "${APPDIR}/${APP_NAME}.desktop"

cat > "${APPDIR}/usr/share/icons/hicolor/256x256/apps/sih-2026.svg" <<'ICON'
<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
<rect width="256" height="256" rx="32"/>
<circle cx="128" cy="128" r="78" fill="none" stroke="#f59e0b" stroke-width="12"/>
<path d="M128 70v116M105 82h38c22 0 35 10 35 27 0 17-13 27-35 27h-38m0-54h43c19 0 30 9 30 23 0 15-11 24-30 24h-43m0 0h45c22 0 35 10 35 27 0 18-13 28-35 28h-45"
fill="none" stroke="white" stroke-width="10" stroke-linecap="round"/>
</svg>
ICON

cp "${APPDIR}/usr/share/icons/hicolor/256x256/apps/sih-2026.svg" \
   "${APPDIR}/sih-2026.svg"

echo "[6/6] Building AppImage..."

mkdir -p "${DIST_DIR}"

rm -f "${DIST_DIR}/${APP_NAME}-x86_64.AppImage"

"${APPIMAGETOOL}" \
    "${APPDIR}" \
    "${DIST_DIR}/${APP_NAME}-x86_64.AppImage"

chmod +x "${DIST_DIR}/${APP_NAME}-x86_64.AppImage"

echo
echo "=== APPIMAGE BUILD COMPLETE ==="
ls -lh "${DIST_DIR}/${APP_NAME}-x86_64.AppImage"