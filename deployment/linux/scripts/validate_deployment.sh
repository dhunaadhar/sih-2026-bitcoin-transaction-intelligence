#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "${ROOT_DIR}"

echo "=== SIH 2026 Linux Deployment Validation ==="

required_paths=(
    "requirements.txt"
    "src/api/app.py"
    "src/monitoring/health_monitor.py"
    "src/security/security_baseline.py"
    "models/production_xgboost/model.joblib"
    "models/production_isolation_forest/model.joblib"
    "data/derived/unified_features_temporal_safe.parquet"
    "data/derived/unified_risk_scores.parquet"
    "data/derived/ranked_alerts.parquet"
    "data/derived/shap_explanations.parquet"
    "data/derived/peeling_chain_indicators.parquet"
    "data/derived/mixing_pattern_indicators.parquet"
    "data/derived/investigation_graph/investigation_graph_nodes.parquet"
    "data/derived/investigation_graph/investigation_graph_edges.parquet"
    "src/dashboard/index.html"
)

for path in "${required_paths[@]}"; do
    if [[ ! -s "${ROOT_DIR}/${path}" ]]; then
        echo "FAIL: missing or empty ${path}"
        exit 1
    fi
done

echo "Required deployment artifacts: PASS"

python3 -B - <<'PY'
from src.api.app import app
from src.monitoring.health_monitor import run_health_check
from src.security.security_baseline import run_security_baseline_check

assert len(app.routes) >= 16

health = run_health_check()
assert health["status"] == "healthy"

security = run_security_baseline_check()
assert security["status"] == "PASS"

print("API: PASS")
print("Health monitoring: PASS")
print("Security baseline: PASS")
print("Offline deployment validation: PASS")
PY

echo "=== VALIDATION COMPLETE ==="