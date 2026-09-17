"""
Authority Report Export
-----------------------
M17.3 controlled investigative report export.

Exports a selected ranked alert into a self-contained JSON
evidence bundle suitable for controlled analyst/authority review.

The report describes investigative indicators and does not
claim identity, illicit activity, ownership, or guilt.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RANKED_ALERTS = ROOT / "data" / "derived" / "ranked_alerts.parquet"
RISK_SCORES = ROOT / "data" / "derived" / "unified_risk_scores.parquet"
SHAP_EXPLANATIONS = ROOT / "data" / "derived" / "shap_explanations.parquet"

REPORT_DIR = ROOT / "reports" / "authority"
EXPORT_DIR = REPORT_DIR / "exports"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_txid(value) -> str:
    """Normalize TXID representation for exact matching."""
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        try:
            numeric = float(text)
            if numeric.is_integer():
                return str(int(numeric))
        except ValueError:
            pass

    return text


def load_single_alert(txid: str) -> dict:
    """
    Load the selected alert and associated risk/explainability data.
    """
    alerts = pd.read_parquet(RANKED_ALERTS)
    alerts["txid_normalized"] = alerts["txid"].map(normalize_txid)

    matches = alerts[
        alerts["txid_normalized"] == normalize_txid(txid)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one ranked alert for TXID {txid}; "
            f"found {len(matches)}."
        )

    alert = matches.iloc[0].to_dict()

    risk = pd.read_parquet(RISK_SCORES)
    risk["txid_normalized"] = risk["txid"].map(normalize_txid)

    risk_match = risk[
        risk["txid_normalized"] == normalize_txid(txid)
    ]

    risk_record = (
        risk_match.iloc[0].to_dict()
        if len(risk_match) == 1
        else {}
    )

    shap_record = {}

    if SHAP_EXPLANATIONS.exists():
        shap = pd.read_parquet(SHAP_EXPLANATIONS)
        shap["txid_normalized"] = shap["txid"].map(normalize_txid)

        shap_match = shap[
            shap["txid_normalized"] == normalize_txid(txid)
        ]

        if len(shap_match) == 1:
            shap_record = shap_match.iloc[0].to_dict()

    return {
        "alert": alert,
        "risk": risk_record,
        "explainability": shap_record,
    }


def sanitize_value(value):
    """Convert pandas/numpy values into JSON-compatible values."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    if isinstance(value, dict):
        return {
            str(k): sanitize_value(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [sanitize_value(v) for v in value]

    return value


def sanitize_record(record: dict) -> dict:
    return {
        str(key): sanitize_value(value)
        for key, value in record.items()
        if key != "txid_normalized"
    }


def build_authority_report(txid: str) -> dict:
    """
    Build a controlled investigative evidence bundle.
    """
    data = load_single_alert(txid)

    alert = sanitize_record(data["alert"])
    risk = sanitize_record(data["risk"])
    explainability = sanitize_record(data["explainability"])

    return {
        "report_type": "controlled_investigative_alert",
        "generated_at_utc": utc_now(),
        "mode": "offline",
        "txid": normalize_txid(txid),
        "investigative_status": {
            "purpose": "analyst_prioritization",
            "identity_established": False,
            "ownership_established": False,
            "illicit_activity_established": False,
            "guilt_established": False,
        },
        "alert": alert,
        "risk_evidence": risk,
        "explainability": explainability,
        "methodology_note": (
            "This report contains model, anomaly, behavioral, "
            "graph/entity, network, and explainability evidence "
            "available to the system. These indicators support "
            "investigative prioritization and require independent "
            "investigative verification."
        ),
        "limitations": [
            "Blockchain addresses are pseudonymous identifiers.",
            "Structural clustering does not establish real-world identity.",
            "Risk scores are investigative prioritization signals.",
            "Model predictions do not independently establish illicit activity.",
            "Network intelligence depends on available offline intelligence feeds.",
        ],
    }


def export_authority_report(txid: str) -> Path:
    """
    Export one controlled authority report.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = build_authority_report(txid)

    path = EXPORT_DIR / f"authority_alert_{normalize_txid(txid)}.json"

    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python -B -m src.monitoring.authority_report <TXID>"
        )
        raise SystemExit(2)

    output_path = export_authority_report(sys.argv[1])

    print(f"AUTHORITY REPORT EXPORT PASS")
    print(f"Output: {output_path}")