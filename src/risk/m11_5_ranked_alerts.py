"""
M11.5 — Ranked Explainable Alerts

Purpose
-------
Convert the M11.4 unified transaction risk scores into a ranked,
investigation-oriented alert queue.

Input
-----
data/derived/unified_risk_scores.parquet

Outputs
-------
data/derived/ranked_alerts.parquet
reports/risk/m11_5_ranked_alerts.json

Important semantics
-------------------
The alert queue represents investigative PRIORITY.

It does NOT establish:
    - illicit activity
    - criminal activity
    - identity
    - wallet ownership
    - IP ownership
    - transaction authorship

The numeric Elliptic++ class labels are preserved without assigning
semantic names to them.

Design principles
-----------------
- Deterministic ranking
- Stable tie-breaking
- One alert row per TXID
- Explicit evidence channels
- Human-readable explanation
- No re-scoring
- No modification of M11.4 scores
- Separate alert artifact from the complete risk-score artifact
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "derived"
    / "unified_risk_scores.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "derived"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "risk"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "ranked_alerts.parquet"
)

REPORT_FILE = (
    REPORT_DIR
    / "m11_5_ranked_alerts.json"
)


# ============================================================================
# Configuration
# ============================================================================

# These are queue thresholds, not probabilities.

ALERT_THRESHOLDS = {
    "VERY_HIGH": 80.0,
    "HIGH": 60.0,
    "MODERATE": 40.0,
    "GUARDED": 20.0,
    "LOW": 0.0,
}


# Number of highest-priority records explicitly marked as alerts.
#
# The complete scored dataset remains available in M11.4.
# M11.5 adds a ranked investigation queue on top of it.

TOP_ALERT_COUNT = 1000


# ============================================================================
# Utility functions
# ============================================================================

def normalize_txid(value) -> str:
    """
    Normalize TXID representations consistently.
    """

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


def require_columns(
    df: pd.DataFrame,
    required: list[str],
) -> None:
    """
    Ensure all required columns are present.
    """

    missing = sorted(
        set(required)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "M11.4 risk-score artifact is missing required "
            f"columns: {missing}"
        )


def finite_float(
    value,
    default: float = 0.0,
) -> float:
    """
    Convert a value to a finite float.
    """

    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except (
        TypeError,
        ValueError,
    ):
        pass

    return default


def threshold_for_level(
    level: str,
) -> float:
    """
    Return the numeric threshold for a risk level.
    """

    return float(
        ALERT_THRESHOLDS.get(
            level,
            0.0,
        )
    )


# ============================================================================
# Loading
# ============================================================================

def load_risk_scores() -> pd.DataFrame:
    """
    Load and validate M11.4 output.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"M11.4 output not found: {INPUT_FILE}"
        )

    print(
        "Reading M11.4 unified risk scores:",
        INPUT_FILE,
    )

    df = pd.read_parquet(
        INPUT_FILE
    )

    required = [
        "txid",
        "time_step",
        "risk_score",
        "risk_level",
        "ml_predicted_class",
        "ml_confidence",
        "anomaly_signal",
        "behavioral_signal",
        "entity_signal",
        "network_signal",
        "effective_evidence_channel_count",
        "risk_explanation",
        "evidence_agreement",
        "peeling_evidence_score",
        "mixing_evidence_score",
        "behavioral_signal_count",
        "multiple_behavioral_signals",
        "behavioral_signal_agreement",
        "network_observation_count",
        "network_unique_source_ip_count",
        "network_unique_destination_ip_count",
        "network_unique_endpoint_count",
        "network_temporal_evidence_level",
        "network_observation_available",
    ]

    require_columns(
        df,
        required,
    )

    df = df.copy()

    df["txid"] = (
        df["txid"]
        .map(normalize_txid)
    )

    if df["txid"].eq("").any():
        raise ValueError(
            "M11.4 output contains empty TXIDs."
        )

    if df["txid"].duplicated().any():
        raise ValueError(
            "M11.4 output contains duplicate TXIDs."
        )

    risk_scores = pd.to_numeric(
        df["risk_score"],
        errors="coerce",
    )

    if risk_scores.isna().any():
        raise ValueError(
            "M11.4 output contains invalid risk scores."
        )

    if (
        risk_scores < 0
    ).any() or (
        risk_scores > 100
    ).any():

        raise ValueError(
            "M11.4 risk scores are outside [0,100]."
        )

    return df


# ============================================================================
# Alert classification
# ============================================================================

def derive_alert_priority(
    score: float,
) -> str:
    """
    Derive priority directly from M11.4 score.

    This is intentionally independent of model class semantics.
    """

    if score >= ALERT_THRESHOLDS["VERY_HIGH"]:
        return "VERY_HIGH"

    if score >= ALERT_THRESHOLDS["HIGH"]:
        return "HIGH"

    if score >= ALERT_THRESHOLDS["MODERATE"]:
        return "MODERATE"

    if score >= ALERT_THRESHOLDS["GUARDED"]:
        return "GUARDED"

    return "LOW"


def derive_alert_status(
    priority: str,
) -> str:
    """
    Determine whether a record belongs to the actionable alert queue.

    LOW records remain in the complete ranked artifact but are not
    promoted into the active queue.
    """

    if priority in {
        "VERY_HIGH",
        "HIGH",
        "MODERATE",
    }:
        return "ACTIVE"

    if priority == "GUARDED":
        return "WATCH"

    return "INFORMATIONAL"


# ============================================================================
# Evidence summaries
# ============================================================================

def build_evidence_summary(
    row: pd.Series,
) -> str:
    """
    Build a compact structured evidence summary.
    """

    parts = []

    ml = finite_float(
        row.get(
            "ml_confidence"
        )
    )

    anomaly = finite_float(
        row.get(
            "anomaly_signal"
        )
    )

    behavioral = finite_float(
        row.get(
            "behavioral_signal"
        )
    )

    entity = finite_float(
        row.get(
            "entity_signal"
        )
    )

    network = finite_float(
        row.get(
            "network_signal"
        )
    )

    if ml >= 0.70:
        parts.append(
            f"ML confidence={ml:.3f}"
        )

    if anomaly >= 0.70:
        parts.append(
            f"anomaly={anomaly:.3f}"
        )

    if behavioral >= 0.70:
        parts.append(
            f"behavioral={behavioral:.3f}"
        )

    elif behavioral >= 0.40:
        parts.append(
            f"behavioral={behavioral:.3f}"
        )

    if entity >= 0.70:
        parts.append(
            f"entity={entity:.3f}"
        )

    elif entity >= 0.40:
        parts.append(
            f"entity={entity:.3f}"
        )

    if network > 0.0:
        parts.append(
            f"network={network:.3f}"
        )

    if not parts:
        return "No strong evidence channel."

    return "; ".join(
        parts
    )


def build_alert_explanation(
    row: pd.Series,
) -> str:
    """
    Create a concise explanation suitable for a dashboard alert card.
    """

    score = finite_float(
        row.get(
            "risk_score"
        )
    )

    priority = str(
        row.get(
            "risk_level",
            "LOW",
        )
    )

    channel_count = int(
        finite_float(
            row.get(
                "effective_evidence_channel_count"
            )
        )
    )

    predicted_class = row.get(
        "ml_predicted_class"
    )

    confidence = finite_float(
        row.get(
            "ml_confidence"
        )
    )

    behavioral = finite_float(
        row.get(
            "behavioral_signal"
        )
    )

    entity = finite_float(
        row.get(
            "entity_signal"
        )
    )

    network = finite_float(
        row.get(
            "network_signal"
        )
    )

    anomaly = finite_float(
        row.get(
            "anomaly_signal"
        )
    )

    contributors = []

    if behavioral >= 0.70:
        contributors.append(
            "strong behavioral evidence"
        )

    elif behavioral >= 0.40:
        contributors.append(
            "moderate behavioral evidence"
        )

    if anomaly >= 0.70:
        contributors.append(
            "strong anomaly evidence"
        )

    elif anomaly >= 0.40:
        contributors.append(
            "moderate anomaly evidence"
        )

    if entity >= 0.70:
        contributors.append(
            "strong historical entity evidence"
        )

    elif entity >= 0.40:
        contributors.append(
            "moderate historical entity evidence"
        )

    if network > 0.0:
        contributors.append(
            "network correlation evidence"
        )

    if confidence >= 0.90:
        contributors.append(
            "high supervised-model confidence"
        )

    if not contributors:
        contributors.append(
            "limited positive evidence"
        )

    contributor_text = ", ".join(
        contributors
    )

    return (
        f"Priority score {score:.2f} ({priority}). "
        f"Contributing signals: {contributor_text}. "
        f"{channel_count} independent evidence channels "
        f"are active. "
        f"M7 predicted class={predicted_class} with "
        f"confidence={confidence:.3f}. "
        f"This is an investigative-priority assessment, "
        f"not proof of illicit activity or identity."
    )


# ============================================================================
# Ranking
# ============================================================================

def build_ranked_alerts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build deterministic ranked alert queue.

    Ranking:
        1. risk_score descending
        2. evidence channel count descending
        3. behavioral signal descending
        4. entity signal descending
        5. anomaly signal descending
        6. network signal descending
        7. time_step ascending
        8. txid ascending

    The additional fields are tie-breakers only and do not alter the
    M11.4 risk score.
    """

    result = df.copy()

    result[
        "alert_priority"
    ] = result[
        "risk_score"
    ].map(
        derive_alert_priority
    )

    result[
        "alert_status"
    ] = result[
        "alert_priority"
    ].map(
        derive_alert_status
    )

    # ---------------------------------------------------------------
    # Sort deterministically.
    # ---------------------------------------------------------------

    result = result.sort_values(
        [
            "risk_score",
            "effective_evidence_channel_count",
            "behavioral_signal",
            "entity_signal",
            "anomaly_signal",
            "network_signal",
            "time_step",
            "txid",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------------
    # Global rank.
    # ---------------------------------------------------------------

    result[
        "investigation_rank"
    ] = np.arange(
        1,
        len(result) + 1,
        dtype=np.int64,
    )

    # ---------------------------------------------------------------
    # Queue rank.
    #
    # Only ACTIVE alerts receive an active queue rank.
    # WATCH and INFORMATIONAL records remain globally ranked.
    # ---------------------------------------------------------------

    active_mask = result[
        "alert_status"
    ] == "ACTIVE"

    result[
        "active_alert_rank"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    active_count = int(
        active_mask.sum()
    )

    if active_count:

        result.loc[
            active_mask,
            "active_alert_rank",
        ] = np.arange(
            1,
            active_count + 1,
            dtype=np.int64,
        )

    # ---------------------------------------------------------------
    # Evidence summary and explanation.
    # ---------------------------------------------------------------

    result[
        "evidence_summary"
    ] = result.apply(
        build_evidence_summary,
        axis=1,
    )

    result[
        "alert_explanation"
    ] = result.apply(
        build_alert_explanation,
        axis=1,
    )

    # ---------------------------------------------------------------
    # Alert selection flag.
    #
    # This identifies the top TOP_ALERT_COUNT records within the
    # ACTIVE queue for the dashboard's initial alert page.
    # ---------------------------------------------------------------

    result[
        "top_alert_queue"
    ] = False

    top_active_index = (
        result.index[
            active_mask
        ][:TOP_ALERT_COUNT]
    )

    result.loc[
        top_active_index,
        "top_alert_queue",
    ] = True

    return result


# ============================================================================
# Output selection
# ============================================================================

def select_output_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select a clean alert-facing schema.
    """

    columns = [
        "investigation_rank",
        "active_alert_rank",
        "top_alert_queue",

        "txid",
        "time_step",

        "risk_score",
        "risk_level",
        "alert_priority",
        "alert_status",

        "ml_predicted_class",
        "ml_confidence",

        "anomaly_signal",

        "peeling_evidence_score",
        "mixing_evidence_score",
        "behavioral_signal",
        "behavioral_signal_count",
        "multiple_behavioral_signals",
        "behavioral_signal_agreement",

        "entity_signal",

        "network_signal",
        "network_observation_count",
        "network_unique_source_ip_count",
        "network_unique_destination_ip_count",
        "network_unique_endpoint_count",
        "network_temporal_evidence_level",
        "network_observation_available",

        "effective_evidence_channel_count",
        "evidence_agreement",

        "evidence_summary",
        "alert_explanation",
        "risk_explanation",
    ]

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Unable to construct M11.5 output; missing columns: "
            f"{missing}"
        )

    return df[
        columns
    ].copy()


# ============================================================================
# Validation
# ============================================================================

def validate_alerts(
    alerts: pd.DataFrame,
    source: pd.DataFrame,
) -> dict:
    """
    Validate M11.5 output integrity.
    """

    errors = []

    # ---------------------------------------------------------------
    # Coverage
    # ---------------------------------------------------------------

    if len(alerts) != len(source):

        errors.append(
            "M11.5 changed the total transaction row count."
        )

    if alerts[
        "txid"
    ].duplicated().any():

        errors.append(
            "M11.5 contains duplicate TXIDs."
        )

    if set(
        alerts["txid"]
    ) != set(
        source["txid"]
    ):

        errors.append(
            "M11.5 changed TXID coverage."
        )

    # ---------------------------------------------------------------
    # Ranking
    # ---------------------------------------------------------------

    expected_rank = np.arange(
        1,
        len(alerts) + 1,
        dtype=np.int64,
    )

    if not np.array_equal(
        alerts[
            "investigation_rank"
        ].to_numpy(
            dtype=np.int64
        ),
        expected_rank,
    ):

        errors.append(
            "Investigation ranks are not contiguous."
        )

    # ---------------------------------------------------------------
    # Risk score preservation
    # ---------------------------------------------------------------

    source_scores = (
        source[
            [
                "txid",
                "risk_score",
            ]
        ]
        .sort_values(
            "txid"
        )
        .reset_index(
            drop=True
        )
    )

    alert_scores = (
        alerts[
            [
                "txid",
                "risk_score",
            ]
        ]
        .sort_values(
            "txid"
        )
        .reset_index(
            drop=True
        )
    )

    if not np.allclose(
        source_scores[
            "risk_score"
        ].to_numpy(
            dtype=float
        ),
        alert_scores[
            "risk_score"
        ].to_numpy(
            dtype=float
        ),
        atol=1e-10,
    ):

        errors.append(
            "M11.5 changed M11.4 risk scores."
        )

    # ---------------------------------------------------------------
    # Priority consistency
    # ---------------------------------------------------------------

    expected_priority = (
        alerts[
            "risk_score"
        ]
        .map(
            derive_alert_priority
        )
    )

    if not (
        expected_priority.to_numpy()
        ==
        alerts[
            "alert_priority"
        ].to_numpy()
    ).all():

        errors.append(
            "Alert priority does not match risk score."
        )

    # ---------------------------------------------------------------
    # Active rank consistency
    # ---------------------------------------------------------------

    active = (
        alerts[
            "alert_status"
        ]
        == "ACTIVE"
    )

    active_ranks = alerts[
        "active_alert_rank"
    ]

    if active.any():

        expected_active_ranks = np.arange(
            1,
            int(active.sum()) + 1,
            dtype=np.int64,
        )

        actual_active_ranks = (
            active_ranks[
                active
            ]
            .astype(int)
            .to_numpy()
        )

        if not np.array_equal(
            actual_active_ranks,
            expected_active_ranks,
        ):

            errors.append(
                "Active alert ranks are not contiguous."
            )

    # ---------------------------------------------------------------
    # Score bounds
    # ---------------------------------------------------------------

    scores = alerts[
        "risk_score"
    ]

    if (
        scores < 0
    ).any() or (
        scores > 100
    ).any():

        errors.append(
            "Risk score outside [0,100]."
        )

    # ---------------------------------------------------------------
    # Evidence count
    # ---------------------------------------------------------------

    counts = alerts[
        "effective_evidence_channel_count"
    ]

    if (
        counts < 0
    ).any() or (
        counts > 5
    ).any():

        errors.append(
            "Evidence channel count outside [0,5]."
        )

    # ---------------------------------------------------------------
    # Top queue
    # ---------------------------------------------------------------

    top_queue_count = int(
        alerts[
            "top_alert_queue"
        ].sum()
    )

    expected_top_count = min(
        TOP_ALERT_COUNT,
        int(active.sum()),
    )

    if top_queue_count != expected_top_count:

        errors.append(
            "Top alert queue size is inconsistent."
        )

    # ---------------------------------------------------------------
    # No-network semantics
    # ---------------------------------------------------------------

    no_network = ~alerts[
        "network_observation_available"
    ]

    if not (
        alerts.loc[
            no_network,
            "network_signal",
        ]
        == 0
    ).all():

        errors.append(
            "Transactions without network observations "
            "have non-zero network signal."
        )

    if errors:

        raise AssertionError(
            "M11.5 validation failed: "
            + " | ".join(
                errors
            )
        )

    return {
        "status": "PASS",
        "rows": int(
            len(alerts)
        ),
        "unique_txids": int(
            alerts[
                "txid"
            ].nunique()
        ),
        "active_alert_count": int(
            active.sum()
        ),
        "watch_count": int(
            (
                alerts[
                    "alert_status"
                ]
                == "WATCH"
            ).sum()
        ),
        "informational_count": int(
            (
                alerts[
                    "alert_status"
                ]
                == "INFORMATIONAL"
            ).sum()
        ),
        "top_alert_queue_count": (
            top_queue_count
        ),
        "very_high_count": int(
            (
                alerts[
                    "alert_priority"
                ]
                == "VERY_HIGH"
            ).sum()
        ),
        "high_count": int(
            (
                alerts[
                    "alert_priority"
                ]
                == "HIGH"
            ).sum()
        ),
        "moderate_count": int(
            (
                alerts[
                    "alert_priority"
                ]
                == "MODERATE"
            ).sum()
        ),
        "guarded_count": int(
            (
                alerts[
                    "alert_priority"
                ]
                == "GUARDED"
            ).sum()
        ),
        "low_count": int(
            (
                alerts[
                    "alert_priority"
                ]
                == "LOW"
            ).sum()
        ),
        "highest_risk_score": float(
            alerts[
                "risk_score"
            ].max()
        ),
        "lowest_risk_score": float(
            alerts[
                "risk_score"
            ].min()
        ),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "M11.5 RANKED EXPLAINABLE ALERTS"
    )
    print("=" * 65)

    print(
        "Input:",
        INPUT_FILE,
    )

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------

    source = load_risk_scores()

    print()
    print(
        f"M11.4 risk-score rows: "
        f"{len(source):,}"
    )

    # ---------------------------------------------------------------
    # Build ranked queue
    # ---------------------------------------------------------------

    alerts_full = build_ranked_alerts(
        source
    )

    # ---------------------------------------------------------------
    # Select clean output
    # ---------------------------------------------------------------

    alerts = select_output_columns(
        alerts_full
    )

    # ---------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------

    print()
    print(
        "Validating ranked alert artifact..."
    )

    validation = validate_alerts(
        alerts,
        source,
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    alerts.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    output_sha256 = hashlib.sha256(
        OUTPUT_FILE.read_bytes()
    ).hexdigest()

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    report = {
        "module": "M11.5",
        "status": validation[
            "status"
        ],
        "purpose": (
            "Convert M11.4 unified risk scores into a deterministic "
            "ranked and explainable investigation queue."
        ),
        "score_semantics": (
            "Investigative priority. Not a probability of illicit "
            "activity and not proof of identity, ownership, "
            "authorship or criminal activity."
        ),
        "class_semantics": (
            "Numeric class IDs are preserved without assigning "
            "semantic meanings not established by the project."
        ),
        "ranking_order": [
            "risk_score descending",
            "effective_evidence_channel_count descending",
            "behavioral_signal descending",
            "entity_signal descending",
            "anomaly_signal descending",
            "network_signal descending",
            "time_step ascending",
            "txid ascending",
        ],
        "alert_thresholds": ALERT_THRESHOLDS,
        "top_alert_count": TOP_ALERT_COUNT,
        "validation": validation,
        "input": str(
            INPUT_FILE
        ),
        "output": str(
            OUTPUT_FILE
        ),
        "output_sha256": output_sha256,
        "network_coverage_note": (
            "Current network evidence remains based on the "
            "synthetic validation fixture and therefore has "
            "very low coverage."
        ),
        "identity_inference": False,
        "ownership_inference": False,
        "illicit_activity_inference": False,
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------

    print()
    print(
        "M11.5 RESULT"
    )
    print("-" * 65)

    print(
        f"OUTPUT ROWS: "
        f"{len(alerts):,}"
    )

    print(
        f"UNIQUE TXIDS: "
        f"{alerts['txid'].nunique():,}"
    )

    print(
        f"ACTIVE ALERTS: "
        f"{validation['active_alert_count']:,}"
    )

    print(
        f"WATCH: "
        f"{validation['watch_count']:,}"
    )

    print(
        f"INFORMATIONAL: "
        f"{validation['informational_count']:,}"
    )

    print(
        f"TOP ALERT QUEUE: "
        f"{validation['top_alert_queue_count']:,}"
    )

    print()
    print(
        "PRIORITY DISTRIBUTION:"
    )

    print(
        alerts[
            "alert_priority"
        ]
        .value_counts()
        .reindex(
            [
                "VERY_HIGH",
                "HIGH",
                "MODERATE",
                "GUARDED",
                "LOW",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print()
    print(
        "TOP 10 RANKED ALERTS:"
    )

    preview_columns = [
        "investigation_rank",
        "txid",
        "time_step",
        "risk_score",
        "alert_priority",
        "ml_predicted_class",
        "ml_confidence",
        "anomaly_signal",
        "behavioral_signal",
        "entity_signal",
        "network_signal",
        "effective_evidence_channel_count",
    ]

    print(
        alerts[
            preview_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "TOP ALERT EXPLANATION:"
    )

    print(
        alerts.iloc[
            0
        ][
            "alert_explanation"
        ]
    )

    print()
    print(
        "OUTPUT:",
        OUTPUT_FILE,
    )

    print(
        "REPORT:",
        REPORT_FILE,
    )

    print()
    print(
        "M11.5 COMPLETE"
    )


if __name__ == "__main__":
    main()