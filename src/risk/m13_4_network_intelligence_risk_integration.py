"""
M13.4 — Network Intelligence Risk Integration

Purpose
-------
Integrate M13 VPN/proxy/Tor/hosting intelligence into the existing
M11 unified investigative priority score.

Important
---------
M13 does NOT create an additional sixth risk channel.

Instead, the existing M10 network channel is enriched with M13
IP-level intelligence. This prevents double-counting network evidence.

The original M11.4 artifact remains unchanged.

Score semantics
---------------
The resulting score is an INVESTIGATIVE PRIORITY SCORE.

It is NOT:
    - probability of illicit activity
    - proof of criminal activity
    - proof of identity
    - proof of wallet ownership
    - proof of IP ownership
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

M11_RISK_FILE = (
    ROOT
    / "data"
    / "derived"
    / "unified_risk_scores.parquet"
)

M13_NETWORK_FILE = (
    ROOT
    / "data"
    / "derived"
    / "transaction_network_intelligence.parquet"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "derived"
    / "unified_risk_scores_m13_integrated.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "risk"
    / "m13_4_network_intelligence_risk_integration.json"
)


# ============================================================================
# Existing M11 weights
# ============================================================================

WEIGHTS = {
    "ml": 0.15,
    "anomaly": 0.20,
    "behavioral": 0.30,
    "entity": 0.20,
    "network": 0.15,
}


AGREEMENT_BONUS = {
    0: 0.00,
    1: 0.00,
    2: 0.02,
    3: 0.04,
    4: 0.06,
    5: 0.08,
}


RISK_BANDS = [
    (80.0, "VERY_HIGH"),
    (60.0, "HIGH"),
    (40.0, "MODERATE"),
    (20.0, "GUARDED"),
    (0.0, "LOW"),
]


# ============================================================================
# Utilities
# ============================================================================

def normalize_txid(value) -> str:
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


def clamp01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    return values.clip(
        0.0,
        1.0,
    )


def risk_band(score: float) -> str:
    for threshold, label in RISK_BANDS:
        if score >= threshold:
            return label

    return "LOW"


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


# ============================================================================
# Loading
# ============================================================================

def load_m11() -> pd.DataFrame:
    if not M11_RISK_FILE.exists():
        raise FileNotFoundError(
            f"M11 unified risk artifact not found:\n{M11_RISK_FILE}"
        )

    df = pd.read_parquet(
        M11_RISK_FILE
    ).copy()

    required = {
        "txid",
        "time_step",
        "ml_signal",
        "anomaly_signal",
        "behavioral_signal",
        "entity_signal",
        "network_signal",
        "effective_evidence_channel_count",
        "risk_score",
        "risk_level",
    }

    require_columns(
        df,
        required,
        "M11 unified risk scores",
    )

    df["txid"] = df["txid"].map(
        normalize_txid
    )

    if df["txid"].eq("").any():
        raise ValueError(
            "M11 artifact contains empty TXIDs."
        )

    if df["txid"].duplicated().any():
        raise ValueError(
            "M11 artifact contains duplicate TXIDs."
        )

    return df


def load_m13() -> pd.DataFrame:
    if not M13_NETWORK_FILE.exists():
        raise FileNotFoundError(
            "M13 transaction network intelligence artifact "
            f"not found:\n{M13_NETWORK_FILE}"
        )

    df = pd.read_parquet(
        M13_NETWORK_FILE
    ).copy()

    required = {
        "txid",
        "network_intelligence_observation_count",
        "network_intelligence_available",
        "network_intelligence_classification",
        "network_intelligence_confidence",
        "vpn_evidence",
        "proxy_evidence",
        "tor_evidence",
        "hosting_evidence",
        "network_intelligence_evidence_score",
        "network_intelligence_evidence_level",
    }

    require_columns(
        df,
        required,
        "M13 transaction network intelligence",
    )

    df["txid"] = df["txid"].map(
        normalize_txid
    )

    if df["txid"].eq("").any():
        raise ValueError(
            "M13 artifact contains empty TXIDs."
        )

    if df["txid"].duplicated().any():
        raise ValueError(
            "M13 artifact contains duplicate TXIDs."
        )

    return df


# ============================================================================
# Integration
# ============================================================================

def merge_evidence(
    m11: pd.DataFrame,
    m13: pd.DataFrame,
) -> pd.DataFrame:
    m13_subset = m13[
        [
            "txid",
            "network_intelligence_observation_count",
            "network_intelligence_available",
            "network_intelligence_classification",
            "network_intelligence_confidence",
            "vpn_evidence",
            "proxy_evidence",
            "tor_evidence",
            "hosting_evidence",
            "network_intelligence_evidence_score",
            "network_intelligence_evidence_level",
        ]
    ].copy()

    result = m11.merge(
        m13_subset,
        on="txid",
        how="left",
        validate="one_to_one",
    )

    # M11 covers all transactions. M13 only covers transactions for
    # which network observations were correlated.
    result[
        "network_intelligence_observation_count"
    ] = (
        pd.to_numeric(
            result[
                "network_intelligence_observation_count"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    # Explicit nullable-boolean conversion avoids pandas FutureWarning
    # associated with fillna(False) on object dtype.
    result[
        "network_intelligence_available"
    ] = (
        result[
            "network_intelligence_available"
        ]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    result[
        "network_intelligence_classification"
    ] = (
        result[
            "network_intelligence_classification"
        ]
        .fillna("UNKNOWN")
        .astype(str)
    )

    result[
        "network_intelligence_confidence"
    ] = clamp01(
        result[
            "network_intelligence_confidence"
        ]
    )

    for column in [
        "vpn_evidence",
        "proxy_evidence",
        "tor_evidence",
        "hosting_evidence",
        "network_intelligence_evidence_score",
    ]:
        result[column] = clamp01(
            result[column]
        )

    result[
        "network_intelligence_evidence_level"
    ] = (
        result[
            "network_intelligence_evidence_level"
        ]
        .fillna("NONE")
        .astype(str)
    )

    return result


def build_enriched_network_signal(
    result: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace the existing M10 network signal with the enriched
    M13 signal when actual IP intelligence is available.

    M10 network correlation remains the fallback.

    This is replacement, NOT addition, so the network channel
    is not double-counted.
    """

    result = result.copy()

    m10_signal = clamp01(
        result[
            "network_signal"
        ]
    )

    m13_signal = clamp01(
        result[
            "network_intelligence_evidence_score"
        ]
    )

    intelligence_available = result[
        "network_intelligence_available"
    ]

    result[
        "m13_original_m10_network_signal"
    ] = m10_signal

    result[
        "m13_enriched_network_signal"
    ] = np.where(
        intelligence_available,
        np.maximum(
            m10_signal,
            m13_signal,
        ),
        m10_signal,
    )

    result[
        "network_signal"
    ] = result[
        "m13_enriched_network_signal"
    ]

    return result


# ============================================================================
# Recalculate unified score
# ============================================================================

def recalculate_score(
    result: pd.DataFrame,
) -> pd.DataFrame:
    result = result.copy()

    signal_columns = [
        "ml_signal",
        "anomaly_signal",
        "behavioral_signal",
        "entity_signal",
        "network_signal",
    ]

    for column in signal_columns:
        result[column] = clamp01(
            result[column]
        )

    # ------------------------------------------------------------------------
    # Recalculate channel availability.
    #
    # M13 does not create an additional channel.
    # The network channel remains one channel.
    # ------------------------------------------------------------------------

    result[
        "ml_evidence_available"
    ] = (
        result["ml_signal"] > 0
    )

    result[
        "anomaly_evidence_available"
    ] = (
        result["anomaly_signal"] > 0
    )

    result[
        "behavioral_evidence_available"
    ] = (
        result["behavioral_signal"] > 0
    )

    result[
        "entity_evidence_available"
    ] = (
        result["entity_signal"] > 0
    )

    result[
        "network_evidence_available"
    ] = (
        result["network_signal"] > 0
    )

    availability_columns = [
        "ml_evidence_available",
        "anomaly_evidence_available",
        "behavioral_evidence_available",
        "entity_evidence_available",
        "network_evidence_available",
    ]

    result[
        "effective_evidence_channel_count"
    ] = (
        result[
            availability_columns
        ]
        .sum(axis=1)
        .astype(int)
    )

    # ------------------------------------------------------------------------
    # Weighted evidence score
    # ------------------------------------------------------------------------

    result[
        "weighted_evidence_score"
    ] = (
        result["ml_signal"]
        * WEIGHTS["ml"]
        +
        result["anomaly_signal"]
        * WEIGHTS["anomaly"]
        +
        result["behavioral_signal"]
        * WEIGHTS["behavioral"]
        +
        result["entity_signal"]
        * WEIGHTS["entity"]
        +
        result["network_signal"]
        * WEIGHTS["network"]
    )

    # ------------------------------------------------------------------------
    # Agreement bonus
    # ------------------------------------------------------------------------

    result[
        "evidence_agreement_bonus"
    ] = (
        result[
            "effective_evidence_channel_count"
        ]
        .map(
            AGREEMENT_BONUS
        )
        .fillna(
            AGREEMENT_BONUS[5]
        )
    )

    result[
        "agreement_adjusted_score"
    ] = (
        result[
            "weighted_evidence_score"
        ]
        *
        (
            1.0
            +
            result[
                "evidence_agreement_bonus"
            ]
        )
    )

    result[
        "risk_score"
    ] = (
        result[
            "agreement_adjusted_score"
        ]
        .clip(
            0.0,
            1.0,
        )
        * 100.0
    )

    result[
        "risk_level"
    ] = result[
        "risk_score"
    ].map(
        risk_band
    )

    return result


# ============================================================================
# Explanations
# ============================================================================

def build_network_explanation(
    row: pd.Series,
) -> str:
    if not bool(
        row[
            "network_intelligence_available"
        ]
    ):
        return (
            "No IP-level VPN/proxy/Tor intelligence was "
            "available for this transaction."
        )

    classification = str(
        row[
            "network_intelligence_classification"
        ]
    )

    confidence = float(
        row[
            "network_intelligence_confidence"
        ]
    )

    evidence_level = str(
        row[
            "network_intelligence_evidence_level"
        ]
    )

    observation_count = int(
        row[
            "network_intelligence_observation_count"
        ]
    )

    return (
        f"IP-level network intelligence classified the "
        f"observed network context as {classification} "
        f"with source confidence {confidence:.3f}. "
        f"Network evidence level: {evidence_level}. "
        f"Correlated observations: {observation_count}. "
        f"This classification does not establish identity, "
        f"ownership, intent, or illicit activity."
    )


def build_integrated_explanation(
    row: pd.Series,
) -> str:
    network_text = build_network_explanation(
        row
    )

    m11_score = float(
        row[
            "m11_original_risk_score"
        ]
    )

    integrated_score = float(
        row[
            "risk_score"
        ]
    )

    delta = integrated_score - m11_score

    if abs(delta) < 1e-9:
        score_change = (
            "The M13 integration did not change the "
            "numeric priority score."
        )
    elif delta > 0:
        score_change = (
            f"The enriched network evidence increased "
            f"the priority score by {delta:.4f} points."
        )
    else:
        score_change = (
            f"The enriched network evidence changed "
            f"the priority score by {delta:.4f} points."
        )

    return (
        f"{network_text} "
        f"{score_change} "
        f"The final score remains an investigative "
        f"priority score rather than a probability of "
        f"illicit activity."
    )


def add_explanations(
    result: pd.DataFrame,
) -> pd.DataFrame:
    result = result.copy()

    result[
        "m13_network_explanation"
    ] = result.apply(
        build_network_explanation,
        axis=1,
    )

    result[
        "risk_explanation"
    ] = result.apply(
        build_integrated_explanation,
        axis=1,
    )

    return result


# ============================================================================
# Validation
# ============================================================================

def validate(
    m11: pd.DataFrame,
    m13: pd.DataFrame,
    result: pd.DataFrame,
) -> dict:
    checks = {}

    checks[
        "m11_row_count_preserved"
    ] = (
        len(result) == len(m11)
    )

    checks[
        "m11_txid_coverage_preserved"
    ] = (
        set(result["txid"])
        == set(m11["txid"])
    )

    checks[
        "m13_txids_unique"
    ] = (
        m13["txid"].nunique()
        == len(m13)
    )

    checks[
        "integrated_txids_unique"
    ] = (
        result["txid"].nunique()
        == len(result)
    )

    checks[
        "risk_score_finite"
    ] = bool(
        np.isfinite(
            result[
                "risk_score"
            ].to_numpy(
                dtype=float
            )
        ).all()
    )

    checks[
        "risk_score_bounded"
    ] = bool(
        (
            result["risk_score"]
            >= 0
        ).all()
        and
        (
            result["risk_score"]
            <= 100
        ).all()
    )

    checks[
        "network_signal_bounded"
    ] = bool(
        (
            result["network_signal"]
            >= 0
        ).all()
        and
        (
            result["network_signal"]
            <= 1
        ).all()
    )

    checks[
        "network_intelligence_evidence_bounded"
    ] = bool(
        (
            result[
                "network_intelligence_evidence_score"
            ]
            >= 0
        ).all()
        and
        (
            result[
                "network_intelligence_evidence_score"
            ]
            <= 1
        ).all()
    )

    # Transactions without M13 intelligence must retain the
    # original M10 network signal.
    no_m13 = ~result[
        "network_intelligence_available"
    ]

    checks[
        "no_m13_fallback_preserved"
    ] = bool(
        np.allclose(
            result.loc[
                no_m13,
                "network_signal",
            ].to_numpy(
                dtype=float
            ),
            result.loc[
                no_m13,
                "m13_original_m10_network_signal",
            ].to_numpy(
                dtype=float
            ),
            atol=1e-12,
        )
    )

    # M13 remains one network channel.
    checks[
        "maximum_evidence_channels_is_five"
    ] = bool(
        (
            result[
                "effective_evidence_channel_count"
            ]
            <= 5
        ).all()
    )

    # Confirm M13 does not become a separate sixth channel.
    checks[
        "m13_not_separate_channel"
    ] = (
        "m13_network_channel_available"
        not in result.columns
    )

    # Confirm network signal uses M13 where intelligence exists.
    m13_available = result[
        "network_intelligence_available"
    ]

    checks[
        "m13_replaces_or_enriches_network_channel"
    ] = bool(
        (
            result.loc[
                m13_available,
                "network_signal",
            ]
            >=
            result.loc[
                m13_available,
                "network_intelligence_evidence_score",
            ]
            - 1e-12
        ).all()
    )

    # Original M11 score is retained.
    checks[
        "original_m11_score_preserved"
    ] = bool(
        np.isfinite(
            result[
                "m11_original_risk_score"
            ].to_numpy(
                dtype=float
            )
        ).all()
    )

    passed = all(
        checks.values()
    )

    return {
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "checks": checks,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print()
    print(
        "=" * 72
    )
    print(
        "M13.4 NETWORK INTELLIGENCE RISK INTEGRATION"
    )
    print(
        "=" * 72
    )

    print(
        "Loading M11 unified risk scores..."
    )

    m11 = load_m11()

    print(
        f"M11 rows: {len(m11):,}"
    )

    print(
        "Loading M13 transaction network intelligence..."
    )

    m13 = load_m13()

    print(
        f"M13 rows: {len(m13):,}"
    )

    print()
    print(
        "Merging M13 network intelligence..."
    )

    result = merge_evidence(
        m11,
        m13,
    )

    print(
        "Building enriched network channel..."
    )

    result = build_enriched_network_signal(
        result
    )

    # Preserve the original M11 score before recalculation.
    result[
        "m11_original_risk_score"
    ] = result[
        "risk_score"
    ].astype(float)

    result[
        "m11_original_risk_level"
    ] = result[
        "risk_level"
    ].astype(str)

    print(
        "Recalculating unified priority score..."
    )

    result = recalculate_score(
        result
    )

    result = add_explanations(
        result
    )

    print()
    print(
        "Validating M13.4 output..."
    )

    validation = validate(
        m11,
        m13,
        result,
    )

    for name, passed in validation[
        "checks"
    ].items():
        print(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if validation["status"] != "PASS":
        raise RuntimeError(
            "M13.4 validation failed."
        )

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    output_sha256 = hashlib.sha256(
        OUTPUT_FILE.read_bytes()
    ).hexdigest()

    m13_available_count = int(
        result[
            "network_intelligence_available"
        ].sum()
    )

    vpn_count = int(
        (
            result[
                "vpn_evidence"
            ]
            > 0
        ).sum()
    )

    proxy_count = int(
        (
            result[
                "proxy_evidence"
            ]
            > 0
        ).sum()
    )

    tor_count = int(
        (
            result[
                "tor_evidence"
            ]
            > 0
        ).sum()
    )

    hosting_count = int(
        (
            result[
                "hosting_evidence"
            ]
            > 0
        ).sum()
    )

    changed_score_count = int(
        (
            np.abs(
                result[
                    "risk_score"
                ]
                -
                result[
                    "m11_original_risk_score"
                ]
            )
            > 1e-9
        ).sum()
    )

    report = {
        "milestone": "M13.4",
        "component": (
            "network_intelligence_risk_integration"
        ),
        "status": "PASS",
        "methodology": {
            "m11_original_artifact_preserved": True,
            "m13_replaces_m10_network_channel": True,
            "additional_sixth_channel": False,
            "network_double_counting": False,
            "identity_inference": False,
            "illicitness_probability": False,
            "offline_only": True,
        },
        "inputs": {
            "m11_risk_file": str(
                M11_RISK_FILE
            ),
            "m13_network_file": str(
                M13_NETWORK_FILE
            ),
            "m11_rows": int(
                len(m11)
            ),
            "m13_rows": int(
                len(m13)
            ),
        },
        "outputs": {
            "rows": int(
                len(result)
            ),
            "unique_txids": int(
                result["txid"].nunique()
            ),
            "transactions_with_m13_intelligence": (
                m13_available_count
            ),
            "vpn_transactions": vpn_count,
            "proxy_transactions": proxy_count,
            "tor_transactions": tor_count,
            "hosting_datacenter_transactions": hosting_count,
            "risk_scores_changed": changed_score_count,
            "mean_original_m11_score": float(
                result[
                    "m11_original_risk_score"
                ].mean()
            ),
            "mean_integrated_score": float(
                result[
                    "risk_score"
                ].mean()
            ),
            "max_original_m11_score": float(
                result[
                    "m11_original_risk_score"
                ].max()
            ),
            "max_integrated_score": float(
                result[
                    "risk_score"
                ].max()
            ),
        },
        "weights": WEIGHTS,
        "agreement_bonus": AGREEMENT_BONUS,
        "validation": validation,
        "limitations": [
            (
                "The current network observations originate "
                "from the synthetic M10 validation fixture."
            ),
            (
                "VPN/proxy/Tor classifications depend on the "
                "quality and temporal validity of the offline "
                "intelligence feed."
            ),
            (
                "Network classification does not establish "
                "real-world identity, ownership, intent, or "
                "illicit activity."
            ),
        ],
        "output": str(
            OUTPUT_FILE
        ),
        "output_sha256": output_sha256,
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 72
    )
    print(
        "M13.4 RESULT"
    )
    print(
        "=" * 72
    )

    print(
        f"OUTPUT ROWS: "
        f"{len(result):,}"
    )

    print(
        f"TRANSACTIONS WITH M13 INTELLIGENCE: "
        f"{m13_available_count:,}"
    )

    print(
        f"VPN TRANSACTIONS: "
        f"{vpn_count:,}"
    )

    print(
        f"PROXY TRANSACTIONS: "
        f"{proxy_count:,}"
    )

    print(
        f"TOR TRANSACTIONS: "
        f"{tor_count:,}"
    )

    print(
        f"HOSTING/DATACENTER TRANSACTIONS: "
        f"{hosting_count:,}"
    )

    print(
        f"RISK SCORES CHANGED: "
        f"{changed_score_count:,}"
    )

    print(
        f"MEAN M11 SCORE: "
        f"{result['m11_original_risk_score'].mean():.6f}"
    )

    print(
        f"MEAN INTEGRATED SCORE: "
        f"{result['risk_score'].mean():.6f}"
    )

    print()
    print(
        "TOP 10 INTEGRATED PRIORITY TRANSACTIONS:"
    )

    columns = [
        "txid",
        "time_step",
        "risk_score",
        "risk_level",
        "m11_original_risk_score",
        "network_intelligence_classification",
        "network_intelligence_confidence",
        "vpn_evidence",
        "proxy_evidence",
        "tor_evidence",
        "hosting_evidence",
        "effective_evidence_channel_count",
    ]

    print(
        result[
            columns
        ]
        .sort_values(
            [
                "risk_score",
                "network_intelligence_confidence",
                "txid",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Output:",
        OUTPUT_FILE,
    )

    print(
        "Report:",
        REPORT_FILE,
    )

    print()
    print(
        "M13.4 NETWORK INTELLIGENCE RISK INTEGRATION COMPLETE"
    )


if __name__ == "__main__":
    main()