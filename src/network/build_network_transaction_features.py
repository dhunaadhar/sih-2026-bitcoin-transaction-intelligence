"""
M10.5 — Network Transaction Feature Integration

Purpose
-------
Convert M10.4 network observation-level correlation data into a
transaction-level network feature artifact aligned with the canonical
blockchain transaction dataset.

Design
------
M10.4 network observations
        ↓
Group by TXID
        ↓
Transaction-level network features
        ↓
Left join with canonical transactions
        ↓
Exactly one row per canonical TXID

Important limitations
---------------------
1. Current network observations are synthetic validation data.
2. No real network observations are fabricated.
3. Absence of an observation means "no supplied network observation",
   not proof that no network activity existed.
4. Network evidence does not establish IP ownership, wallet ownership,
   transaction authorship, identity, or illicit activity.
5. Elliptic++ time_step is retained only as dataset-relative context.
6. This module does not modify the validated M6/M7 ML feature set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

NETWORK_INPUT = (
    ROOT
    / "data"
    / "derived"
    / "temporal_network_correlation.parquet"
)

CANONICAL_INPUT = (
    ROOT
    / "data"
    / "canonical"
    / "canonical_transactions.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "derived"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "network"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "network_transaction_features.parquet"
)

REPORT_FILE = (
    REPORT_DIR
    / "m10_5_network_transaction_features.json"
)


# ---------------------------------------------------------------------------
# TXID normalization
# ---------------------------------------------------------------------------

def normalize_txid(value) -> str:
    """
    Normalize TXID representations consistently.

    Examples
    --------
    4304541
        -> "4304541"

    "4304541"
        -> "4304541"

    "4304541.0"
        -> "4304541"
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):

        try:
            numeric = float(text)

            if numeric.is_integer():
                text = str(int(numeric))

        except ValueError:
            pass

    return text


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def load_network_data() -> pd.DataFrame:
    """
    Load M10.4 temporal network correlation output.
    """

    if not NETWORK_INPUT.exists():
        raise FileNotFoundError(
            f"M10.4 output not found: {NETWORK_INPUT}"
        )

    print(
        "Reading M10.4 network correlation:",
        NETWORK_INPUT,
    )

    df = pd.read_parquet(
        NETWORK_INPUT
    )

    required_columns = {
        "network_event_id",
        "txid",
        "txid_match",
        "timestamp",
        "src_ip",
        "dst_ip",
        "observation_source",
        "network_observation_sequence",
        "network_observation_count",
        "network_source_ip_count",
        "network_destination_ip_count",
        "network_unique_endpoint_count",
        "network_observation_span_seconds",
        "temporal_observation_evidence",
        "temporal_evidence_level",
        "network_temporal_context_available",
        "network_multi_observation_context",
    }

    missing = sorted(
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "M10.4 output is missing required columns: "
            f"{missing}"
        )

    df = df.copy()

    df["txid"] = (
        df["txid"]
        .map(normalize_txid)
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    if df["txid"].eq("").any():
        raise ValueError(
            "M10.4 output contains empty TXIDs."
        )

    if df["timestamp"].isna().any():
        raise ValueError(
            "M10.4 output contains invalid timestamps."
        )

    if df["network_event_id"].duplicated().any():
        raise ValueError(
            "M10.4 output contains duplicate network_event_id values."
        )

    return df


def load_canonical_data() -> pd.DataFrame:
    """
    Load canonical blockchain transaction dataset.
    """

    if not CANONICAL_INPUT.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {CANONICAL_INPUT}"
        )

    print(
        "Reading canonical transactions:",
        CANONICAL_INPUT,
    )

    df = pd.read_parquet(
        CANONICAL_INPUT
    )

    required = {
        "txid",
        "time_step",
    }

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Canonical dataset is missing required columns: "
            f"{missing}"
        )

    df = df.copy()

    df["txid"] = (
        df["txid"]
        .map(normalize_txid)
    )

    if df["txid"].eq("").any():
        raise ValueError(
            "Canonical dataset contains empty TXIDs."
        )

    if df["txid"].duplicated().any():
        raise ValueError(
            "Canonical dataset contains duplicate TXIDs."
        )

    return df


# ---------------------------------------------------------------------------
# Transaction-level aggregation
# ---------------------------------------------------------------------------

def aggregate_network_features(
    network_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate M10.4 observation-level information to one row per TXID.

    Only exact TXID matches contribute network features.

    Unmatched network observations are retained separately in the M10.4
    artifact but do not create a blockchain transaction feature row.
    """

    matched = network_df[
        network_df["txid_match"]
    ].copy()

    if matched.empty:

        return pd.DataFrame(
            columns=[
                "txid",
                "network_observation_count",
                "network_unique_source_ip_count",
                "network_unique_destination_ip_count",
                "network_unique_endpoint_count",
                "network_observation_span_seconds",
                "network_first_observation",
                "network_last_observation",
                "network_unique_asn_count",
                "network_unique_country_count",
                "network_exact_txid_match",
                "network_temporal_context_available",
                "network_multi_observation_context",
                "network_temporal_evidence_score",
                "network_temporal_evidence_level",
            ]
        )

    rows = []

    for txid, group in matched.groupby(
        "txid",
        sort=False,
    ):

        group = group.sort_values(
            [
                "timestamp",
                "network_event_id",
            ]
        )

        # ---------------------------------------------------------------
        # IP diversity
        # ---------------------------------------------------------------

        source_ip_count = int(
            group[
                "src_ip"
            ].nunique()
        )

        destination_ip_count = int(
            group[
                "dst_ip"
            ].nunique()
        )

        endpoint_values = pd.concat(
            [
                group["src_ip"],
                group["dst_ip"],
            ],
            ignore_index=True,
        )

        unique_endpoint_count = int(
            endpoint_values.nunique()
        )

        # ---------------------------------------------------------------
        # Observation timing
        # ---------------------------------------------------------------

        first_observation = (
            group["timestamp"].min()
        )

        last_observation = (
            group["timestamp"].max()
        )

        span_seconds = float(
            (
                last_observation
                - first_observation
            ).total_seconds()
        )

        # ---------------------------------------------------------------
        # ASN diversity
        # ---------------------------------------------------------------

        if "asn" in group.columns:

            asn_count = int(
                group["asn"]
                .dropna()
                .astype(str)
                .nunique()
            )

        else:

            asn_count = 0

        # ---------------------------------------------------------------
        # Country diversity
        # ---------------------------------------------------------------

        if "geo_country" in group.columns:

            country_count = int(
                group["geo_country"]
                .dropna()
                .astype(str)
                .nunique()
            )

        else:

            country_count = 0

        # ---------------------------------------------------------------
        # Temporal evidence
        # ---------------------------------------------------------------

        evidence_values = (
            pd.to_numeric(
                group[
                    "temporal_observation_evidence"
                ],
                errors="coerce",
            )
            .fillna(0.0)
        )

        temporal_evidence_score = float(
            evidence_values.max()
        )

        evidence_levels = (
            group[
                "temporal_evidence_level"
            ]
            .dropna()
            .astype(str)
        )

        if not evidence_levels.empty:

            level_order = {
                "NONE": 0,
                "LOW": 1,
                "MODERATE": 2,
                "HIGH": 3,
                "VERY_HIGH": 4,
            }

            strongest_level = max(
                evidence_levels,
                key=lambda value: level_order.get(
                    value,
                    0,
                ),
            )

        else:

            strongest_level = "NONE"

        # ---------------------------------------------------------------
        # Observation count
        # ---------------------------------------------------------------

        observation_count = int(
            len(group)
        )

        # ---------------------------------------------------------------
        # Create transaction-level feature row
        # ---------------------------------------------------------------

        rows.append(
            {
                "txid": txid,

                "network_observation_count": (
                    observation_count
                ),

                "network_unique_source_ip_count": (
                    source_ip_count
                ),

                "network_unique_destination_ip_count": (
                    destination_ip_count
                ),

                "network_unique_endpoint_count": (
                    unique_endpoint_count
                ),

                "network_observation_span_seconds": (
                    span_seconds
                ),

                "network_first_observation": (
                    first_observation
                ),

                "network_last_observation": (
                    last_observation
                ),

                "network_unique_asn_count": (
                    asn_count
                ),

                "network_unique_country_count": (
                    country_count
                ),

                "network_exact_txid_match": True,

                "network_temporal_context_available": True,

                "network_multi_observation_context": (
                    observation_count > 1
                ),

                "network_temporal_evidence_score": (
                    temporal_evidence_score
                ),

                "network_temporal_evidence_level": (
                    strongest_level
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------------
# Canonical integration
# ---------------------------------------------------------------------------

def integrate_with_canonical(
    canonical_df: pd.DataFrame,
    network_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left join network transaction features onto the canonical dataset.

    This guarantees complete canonical transaction coverage.
    """

    result = canonical_df[
        [
            "txid",
            "time_step",
        ]
    ].copy()

    result = result.merge(
        network_features,
        on="txid",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # Explicit no-network-observation state
    # ---------------------------------------------------------------

    observation_count_missing = (
        result[
            "network_observation_count"
        ].isna()
    )

    numeric_zero_columns = [
        "network_observation_count",
        "network_unique_source_ip_count",
        "network_unique_destination_ip_count",
        "network_unique_endpoint_count",
        "network_observation_span_seconds",
        "network_unique_asn_count",
        "network_unique_country_count",
        "network_temporal_evidence_score",
    ]

    for column in numeric_zero_columns:

        result.loc[
            observation_count_missing,
            column,
        ] = 0

    # ---------------------------------------------------------------
    # Explicit boolean handling
    #
    # Use pandas BooleanDtype before fillna to avoid the deprecated
    # object-dtype downcasting behavior.
    # ---------------------------------------------------------------

    boolean_false_columns = [
        "network_exact_txid_match",
        "network_temporal_context_available",
        "network_multi_observation_context",
    ]

    for column in boolean_false_columns:

        result[column] = (
            result[column]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )

    # ---------------------------------------------------------------
    # Evidence level
    # ---------------------------------------------------------------

    result[
        "network_temporal_evidence_level"
    ] = (
        result[
            "network_temporal_evidence_level"
        ]
        .fillna("NONE")
        .astype(str)
    )

    # ---------------------------------------------------------------
    # Observation availability
    # ---------------------------------------------------------------

    result[
        "network_observation_available"
    ] = (
        result[
            "network_observation_count"
        ]
        > 0
    )

    result[
        "network_observation_available"
    ] = (
        result[
            "network_observation_available"
        ]
        .astype(bool)
    )

    # ---------------------------------------------------------------
    # Deterministic ordering
    # ---------------------------------------------------------------

    result = result.sort_values(
        [
            "time_step",
            "txid",
        ]
    ).reset_index(
        drop=True
    )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        "M10.5 NETWORK TRANSACTION FEATURE INTEGRATION"
    )
    print("=" * 65)

    print(
        "Network input:",
        NETWORK_INPUT,
    )

    print(
        "Canonical input:",
        CANONICAL_INPUT,
    )

    # ---------------------------------------------------------------
    # Load inputs
    # ---------------------------------------------------------------

    network_df = load_network_data()

    canonical_df = load_canonical_data()

    print()
    print(
        f"M10.4 network rows: "
        f"{len(network_df):,}"
    )

    print(
        f"Canonical transactions: "
        f"{len(canonical_df):,}"
    )

    # ---------------------------------------------------------------
    # Aggregate network information
    # ---------------------------------------------------------------

    network_features = (
        aggregate_network_features(
            network_df
        )
    )

    print()
    print(
        "Matched transaction feature rows:",
        f"{len(network_features):,}",
    )

    # ---------------------------------------------------------------
    # Integrate with canonical transactions
    # ---------------------------------------------------------------

    result = integrate_with_canonical(
        canonical_df,
        network_features,
    )

    # ---------------------------------------------------------------
    # Integrity checks
    # ---------------------------------------------------------------

    if len(result) != len(canonical_df):

        raise AssertionError(
            "M10.5 changed canonical transaction row count."
        )

    if result[
        "txid"
    ].duplicated().any():

        raise AssertionError(
            "M10.5 produced duplicate TXIDs."
        )

    if set(
        result["txid"]
    ) != set(
        canonical_df["txid"]
    ):

        raise AssertionError(
            "M10.5 changed canonical TXID coverage."
        )

    # Network observations cannot exceed source rows.
    if (
        result[
            "network_observation_count"
        ]
        > len(network_df)
    ).any():

        raise AssertionError(
            "Network observation count exceeds "
            "input observations."
        )

    # Evidence must remain within [0,1].
    evidence = result[
        "network_temporal_evidence_score"
    ]

    if (
        evidence < 0
    ).any() or (
        evidence > 1
    ).any():

        raise AssertionError(
            "Network temporal evidence is outside [0,1]."
        )

    # Transactions without observations must have zero evidence.
    no_network = ~result[
        "network_observation_available"
    ]

    if not (
        result.loc[
            no_network,
            "network_temporal_evidence_score",
        ]
        == 0
    ).all():

        raise AssertionError(
            "Transactions without network observations "
            "received non-zero network evidence."
        )

    # Transactions without observations must have NONE evidence level.
    if not (
        result.loc[
            no_network,
            "network_temporal_evidence_level",
        ]
        == "NONE"
    ).all():

        raise AssertionError(
            "Transactions without network observations "
            "received a non-NONE evidence level."
        )

    # Network availability must be consistent with observation count.
    expected_availability = (
        result[
            "network_observation_count"
        ]
        > 0
    )

    if not (
        result[
            "network_observation_available"
        ]
        == expected_availability
    ).all():

        raise AssertionError(
            "Network observation availability is inconsistent "
            "with network observation count."
        )

    # ---------------------------------------------------------------
    # Save parquet
    # ---------------------------------------------------------------

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    output_sha256 = hashlib.sha256(
        OUTPUT_FILE.read_bytes()
    ).hexdigest()

    # ---------------------------------------------------------------
    # Report statistics
    # ---------------------------------------------------------------

    network_available_count = int(
        result[
            "network_observation_available"
        ].sum()
    )

    network_unavailable_count = int(
        (
            ~result[
                "network_observation_available"
            ]
        ).sum()
    )

    exact_match_count = int(
        result[
            "network_exact_txid_match"
        ].sum()
    )

    multi_observation_count = int(
        result[
            "network_multi_observation_context"
        ].sum()
    )

    report = {
        "module": "M10.5",
        "status": "PASS",
        "purpose": (
            "Convert M10.4 network observations into "
            "transaction-level network features aligned "
            "with the canonical transaction dataset."
        ),
        "network_input": str(
            NETWORK_INPUT
        ),
        "canonical_input": str(
            CANONICAL_INPUT
        ),
        "network_input_rows": int(
            len(network_df)
        ),
        "canonical_rows": int(
            len(canonical_df)
        ),
        "output_rows": int(
            len(result)
        ),
        "network_transactions_with_observations": (
            network_available_count
        ),
        "canonical_transactions_without_supplied_network_observation": (
            network_unavailable_count
        ),
        "exact_txid_matched_transactions": (
            exact_match_count
        ),
        "transactions_with_multiple_network_observations": (
            multi_observation_count
        ),
        "output_unique_txids": int(
            result["txid"].nunique()
        ),
        "output_columns": int(
            len(result.columns)
        ),
        "canonical_txid_coverage": (
            float(
                result["txid"].nunique()
                / len(canonical_df)
            )
            if len(canonical_df)
            else 0.0
        ),
        "network_observation_coverage": (
            float(
                network_available_count
                / len(canonical_df)
            )
            if len(canonical_df)
            else 0.0
        ),
        "network_data_is_synthetic": True,
        "identity_inference_performed": False,
        "wallet_ownership_inference_performed": False,
        "transaction_authorship_inference_performed": False,
        "illicit_activity_inference_performed": False,
        "absence_semantics": (
            "No supplied network observation; not proof "
            "that no network activity existed."
        ),
        "score_semantics": (
            "Network temporal contextual evidence in [0,1]; "
            "not a probability or illicit-activity score."
        ),
        "notes": [
            (
                "M10.5 creates an independent network feature "
                "artifact and does not modify the validated "
                "M6/M7 temporal-safe ML feature set."
            ),
            (
                "Only exact TXID-correlated network observations "
                "contribute transaction-level network evidence."
            ),
            (
                "The current network dataset is synthetic and "
                "exists only for pipeline validation."
            ),
            (
                "Network absence is represented explicitly and "
                "must not be interpreted as proof of no activity."
            ),
            (
                "No IP ownership or real-world identity inference "
                "is performed."
            ),
        ],
        "output": str(
            OUTPUT_FILE
        ),
        "output_sha256": output_sha256,
    }

    # ---------------------------------------------------------------
    # Save report
    # ---------------------------------------------------------------

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
        "CANONICAL OUTPUT ROWS:",
        len(result),
    )

    print(
        "UNIQUE OUTPUT TXIDS:",
        result[
            "txid"
        ].nunique(),
    )

    print(
        "TRANSACTIONS WITH NETWORK OBSERVATIONS:",
        network_available_count,
    )

    print(
        "TRANSACTIONS WITHOUT SUPPLIED NETWORK OBSERVATION:",
        network_unavailable_count,
    )

    print(
        "EXACT TXID MATCHED TRANSACTIONS:",
        exact_match_count,
    )

    print(
        "TRANSACTIONS WITH MULTIPLE NETWORK OBSERVATIONS:",
        multi_observation_count,
    )

    print()
    print(
        "Network feature preview:"
    )

    print("-" * 65)

    preview_columns = [
        "txid",
        "time_step",
        "network_observation_available",
        "network_observation_count",
        "network_unique_source_ip_count",
        "network_unique_destination_ip_count",
        "network_unique_endpoint_count",
        "network_observation_span_seconds",
        "network_temporal_evidence_score",
        "network_temporal_evidence_level",
    ]

    print(
        result[
            preview_columns
        ]
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
        "M10.5 COMPLETE"
    )


if __name__ == "__main__":
    main()