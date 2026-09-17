"""
M10.4 — Temporal Network Correlation

Purpose
-------
Derive conservative temporal/network-context evidence from the
M10.3 network ↔ blockchain correlation output.

Important limitation
--------------------
The Elliptic++ blockchain dataset provides `time_step`, which is an
ordered dataset-relative temporal index rather than a wall-clock
timestamp.

Therefore this module DOES NOT fabricate a conversion between
blockchain time_step and network wall-clock timestamps.

Instead, M10.4 analyzes temporal structure entirely within the
network observations associated with each blockchain transaction.

For each matched transaction, the module derives:

    - network observation count
    - first network observation
    - last network observation
    - observation span
    - source IP count
    - destination IP count
    - source/destination IP diversity
    - observation density
    - network observation ordering
    - repeated observation indicators
    - conservative temporal evidence score

The resulting score represents temporal/network contextual strength.
It does NOT represent:

    - identity
    - IP ownership
    - wallet ownership
    - transaction authorship
    - probability of illicit activity
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "derived"
    / "network_blockchain_correlations.parquet"
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
    / "temporal_network_correlation.parquet"
)

REPORT_FILE = (
    REPORT_DIR
    / "m10_4_temporal_network_correlation.json"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Observation-span normalization.

# A span of 60 seconds or more is treated as the upper reference point
# for the temporal-context component.
REFERENCE_SPAN_SECONDS = 60.0

# Number of observations used as the saturation point for observation
# density. More observations can still be reported, but the evidence
# component does not increase indefinitely.
REFERENCE_OBSERVATION_COUNT = 5


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Element-wise safe division.
    """

    result = numerator.astype(float) / denominator.astype(float)

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)


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
                text = str(int(numeric))

        except ValueError:
            pass

    return text


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_correlation_data(
    path: Path,
) -> pd.DataFrame:
    """
    Load and validate M10.3 output.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"M10.3 correlation output not found: {path}"
        )

    print(
        "Reading M10.3 correlation data:",
        path,
    )

    df = pd.read_parquet(path)

    required_columns = {
        "network_event_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "txid",
        "txid_match",
        "blockchain_time_step",
        "observation_source",
    }

    missing_columns = sorted(
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "M10.3 output is missing required columns: "
            f"{missing_columns}"
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
            "M10.3 output contains empty TXIDs."
        )

    if df["timestamp"].isna().any():
        raise ValueError(
            "M10.3 output contains invalid timestamps."
        )

    if df["network_event_id"].duplicated().any():
        raise ValueError(
            "M10.3 output contains duplicate "
            "network_event_id values."
        )

    return df


# ---------------------------------------------------------------------------
# Per-TX temporal statistics
# ---------------------------------------------------------------------------

def build_transaction_temporal_statistics(
    matched_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build temporal/network statistics for each matched blockchain TXID.

    Only observations with an exact TXID match are included.
    """

    if matched_df.empty:

        return pd.DataFrame(
            columns=[
                "txid",
                "network_observation_count",
                "network_first_observation",
                "network_last_observation",
                "network_observation_span_seconds",
                "network_source_ip_count",
                "network_destination_ip_count",
                "network_unique_endpoint_count",
            ]
        )

    rows = []

    for txid, group in matched_df.groupby(
        "txid",
        sort=False,
    ):

        group = group.sort_values(
            "timestamp"
        )

        first_timestamp = group[
            "timestamp"
        ].min()

        last_timestamp = group[
            "timestamp"
        ].max()

        span_seconds = (
            last_timestamp
            - first_timestamp
        ).total_seconds()

        source_ip_count = (
            group["src_ip"]
            .nunique()
        )

        destination_ip_count = (
            group["dst_ip"]
            .nunique()
        )

        endpoint_values = pd.concat(
            [
                group["src_ip"],
                group["dst_ip"],
            ],
            ignore_index=True,
        )

        unique_endpoint_count = (
            endpoint_values.nunique()
        )

        rows.append(
            {
                "txid": txid,
                "network_observation_count": int(
                    len(group)
                ),
                "network_first_observation": (
                    first_timestamp
                ),
                "network_last_observation": (
                    last_timestamp
                ),
                "network_observation_span_seconds": (
                    float(span_seconds)
                ),
                "network_source_ip_count": int(
                    source_ip_count
                ),
                "network_destination_ip_count": int(
                    destination_ip_count
                ),
                "network_unique_endpoint_count": int(
                    unique_endpoint_count
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Observation-level temporal context
# ---------------------------------------------------------------------------

def build_observation_context(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add temporal ordering information to every network observation.

    The ordering is calculated only from supplied network timestamps.
    """

    result = df.copy()

    result = result.sort_values(
        [
            "txid",
            "timestamp",
            "network_event_id",
        ]
    ).reset_index(
        drop=True
    )

    result[
        "network_observation_sequence"
    ] = (
        result.groupby(
            "txid"
        ).cumcount()
        + 1
    )

    result[
        "network_observations_for_tx"
    ] = (
        result.groupby(
            "txid"
        )["network_event_id"]
        .transform("count")
    )

    result[
        "network_is_first_observation"
    ] = (
        result[
            "network_observation_sequence"
        ]
        == 1
    )

    result[
        "network_is_last_observation"
    ] = (
        result[
            "network_observation_sequence"
        ]
        == result[
            "network_observations_for_tx"
        ]
    )

    # Time difference from the previous network observation for the
    # same TXID.
    result[
        "previous_network_observation_timestamp"
    ] = (
        result.groupby(
            "txid"
        )["timestamp"]
        .shift(1)
    )

    result[
        "network_gap_from_previous_seconds"
    ] = (
        result["timestamp"]
        - result[
            "previous_network_observation_timestamp"
        ]
    ).dt.total_seconds()

    result[
        "network_gap_from_previous_seconds"
    ] = (
        result[
            "network_gap_from_previous_seconds"
        ]
        .fillna(0.0)
    )

    result[
        "network_repeated_tx_observation"
    ] = (
        result[
            "network_observations_for_tx"
        ]
        > 1
    )

    return result


# ---------------------------------------------------------------------------
# Evidence scoring
# ---------------------------------------------------------------------------

def calculate_temporal_evidence(
    stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate a conservative temporal/network contextual evidence score.

    Components
    ----------
    1. Observation count
       More independent observations provide stronger network context,
       with saturation after REFERENCE_OBSERVATION_COUNT.

    2. Endpoint diversity
       Multiple distinct source/destination endpoints provide stronger
       network context than a single observation.

    3. Observation span
       A non-zero span demonstrates temporal persistence in the supplied
       network observations.

    The final score is bounded to [0, 1].

    This is an evidence/context score, not a probability.
    """

    result = stats.copy()

    # ---------------------------------------------------------------
    # Observation-count component
    # ---------------------------------------------------------------

    observation_component = (
        1.0
        - np.exp(
            -result[
                "network_observation_count"
            ].astype(float)
            / REFERENCE_OBSERVATION_COUNT
        )
    )

    # ---------------------------------------------------------------
    # Endpoint-diversity component
    # ---------------------------------------------------------------

    endpoint_component = (
        1.0
        - np.exp(
            -result[
                "network_unique_endpoint_count"
            ].astype(float)
            / 4.0
        )
    )

    # ---------------------------------------------------------------
    # Temporal-span component
    # ---------------------------------------------------------------

    span_component = np.minimum(
        result[
            "network_observation_span_seconds"
        ].astype(float)
        / REFERENCE_SPAN_SECONDS,
        1.0,
    )

    # ---------------------------------------------------------------
    # Combined contextual score
    # ---------------------------------------------------------------

    result[
        "temporal_observation_evidence"
    ] = (
        0.40
        * observation_component
        + 0.30
        * endpoint_component
        + 0.30
        * span_component
    )

    result[
        "temporal_observation_evidence"
    ] = (
        result[
            "temporal_observation_evidence"
        ]
        .clip(0.0, 1.0)
        .round(6)
    )

    # Evidence level is descriptive only.
    result[
        "temporal_evidence_level"
    ] = pd.cut(
        result[
            "temporal_observation_evidence"
        ],
        bins=[
            -0.000001,
            0.25,
            0.50,
            0.75,
            1.000001,
        ],
        labels=[
            "LOW",
            "MODERATE",
            "HIGH",
            "VERY_HIGH",
        ],
    ).astype(str)

    return result


# ---------------------------------------------------------------------------
# Build complete temporal correlation
# ---------------------------------------------------------------------------

def build_temporal_correlation(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Build the final M10.4 temporal correlation artifact.
    """

    # ---------------------------------------------------------------
    # Exact matches only
    # ---------------------------------------------------------------

    matched_df = df[
        df["txid_match"]
    ].copy()

    # ---------------------------------------------------------------
    # Transaction-level temporal statistics
    # ---------------------------------------------------------------

    stats = (
        build_transaction_temporal_statistics(
            matched_df
        )
    )

    if not stats.empty:

        stats = (
            calculate_temporal_evidence(
                stats
            )
        )

    # ---------------------------------------------------------------
    # Add observation-level context
    # ---------------------------------------------------------------

    result = build_observation_context(
        df
    )

    # ---------------------------------------------------------------
    # Merge transaction-level temporal statistics
    # ---------------------------------------------------------------

    result = result.merge(
        stats,
        on="txid",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Matched/unmatched handling
    # ---------------------------------------------------------------

    unmatched_mask = ~result[
        "txid_match"
    ]

    temporal_columns = [
        "network_observation_count",
        "network_source_ip_count",
        "network_destination_ip_count",
        "network_unique_endpoint_count",
        "network_observation_span_seconds",
        "temporal_observation_evidence",
    ]

    for column in temporal_columns:

        result.loc[
            unmatched_mask,
            column,
        ] = 0

    result.loc[
        unmatched_mask,
        "temporal_evidence_level",
    ] = "NONE"

    # ---------------------------------------------------------------
    # Network-relative temporal features
    # ---------------------------------------------------------------

    result[
        "network_observation_density_per_second"
    ] = safe_divide(
        result[
            "network_observation_count"
        ],
        result[
            "network_observation_span_seconds"
        ].replace(
            0,
            np.nan,
        ),
    )

    # For a single observation, density is not meaningful.
    result.loc[
        result[
            "network_observation_count"
        ]
        <= 1,
        "network_observation_density_per_second",
    ] = 0.0

    result[
        "network_temporal_context_available"
    ] = (
        result[
            "txid_match"
        ]
        & (
            result[
                "network_observation_count"
            ]
            > 0
        )
    )

    result[
        "network_multi_observation_context"
    ] = (
        result[
            "network_observation_count"
        ]
        > 1
    )

    # ---------------------------------------------------------------
    # Conservative interpretation
    # ---------------------------------------------------------------

    result[
        "temporal_correlation_interpretation"
    ] = np.where(
        result[
            "txid_match"
        ],
        (
            "Network observations are temporally "
            "contextualized for the exact matched TXID. "
            "This does not establish IP ownership, "
            "transaction authorship, or identity."
        ),
        (
            "No exact blockchain TXID match; "
            "no transaction-level temporal context "
            "is assigned."
        ),
    )

    # ---------------------------------------------------------------
    # Sort for deterministic output
    # ---------------------------------------------------------------

    result = result.sort_values(
        [
            "timestamp",
            "txid",
            "network_event_id",
        ]
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    matched_count = int(
        result[
            "txid_match"
        ].sum()
    )

    unmatched_count = int(
        (~result[
            "txid_match"
        ]).sum()
    )

    unique_matched_txids = int(
        result.loc[
            result["txid_match"],
            "txid",
        ].nunique()
    )

    multi_observation_txids = int(
        stats.loc[
            stats[
                "network_observation_count"
            ]
            > 1,
            "txid",
        ].nunique()
    ) if not stats.empty else 0

    report = {
        "module": "M10.4",
        "status": "PASS",
        "purpose": (
            "Conservative temporal/network contextualization "
            "of M10.3 exact TXID correlations."
        ),
        "input": str(
            INPUT_FILE
        ),
        "input_rows": int(
            len(df)
        ),
        "matched_network_observations": matched_count,
        "unmatched_network_observations": unmatched_count,
        "unique_matched_blockchain_txids": (
            unique_matched_txids
        ),
        "transactions_with_multiple_network_observations": (
            multi_observation_txids
        ),
        "wall_clock_blockchain_timestamp_available": False,
        "blockchain_time_step_converted_to_wall_clock": False,
        "fabricated_temporal_delta": False,
        "identity_inference_performed": False,
        "wallet_ownership_inference_performed": False,
        "transaction_authorship_inference_performed": False,
        "illicit_activity_inference_performed": False,
        "score_semantics": (
            "Contextual evidence score in [0,1], "
            "not a probability and not proof of illicit activity."
        ),
        "score_components": {
            "observation_count": 0.40,
            "endpoint_diversity": 0.30,
            "network_observation_span": 0.30,
        },
        "reference_parameters": {
            "reference_span_seconds": (
                REFERENCE_SPAN_SECONDS
            ),
            "reference_observation_count": (
                REFERENCE_OBSERVATION_COUNT
            ),
        },
        "notes": [
            (
                "M10.4 operates only on supplied network "
                "timestamps and exact TXID relationships."
            ),
            (
                "Elliptic++ time_step is retained as "
                "dataset-relative ordered context."
            ),
            (
                "No conversion from time_step to wall-clock "
                "time is performed."
            ),
            (
                "Temporal evidence does not establish IP "
                "ownership or transaction authorship."
            ),
            (
                "Current network observations are synthetic "
                "pipeline-validation data."
            ),
        ],
        "output": str(
            OUTPUT_FILE
        ),
    }

    return result, report


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
        "M10.4 TEMPORAL NETWORK CORRELATION"
    )
    print("=" * 60)

    print(
        "Input:",
        INPUT_FILE,
    )

    print(
        "Output:",
        OUTPUT_FILE,
    )

    # ---------------------------------------------------------------
    # Load M10.3 output
    # ---------------------------------------------------------------

    df = load_correlation_data(
        INPUT_FILE
    )

    print()
    print(
        f"M10.3 correlation rows: "
        f"{len(df):,}"
    )

    print(
        "Exact TXID matches:",
        int(
            df[
                "txid_match"
            ].sum()
        ),
    )

    # ---------------------------------------------------------------
    # Build temporal correlation
    # ---------------------------------------------------------------

    result, report = (
        build_temporal_correlation(
            df
        )
    )

    # ---------------------------------------------------------------
    # Integrity checks
    # ---------------------------------------------------------------

    # M10.4 must preserve the M10.3 row count.
    if len(result) != len(df):
        raise AssertionError(
            "M10.4 changed the number of "
            "network observations."
        )

    # Network event IDs must remain unique.
    if result[
        "network_event_id"
    ].duplicated().any():

        raise AssertionError(
            "M10.4 produced duplicate "
            "network_event_id values."
        )

    # TXID values must remain unchanged.
    if set(result["txid"]) != set(df["txid"]):
        raise AssertionError(
            "M10.4 changed the set of TXIDs."
        )

    # Evidence must remain bounded.
    evidence = result[
        "temporal_observation_evidence"
    ]

    if (
        evidence < 0
    ).any() or (
        evidence > 1
    ).any():

        raise AssertionError(
            "Temporal evidence score is outside [0,1]."
        )

    # Unmatched rows must have zero contextual evidence.
    unmatched_evidence = result.loc[
        ~result["txid_match"],
        "temporal_observation_evidence",
    ]

    if not (
        unmatched_evidence == 0
    ).all():

        raise AssertionError(
            "Unmatched observations received "
            "non-zero temporal evidence."
        )

    # No fabricated blockchain wall-clock time.
    if report[
        "blockchain_time_step_converted_to_wall_clock"
    ]:
        raise AssertionError(
            "Invalid temporal conversion flag."
        )

    # ---------------------------------------------------------------
    # Save output
    # ---------------------------------------------------------------

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    output_sha256 = hashlib.sha256(
        OUTPUT_FILE.read_bytes()
    ).hexdigest()

    report[
        "output_sha256"
    ] = output_sha256

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
        "MATCHED NETWORK OBSERVATIONS:",
        report[
            "matched_network_observations"
        ],
    )

    print(
        "UNMATCHED NETWORK OBSERVATIONS:",
        report[
            "unmatched_network_observations"
        ],
    )

    print(
        "UNIQUE MATCHED TXIDS:",
        report[
            "unique_matched_blockchain_txids"
        ],
    )

    print(
        "TXIDS WITH MULTIPLE NETWORK OBSERVATIONS:",
        report[
            "transactions_with_multiple_network_observations"
        ],
    )

    print(
        "BLOCKCHAIN TIME_STEP → WALL CLOCK: NO"
    )

    print(
        "FABRICATED TEMPORAL DELTA: NO"
    )

    print()
    print(
        "Temporal correlation preview:"
    )

    print("-" * 60)

    preview_columns = [
        "network_event_id",
        "txid",
        "txid_match",
        "blockchain_time_step",
        "timestamp",
        "network_observation_sequence",
        "network_observation_count",
        "network_source_ip_count",
        "network_destination_ip_count",
        "network_observation_span_seconds",
        "temporal_observation_evidence",
        "temporal_evidence_level",
    ]

    print(
        result[
            preview_columns
        ].to_string(
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
        "M10.4 COMPLETE"
    )


if __name__ == "__main__":
    main()