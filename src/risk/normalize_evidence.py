"""
M11.3 — Evidence Signal Normalization

Purpose
-------
Normalize independent investigative evidence channels into bounded
[0, 1] signals before unified risk/evidence fusion.

Evidence channels
-----------------
1. Peeling-chain evidence          [0, 1]
2. Mixing-pattern evidence         [0, 1]
3. Combined behavioral evidence    [0, 1]
4. Temporal entity/graph evidence  [0, 1]
5. Network evidence                [0, 1]

M7 supervised classification and M8 anomaly detection are deliberately
not fabricated here. Their transaction-level inference outputs will be
attached in a later M11 stage.

Important semantic constraint
------------------------------
These signals represent structural/statistical investigative evidence.
They are NOT probabilities of illicit activity, identity, ownership,
criminality, or guilt.
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

BEHAVIORAL_INPUT = (
    ROOT
    / "data"
    / "derived"
    / "behavioral_evidence.parquet"
)

ENTITY_INPUT = (
    ROOT
    / "data"
    / "derived"
    / "temporal_transaction_entity_evidence.parquet"
)

NETWORK_INPUT = (
    ROOT
    / "data"
    / "derived"
    / "network_transaction_features.parquet"
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
    / "normalized_evidence.parquet"
)

REPORT_FILE = (
    REPORT_DIR
    / "m11_3_signal_normalization.json"
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPSILON = 1e-12


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    """
    Validate required columns.
    """

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            f"{missing}"
        )


def bounded_unit_interval(
    series: pd.Series,
) -> pd.Series:
    """
    Convert numeric values into a finite [0,1] series.

    Existing evidence scores are expected to already be bounded.
    This function provides defensive clipping and NaN handling.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    values = values.fillna(0.0)

    return values.clip(
        lower=0.0,
        upper=1.0,
    )


def robust_percentile_signal(
    series: pd.Series,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.Series:
    """
    Convert an unbounded non-negative structural feature into [0,1]
    using robust empirical quantile bounds.

    Values below the lower quantile map to 0.
    Values above the upper quantile map to 1.

    This is used only for temporal entity evidence where the original
    graph features have heterogeneous scales.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    values = values.fillna(0.0)

    if values.empty:
        return pd.Series(
            0.0,
            index=series.index,
            dtype=float,
        )

    low = float(
        values.quantile(
            lower_quantile
        )
    )

    high = float(
        values.quantile(
            upper_quantile
        )
    )

    if not np.isfinite(low):
        low = 0.0

    if not np.isfinite(high):
        high = low

    if high <= low + EPSILON:

        # Constant signal.
        if high > 0:
            return pd.Series(
                1.0,
                index=series.index,
                dtype=float,
            )

        return pd.Series(
            0.0,
            index=series.index,
            dtype=float,
        )

    normalized = (
        (values - low)
        / (high - low)
    )

    return normalized.clip(
        lower=0.0,
        upper=1.0,
    )


# ---------------------------------------------------------------------------
# Load behavioral evidence
# ---------------------------------------------------------------------------

def load_behavioral() -> pd.DataFrame:
    """
    Load M9 behavioral evidence.
    """

    if not BEHAVIORAL_INPUT.exists():
        raise FileNotFoundError(
            f"Behavioral evidence not found: "
            f"{BEHAVIORAL_INPUT}"
        )

    df = pd.read_parquet(
        BEHAVIORAL_INPUT
    ).copy()

    require_columns(
        df,
        {
            "txid",
            "peeling_evidence_score",
            "mixing_evidence_score",
            "behavioral_evidence_score",
            "behavioral_signal_count",
            "behavioral_signal_agreement",
            "multiple_behavioral_signals",
            "peeling_mixing_interaction",
        },
        "behavioral_evidence",
    )

    df["txid"] = (
        df["txid"]
        .map(normalize_txid)
    )

    if df["txid"].eq("").any():
        raise ValueError(
            "Behavioral evidence contains empty TXIDs."
        )

    if df["txid"].duplicated().any():
        raise ValueError(
            "Behavioral evidence contains duplicate TXIDs."
        )

    return df


# ---------------------------------------------------------------------------
# Load entity evidence
# ---------------------------------------------------------------------------

def load_entity() -> pd.DataFrame:
    """
    Load M6 temporal-safe transaction/entity evidence.
    """

    if not ENTITY_INPUT.exists():
        raise FileNotFoundError(
            f"Temporal entity evidence not found: "
            f"{ENTITY_INPUT}"
        )

    df = pd.read_parquet(
        ENTITY_INPUT
    ).copy()

    require_columns(
        df,
        {
            "txid",
            "time_step",
            "input_repeated_cosponsor_count",
            "input_repeated_cosponsor_ratio",
            "output_repeated_cosponsor_count",
            "output_repeated_cosponsor_ratio",
            "input_max_repeated_cosponsor_degree",
            "output_max_repeated_cosponsor_degree",
            "input_max_shared_transaction_count",
            "output_max_shared_transaction_count",
            "input_distinct_clusters_t1",
            "output_distinct_clusters_t1",
            "input_max_cluster_size_t1",
            "output_max_cluster_size_t1",
            "entity_input_address_count",
            "entity_output_address_count",
        },
        "temporal_transaction_entity_evidence",
    )

    df["txid"] = (
        df["txid"]
        .map(normalize_txid)
    )

    if df["txid"].eq("").any():
        raise ValueError(
            "Temporal entity evidence contains empty TXIDs."
        )

    if df["txid"].duplicated().any():
        raise ValueError(
            "Temporal entity evidence contains duplicate TXIDs."
        )

    return df


# ---------------------------------------------------------------------------
# Load network evidence
# ---------------------------------------------------------------------------

def load_network() -> pd.DataFrame:
    """
    Load M10 transaction-level network features.
    """

    if not NETWORK_INPUT.exists():
        raise FileNotFoundError(
            f"Network transaction features not found: "
            f"{NETWORK_INPUT}"
        )

    df = pd.read_parquet(
        NETWORK_INPUT
    ).copy()

    require_columns(
        df,
        {
            "txid",
            "network_observation_count",
            "network_temporal_evidence_score",
            "network_observation_available",
            "network_exact_txid_match",
            "network_multi_observation_context",
        },
        "network_transaction_features",
    )

    df["txid"] = (
        df["txid"]
        .map(normalize_txid)
    )

    if df["txid"].eq("").any():
        raise ValueError(
            "Network features contain empty TXIDs."
        )

    if df["txid"].duplicated().any():
        raise ValueError(
            "Network features contain duplicate TXIDs."
        )

    return df


# ---------------------------------------------------------------------------
# Build entity signal
# ---------------------------------------------------------------------------

def build_entity_signal(
    entity: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Construct a causal temporal entity evidence signal.

    The signal combines several historical graph indicators:

    - repeated co-sponsor ratio
    - repeated co-sponsor degree
    - shared transaction count
    - distinct historical cluster count
    - historical cluster size
    - historically observed input/output address count

    Each component is independently normalized and then combined.

    No globally computed M5 graph feature is used.
    """

    df = entity[
        [
            "txid",
            "time_step",
            "input_repeated_cosponsor_ratio",
            "output_repeated_cosponsor_ratio",
            "input_max_repeated_cosponsor_degree",
            "output_max_repeated_cosponsor_degree",
            "input_max_shared_transaction_count",
            "output_max_shared_transaction_count",
            "input_distinct_clusters_t1",
            "output_distinct_clusters_t1",
            "input_max_cluster_size_t1",
            "output_max_cluster_size_t1",
            "entity_input_address_count",
            "entity_output_address_count",
        ]
    ].copy()

    # ---------------------------------------------------------------
    # Component signals
    # ---------------------------------------------------------------

    df[
        "entity_repeated_cosponsor_signal"
    ] = (
        (
            bounded_unit_interval(
                df[
                    "input_repeated_cosponsor_ratio"
                ]
            )
            +
            bounded_unit_interval(
                df[
                    "output_repeated_cosponsor_ratio"
                ]
            )
        )
        / 2.0
    )

    repeated_degree_input = robust_percentile_signal(
        df[
            "input_max_repeated_cosponsor_degree"
        ]
    )

    repeated_degree_output = robust_percentile_signal(
        df[
            "output_max_repeated_cosponsor_degree"
        ]
    )

    df[
        "entity_repeated_degree_signal"
    ] = (
        repeated_degree_input
        + repeated_degree_output
    ) / 2.0

    shared_input = robust_percentile_signal(
        df[
            "input_max_shared_transaction_count"
        ]
    )

    shared_output = robust_percentile_signal(
        df[
            "output_max_shared_transaction_count"
        ]
    )

    df[
        "entity_shared_transaction_signal"
    ] = (
        shared_input
        + shared_output
    ) / 2.0

    cluster_input = robust_percentile_signal(
        df[
            "input_distinct_clusters_t1"
        ]
    )

    cluster_output = robust_percentile_signal(
        df[
            "output_distinct_clusters_t1"
        ]
    )

    df[
        "entity_cluster_diversity_signal"
    ] = (
        cluster_input
        + cluster_output
    ) / 2.0

    cluster_size_input = robust_percentile_signal(
        df[
            "input_max_cluster_size_t1"
        ]
    )

    cluster_size_output = robust_percentile_signal(
        df[
            "output_max_cluster_size_t1"
        ]
    )

    df[
        "entity_cluster_size_signal"
    ] = (
        cluster_size_input
        + cluster_size_output
    ) / 2.0

    address_history_input = robust_percentile_signal(
        df[
            "entity_input_address_count"
        ]
    )

    address_history_output = robust_percentile_signal(
        df[
            "entity_output_address_count"
        ]
    )

    df[
        "entity_historical_address_signal"
    ] = (
        address_history_input
        + address_history_output
    ) / 2.0

    # ---------------------------------------------------------------
    # Unified entity signal
    #
    # Equal component weighting keeps the construction transparent
    # and prevents a single heterogeneous graph feature from
    # dominating the score.
    # ---------------------------------------------------------------

    entity_components = [
        "entity_repeated_cosponsor_signal",
        "entity_repeated_degree_signal",
        "entity_shared_transaction_signal",
        "entity_cluster_diversity_signal",
        "entity_cluster_size_signal",
        "entity_historical_address_signal",
    ]

    df[
        "entity_signal"
    ] = (
        df[
            entity_components
        ]
        .mean(axis=1)
        .clip(0.0, 1.0)
    )

    component_metadata = {
        "components": entity_components,
        "method": (
            "Robust empirical quantile normalization followed "
            "by equal-weight component averaging."
        ),
        "quantiles": {
            "lower": 0.01,
            "upper": 0.99,
        },
        "temporal_safety": (
            "Uses M6 temporal-safe entity evidence only."
        ),
    }

    return df[
        [
            "txid",
            "time_step",
            "entity_repeated_cosponsor_signal",
            "entity_repeated_degree_signal",
            "entity_shared_transaction_signal",
            "entity_cluster_diversity_signal",
            "entity_cluster_size_signal",
            "entity_historical_address_signal",
            "entity_signal",
        ]
    ], component_metadata


# ---------------------------------------------------------------------------
# Build complete normalized evidence
# ---------------------------------------------------------------------------

def build_normalized_evidence(
    behavioral: pd.DataFrame,
    entity_signal: pd.DataFrame,
    network: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one transaction-level normalized evidence table.
    """

    # ---------------------------------------------------------------
    # Behavioral signals
    # ---------------------------------------------------------------

    behavioral_out = behavioral[
        [
            "txid",
            "time_step",
            "peeling_evidence_score",
            "mixing_evidence_score",
            "behavioral_evidence_score",
            "behavioral_signal_count",
            "behavioral_signal_agreement",
            "multiple_behavioral_signals",
            "peeling_mixing_interaction",
        ]
    ].copy()

    behavioral_out[
        "peeling_signal"
    ] = bounded_unit_interval(
        behavioral_out[
            "peeling_evidence_score"
        ]
    )

    behavioral_out[
        "mixing_signal"
    ] = bounded_unit_interval(
        behavioral_out[
            "mixing_evidence_score"
        ]
    )

    behavioral_out[
        "behavioral_signal"
    ] = bounded_unit_interval(
        behavioral_out[
            "behavioral_evidence_score"
        ]
    )

    behavioral_out[
        "behavioral_agreement_signal"
    ] = bounded_unit_interval(
        behavioral_out[
            "behavioral_signal_agreement"
        ]
    )

    behavioral_out[
        "behavioral_interaction_signal"
    ] = bounded_unit_interval(
        behavioral_out[
            "peeling_mixing_interaction"
        ]
    )

    # ---------------------------------------------------------------
    # Network signals
    # ---------------------------------------------------------------

    network_out = network[
        [
            "txid",
            "network_observation_count",
            "network_unique_source_ip_count",
            "network_unique_destination_ip_count",
            "network_unique_endpoint_count",
            "network_observation_span_seconds",
            "network_unique_asn_count",
            "network_unique_country_count",
            "network_exact_txid_match",
            "network_temporal_context_available",
            "network_multi_observation_context",
            "network_temporal_evidence_score",
            "network_observation_available",
            "network_temporal_evidence_level",
        ]
    ].copy()

    network_out[
        "network_signal"
    ] = bounded_unit_interval(
        network_out[
            "network_temporal_evidence_score"
        ]
    )

    # Explicitly enforce zero network evidence when there is no
    # supplied observation.
    network_out.loc[
        ~network_out[
            "network_observation_available"
        ],
        "network_signal",
    ] = 0.0

    # ---------------------------------------------------------------
    # Merge all evidence channels
    # ---------------------------------------------------------------

    result = behavioral_out.merge(
        entity_signal,
        on=[
            "txid",
            "time_step",
        ],
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        network_out,
        on="txid",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # Entity integrity
    # ---------------------------------------------------------------

    entity_signal_columns = [
        "entity_repeated_cosponsor_signal",
        "entity_repeated_degree_signal",
        "entity_shared_transaction_signal",
        "entity_cluster_diversity_signal",
        "entity_cluster_size_signal",
        "entity_historical_address_signal",
        "entity_signal",
    ]

    for column in entity_signal_columns:

        result[column] = (
            pd.to_numeric(
                result[column],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

    # ---------------------------------------------------------------
    # Network integrity
    # ---------------------------------------------------------------

    network_numeric_columns = [
        "network_observation_count",
        "network_unique_source_ip_count",
        "network_unique_destination_ip_count",
        "network_unique_endpoint_count",
        "network_observation_span_seconds",
        "network_unique_asn_count",
        "network_unique_country_count",
    ]

    for column in network_numeric_columns:

        result[column] = (
            pd.to_numeric(
                result[column],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

    result[
        "network_signal"
    ] = (
        pd.to_numeric(
            result["network_signal"],
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    result[
        "network_observation_available"
    ] = (
        result[
            "network_observation_available"
        ]
        .fillna(False)
        .astype(bool)
    )

    result[
        "network_exact_txid_match"
    ] = (
        result[
            "network_exact_txid_match"
        ]
        .fillna(False)
        .astype(bool)
    )

    result[
        "network_temporal_context_available"
    ] = (
        result[
            "network_temporal_context_available"
        ]
        .fillna(False)
        .astype(bool)
    )

    result[
        "network_multi_observation_context"
    ] = (
        result[
            "network_multi_observation_context"
        ]
        .fillna(False)
        .astype(bool)
    )

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
    # Behavioral integrity
    # ---------------------------------------------------------------

    behavioral_signal_columns = [
        "peeling_signal",
        "mixing_signal",
        "behavioral_signal",
        "behavioral_agreement_signal",
        "behavioral_interaction_signal",
    ]

    for column in behavioral_signal_columns:

        result[column] = bounded_unit_interval(
            result[column]
        )

    # ---------------------------------------------------------------
    # Evidence channel availability
    # ---------------------------------------------------------------

    result[
        "behavioral_evidence_available"
    ] = (
        result[
            "behavioral_signal"
        ]
        > 0
    )

    result[
        "entity_evidence_available"
    ] = (
        result[
            "entity_signal"
        ]
        > 0
    )

    result[
        "network_evidence_available"
    ] = (
        result[
            "network_signal"
        ]
        > 0
    )

    # ---------------------------------------------------------------
    # Count independent positive channels
    #
    # ML and anomaly are not included yet because M7/M8 outputs are
    # added in a later M11 stage.
    # ---------------------------------------------------------------

    result[
        "available_evidence_channel_count"
    ] = (
        result[
            [
                "behavioral_evidence_available",
                "entity_evidence_available",
                "network_evidence_available",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )

    # ---------------------------------------------------------------
    # Final deterministic ordering
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
# Validation
# ---------------------------------------------------------------------------

def validate_result(
    result: pd.DataFrame,
    behavioral: pd.DataFrame,
    entity: pd.DataFrame,
    network: pd.DataFrame,
) -> dict:
    """
    Validate normalized evidence output.
    """

    if result.empty:
        raise AssertionError(
            "Normalized evidence output is empty."
        )

    if result["txid"].duplicated().any():
        raise AssertionError(
            "Normalized evidence contains duplicate TXIDs."
        )

    expected_txids = set(
        behavioral["txid"]
    )

    actual_txids = set(
        result["txid"]
    )

    if expected_txids != actual_txids:
        raise AssertionError(
            "Normalized evidence TXID coverage does not "
            "match behavioral evidence."
        )

    signal_columns = [
        "peeling_signal",
        "mixing_signal",
        "behavioral_signal",
        "behavioral_agreement_signal",
        "behavioral_interaction_signal",
        "entity_repeated_cosponsor_signal",
        "entity_repeated_degree_signal",
        "entity_shared_transaction_signal",
        "entity_cluster_diversity_signal",
        "entity_cluster_size_signal",
        "entity_historical_address_signal",
        "entity_signal",
        "network_signal",
    ]

    infinite_values = (
        result[
            signal_columns
        ]
        .isin(
            [np.inf, -np.inf]
        )
        .sum()
        .sum()
    )

    if infinite_values != 0:
        raise AssertionError(
            "Normalized evidence contains infinite values."
        )

    for column in signal_columns:

        values = result[
            column
        ]

        if (
            values < -EPSILON
        ).any() or (
            values > 1.0 + EPSILON
        ).any():

            raise AssertionError(
                f"Signal {column} is outside [0,1]."
            )

    # ---------------------------------------------------------------
    # Network absence semantics
    # ---------------------------------------------------------------

    no_network = ~result[
        "network_observation_available"
    ]

    if not (
        result.loc[
            no_network,
            "network_signal",
        ]
        == 0.0
    ).all():

        raise AssertionError(
            "Transactions without supplied network observations "
            "have non-zero network signal."
        )

    # ---------------------------------------------------------------
    # Entity temporal-source check
    # ---------------------------------------------------------------

    if len(entity) != len(result):
        raise AssertionError(
            "Entity evidence row count differs from "
            "normalized output."
        )

    # ---------------------------------------------------------------
    # Network coverage statistics
    # ---------------------------------------------------------------

    network_available = int(
        result[
            "network_evidence_available"
        ].sum()
    )

    behavioral_positive = int(
        (
            result[
                "behavioral_signal"
            ]
            > 0
        ).sum()
    )

    entity_positive = int(
        (
            result[
                "entity_signal"
            ]
            > 0
        ).sum()
    )

    return {
        "status": "PASS",
        "rows": int(
            len(result)
        ),
        "unique_txids": int(
            result["txid"].nunique()
        ),
        "behavioral_positive_transactions": (
            behavioral_positive
        ),
        "entity_positive_transactions": (
            entity_positive
        ),
        "network_positive_transactions": (
            network_available
        ),
        "network_observation_coverage": (
            float(
                network_available
                / len(result)
            )
            if len(result)
            else 0.0
        ),
        "max_available_evidence_channels": int(
            result[
                "available_evidence_channel_count"
            ].max()
        ),
        "mean_behavioral_signal": float(
            result[
                "behavioral_signal"
            ].mean()
        ),
        "mean_entity_signal": float(
            result[
                "entity_signal"
            ].mean()
        ),
        "mean_network_signal": float(
            result[
                "network_signal"
            ].mean()
        ),
    }


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
        "M11.3 EVIDENCE SIGNAL NORMALIZATION"
    )
    print("=" * 65)

    print(
        "Behavioral input:",
        BEHAVIORAL_INPUT,
    )

    print(
        "Entity input:",
        ENTITY_INPUT,
    )

    print(
        "Network input:",
        NETWORK_INPUT,
    )

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------

    behavioral = load_behavioral()

    entity = load_entity()

    network = load_network()

    print()
    print(
        f"Behavioral rows: {len(behavioral):,}"
    )

    print(
        f"Entity rows:     {len(entity):,}"
    )

    print(
        f"Network rows:    {len(network):,}"
    )

    # ---------------------------------------------------------------
    # Build entity signal
    # ---------------------------------------------------------------

    entity_signal, entity_metadata = (
        build_entity_signal(
            entity
        )
    )

    # ---------------------------------------------------------------
    # Build unified normalized evidence
    # ---------------------------------------------------------------

    result = build_normalized_evidence(
        behavioral=behavioral,
        entity_signal=entity_signal,
        network=network,
    )

    # ---------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------

    validation = validate_result(
        result=result,
        behavioral=behavioral,
        entity=entity,
        network=network,
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

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    report = {
        "module": "M11.3",
        "status": validation["status"],
        "purpose": (
            "Normalize independent behavioral, temporal entity, "
            "and network evidence channels into bounded [0,1] "
            "signals for subsequent evidence fusion."
        ),
        "inputs": {
            "behavioral": str(
                BEHAVIORAL_INPUT
            ),
            "temporal_entity": str(
                ENTITY_INPUT
            ),
            "network": str(
                NETWORK_INPUT
            ),
        },
        "input_rows": {
            "behavioral": int(
                len(behavioral)
            ),
            "temporal_entity": int(
                len(entity)
            ),
            "network": int(
                len(network)
            ),
        },
        "output": str(
            OUTPUT_FILE
        ),
        "output_sha256": output_sha256,
        "output_rows": int(
            len(result)
        ),
        "output_unique_txids": int(
            result["txid"].nunique()
        ),
        "signal_definitions": {
            "peeling_signal": (
                "Bounded M9 peeling-chain evidence."
            ),
            "mixing_signal": (
                "Bounded M9 mixing-pattern evidence."
            ),
            "behavioral_signal": (
                "Bounded M9 combined behavioral evidence."
            ),
            "behavioral_agreement_signal": (
                "Bounded M9 agreement between behavioral channels."
            ),
            "behavioral_interaction_signal": (
                "Bounded M9 peeling/mixing interaction evidence."
            ),
            "entity_signal": (
                "Equal-weight fusion of robustly normalized "
                "temporal-safe historical entity indicators."
            ),
            "network_signal": (
                "M10 temporal network contextual evidence; "
                "zero when no supplied network observation exists."
            ),
        },
        "entity_normalization": entity_metadata,
        "validation": validation,
        "ml_signal_status": (
            "Not included yet. M7 production inference outputs "
            "will be attached in a later M11 stage."
        ),
        "anomaly_signal_status": (
            "Not included yet. M8 anomaly inference outputs "
            "will be attached in a later M11 stage."
        ),
        "semantic_constraints": [
            (
                "Normalized evidence values are not probabilities "
                "of illicit activity."
            ),
            (
                "No wallet ownership or real-world identity "
                "inference is performed."
            ),
            (
                "No criminality or guilt inference is performed."
            ),
            (
                "Temporal entity evidence uses M6 causal "
                "historical features rather than globally "
                "computed M5 graph features."
            ),
            (
                "Network absence means no supplied network "
                "observation, not proof of no network activity."
            ),
        ],
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
        "OUTPUT ROWS:",
        len(result),
    )

    print(
        "UNIQUE TXIDS:",
        result["txid"].nunique(),
    )

    print(
        "MEAN BEHAVIORAL SIGNAL:",
        f"{result['behavioral_signal'].mean():.6f}",
    )

    print(
        "MEAN ENTITY SIGNAL:",
        f"{result['entity_signal'].mean():.6f}",
    )

    print(
        "MEAN NETWORK SIGNAL:",
        f"{result['network_signal'].mean():.6f}",
    )

    print(
        "NETWORK POSITIVE TRANSACTIONS:",
        int(
            result[
                "network_evidence_available"
            ].sum()
        ),
    )

    print(
        "MAX AVAILABLE EVIDENCE CHANNELS:",
        int(
            result[
                "available_evidence_channel_count"
            ].max()
        ),
    )

    print()
    print(
        "Normalized evidence preview:"
    )

    print("-" * 65)

    preview_columns = [
        "txid",
        "time_step",
        "peeling_signal",
        "mixing_signal",
        "behavioral_signal",
        "entity_signal",
        "network_signal",
        "available_evidence_channel_count",
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
        "M11.3 COMPLETE"
    )


if __name__ == "__main__":
    main()