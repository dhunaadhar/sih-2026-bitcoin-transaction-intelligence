"""
M6.15 — BUILD TEMPORAL-SAFE UNIFIED FEATURES

Replaces the 22 globally constructed M5 graph/entity features in
unified_features.parquet with the 22 strictly causal M6.13 features.

The original unified_features.parquet is NOT modified.

Output:
    data/derived/unified_features_temporal_safe.parquet

Validation guarantees:
    - one row per transaction
    - exact TXID coverage
    - exact time-step alignment
    - 93 total columns
    - 22 temporal graph features
    - no infinite values
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

BASE_FEATURES_PATH = (
    ROOT
    / "data"
    / "derived"
    / "unified_features.parquet"
)

TEMPORAL_GRAPH_PATH = (
    ROOT
    / "data"
    / "derived"
    / "temporal_transaction_entity_evidence.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "derived"
    / "unified_features_temporal_safe.parquet"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "ml"
    / "temporal_safe_unified_features.json"
)


# =====================================================================
# THE 22 GRAPH FEATURES REPLACED FROM M5
# =====================================================================

GRAPH_FEATURES = [
    "input_repeated_cosponsor_count",
    "input_repeated_cosponsor_ratio",
    "output_repeated_cosponsor_count",
    "output_repeated_cosponsor_ratio",

    "input_max_repeated_cosponsor_degree",
    "output_max_repeated_cosponsor_degree",

    "input_max_shared_transaction_count",
    "output_max_shared_transaction_count",

    "input_distinct_clusters_t1",
    "input_distinct_clusters_t2",
    "input_distinct_clusters_t3",

    "output_distinct_clusters_t1",
    "output_distinct_clusters_t2",
    "output_distinct_clusters_t3",

    "input_max_cluster_size_t1",
    "input_max_cluster_size_t2",
    "input_max_cluster_size_t3",

    "output_max_cluster_size_t1",
    "output_max_cluster_size_t2",
    "output_max_cluster_size_t3",

    "entity_input_address_count",
    "entity_output_address_count",
]


# =====================================================================
# HELPERS
# =====================================================================

def normalize_txid(value):
    """
    Normalize TXIDs so numeric representations such as 123.0 and
    string representations such as "123" are treated identically.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.endswith(".0"):

        try:
            numeric = float(text)

            if numeric.is_integer():
                text = str(int(numeric))

        except ValueError:
            pass

    return text


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print(
        "M6.15 — BUILD TEMPORAL-SAFE UNIFIED FEATURES"
    )
    print("=" * 70)

    # -------------------------------------------------------------
    # Source validation.
    # -------------------------------------------------------------

    print()
    print("Checking source artifacts...")

    for path in [
        BASE_FEATURES_PATH,
        TEMPORAL_GRAPH_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

        print(
            f"  OK: {path}"
        )

    # -------------------------------------------------------------
    # Load base features.
    # -------------------------------------------------------------

    print()
    print(
        "Loading base unified features..."
    )

    base = pd.read_parquet(
        BASE_FEATURES_PATH
    )

    print(
        f"Base shape: "
        f"{base.shape}"
    )

    if "txid" not in base.columns:

        raise ValueError(
            "Base unified features do not contain txid."
        )

    # Preserve the ORIGINAL base column count before adding the
    # temporary normalization helper.
    original_base_column_count = len(
        base.columns
    )

    base["__txid_norm"] = base[
        "txid"
    ].map(
        normalize_txid
    )

    if base["__txid_norm"].isna().any():

        raise ValueError(
            "Base unified features contain null TXIDs."
        )

    if base["__txid_norm"].duplicated().any():

        raise ValueError(
            "Base unified features contain duplicate TXIDs."
        )

    # -------------------------------------------------------------
    # Load temporal graph features.
    # -------------------------------------------------------------

    print()
    print(
        "Loading temporal graph evidence..."
    )

    temporal = pd.read_parquet(
        TEMPORAL_GRAPH_PATH
    )

    print(
        f"Temporal graph shape: "
        f"{temporal.shape}"
    )

    if "txid" not in temporal.columns:

        raise ValueError(
            "Temporal graph artifact does not contain txid."
        )

    temporal["__txid_norm"] = temporal[
        "txid"
    ].map(
        normalize_txid
    )

    if temporal["__txid_norm"].isna().any():

        raise ValueError(
            "Temporal graph artifact contains null TXIDs."
        )

    if temporal["__txid_norm"].duplicated().any():

        raise ValueError(
            "Temporal graph artifact contains duplicate TXIDs."
        )

    # -------------------------------------------------------------
    # Validate temporal graph feature columns.
    # -------------------------------------------------------------

    missing_temporal = [
        feature
        for feature in GRAPH_FEATURES
        if feature not in temporal.columns
    ]

    if missing_temporal:

        raise ValueError(
            "Missing temporal graph features: "
            + ", ".join(missing_temporal)
        )

    # -------------------------------------------------------------
    # Validate base graph feature columns.
    # -------------------------------------------------------------

    missing_base = [
        feature
        for feature in GRAPH_FEATURES
        if feature not in base.columns
    ]

    if missing_base:

        raise ValueError(
            "Expected graph features missing from base dataset: "
            + ", ".join(missing_base)
        )

    # -------------------------------------------------------------
    # Coverage validation.
    # -------------------------------------------------------------

    print()
    print(
        "Validating TXID coverage..."
    )

    base_ids = set(
        base["__txid_norm"]
    )

    temporal_ids = set(
        temporal["__txid_norm"]
    )

    base_only = base_ids - temporal_ids
    temporal_only = temporal_ids - base_ids

    print(
        f"Base-only TXIDs:      {len(base_only):,}"
    )

    print(
        f"Temporal-only TXIDs:  {len(temporal_only):,}"
    )

    if base_only or temporal_only:

        raise ValueError(
            "TXID coverage mismatch between base and temporal graph."
        )

    # -------------------------------------------------------------
    # Time-step validation.
    # -------------------------------------------------------------

    print()
    print(
        "Validating time-step alignment..."
    )

    base_time = (
        base[
            [
                "__txid_norm",
                "time_step",
            ]
        ]
        .set_index("__txid_norm")[
            "time_step"
        ]
    )

    temporal_time = (
        temporal[
            [
                "__txid_norm",
                "time_step",
            ]
        ]
        .set_index("__txid_norm")[
            "time_step"
        ]
    )

    aligned_temporal_time = temporal_time.reindex(
        base_time.index
    )

    time_mismatch = int(
        (
            base_time.to_numpy()
            != aligned_temporal_time.to_numpy()
        ).sum()
    )

    print(
        f"Time-step mismatches: "
        f"{time_mismatch:,}"
    )

    if time_mismatch != 0:

        raise ValueError(
            "Time-step mismatch detected."
        )

    # -------------------------------------------------------------
    # Preserve base dataset.
    # -------------------------------------------------------------

    print()
    print(
        "Replacing graph features..."
    )

    result = base.copy()

    # Remove the globally constructed graph features.
    result = result.drop(
        columns=GRAPH_FEATURES
    )

    # -------------------------------------------------------------
    # Select causal temporal features.
    # -------------------------------------------------------------

    temporal_selected = temporal[
        [
            "__txid_norm",
            *GRAPH_FEATURES,
        ]
    ].copy()

    # -------------------------------------------------------------
    # Merge causal graph features.
    # -------------------------------------------------------------

    result = result.merge(
        temporal_selected,
        on="__txid_norm",
        how="left",
        validate="one_to_one",
    )

    # -------------------------------------------------------------
    # Validate no missing temporal graph features.
    # -------------------------------------------------------------

    missing_counts = (
        result[GRAPH_FEATURES]
        .isna()
        .sum()
    )

    missing_total = int(
        missing_counts.sum()
    )

    print(
        f"Missing temporal graph cells: "
        f"{missing_total:,}"
    )

    if missing_total != 0:

        missing_features = {
            feature: int(count)
            for feature, count
            in missing_counts.items()
            if count > 0
        }

        raise ValueError(
            "Temporal graph features contain missing values: "
            + str(missing_features)
        )

    # -------------------------------------------------------------
    # Numeric validation.
    # -------------------------------------------------------------

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    infinite_count = int(
        np.isinf(
            result[numeric_columns].to_numpy(
                dtype=float
            )
        ).sum()
    )

    print(
        f"Infinite values: "
        f"{infinite_count}"
    )

    if infinite_count != 0:

        raise ValueError(
            "Infinite values detected."
        )

    # -------------------------------------------------------------
    # Remove temporary helper.
    # -------------------------------------------------------------

    result = result.drop(
        columns=["__txid_norm"]
    )

    # -------------------------------------------------------------
    # Final dataset validation.
    # -------------------------------------------------------------

    print()
    print(
        "Final dataset validation..."
    )

    expected_columns = (
        original_base_column_count
    )

    actual_columns = (
        len(result.columns)
    )

    print(
        f"Base columns:       {expected_columns:,}"
    )

    print(
        f"Final columns:      {actual_columns:,}"
    )

    if actual_columns != expected_columns:

        raise ValueError(
            "Final column count changed unexpectedly."
        )

    if len(result) != len(base):

        raise ValueError(
            "Final row count changed unexpectedly."
        )

    if result["txid"].nunique() != len(result):

        raise ValueError(
            "Final dataset contains duplicate TXIDs."
        )

    # -------------------------------------------------------------
    # Ensure time-step column is preserved.
    # -------------------------------------------------------------

    if "time_step" not in result.columns:

        raise ValueError(
            "Final dataset lost time_step."
        )

    # -------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Verify saved artifact.
    # -------------------------------------------------------------

    saved = pd.read_parquet(
        OUTPUT_PATH
    )

    if saved.shape != result.shape:

        raise ValueError(
            "Saved artifact shape mismatch."
        )

    if saved["txid"].nunique() != len(saved):

        raise ValueError(
            "Saved artifact contains duplicate TXIDs."
        )

    if len(saved.columns) != expected_columns:

        raise ValueError(
            "Saved artifact column count mismatch."
        )

    # -------------------------------------------------------------
    # Compare replaced features.
    # -------------------------------------------------------------

    replacement_statistics = {}

    for feature in GRAPH_FEATURES:

        old_values = pd.to_numeric(
            base[feature],
            errors="coerce",
        )

        new_values = pd.to_numeric(
            saved[feature],
            errors="coerce",
        )

        changed = int(
            (
                old_values.to_numpy()
                != new_values.to_numpy()
            ).sum()
        )

        replacement_statistics[
            feature
        ] = {
            "changed_rows": changed,
            "old_nonzero": int(
                (old_values != 0).sum()
            ),
            "new_nonzero": int(
                (new_values != 0).sum()
            ),
        }

    # -------------------------------------------------------------
    # Report.
    # -------------------------------------------------------------

    report = {
        "stage": "M6.15",

        "task": (
            "build_temporal_safe_unified_features"
        ),

        "status": "PASS",

        "base_artifact": str(
            BASE_FEATURES_PATH
        ),

        "temporal_graph_artifact": str(
            TEMPORAL_GRAPH_PATH
        ),

        "output_artifact": str(
            OUTPUT_PATH
        ),

        "statistics": {
            "base_rows": int(
                len(base)
            ),

            "output_rows": int(
                len(result)
            ),

            "base_columns": int(
                original_base_column_count
            ),

            "output_columns": int(
                len(result.columns)
            ),

            "graph_features_replaced": int(
                len(GRAPH_FEATURES)
            ),

            "txid_coverage_base_only": int(
                len(base_only)
            ),

            "txid_coverage_temporal_only": int(
                len(temporal_only)
            ),

            "time_step_mismatches": int(
                time_mismatch
            ),

            "missing_temporal_graph_cells": int(
                missing_total
            ),

            "infinite_values": int(
                infinite_count
            ),
        },

        "temporal_graph_features": GRAPH_FEATURES,

        "replacement_statistics": (
            replacement_statistics
        ),

        "methodology": {
            "global_graph_features_removed": True,

            "temporal_graph_features_inserted": True,

            "future_graph_information_used": False,

            "same_time_step_graph_information_used": False,

            "temporal_policy": (
                "graph features at time t use only "
                "transactions from time steps < t"
            ),
        },

        "next_step": (
            "Build the temporal-safe benchmark train, "
            "validation, and test datasets using the "
            "existing chronological split."
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Final output.
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "M6.15 COMPLETE"
    )
    print("=" * 70)

    print(
        f"Output shape: "
        f"{result.shape}"
    )

    print(
        f"Graph features replaced: "
        f"{len(GRAPH_FEATURES)}"
    )

    print(
        f"TXID coverage mismatch: "
        f"{len(base_only) + len(temporal_only)}"
    )

    print(
        f"Time-step mismatches: "
        f"{time_mismatch}"
    )

    print(
        f"Missing temporal graph cells: "
        f"{missing_total}"
    )

    print(
        f"Infinite values: "
        f"{infinite_count}"
    )

    print()
    print(
        f"Output:"
        f"\n  {OUTPUT_PATH}"
    )

    print(
        f"Report:"
        f"\n  {REPORT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()