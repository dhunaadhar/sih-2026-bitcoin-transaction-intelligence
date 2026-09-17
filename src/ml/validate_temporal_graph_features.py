from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TEMPORAL_TX = (
    ROOT
    / "data"
    / "derived"
    / "temporal_transaction_entity_evidence.parquet"
)

CANONICAL = (
    ROOT
    / "data"
    / "canonical"
    / "canonical_transactions.parquet"
)

OUTPUT = (
    ROOT
    / "reports"
    / "ml"
    / "temporal_graph_validation.json"
)


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


# These features legitimately equal 1 for an address with no
# historical graph edges: the singleton component contains the
# address itself.
BASELINE_ONE_FEATURES = {
    "input_max_cluster_size_t1",
    "input_max_cluster_size_t2",
    "input_max_cluster_size_t3",
    "output_max_cluster_size_t1",
    "output_max_cluster_size_t2",
    "output_max_cluster_size_t3",
}


def normalize_txid(value):

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.endswith(".0"):

        try:
            return str(
                int(
                    float(text)
                )
            )
        except ValueError:
            pass

    return text


def validate_columns(df):

    missing = [
        column
        for column in GRAPH_FEATURES
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Temporal artifact missing required graph "
            f"features: {missing}"
        )


def main():

    print("=" * 70)
    print(
        "M6.14 — TEMPORAL GRAPH ARTIFACT VALIDATION"
    )
    print("=" * 70)

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    temporal = pd.read_parquet(
        TEMPORAL_TX
    )

    canonical = pd.read_parquet(
        CANONICAL,
        columns=[
            "txid",
            "time_step",
        ],
    )

    validate_columns(
        temporal
    )

    # -------------------------------------------------------------
    # Normalize TXIDs for coverage comparison
    # -------------------------------------------------------------

    temporal["__txid_norm"] = (
        temporal["txid"]
        .map(normalize_txid)
    )

    canonical["__txid_norm"] = (
        canonical["txid"]
        .map(normalize_txid)
    )

    temporal_txids = set(
        temporal["__txid_norm"].dropna()
    )

    canonical_txids = set(
        canonical["__txid_norm"].dropna()
    )

    missing_from_temporal = (
        canonical_txids
        - temporal_txids
    )

    unexpected_temporal = (
        temporal_txids
        - canonical_txids
    )

    # -------------------------------------------------------------
    # Basic integrity
    # -------------------------------------------------------------

    print()
    print("TEMPORAL ARTIFACT")
    print("-" * 70)

    print(
        f"Rows:              {len(temporal):,}"
    )

    print(
        f"Unique TXIDs:      "
        f"{temporal['__txid_norm'].nunique():,}"
    )

    print(
        f"Time range:        "
        f"{temporal['time_step'].min()}–"
        f"{temporal['time_step'].max()}"
    )

    print()
    print("CANONICAL REFERENCE")
    print("-" * 70)

    print(
        f"Rows:              {len(canonical):,}"
    )

    print(
        f"Unique TXIDs:      "
        f"{canonical['__txid_norm'].nunique():,}"
    )

    print(
        f"Time range:        "
        f"{canonical['time_step'].min()}–"
        f"{canonical['time_step'].max()}"
    )

    # -------------------------------------------------------------
    # Coverage
    # -------------------------------------------------------------

    print()
    print("-" * 70)
    print("TXID COVERAGE")
    print("-" * 70)

    print(
        f"Canonical-only TXIDs: "
        f"{len(missing_from_temporal):,}"
    )

    print(
        f"Temporal-only TXIDs:  "
        f"{len(unexpected_temporal):,}"
    )

    # -------------------------------------------------------------
    # Duplicate TXIDs
    # -------------------------------------------------------------

    duplicate_temporal = int(
        temporal["__txid_norm"]
        .duplicated()
        .sum()
    )

    duplicate_canonical = int(
        canonical["__txid_norm"]
        .duplicated()
        .sum()
    )

    print()
    print(
        f"Duplicate temporal TXIDs:  "
        f"{duplicate_temporal:,}"
    )

    print(
        f"Duplicate canonical TXIDs: "
        f"{duplicate_canonical:,}"
    )

    # -------------------------------------------------------------
    # Time consistency
    # -------------------------------------------------------------

    canonical_time = (
        canonical
        .drop_duplicates("__txid_norm")
        .set_index("__txid_norm")[
            "time_step"
        ]
    )

    temporal_time = (
        temporal
        .drop_duplicates("__txid_norm")
        .set_index("__txid_norm")[
            "time_step"
        ]
    )

    common_txids = (
        canonical_txids
        & temporal_txids
    )

    common_list = list(
        common_txids
    )

    time_mismatches = int(
        (
            canonical_time.loc[
                common_list
            ].to_numpy()
            != temporal_time.loc[
                common_list
            ].to_numpy()
        ).sum()
    )

    print(
        f"Time-step mismatches: "
        f"{time_mismatches:,}"
    )

    # -------------------------------------------------------------
    # Numeric validation
    # -------------------------------------------------------------

    numeric = temporal[
        GRAPH_FEATURES
    ]

    infinite_values = int(
        np.isinf(
            numeric.to_numpy(
                dtype=float
            )
        ).sum()
    )

    negative_values = int(
        (
            numeric < 0
        ).sum().sum()
    )

    print()
    print("-" * 70)
    print("NUMERIC VALIDATION")
    print("-" * 70)

    print(
        f"Infinite values: "
        f"{infinite_values:,}"
    )

    print(
        f"Negative values: "
        f"{negative_values:,}"
    )

    # -------------------------------------------------------------
    # Feature statistics
    # -------------------------------------------------------------

    print()
    print("-" * 70)
    print("TEMPORAL GRAPH FEATURE STATISTICS")
    print("-" * 70)

    feature_stats = {}

    for column in GRAPH_FEATURES:

        series = temporal[column]

        nonzero = int(
            (
                series != 0
            ).sum()
        )

        missing = int(
            series.isna().sum()
        )

        feature_stats[column] = {
            "nonzero": nonzero,
            "nonzero_rate": float(
                nonzero / len(temporal)
            ),
            "missing": missing,
            "mean": float(
                series.mean()
            ),
            "max": float(
                series.max()
            ),
        }

        print()
        print(column)

        print(
            f"  Nonzero: "
            f"{nonzero:,} "
            f"({nonzero / len(temporal):.4%})"
        )

        print(
            f"  Missing: "
            f"{missing:,}"
        )

        print(
            f"  Mean: "
            f"{series.mean():.6f}"
        )

        print(
            f"  Max: "
            f"{series.max():.2f}"
        )

    # -------------------------------------------------------------
    # Time-step 1 causality check
    #
    # At time 1 there is no historical relationship.
    #
    # Therefore:
    #   count / ratio / degree / shared / cluster-count features
    #       must be zero.
    #
    #   cluster-size features
    #       must be exactly 1 whenever an address exists.
    #
    #   entity historical-address counts
    #       must be zero.
    # -------------------------------------------------------------

    time1 = temporal[
        temporal["time_step"] == 1
    ]

    zero_expected_features = [
        column
        for column in GRAPH_FEATURES
        if column not in BASELINE_ONE_FEATURES
    ]

    time1_zero_violations = {}

    for column in zero_expected_features:

        count = int(
            (
                time1[column] != 0
            ).sum()
        )

        if count > 0:
            time1_zero_violations[
                column
            ] = count

    one_expected_features = [
        column
        for column in GRAPH_FEATURES
        if column in BASELINE_ONE_FEATURES
    ]

    time1_one_violations = {}

    for column in one_expected_features:

        count = int(
            (
                time1[column] != 1
            ).sum()
        )

        if count > 0:
            time1_one_violations[
                column
            ] = count

    print()
    print("-" * 70)
    print("TIME-STEP 1 CAUSALITY CHECK")
    print("-" * 70)

    print(
        f"Time-step 1 rows: "
        f"{len(time1):,}"
    )

    print(
        "Expected zero-state violations: "
        f"{sum(time1_zero_violations.values()):,}"
    )

    print(
        "Expected singleton-size violations: "
        f"{sum(time1_one_violations.values()):,}"
    )

    # -------------------------------------------------------------
    # Temporal evolution
    # -------------------------------------------------------------

    temporal_evolution = {}

    for column in GRAPH_FEATURES:

        grouped = (
            temporal
            .groupby("time_step")[column]
            .agg(
                nonzero_count=lambda x: int(
                    (x != 0).sum()
                ),
                mean="mean",
                maximum="max",
            )
            .reset_index()
        )

        temporal_evolution[column] = (
            grouped.to_dict(
                orient="records"
            )
        )

    # -------------------------------------------------------------
    # Checks
    # -------------------------------------------------------------

    checks = {
        "row_count_match": (
            len(temporal)
            == len(canonical)
        ),

        "unique_txid_count_match": (
            temporal["__txid_norm"].nunique()
            == canonical["__txid_norm"].nunique()
        ),

        "missing_canonical_txids": (
            len(missing_from_temporal) == 0
        ),

        "unexpected_temporal_txids": (
            len(unexpected_temporal) == 0
        ),

        "duplicate_temporal_txids": (
            duplicate_temporal == 0
        ),

        "duplicate_canonical_txids": (
            duplicate_canonical == 0
        ),

        "time_step_mismatches": (
            time_mismatches == 0
        ),

        "infinite_values": (
            infinite_values == 0
        ),

        "negative_values": (
            negative_values == 0
        ),

        "time1_zero_state": (
            len(time1_zero_violations) == 0
        ),

        "time1_singleton_state": (
            len(time1_one_violations) == 0
        ),
    }

    status = (
        "PASS"
        if all(checks.values())
        else "REVIEW_REQUIRED"
    )

    # -------------------------------------------------------------
    # Report
    # -------------------------------------------------------------

    report = {
        "stage": "M6.14",

        "task": (
            "temporal_graph_artifact_validation"
        ),

        "status": status,

        "checks": checks,

        "coverage": {
            "temporal_rows": int(
                len(temporal)
            ),
            "canonical_rows": int(
                len(canonical)
            ),
            "temporal_unique_txids": int(
                temporal["__txid_norm"].nunique()
            ),
            "canonical_unique_txids": int(
                canonical["__txid_norm"].nunique()
            ),
            "missing_canonical_txids": int(
                len(missing_from_temporal)
            ),
            "unexpected_temporal_txids": int(
                len(unexpected_temporal)
            ),
        },

        "numeric_validation": {
            "infinite_values": infinite_values,
            "negative_values": negative_values,
        },

        "time1_causality": {
            "rows": int(
                len(time1)
            ),
            "zero_state_violations": (
                time1_zero_violations
            ),
            "singleton_state_violations": (
                time1_one_violations
            ),
            "singleton_size_features": sorted(
                BASELINE_ONE_FEATURES
            ),
        },

        "feature_statistics": feature_stats,

        "temporal_evolution": temporal_evolution,

        "methodology": {
            "future_information_used": False,
            "current_transaction_used_before_features": False,
            "historical_condition": (
                "only strictly earlier time steps "
                "contribute to graph state"
            ),
            "global_m5_artifact_required": False,
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print(
        f"STATUS: {status}"
    )
    print(
        f"Report: {OUTPUT}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()