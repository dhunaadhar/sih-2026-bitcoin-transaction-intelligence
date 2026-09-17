from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TEMPORAL_PATH = ROOT / "data" / "derived" / "temporal_transaction_entity_evidence.parquet"
CANONICAL_PATH = ROOT / "data" / "canonical" / "canonical_transactions.parquet"
REPORT_PATH = ROOT / "reports" / "ml" / "temporal_graph_causality_diagnostic.json"


ZERO_STATE_FEATURES = [
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
    "entity_input_address_count",
    "entity_output_address_count",
]

SIZE_FEATURES = [
    "input_max_cluster_size_t1",
    "input_max_cluster_size_t2",
    "input_max_cluster_size_t3",
    "output_max_cluster_size_t1",
    "output_max_cluster_size_t2",
    "output_max_cluster_size_t3",
]


def normalize_txid(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.endswith(".0"):
        try:
            numeric = float(text)
            if numeric.is_integer():
                text = str(int(numeric))
        except ValueError:
            pass

    return text


def main():
    print("=" * 70)
    print("M6.14.1 — TEMPORAL GRAPH CAUSALITY DIAGNOSTIC")
    print("=" * 70)

    temporal = pd.read_parquet(TEMPORAL_PATH)
    canonical = pd.read_parquet(CANONICAL_PATH)

    temporal["__txid_norm"] = temporal["txid"].map(normalize_txid)
    canonical["__txid_norm"] = canonical["txid"].map(normalize_txid)

    temporal["__time_norm"] = pd.to_numeric(
        temporal["time_step"], errors="coerce"
    )
    canonical["__time_norm"] = pd.to_numeric(
        canonical["time_step"], errors="coerce"
    )

    time1 = temporal[temporal["__time_norm"] == 1].copy()

    print()
    print("DATASET")
    print("-" * 70)
    print(f"Temporal rows:       {len(temporal):,}")
    print(f"Time-step-1 rows:    {len(time1):,}")
    print(f"Temporal columns:    {len(temporal.columns) - 4:,}")

    print()
    print("TIME-STEP 1 ZERO-STATE FEATURES")
    print("-" * 70)

    zero_results = {}

    for feature in ZERO_STATE_FEATURES:
        values = pd.to_numeric(time1[feature], errors="coerce")

        nonzero_mask = values.fillna(0) != 0
        nonzero_count = int(nonzero_mask.sum())

        zero_results[feature] = {
            "nonzero_count": nonzero_count,
            "max": float(values.max()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
            "sample_txids": time1.loc[
                nonzero_mask, "__txid_norm"
            ].head(10).tolist(),
            "sample_values": values.loc[
                nonzero_mask
            ].head(10).tolist(),
        }

        print(
            f"{feature:<45} "
            f"nonzero={nonzero_count:>6,} "
            f"max={values.max():>10.2f}"
        )

    print()
    print("TIME-STEP 1 CLUSTER SIZE FEATURES")
    print("-" * 70)

    size_results = {}

    for feature in SIZE_FEATURES:
        values = pd.to_numeric(time1[feature], errors="coerce")

        not_one_mask = values != 1
        violation_count = int(not_one_mask.sum())

        size_results[feature] = {
            "not_equal_to_one": violation_count,
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
            "sample_txids": time1.loc[
                not_one_mask, "__txid_norm"
            ].head(10).tolist(),
            "sample_values": values.loc[
                not_one_mask
            ].head(10).tolist(),
        }

        print(
            f"{feature:<40} "
            f"not_one={violation_count:>6,} "
            f"min={values.min():>8.2f} "
            f"max={values.max():>8.2f}"
        )

    print()
    print("TIME-STEP 1 SAMPLE ROWS")
    print("-" * 70)

    display_columns = [
        "__txid_norm",
        "__time_norm",
    ] + ZERO_STATE_FEATURES + SIZE_FEATURES

    print(
        time1[display_columns]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("CANONICAL TIME-STEP CROSS-CHECK")
    print("-" * 70)

    merged = time1[
        ["__txid_norm", "__time_norm"]
    ].merge(
        canonical[
            ["__txid_norm", "__time_norm"]
        ],
        on="__txid_norm",
        how="left",
        suffixes=("_temporal", "_canonical"),
        indicator=True,
    )

    missing_canonical = int(
        (merged["_merge"] != "both").sum()
    )

    mismatched_time = int(
        (
            (merged["_merge"] == "both")
            & (
                merged["__time_norm_temporal"]
                != merged["__time_norm_canonical"]
            )
        ).sum()
    )

    print(f"Time-1 TXIDs missing canonical: {missing_canonical:,}")
    print(f"Time-1 time mismatches:         {mismatched_time:,}")

    print()
    print("INTERPRETATION")
    print("-" * 70)

    total_zero_violations = sum(
        result["nonzero_count"]
        for result in zero_results.values()
    )

    total_size_violations = sum(
        result["not_equal_to_one"]
        for result in size_results.values()
    )

    print(f"Total zero-state violations:     {total_zero_violations:,}")
    print(f"Total singleton-size violations: {total_size_violations:,}")

    if total_zero_violations == 0 and total_size_violations == 0:
        status = "PASS"
        interpretation = (
            "Time-step 1 contains no historical graph-state contamination."
        )
    else:
        status = "REVIEW_REQUIRED"

        if total_zero_violations > 0:
            interpretation = (
                "One or more historical graph features are nonzero at "
                "time-step 1. This indicates that historical graph state "
                "is being populated before the first observation or that "
                "same-time-step data is entering the feature calculation."
            )
        else:
            interpretation = (
                "Zero-state features are clean, but cluster-size features "
                "are not initialized as singleton components at time-step 1."
            )

    print(f"Status: {status}")
    print(f"Interpretation: {interpretation}")

    report = {
        "status": status,
        "interpretation": interpretation,
        "temporal_rows": int(len(temporal)),
        "time1_rows": int(len(time1)),
        "zero_state_results": zero_results,
        "size_feature_results": size_results,
        "canonical_cross_check": {
            "missing_canonical": missing_canonical,
            "time_mismatches": mismatched_time,
        },
        "total_zero_state_violations": total_zero_violations,
        "total_singleton_size_violations": total_size_violations,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 70)
    print(f"Report: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()