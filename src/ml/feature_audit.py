from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = PROJECT_ROOT / "data" / "derived" / "unified_features.parquet"
REPORT_DIR = PROJECT_ROOT / "reports" / "ml"
REPORT_PATH = REPORT_DIR / "feature_audit.json"


IDENTIFIER_COLUMNS = {
    "txid",
    "txId",
    "transaction_id",
    "address",
    "input_address",
    "output_address",
}

KNOWN_TEMPORAL_COLUMNS = {
    "time_step",
    "Time step",
    "timestamp",
}

SUSPICIOUS_DISTRIBUTION_COLUMNS = {
    "transactions_in_time_step",
    "time_step_activity_ratio",
}


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Unified feature matrix not found: {FEATURE_PATH}"
        )

    df = pd.read_parquet(FEATURE_PATH)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    identifier_columns = [
        column for column in df.columns
        if column in IDENTIFIER_COLUMNS
    ]

    temporal_columns = [
        column for column in df.columns
        if column in KNOWN_TEMPORAL_COLUMNS
        or "time" in column.lower()
        or "timestamp" in column.lower()
    ]

    constant_features = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    infinite_counts = {}
    for column in numeric_columns:
        count = int(np.isinf(df[column].to_numpy()).sum())
        if count:
            infinite_counts[column] = count

    missing_counts = df.isna().sum()
    missing_features = {
        column: int(count)
        for column, count in missing_counts.items()
        if count > 0
    }

    suspicious_distribution_features = [
        column
        for column in df.columns
        if column in SUSPICIOUS_DISTRIBUTION_COLUMNS
    ]

    feature_columns = [
        column
        for column in numeric_columns
        if column not in identifier_columns
        and column not in constant_features
    ]

    audit = {
        "dataset": {
            "path": str(FEATURE_PATH.relative_to(PROJECT_ROOT)),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": {
            "all": df.columns.tolist(),
            "numeric": numeric_columns,
            "non_numeric": non_numeric_columns,
            "identifier": identifier_columns,
            "temporal": temporal_columns,
            "candidate_ml_features": feature_columns,
        },
        "counts": {
            "numeric": len(numeric_columns),
            "non_numeric": len(non_numeric_columns),
            "identifier": len(identifier_columns),
            "temporal": len(temporal_columns),
            "candidate_ml_features": len(feature_columns),
        },
        "quality": {
            "duplicate_columns": (
                df.columns[df.columns.duplicated()].tolist()
            ),
            "duplicate_txid_count": (
                int(df["txid"].duplicated().sum())
                if "txid" in df.columns
                else None
            ),
            "constant_features": constant_features,
            "infinite_values": infinite_counts,
            "missing_features": missing_features,
            "total_missing_cells": int(df.isna().sum().sum()),
        },
        "temporal": {
            column: {
                "min": (
                    df[column].min()
                    if pd.api.types.is_numeric_dtype(df[column])
                    else None
                ),
                "max": (
                    df[column].max()
                    if pd.api.types.is_numeric_dtype(df[column])
                    else None
                ),
                "unique": int(df[column].nunique(dropna=True)),
            }
            for column in temporal_columns
        },
        "leakage_review": {
            "suspicious_distribution_dependent_features": (
                suspicious_distribution_features
            ),
            "note": (
                "Distribution-dependent temporal features require "
                "training-only computation or explicit exclusion during "
                "supervised benchmarking."
            ),
        },
        "status": "AUDIT_ONLY_NO_LABELS",
    }

    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, default=str)

    print("=" * 70)
    print("M6.1 — ML FEATURE AUDIT")
    print("=" * 70)

    print(f"Dataset:              {FEATURE_PATH}")
    print(f"Shape:                {df.shape}")
    print(f"Numeric columns:      {len(numeric_columns)}")
    print(f"Non-numeric columns:  {len(non_numeric_columns)}")
    print(f"Identifier columns:   {len(identifier_columns)}")
    print(f"Temporal columns:     {len(temporal_columns)}")
    print(f"Candidate ML features:{len(feature_columns)}")
    print(f"Missing cells:        {int(df.isna().sum().sum())}")
    print(f"Constant features:    {len(constant_features)}")
    print(f"Infinite features:    {len(infinite_counts)}")

    print("\nIdentifier columns:")
    for column in identifier_columns:
        print(f"  - {column}")

    print("\nNon-numeric columns:")
    for column in non_numeric_columns:
        print(f"  - {column}")

    print("\nTemporal columns:")
    for column in temporal_columns:
        print(f"  - {column}")

    print("\nMissing features:")
    for column, count in sorted(
        missing_features.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  - {column}: {count}")

    print("\nPotential leakage-sensitive features:")
    for column in suspicious_distribution_features:
        print(f"  - {column}")

    print("\nAudit report:")
    print(REPORT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()