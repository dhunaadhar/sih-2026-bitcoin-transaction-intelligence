from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = PROJECT_ROOT / "data" / "derived" / "unified_features.parquet"
AUDIT_PATH = PROJECT_ROOT / "reports" / "ml" / "feature_audit.json"
REPORT_DIR = PROJECT_ROOT / "reports" / "ml"
REPORT_PATH = REPORT_DIR / "feature_manifest.json"


IDENTIFIER_COLUMNS = {
    "txid",
    "txId",
    "transaction_id",
    "address",
    "input_address",
    "output_address",
}

SPLIT_ONLY_COLUMNS = {
    "time_step",
}

LEAKAGE_SENSITIVE_COLUMNS = {
    "transactions_in_time_step",
    "time_step_activity_ratio",
    "time_step_activity_deviation",
    "time_step_activity_log1p",
}


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {FEATURE_PATH}"
        )

    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Feature audit not found: {AUDIT_PATH}"
        )

    df = pd.read_parquet(FEATURE_PATH)

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    non_numeric_columns = df.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    duplicate_columns = df.columns[
        df.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        raise ValueError(
            f"Duplicate columns detected: {duplicate_columns}"
        )

    if "txid" not in df.columns:
        raise ValueError("Required identifier 'txid' is missing.")

    if df["txid"].duplicated().any():
        raise ValueError("Duplicate txid values detected.")

    if "time_step" not in df.columns:
        raise ValueError("Required split column 'time_step' is missing.")

    if df["time_step"].isna().any():
        raise ValueError("Missing time_step values detected.")

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    infinite_columns = []

    for column in numeric_columns:
        values = df[column].to_numpy()

        if np.isinf(values).any():
            infinite_columns.append(column)

    classifications = {}

    for column in df.columns:
        if column in IDENTIFIER_COLUMNS:
            classification = "EXCLUDE_IDENTIFIER"

        elif column in SPLIT_ONLY_COLUMNS:
            classification = "SPLIT_CONTROL"

        elif column in LEAKAGE_SENSITIVE_COLUMNS:
            classification = "LEAKAGE_SENSITIVE"

        elif column in constant_columns:
            classification = "EXCLUDE_CONSTANT"

        elif column in non_numeric_columns:
            classification = "EXCLUDE_NON_NUMERIC"

        elif column in infinite_columns:
            classification = "EXCLUDE_INFINITE"

        elif column in numeric_columns:
            classification = "MODEL_CANDIDATE"

        else:
            classification = "REVIEW"

        classifications[column] = classification

    model_candidates = [
        column
        for column, classification in classifications.items()
        if classification == "MODEL_CANDIDATE"
    ]

    leakage_sensitive = [
        column
        for column, classification in classifications.items()
        if classification == "LEAKAGE_SENSITIVE"
    ]

    excluded = [
        column
        for column, classification in classifications.items()
        if classification.startswith("EXCLUDE")
    ]

    split_controls = [
        column
        for column, classification in classifications.items()
        if classification == "SPLIT_CONTROL"
    ]

    missing_counts = df.isna().sum()

    missingness = {
        column: int(count)
        for column, count in missing_counts.items()
        if count > 0
    }

    manifest = {
        "dataset": {
            "path": str(FEATURE_PATH.relative_to(PROJECT_ROOT)),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
        "protocol": {
            "identifier_columns": sorted(IDENTIFIER_COLUMNS),
            "split_control_columns": sorted(SPLIT_ONLY_COLUMNS),
            "leakage_sensitive_columns": sorted(
                LEAKAGE_SENSITIVE_COLUMNS
            ),
            "constant_columns": constant_columns,
            "non_numeric_columns": non_numeric_columns,
            "infinite_columns": infinite_columns,
        },
        "classification": classifications,
        "counts": {
            "total_columns": int(len(df.columns)),
            "model_candidates": len(model_candidates),
            "leakage_sensitive": len(leakage_sensitive),
            "split_controls": len(split_controls),
            "excluded": len(excluded),
        },
        "model_candidates": model_candidates,
        "leakage_sensitive": leakage_sensitive,
        "split_controls": split_controls,
        "excluded": excluded,
        "missingness": missingness,
        "decisions": {
            "txid": (
                "Excluded from all ML models because it is an "
                "identifier rather than a behavioral feature."
            ),
            "time_step": (
                "Used for chronological train/validation/test "
                "partitioning. It is not included as a predictive "
                "feature in the initial benchmark."
            ),
            "distribution_dependent_temporal_features": (
                "Excluded from the initial benchmark until they can "
                "be recomputed using training-period statistics only."
            ),
            "missing_values": (
                "Missingness is preserved for downstream preprocessing. "
                "Models must use an explicitly defined imputation and/or "
                "missingness-indicator policy."
            ),
        },
        "status": "FEATURE_MANIFEST_READY",
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("=" * 70)
    print("M6.3 — ML FEATURE MANIFEST + LEAKAGE GATE")
    print("=" * 70)

    print(f"Dataset rows:          {len(df):,}")
    print(f"Total columns:         {len(df.columns)}")
    print(f"Model candidates:      {len(model_candidates)}")
    print(f"Leakage-sensitive:     {len(leakage_sensitive)}")
    print(f"Split controls:        {len(split_controls)}")
    print(f"Excluded:              {len(excluded)}")
    print(f"Missing-feature count: {len(missingness)}")

    print("\nMODEL CANDIDATES:")
    for column in model_candidates:
        print(f"  + {column}")

    print("\nLEAKAGE-SENSITIVE:")
    for column in leakage_sensitive:
        print(f"  ! {column}")

    print("\nSPLIT CONTROL:")
    for column in split_controls:
        print(f"  # {column}")

    print("\nEXCLUDED:")
    for column in excluded:
        print(f"  - {column}")

    print("\nLeakage policy:")
    print("  ✓ txid excluded")
    print("  ✓ time_step excluded from predictive features")
    print("  ✓ distribution-dependent temporal features isolated")
    print("  ✓ constant/non-numeric/infinite features excluded")
    print("  ✓ missingness explicitly recorded")

    print(f"\nManifest: {REPORT_PATH}")

    print("=" * 70)


if __name__ == "__main__":
    main()