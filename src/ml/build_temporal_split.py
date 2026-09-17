from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = PROJECT_ROOT / "data" / "derived" / "unified_features.parquet"
REPORT_DIR = PROJECT_ROOT / "reports" / "ml"
REPORT_PATH = REPORT_DIR / "temporal_split.json"


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {FEATURE_PATH}"
        )

    df = pd.read_parquet(FEATURE_PATH)

    required_columns = {"txid", "time_step"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Required columns missing: {sorted(missing)}"
        )

    if df["txid"].duplicated().any():
        raise ValueError("Duplicate txid values detected.")

    if df["time_step"].isna().any():
        raise ValueError("Missing time_step values detected.")

    time_steps = sorted(df["time_step"].unique().tolist())

    if len(time_steps) < 3:
        raise ValueError(
            "At least three distinct time steps are required."
        )

    # Chronological 60/20/20 split by unique time steps.
    n_steps = len(time_steps)

    train_end = int(n_steps * 0.60)
    validation_end = int(n_steps * 0.80)

    # Protect against degenerate boundaries.
    train_end = max(train_end, 1)
    validation_end = max(validation_end, train_end + 1)

    if validation_end >= n_steps:
        validation_end = n_steps - 1

    train_steps = time_steps[:train_end]
    validation_steps = time_steps[train_end:validation_end]
    test_steps = time_steps[validation_end:]

    if not train_steps or not validation_steps or not test_steps:
        raise ValueError(
            "Temporal split produced an empty partition."
        )

    train_df = df[df["time_step"].isin(train_steps)]
    validation_df = df[df["time_step"].isin(validation_steps)]
    test_df = df[df["time_step"].isin(test_steps)]

    # Verify strict chronology.
    if max(train_steps) >= min(validation_steps):
        raise AssertionError(
            "Training and validation periods overlap."
        )

    if max(validation_steps) >= min(test_steps):
        raise AssertionError(
            "Validation and test periods overlap."
        )

    if (
        train_df["txid"].isin(validation_df["txid"]).any()
        or train_df["txid"].isin(test_df["txid"]).any()
        or validation_df["txid"].isin(test_df["txid"]).any()
    ):
        raise AssertionError(
            "Transaction overlap detected between partitions."
        )

    report = {
        "dataset": {
            "path": str(FEATURE_PATH.relative_to(PROJECT_ROOT)),
            "rows": int(len(df)),
            "unique_txids": int(df["txid"].nunique()),
            "unique_time_steps": len(time_steps),
        },
        "protocol": {
            "type": "chronological_unique_time_step_split",
            "train_fraction": 0.60,
            "validation_fraction": 0.20,
            "test_fraction": 0.20,
            "randomization": False,
            "transaction_overlap": False,
        },
        "partitions": {
            "train": {
                "rows": int(len(train_df)),
                "time_steps": train_steps,
                "min_time_step": int(min(train_steps)),
                "max_time_step": int(max(train_steps)),
            },
            "validation": {
                "rows": int(len(validation_df)),
                "time_steps": validation_steps,
                "min_time_step": int(min(validation_steps)),
                "max_time_step": int(max(validation_steps)),
            },
            "test": {
                "rows": int(len(test_df)),
                "time_steps": test_steps,
                "min_time_step": int(min(test_steps)),
                "max_time_step": int(max(test_steps)),
            },
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=" * 70)
    print("M6.2 — REPRODUCIBLE TEMPORAL SPLIT")
    print("=" * 70)

    print(f"Total rows:       {len(df):,}")
    print(f"Unique time steps:{len(time_steps)}")

    print("\nTRAIN")
    print(
        f"  Time: {min(train_steps)} → {max(train_steps)}"
    )
    print(f"  Rows: {len(train_df):,}")

    print("\nVALIDATION")
    print(
        f"  Time: {min(validation_steps)} → {max(validation_steps)}"
    )
    print(f"  Rows: {len(validation_df):,}")

    print("\nTEST")
    print(
        f"  Time: {min(test_steps)} → {max(test_steps)}"
    )
    print(f"  Rows: {len(test_df):,}")

    print("\nLeakage checks:")
    print("  ✓ No transaction overlap")
    print("  ✓ Strict chronological ordering")
    print("  ✓ Split boundaries based on unique time steps")

    print(f"\nManifest: {REPORT_PATH}")

    print("=" * 70)


if __name__ == "__main__":
    main()