from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "unified_features.parquet"
)

LABEL_PATH = (
    PROJECT_ROOT.parent
    / "data"
    / "external"
    / "elipticpp"
    / "transactions"
    / "txs_classes.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "ml"
REPORT_PATH = REPORT_DIR / "label_audit.json"


def main() -> None:
    print("=" * 70)
    print("M6.4 — LABEL INTEGRITY AUDIT")
    print("=" * 70)

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {FEATURE_PATH}"
        )

    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Label file not found: {LABEL_PATH}"
        )

    print(f"Feature matrix: {FEATURE_PATH}")
    print(f"Label source:   {LABEL_PATH}")

    features = pd.read_parquet(FEATURE_PATH)
    labels = pd.read_csv(LABEL_PATH)

    print("\nRaw label schema:")
    print(f"  Shape: {labels.shape}")
    print(f"  Columns: {labels.columns.tolist()}")

    if len(labels.columns) < 2:
        raise ValueError(
            "Label file must contain TXID and label columns."
        )

    txid_column = labels.columns[0]
    label_column = labels.columns[1]

    if "txid" not in features.columns:
        raise ValueError(
            "Unified feature matrix does not contain txid."
        )

    if features["txid"].duplicated().any():
        raise ValueError(
            "Duplicate txid values found in feature matrix."
        )

    if labels[txid_column].duplicated().any():
        duplicate_count = int(
            labels[txid_column].duplicated().sum()
        )
        raise ValueError(
            f"Duplicate TXIDs found in label file: {duplicate_count}"
        )

    feature_txids = set(features["txid"])
    label_txids = set(labels[txid_column])

    feature_only = feature_txids - label_txids
    label_only = label_txids - feature_txids
    intersection = feature_txids & label_txids

    merged = features[["txid", "time_step"]].merge(
        labels[[txid_column, label_column]],
        left_on="txid",
        right_on=txid_column,
        how="left",
        validate="one_to_one",
    )

    missing_labels = int(
        merged[label_column].isna().sum()
    )

    label_counts = (
        labels[label_column]
        .value_counts(dropna=False)
        .sort_index()
    )

    temporal_label_counts = (
        merged.dropna(subset=[label_column])
        .groupby(["time_step", label_column])
        .size()
        .unstack(fill_value=0)
    )

    report = {
        "source": {
            "label_path": str(LABEL_PATH),
            "feature_path": str(
                FEATURE_PATH.relative_to(PROJECT_ROOT)
            ),
        },
        "schema": {
            "label_columns": labels.columns.tolist(),
            "txid_column": txid_column,
            "label_column": label_column,
        },
        "integrity": {
            "feature_rows": int(len(features)),
            "feature_unique_txids": int(
                features["txid"].nunique()
            ),
            "label_rows": int(len(labels)),
            "label_unique_txids": int(
                labels[txid_column].nunique()
            ),
            "txids_in_both": int(len(intersection)),
            "feature_only_txids": int(len(feature_only)),
            "label_only_txids": int(len(label_only)),
            "missing_labels_after_join": missing_labels,
        },
        "labels": {
            "unique_values": [
                str(value)
                for value in labels[label_column]
                .dropna()
                .unique()
                .tolist()
            ],
            "counts": {
                str(index): int(value)
                for index, value in label_counts.items()
            },
        },
        "temporal_label_counts": {
            str(time_step): {
                str(label): int(count)
                for label, count in row.items()
            }
            for time_step, row in temporal_label_counts.iterrows()
        },
        "status": (
            "PASS"
            if (
                len(feature_only) == 0
                and len(label_only) == 0
                and missing_labels == 0
            )
            else "REVIEW"
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nIntegrity:")
    print(f"  Feature rows:       {len(features):,}")
    print(f"  Label rows:         {len(labels):,}")
    print(
        f"  TXIDs in both:      {len(intersection):,}"
    )
    print(
        f"  Feature-only TXIDs: {len(feature_only):,}"
    )
    print(
        f"  Label-only TXIDs:   {len(label_only):,}"
    )
    print(
        f"  Missing labels:     {missing_labels:,}"
    )

    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  Class {label}: {count:,}")

    print("\nTemporal label distribution:")
    print(temporal_label_counts.to_string())

    print("\nStatus:", report["status"])
    print(f"\nReport: {REPORT_PATH}")

    print("=" * 70)


if __name__ == "__main__":
    main()