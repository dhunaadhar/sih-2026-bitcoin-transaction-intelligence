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

MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "feature_manifest.json"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "temporal_split.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    print("=" * 70)
    print("M6.5 — REPRODUCIBLE BENCHMARK DATASET")
    print("=" * 70)

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {FEATURE_PATH}"
        )

    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Label file not found: {LABEL_PATH}"
        )

    features = pd.read_parquet(FEATURE_PATH)
    labels = pd.read_csv(LABEL_PATH)

    manifest = load_json(MANIFEST_PATH)
    split_manifest = load_json(SPLIT_PATH)

    txid_column = labels.columns[0]
    label_column = labels.columns[1]

    if txid_column != "txId":
        raise ValueError(
            f"Expected label TXID column 'txId', got '{txid_column}'"
        )

    if label_column != "class":
        raise ValueError(
            f"Expected label column 'class', got '{label_column}'"
        )

    if features["txid"].duplicated().any():
        raise ValueError("Duplicate txid values in features.")

    if labels["txId"].duplicated().any():
        raise ValueError("Duplicate txId values in labels.")

    # Join verified labels to the feature matrix.
    dataset = features.merge(
        labels,
        left_on="txid",
        right_on="txId",
        how="inner",
        validate="one_to_one",
    )

    if len(dataset) != len(features):
        raise ValueError(
            "Label join changed dataset cardinality."
        )

    if dataset["class"].isna().any():
        raise ValueError(
            "Missing class labels after join."
        )

    if not (
        dataset["txid"].astype(str)
        == dataset["txId"].astype(str)
    ).all():
        raise ValueError(
            "TXID mismatch detected after label join."
        )

    dataset = dataset.drop(columns=["txId"])

    # Obtain the authoritative candidate feature list.
    candidate_features = manifest["model_candidates"]

    missing_features = [
        column
        for column in candidate_features
        if column not in dataset.columns
    ]

    if missing_features:
        raise ValueError(
            "Manifest features missing from dataset: "
            f"{missing_features}"
        )

    # Initial benchmark excludes leakage-sensitive features.
    leakage_sensitive = manifest["leakage_sensitive"]

    benchmark_features = [
        column
        for column in candidate_features
        if column not in leakage_sensitive
    ]

    if not benchmark_features:
        raise ValueError(
            "No benchmark features remain after leakage exclusion."
        )

    # Explicitly retain txid and time_step as metadata only.
    metadata_columns = ["txid", "time_step", "class"]

    benchmark_dataset = dataset[
        metadata_columns + benchmark_features
    ].copy()

    # Validate classes.
    classes = sorted(
        benchmark_dataset["class"].dropna().unique().tolist()
    )

    if classes != [1, 2, 3]:
        raise ValueError(
            f"Unexpected class set: {classes}"
        )

    # Validate temporal split.
    train_steps = set(
        split_manifest["partitions"]["train"]["time_steps"]
    )
    validation_steps = set(
        split_manifest["partitions"]["validation"]["time_steps"]
    )
    test_steps = set(
        split_manifest["partitions"]["test"]["time_steps"]
    )

    train = benchmark_dataset[
        benchmark_dataset["time_step"].isin(train_steps)
    ]

    validation = benchmark_dataset[
        benchmark_dataset["time_step"].isin(validation_steps)
    ]

    test = benchmark_dataset[
        benchmark_dataset["time_step"].isin(test_steps)
    ]

    if len(train) + len(validation) + len(test) != len(
        benchmark_dataset
    ):
        raise ValueError(
            "Temporal partitions do not cover the full dataset."
        )

    if set(train["txid"]) & set(validation["txid"]):
        raise ValueError("Train/validation TXID overlap.")

    if set(train["txid"]) & set(test["txid"]):
        raise ValueError("Train/test TXID overlap.")

    if set(validation["txid"]) & set(test["txid"]):
        raise ValueError("Validation/test TXID overlap.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # These are evaluation artifacts and are ignored by Git.
    train_path = OUTPUT_DIR / "train.parquet"
    validation_path = OUTPUT_DIR / "validation.parquet"
    test_path = OUTPUT_DIR / "test.parquet"

    train.to_parquet(train_path, index=False)
    validation.to_parquet(validation_path, index=False)
    test.to_parquet(test_path, index=False)

    report = {
        "protocol": {
            "task": "three_class_supervised_classification",
            "classes": classes,
            "leakage_sensitive_features_excluded": leakage_sensitive,
            "split_type": "chronological",
        },
        "features": {
            "total_benchmark_features": len(benchmark_features),
            "features": benchmark_features,
        },
        "dataset": {
            "rows": len(benchmark_dataset),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        "class_distribution": {
            "overall": {
                str(k): int(v)
                for k, v in benchmark_dataset["class"]
                .value_counts()
                .sort_index()
                .items()
            },
            "train": {
                str(k): int(v)
                for k, v in train["class"]
                .value_counts()
                .sort_index()
                .items()
            },
            "validation": {
                str(k): int(v)
                for k, v in validation["class"]
                .value_counts()
                .sort_index()
                .items()
            },
            "test": {
                str(k): int(v)
                for k, v in test["class"]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "artifacts": {
            "train": str(train_path.relative_to(PROJECT_ROOT)),
            "validation": str(
                validation_path.relative_to(PROJECT_ROOT)
            ),
            "test": str(test_path.relative_to(PROJECT_ROOT)),
        },
        "status": "PASS",
    }

    report_path = (
        PROJECT_ROOT
        / "reports"
        / "ml"
        / "benchmark_dataset.json"
    )

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nJoined rows:           {len(benchmark_dataset):,}")
    print(f"Benchmark features:   {len(benchmark_features)}")

    print("\nClasses:")
    for class_id in classes:
        count = int(
            (benchmark_dataset["class"] == class_id).sum()
        )
        print(f"  Class {class_id}: {count:,}")

    print("\nPartitions:")
    print(f"  Train:      {len(train):,}")
    print(f"  Validation: {len(validation):,}")
    print(f"  Test:       {len(test):,}")

    print("\nLeakage-sensitive features excluded:")
    for feature in leakage_sensitive:
        print(f"  - {feature}")

    print("\nArtifacts:")
    print(f"  {train_path}")
    print(f"  {validation_path}")
    print(f"  {test_path}")

    print(f"\nReport: {report_path}")

    print("\nStatus: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()