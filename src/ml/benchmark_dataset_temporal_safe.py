"""
M6.16 — BUILD TEMPORAL-SAFE BENCHMARK DATASET

Builds train / validation / test benchmark datasets from the
M6.15 temporal-safe unified feature matrix.

Frozen chronological split:

    Train       : time steps 1–29
    Validation  : time steps 30–39
    Test        : time steps 40–49

Labels are taken from the authoritative Elliptic++ transaction
class file.

This stage does NOT perform:
    - random splitting
    - resampling
    - class balancing
    - feature selection
    - model training

Outputs:

    data/evaluation/train_temporal_safe.parquet
    data/evaluation/validation_temporal_safe.parquet
    data/evaluation/test_temporal_safe.parquet

    reports/ml/temporal_safe_benchmark_dataset.json
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    ROOT
    / "data"
    / "derived"
    / "unified_features_temporal_safe.parquet"
)

LABEL_PATH = (
    ROOT.parent
    / "data"
    / "external"
    / "elipticpp"
    / "transactions"
    / "txs_classes.csv"
)

FEATURE_MANIFEST_PATH = (
    ROOT
    / "reports"
    / "ml"
    / "feature_manifest.json"
)

TEMPORAL_SPLIT_PATH = (
    ROOT
    / "reports"
    / "ml"
    / "temporal_split.json"
)

EVALUATION_DIR = (
    ROOT
    / "data"
    / "evaluation"
)

TRAIN_OUTPUT = (
    EVALUATION_DIR
    / "train_temporal_safe.parquet"
)

VALIDATION_OUTPUT = (
    EVALUATION_DIR
    / "validation_temporal_safe.parquet"
)

TEST_OUTPUT = (
    EVALUATION_DIR
    / "test_temporal_safe.parquet"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "ml"
    / "temporal_safe_benchmark_dataset.json"
)


# =====================================================================
# FROZEN CHRONOLOGICAL SPLIT
# =====================================================================

EXPECTED_TRAIN_RANGE = (1, 29)
EXPECTED_VALIDATION_RANGE = (30, 39)
EXPECTED_TEST_RANGE = (40, 49)

EXPECTED_TRAIN_ROWS = 120804
EXPECTED_VALIDATION_ROWS = 36318
EXPECTED_TEST_ROWS = 46647

EXPECTED_TOTAL_ROWS = (
    EXPECTED_TRAIN_ROWS
    + EXPECTED_VALIDATION_ROWS
    + EXPECTED_TEST_ROWS
)


# =====================================================================
# HELPERS
# =====================================================================

def normalize_txid(value):
    """
    Normalize TXID representations so equivalent textual/numeric
    representations match exactly.
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


def validate_time_range(
    dataframe,
    expected_range,
    split_name,
):
    """
    Validate that a split contains exactly the expected inclusive
    time-step range.
    """

    if "time_step" not in dataframe.columns:

        raise ValueError(
            f"{split_name} is missing time_step."
        )

    values = sorted(
        dataframe["time_step"]
        .dropna()
        .unique()
        .tolist()
    )

    expected = list(
        range(
            expected_range[0],
            expected_range[1] + 1,
        )
    )

    if values != expected:

        raise ValueError(
            f"{split_name} time steps are incorrect.\n"
            f"Expected: {expected}\n"
            f"Observed: {values}"
        )


def class_distribution(dataframe):
    """
    Return deterministic class counts.
    """

    counts = (
        dataframe["label"]
        .value_counts()
        .sort_index()
    )

    return {
        str(int(label)): int(count)
        for label, count in counts.items()
    }


def verify_numeric_features(
    dataframe,
    feature_columns,
    split_name,
):
    """
    Ensure all model features are numeric and contain no infinite
    values.

    NaN values are allowed because the downstream model pipelines
    perform median imputation and missingness-indicator handling.
    """

    non_numeric = []

    for column in feature_columns:

        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):

            non_numeric.append(column)

    if non_numeric:

        raise ValueError(
            f"{split_name} contains non-numeric model features: "
            + ", ".join(non_numeric)
        )

    numeric_values = dataframe[
        feature_columns
    ].to_numpy(
        dtype=float
    )

    infinite_count = int(
        np.isinf(numeric_values).sum()
    )

    if infinite_count != 0:

        raise ValueError(
            f"{split_name} contains "
            f"{infinite_count:,} infinite feature values."
        )

    return infinite_count


def extract_split_metadata(split_report):
    """
    Extract the frozen split information from the existing M6.2
    temporal_split.json.

    M6.2 reports may store the information under different nested
    structures. This function recursively searches the JSON for
    recognizable train / validation / test row counts and time
    ranges.

    It does NOT invent a split. The hard-coded values below are the
    already-authoritative M6.2 split and are checked against whatever
    metadata is actually present.
    """

    result = {
        "train_rows": None,
        "validation_rows": None,
        "test_rows": None,

        "train_min_time": None,
        "train_max_time": None,

        "validation_min_time": None,
        "validation_max_time": None,

        "test_min_time": None,
        "test_max_time": None,
    }

    def walk(obj):

        if isinstance(obj, dict):

            lowered = {
                str(key).lower(): value
                for key, value in obj.items()
            }

            # -----------------------------------------------------
            # Row-count candidates.
            # -----------------------------------------------------

            for key, value in lowered.items():

                if not isinstance(
                    value,
                    (int, float),
                ):
                    continue

                if isinstance(value, bool):
                    continue

                integer_value = int(value)

                if (
                    "train" in key
                    and "row" in key
                    and result["train_rows"] is None
                ):

                    result["train_rows"] = integer_value

                elif (
                    (
                        "validation" in key
                        or "val" in key
                    )
                    and "row" in key
                    and result["validation_rows"] is None
                ):

                    result["validation_rows"] = integer_value

                elif (
                    "test" in key
                    and "row" in key
                    and result["test_rows"] is None
                ):

                    result["test_rows"] = integer_value

            # -----------------------------------------------------
            # Recursive processing of split dictionaries.
            # -----------------------------------------------------

            for key, value in obj.items():

                key_lower = str(key).lower()

                if not isinstance(
                    value,
                    dict,
                ):
                    continue

                if (
                    "train" in key_lower
                    and result["train_rows"] is None
                ):

                    candidate = value

                    result["train_rows"] = (
                        candidate.get("rows")
                        or candidate.get("row_count")
                        or candidate.get("n_rows")
                    )

                    result["train_min_time"] = (
                        candidate.get("min_time_step")
                        or candidate.get("time_min")
                        or candidate.get("start_time_step")
                        or candidate.get("min_time")
                    )

                    result["train_max_time"] = (
                        candidate.get("max_time_step")
                        or candidate.get("time_max")
                        or candidate.get("end_time_step")
                        or candidate.get("max_time")
                    )

                elif (
                    (
                        "validation" in key_lower
                        or key_lower in {"val", "valid"}
                    )
                    and result["validation_rows"] is None
                ):

                    candidate = value

                    result["validation_rows"] = (
                        candidate.get("rows")
                        or candidate.get("row_count")
                        or candidate.get("n_rows")
                    )

                    result["validation_min_time"] = (
                        candidate.get("min_time_step")
                        or candidate.get("time_min")
                        or candidate.get("start_time_step")
                        or candidate.get("min_time")
                    )

                    result["validation_max_time"] = (
                        candidate.get("max_time_step")
                        or candidate.get("time_max")
                        or candidate.get("end_time_step")
                        or candidate.get("max_time")
                    )

                elif (
                    "test" in key_lower
                    and result["test_rows"] is None
                ):

                    candidate = value

                    result["test_rows"] = (
                        candidate.get("rows")
                        or candidate.get("row_count")
                        or candidate.get("n_rows")
                    )

                    result["test_min_time"] = (
                        candidate.get("min_time_step")
                        or candidate.get("time_min")
                        or candidate.get("start_time_step")
                        or candidate.get("min_time")
                    )

                    result["test_max_time"] = (
                        candidate.get("max_time_step")
                        or candidate.get("time_max")
                        or candidate.get("end_time_step")
                        or candidate.get("max_time")
                    )

            for value in obj.values():
                walk(value)

        elif isinstance(obj, list):

            for value in obj:
                walk(value)

    walk(split_report)

    return result


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print(
        "M6.16 — BUILD TEMPORAL-SAFE BENCHMARK DATASET"
    )
    print("=" * 70)

    # -------------------------------------------------------------
    # Check required artifacts.
    # -------------------------------------------------------------

    print()
    print(
        "Checking source artifacts..."
    )

    required_paths = [
        FEATURE_PATH,
        LABEL_PATH,
        FEATURE_MANIFEST_PATH,
        TEMPORAL_SPLIT_PATH,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required artifact not found:\n{path}"
            )

        print(
            f"  OK: {path}"
        )

    # -------------------------------------------------------------
    # Load temporal-safe features.
    # -------------------------------------------------------------

    print()
    print(
        "Loading temporal-safe unified features..."
    )

    features = pd.read_parquet(
        FEATURE_PATH
    )

    print(
        f"Feature shape: "
        f"{features.shape}"
    )

    if "txid" not in features.columns:

        raise ValueError(
            "Feature matrix does not contain txid."
        )

    if "time_step" not in features.columns:

        raise ValueError(
            "Feature matrix does not contain time_step."
        )

    features["__txid_norm"] = (
        features["txid"]
        .map(normalize_txid)
    )

    if features["__txid_norm"].isna().any():

        raise ValueError(
            "Feature matrix contains null TXIDs."
        )

    duplicate_features = int(
        features["__txid_norm"]
        .duplicated()
        .sum()
    )

    if duplicate_features != 0:

        raise ValueError(
            f"Feature matrix contains "
            f"{duplicate_features:,} duplicate TXIDs."
        )

    # -------------------------------------------------------------
    # Load labels.
    # -------------------------------------------------------------

    print()
    print(
        "Loading authoritative transaction labels..."
    )

    labels = pd.read_csv(
        LABEL_PATH
    )

    print(
        f"Label shape: "
        f"{labels.shape}"
    )

    required_label_columns = {
        "txId",
        "class",
    }

    missing_label_columns = (
        required_label_columns
        - set(labels.columns)
    )

    if missing_label_columns:

        raise ValueError(
            "Label file is missing columns: "
            + ", ".join(
                sorted(missing_label_columns)
            )
        )

    labels = labels[
        [
            "txId",
            "class",
        ]
    ].copy()

    labels = labels.rename(
        columns={
            "txId": "txid",
            "class": "label",
        }
    )

    labels["__txid_norm"] = (
        labels["txid"]
        .map(normalize_txid)
    )

    if labels["__txid_norm"].isna().any():

        raise ValueError(
            "Label file contains null TXIDs."
        )

    duplicate_labels = int(
        labels["__txid_norm"]
        .duplicated()
        .sum()
    )

    if duplicate_labels != 0:

        raise ValueError(
            f"Label file contains "
            f"{duplicate_labels:,} duplicate TXIDs."
        )

    # -------------------------------------------------------------
    # Validate label values.
    # -------------------------------------------------------------

    labels["label"] = pd.to_numeric(
        labels["label"],
        errors="coerce",
    )

    if labels["label"].isna().any():

        raise ValueError(
            "Label file contains invalid class values."
        )

    labels["label"] = (
        labels["label"]
        .astype(int)
    )

    observed_classes = sorted(
        labels["label"]
        .unique()
        .tolist()
    )

    expected_classes = [1, 2, 3]

    if observed_classes != expected_classes:

        raise ValueError(
            "Unexpected label classes.\n"
            f"Expected: {expected_classes}\n"
            f"Observed: {observed_classes}"
        )

    # -------------------------------------------------------------
    # Feature/label coverage.
    # -------------------------------------------------------------

    print()
    print(
        "Validating feature/label coverage..."
    )

    feature_ids = set(
        features["__txid_norm"]
    )

    label_ids = set(
        labels["__txid_norm"]
    )

    feature_only = (
        feature_ids - label_ids
    )

    label_only = (
        label_ids - feature_ids
    )

    print(
        f"Feature-only TXIDs: "
        f"{len(feature_only):,}"
    )

    print(
        f"Label-only TXIDs:   "
        f"{len(label_only):,}"
    )

    if feature_only or label_only:

        raise ValueError(
            "Feature/label TXID coverage mismatch."
        )

    # -------------------------------------------------------------
    # Join labels.
    # -------------------------------------------------------------

    print()
    print(
        "Joining labels to temporal-safe features..."
    )

    dataset = features.merge(
        labels[
            [
                "__txid_norm",
                "label",
            ]
        ],
        on="__txid_norm",
        how="left",
        validate="one_to_one",
    )

    if len(dataset) != len(features):

        raise ValueError(
            "Join changed the feature row count."
        )

    missing_labels = int(
        dataset["label"].isna().sum()
    )

    print(
        f"Missing labels after join: "
        f"{missing_labels:,}"
    )

    if missing_labels != 0:

        raise ValueError(
            "Joined dataset contains missing labels."
        )

    # -------------------------------------------------------------
    # Load feature manifest.
    # -------------------------------------------------------------

    print()
    print(
        "Loading feature manifest..."
    )

    with open(
        FEATURE_MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        manifest = json.load(f)

    if "model_candidates" not in manifest:

        raise ValueError(
            "Feature manifest does not contain model_candidates."
        )

    if "leakage_sensitive" not in manifest:

        raise ValueError(
            "Feature manifest does not contain leakage_sensitive."
        )

    model_candidates = list(
        manifest["model_candidates"]
    )

    leakage_sensitive = list(
        manifest["leakage_sensitive"]
    )

    print(
        f"Manifest model candidates: "
        f"{len(model_candidates):,}"
    )

    print(
        f"Leakage-sensitive features: "
        f"{len(leakage_sensitive):,}"
    )

    # -------------------------------------------------------------
    # Validate model feature availability.
    # -------------------------------------------------------------

    missing_model_features = [
        feature
        for feature in model_candidates
        if feature not in dataset.columns
    ]

    if missing_model_features:

        raise ValueError(
            "Model candidate features missing from "
            "temporal-safe dataset:\n"
            + "\n".join(
                missing_model_features
            )
        )

    overlapping_features = (
        set(model_candidates)
        & set(leakage_sensitive)
    )

    if overlapping_features:

        raise ValueError(
            "Leakage-sensitive features are incorrectly "
            "present in model_candidates:\n"
            + "\n".join(
                sorted(overlapping_features)
            )
        )

    # -------------------------------------------------------------
    # Load frozen M6.2 split report.
    # -------------------------------------------------------------

    print()
    print(
        "Loading frozen chronological split..."
    )

    with open(
        TEMPORAL_SPLIT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        split_report = json.load(f)

    split_metadata = extract_split_metadata(
        split_report
    )

    # -------------------------------------------------------------
    # M6.2 authoritative split verification.
    #
    # The actual split itself is constructed from the frozen
    # chronological ranges below. The report is used as an
    # independent consistency check.
    # -------------------------------------------------------------

    print()

    if split_metadata["train_rows"] is not None:

        print(
            "  M6.2 reported train rows: "
            f"{int(split_metadata['train_rows']):,}"
        )

        if int(
            split_metadata["train_rows"]
        ) != EXPECTED_TRAIN_ROWS:

            raise ValueError(
                "M6.2 train row count conflicts with "
                "the frozen benchmark split."
            )

    else:

        print(
            "  M6.2 train row count: not explicitly stored"
        )

    if split_metadata["validation_rows"] is not None:

        print(
            "  M6.2 reported validation rows: "
            f"{int(split_metadata['validation_rows']):,}"
        )

        if int(
            split_metadata["validation_rows"]
        ) != EXPECTED_VALIDATION_ROWS:

            raise ValueError(
                "M6.2 validation row count conflicts with "
                "the frozen benchmark split."
            )

    else:

        print(
            "  M6.2 validation row count: "
            "not explicitly stored"
        )

    if split_metadata["test_rows"] is not None:

        print(
            "  M6.2 reported test rows: "
            f"{int(split_metadata['test_rows']):,}"
        )

        if int(
            split_metadata["test_rows"]
        ) != EXPECTED_TEST_ROWS:

            raise ValueError(
                "M6.2 test row count conflicts with "
                "the frozen benchmark split."
            )

    else:

        print(
            "  M6.2 test row count: not explicitly stored"
        )

    print()
    print(
        "Frozen chronological split:"
    )

    print(
        "  Train:       time 1–29"
    )

    print(
        "  Validation:  time 30–39"
    )

    print(
        "  Test:        time 40–49"
    )

    # -------------------------------------------------------------
    # Build chronological partitions.
    # -------------------------------------------------------------

    print()
    print(
        "Building chronological partitions..."
    )

    train = dataset[
        dataset["time_step"].between(
            EXPECTED_TRAIN_RANGE[0],
            EXPECTED_TRAIN_RANGE[1],
        )
    ].copy()

    validation = dataset[
        dataset["time_step"].between(
            EXPECTED_VALIDATION_RANGE[0],
            EXPECTED_VALIDATION_RANGE[1],
        )
    ].copy()

    test = dataset[
        dataset["time_step"].between(
            EXPECTED_TEST_RANGE[0],
            EXPECTED_TEST_RANGE[1],
        )
    ].copy()

    print(
        f"Train rows:       {len(train):,}"
    )

    print(
        f"Validation rows:  {len(validation):,}"
    )

    print(
        f"Test rows:        {len(test):,}"
    )

    # -------------------------------------------------------------
    # Exact row-count validation.
    # -------------------------------------------------------------

    if len(train) != EXPECTED_TRAIN_ROWS:

        raise ValueError(
            "Train row count differs from frozen M6.2 split.\n"
            f"Expected: {EXPECTED_TRAIN_ROWS:,}\n"
            f"Observed: {len(train):,}"
        )

    if len(validation) != EXPECTED_VALIDATION_ROWS:

        raise ValueError(
            "Validation row count differs from frozen M6.2 split.\n"
            f"Expected: {EXPECTED_VALIDATION_ROWS:,}\n"
            f"Observed: {len(validation):,}"
        )

    if len(test) != EXPECTED_TEST_ROWS:

        raise ValueError(
            "Test row count differs from frozen M6.2 split.\n"
            f"Expected: {EXPECTED_TEST_ROWS:,}\n"
            f"Observed: {len(test):,}"
        )

    # -------------------------------------------------------------
    # Exact time-range validation.
    # -------------------------------------------------------------

    validate_time_range(
        train,
        EXPECTED_TRAIN_RANGE,
        "Train",
    )

    validate_time_range(
        validation,
        EXPECTED_VALIDATION_RANGE,
        "Validation",
    )

    validate_time_range(
        test,
        EXPECTED_TEST_RANGE,
        "Test",
    )

    # -------------------------------------------------------------
    # Strict chronological ordering.
    # -------------------------------------------------------------

    max_train_time = int(
        train["time_step"].max()
    )

    min_validation_time = int(
        validation["time_step"].min()
    )

    max_validation_time = int(
        validation["time_step"].max()
    )

    min_test_time = int(
        test["time_step"].min()
    )

    if not (
        max_train_time
        < min_validation_time
        <= max_validation_time
        < min_test_time
    ):

        raise ValueError(
            "Chronological split ordering is invalid."
        )

    # -------------------------------------------------------------
    # Validate no TXID overlap.
    # -------------------------------------------------------------

    train_ids = set(
        train["__txid_norm"]
    )

    validation_ids = set(
        validation["__txid_norm"]
    )

    test_ids = set(
        test["__txid_norm"]
    )

    train_validation_overlap = (
        train_ids
        & validation_ids
    )

    train_test_overlap = (
        train_ids
        & test_ids
    )

    validation_test_overlap = (
        validation_ids
        & test_ids
    )

    print()
    print(
        "TXID overlap checks:"
    )

    print(
        f"  Train ∩ Validation: "
        f"{len(train_validation_overlap):,}"
    )

    print(
        f"  Train ∩ Test:       "
        f"{len(train_test_overlap):,}"
    )

    print(
        f"  Validation ∩ Test:  "
        f"{len(validation_test_overlap):,}"
    )

    if (
        train_validation_overlap
        or train_test_overlap
        or validation_test_overlap
    ):

        raise ValueError(
            "TXID overlap detected between benchmark partitions."
        )

    # -------------------------------------------------------------
    # Validate complete partition coverage.
    # -------------------------------------------------------------

    split_ids = (
        train_ids
        | validation_ids
        | test_ids
    )

    dataset_ids = set(
        dataset["__txid_norm"]
    )

    print()
    print(
        "Checking complete partition coverage..."
    )

    complete_partition_coverage = (
        split_ids == dataset_ids
    )

    print(
        f"Partition coverage: "
        f"{complete_partition_coverage}"
    )

    if not complete_partition_coverage:

        raise ValueError(
            "Benchmark partitions do not cover the "
            "complete temporal-safe dataset."
        )

    if len(dataset) != EXPECTED_TOTAL_ROWS:

        raise ValueError(
            "Total dataset row count differs from frozen M6.2 split."
        )

    # -------------------------------------------------------------
    # Validate numeric model features.
    # -------------------------------------------------------------

    print()
    print(
        "Validating model feature matrices..."
    )

    train_infinite = verify_numeric_features(
        train,
        model_candidates,
        "Train",
    )

    validation_infinite = verify_numeric_features(
        validation,
        model_candidates,
        "Validation",
    )

    test_infinite = verify_numeric_features(
        test,
        model_candidates,
        "Test",
    )

    print(
        f"  Train infinite values:      "
        f"{train_infinite}"
    )

    print(
        f"  Validation infinite values: "
        f"{validation_infinite}"
    )

    print(
        f"  Test infinite values:       "
        f"{test_infinite}"
    )

    # -------------------------------------------------------------
    # Class distributions.
    # -------------------------------------------------------------

    print()
    print(
        "Class distributions:"
    )

    train_classes = class_distribution(
        train
    )

    validation_classes = class_distribution(
        validation
    )

    test_classes = class_distribution(
        test
    )

    full_classes = class_distribution(
        dataset
    )

    print(
        f"  Full:        {full_classes}"
    )

    print(
        f"  Train:       {train_classes}"
    )

    print(
        f"  Validation:  {validation_classes}"
    )

    print(
        f"  Test:        {test_classes}"
    )

    expected_full_classes = {
        "1": 4545,
        "2": 42019,
        "3": 157205,
    }

    if full_classes != expected_full_classes:

        raise ValueError(
            "Full class distribution differs from "
            "the audited labels.\n"
            f"Expected: {expected_full_classes}\n"
            f"Observed: {full_classes}"
        )

    # -------------------------------------------------------------
    # Save clean partitions.
    # -------------------------------------------------------------

    train = train.drop(
        columns=["__txid_norm"]
    )

    validation = validation.drop(
        columns=["__txid_norm"]
    )

    test = test.drop(
        columns=["__txid_norm"]
    )

    print()
    print(
        "Saving temporal-safe benchmark datasets..."
    )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_parquet(
        TRAIN_OUTPUT,
        index=False,
    )

    validation.to_parquet(
        VALIDATION_OUTPUT,
        index=False,
    )

    test.to_parquet(
        TEST_OUTPUT,
        index=False,
    )

    print(
        f"  {TRAIN_OUTPUT}"
    )

    print(
        f"  {VALIDATION_OUTPUT}"
    )

    print(
        f"  {TEST_OUTPUT}"
    )

    # -------------------------------------------------------------
    # Reload and verify saved artifacts.
    # -------------------------------------------------------------

    print()
    print(
        "Reloading saved artifacts for final verification..."
    )

    saved_train = pd.read_parquet(
        TRAIN_OUTPUT
    )

    saved_validation = pd.read_parquet(
        VALIDATION_OUTPUT
    )

    saved_test = pd.read_parquet(
        TEST_OUTPUT
    )

    saved_splits = {
        "train": saved_train,
        "validation": saved_validation,
        "test": saved_test,
    }

    expected_split_sizes = {
        "train": EXPECTED_TRAIN_ROWS,
        "validation": EXPECTED_VALIDATION_ROWS,
        "test": EXPECTED_TEST_ROWS,
    }

    for split_name, frame in saved_splits.items():

        expected_rows = expected_split_sizes[
            split_name
        ]

        if len(frame) != expected_rows:

            raise ValueError(
                f"Saved {split_name} row count mismatch.\n"
                f"Expected: {expected_rows:,}\n"
                f"Observed: {len(frame):,}"
            )

        if "txid" not in frame.columns:

            raise ValueError(
                f"Saved {split_name} is missing txid."
            )

        if "time_step" not in frame.columns:

            raise ValueError(
                f"Saved {split_name} is missing time_step."
            )

        if "label" not in frame.columns:

            raise ValueError(
                f"Saved {split_name} is missing label."
            )

        if frame["txid"].isna().any():

            raise ValueError(
                f"Saved {split_name} contains null TXIDs."
            )

        if frame["txid"].duplicated().any():

            raise ValueError(
                f"Saved {split_name} contains duplicate TXIDs."
            )

    # -------------------------------------------------------------
    # Validate saved time ranges.
    # -------------------------------------------------------------

    validate_time_range(
        saved_train,
        EXPECTED_TRAIN_RANGE,
        "Saved train",
    )

    validate_time_range(
        saved_validation,
        EXPECTED_VALIDATION_RANGE,
        "Saved validation",
    )

    validate_time_range(
        saved_test,
        EXPECTED_TEST_RANGE,
        "Saved test",
    )

    # -------------------------------------------------------------
    # Final report.
    # -------------------------------------------------------------

    report = {
        "stage": "M6.16",

        "task": (
            "build_temporal_safe_benchmark_dataset"
        ),

        "status": "PASS",

        "input_feature_artifact": str(
            FEATURE_PATH
        ),

        "label_artifact": str(
            LABEL_PATH
        ),

        "feature_manifest": str(
            FEATURE_MANIFEST_PATH
        ),

        "temporal_split": str(
            TEMPORAL_SPLIT_PATH
        ),

        "output_artifacts": {
            "train": str(
                TRAIN_OUTPUT
            ),
            "validation": str(
                VALIDATION_OUTPUT
            ),
            "test": str(
                TEST_OUTPUT
            ),
        },

        "feature_statistics": {
            "total_input_columns": int(
                len(features.columns) - 1
            ),

            "model_candidate_features": int(
                len(model_candidates)
            ),

            "leakage_sensitive_features_excluded": int(
                len(leakage_sensitive)
            ),
        },

        "dataset_statistics": {
            "total_rows": int(
                len(dataset)
            ),

            "train_rows": int(
                len(train)
            ),

            "validation_rows": int(
                len(validation)
            ),

            "test_rows": int(
                len(test)
            ),
        },

        "time_ranges": {
            "train": [
                EXPECTED_TRAIN_RANGE[0],
                EXPECTED_TRAIN_RANGE[1],
            ],

            "validation": [
                EXPECTED_VALIDATION_RANGE[0],
                EXPECTED_VALIDATION_RANGE[1],
            ],

            "test": [
                EXPECTED_TEST_RANGE[0],
                EXPECTED_TEST_RANGE[1],
            ],
        },

        "class_distribution": {
            "full": full_classes,
            "train": train_classes,
            "validation": validation_classes,
            "test": test_classes,
        },

        "integrity": {
            "feature_only_txids": int(
                len(feature_only)
            ),

            "label_only_txids": int(
                len(label_only)
            ),

            "train_validation_overlap": int(
                len(train_validation_overlap)
            ),

            "train_test_overlap": int(
                len(train_test_overlap)
            ),

            "validation_test_overlap": int(
                len(validation_test_overlap)
            ),

            "complete_partition_coverage": bool(
                complete_partition_coverage
            ),

            "infinite_values_train": int(
                train_infinite
            ),

            "infinite_values_validation": int(
                validation_infinite
            ),

            "infinite_values_test": int(
                test_infinite
            ),
        },

        "methodology": {
            "split_type": "chronological",

            "train_time_steps": "1-29",

            "validation_time_steps": "30-39",

            "test_time_steps": "40-49",

            "random_split": False,

            "resampling": False,

            "class_balancing": False,

            "leakage_sensitive_features_excluded": True,

            "temporal_safe_graph_features_used": True,

            "future_time_steps_in_training_features": False,

            "same_time_step_graph_information_used": False,
        },

        "next_step": (
            "Benchmark Logistic Regression, Random Forest, "
            "HistGradientBoosting, and XGBoost using these "
            "temporal-safe partitions."
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
        "M6.16 COMPLETE"
    )
    print("=" * 70)

    print(
        f"Total rows:              "
        f"{len(dataset):,}"
    )

    print(
        f"Model features:          "
        f"{len(model_candidates):,}"
    )

    print(
        f"Train:                   "
        f"{len(train):,}"
    )

    print(
        f"Validation:              "
        f"{len(validation):,}"
    )

    print(
        f"Test:                    "
        f"{len(test):,}"
    )

    print(
        f"Train/Test overlap:      "
        f"{len(train_test_overlap):,}"
    )

    print(
        f"Feature-only TXIDs:      "
        f"{len(feature_only):,}"
    )

    print(
        f"Label-only TXIDs:        "
        f"{len(label_only):,}"
    )

    print(
        f"Partition coverage:      "
        f"{complete_partition_coverage}"
    )

    print()
    print(
        "Output artifacts:"
    )

    print(
        f"  {TRAIN_OUTPUT}"
    )

    print(
        f"  {VALIDATION_OUTPUT}"
    )

    print(
        f"  {TEST_OUTPUT}"
    )

    print()
    print(
        "Report:"
    )

    print(
        f"  {REPORT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()