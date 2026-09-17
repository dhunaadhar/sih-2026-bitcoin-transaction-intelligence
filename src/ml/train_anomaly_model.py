from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    ROOT
    / "data"
    / "evaluation"
    / "train_temporal_safe.parquet"
)

VALIDATION_FILE = (
    ROOT
    / "data"
    / "evaluation"
    / "validation_temporal_safe.parquet"
)

MANIFEST_FILE = (
    ROOT
    / "reports"
    / "ml"
    / "feature_manifest.json"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "production_isolation_forest"
)

MODEL_FILE = MODEL_DIR / "model.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
METADATA_FILE = MODEL_DIR / "metadata.json"

REPORT_DIR = ROOT / "reports" / "ml"
REPORT_FILE = (
    REPORT_DIR
    / "m8_anomaly_model.json"
)

TXID_COLUMN = "txid"
LABEL_COLUMN = "label"
TIME_COLUMN = "time_step"

RANDOM_STATE = 42

# A small contamination value is deliberately used because
# the detector is intended as an anomaly-ranking layer rather
# than a hard classifier.
CONTAMINATION = 0.01

N_ESTIMATORS = 300
MAX_SAMPLES = "auto"
N_JOBS = -1


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def validate_dataset(
    df: pd.DataFrame,
    name: str,
) -> None:
    required = {
        TXID_COLUMN,
        LABEL_COLUMN,
        TIME_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{sorted(missing)}"
        )

    if df[TXID_COLUMN].isna().any():
        raise ValueError(
            f"{name} contains null TXIDs."
        )

    if df[TXID_COLUMN].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate TXIDs."
        )

    numeric = df.select_dtypes(
        include=[np.number]
    )

    if np.isinf(
        numeric.to_numpy()
    ).any():
        raise ValueError(
            f"{name} contains infinite values."
        )


def load_model_features() -> tuple[
    list[str],
    list[str],
]:
    manifest = load_json(
        MANIFEST_FILE
    )

    model_features = manifest.get(
        "model_candidates"
    )

    leakage_sensitive = manifest.get(
        "leakage_sensitive",
        [],
    )

    if not isinstance(
        model_features,
        list,
    ):
        raise ValueError(
            "feature_manifest.json does not contain "
            "a valid model_candidates list."
        )

    if not model_features:
        raise ValueError(
            "No model candidate features found."
        )

    return (
        model_features,
        leakage_sensitive,
    )


def validate_features(
    df: pd.DataFrame,
    model_features: list[str],
    dataset_name: str,
) -> None:
    missing = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing model features: "
            f"{missing}"
        )


def validate_temporal_order(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    train_max = int(
        train[TIME_COLUMN].max()
    )

    validation_min = int(
        validation[TIME_COLUMN].min()
    )

    if train_max >= validation_min:
        raise ValueError(
            "Temporal ordering violated: "
            f"train max={train_max}, "
            f"validation min={validation_min}"
        )


def prepare_numeric_features(
    df: pd.DataFrame,
    model_features: list[str],
) -> pd.DataFrame:
    X = df[
        model_features
    ].copy()

    for column in model_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    values = X.to_numpy(
        dtype=float
    )

    if np.isinf(values).any():
        raise ValueError(
            "Infinite values detected "
            "in model features."
        )

    return X


def calculate_score_statistics(
    scores: np.ndarray,
) -> dict:
    return {
        "count": int(len(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "p01": float(np.percentile(scores, 1)),
        "p05": float(np.percentile(scores, 5)),
        "p25": float(np.percentile(scores, 25)),
        "p75": float(np.percentile(scores, 75)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


def main() -> None:
    print("=" * 72)
    print("M8.1 PRODUCTION ISOLATION FOREST TRAINING")
    print("=" * 72)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: "
            f"{TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: "
            f"{VALIDATION_FILE}"
        )

    train = pd.read_parquet(
        TRAIN_FILE
    )

    validation = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    validate_dataset(
        train,
        "Training dataset",
    )

    validate_dataset(
        validation,
        "Validation dataset",
    )

    model_features, leakage_sensitive = (
        load_model_features()
    )

    print(
        f"Model features: "
        f"{len(model_features)}"
    )

    print(
        f"Leakage-sensitive features excluded: "
        f"{len(leakage_sensitive)}"
    )

    validate_features(
        train,
        model_features,
        "Training dataset",
    )

    validate_features(
        validation,
        model_features,
        "Validation dataset",
    )

    validate_temporal_order(
        train,
        validation,
    )

    train_ids = set(
        train[TXID_COLUMN]
    )

    validation_ids = set(
        validation[TXID_COLUMN]
    )

    overlap = train_ids.intersection(
        validation_ids
    )

    if overlap:
        raise ValueError(
            "Training/validation TXID overlap: "
            f"{len(overlap)}"
        )

    X_train = prepare_numeric_features(
        train,
        model_features,
    )

    X_validation = prepare_numeric_features(
        validation,
        model_features,
    )

    print("\nFitting imputer...")

    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )

    X_train_transformed = (
        imputer.fit_transform(
            X_train
        )
    )

    X_validation_transformed = (
        imputer.transform(
            X_validation
        )
    )

    print(
        "Transformed training shape: "
        f"{X_train_transformed.shape}"
    )

    print(
        "Transformed validation shape: "
        f"{X_validation_transformed.shape}"
    )

    print("\nTraining Isolation Forest...")

    detector = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    detector.fit(
        X_train_transformed
    )

    print("Training complete.")

    # sklearn's decision_function:
    #   larger = more normal
    #
    # We expose anomaly_score as:
    #   larger = more anomalous
    #
    # This makes downstream risk fusion intuitive.
    train_normality = detector.decision_function(
        X_train_transformed
    )

    validation_normality = detector.decision_function(
        X_validation_transformed
    )

    train_anomaly_score = (
        -train_normality
    )

    validation_anomaly_score = (
        -validation_normality
    )

    train_predictions = detector.predict(
        X_train_transformed
    )

    validation_predictions = detector.predict(
        X_validation_transformed
    )

    train_anomaly_count = int(
        np.sum(train_predictions == -1)
    )

    validation_anomaly_count = int(
        np.sum(validation_predictions == -1)
    )

    transformed_feature_names = (
        imputer
        .get_feature_names_out(
            model_features
        )
        .tolist()
    )

    model_package = {
        "imputer": imputer,
        "detector": detector,
        "model_features": model_features,
        "transformed_feature_names": (
            transformed_feature_names
        ),
        "random_state": RANDOM_STATE,
        "contamination": CONTAMINATION,
        "model_type": "IsolationForest",
        "milestone": "M8.1",
    }

    joblib.dump(
        model_package,
        MODEL_FILE,
        compress=3,
    )

    schema = {
        "schema_version": "1.0",
        "model_type": "IsolationForest",
        "original_feature_count": len(
            model_features
        ),
        "transformed_feature_count": len(
            transformed_feature_names
        ),
        "model_features": model_features,
        "transformed_feature_names": (
            transformed_feature_names
        ),
        "leakage_sensitive_features_excluded": (
            leakage_sensitive
        ),
        "required_input_columns": (
            model_features
        ),
        "transaction_id_column": (
            TXID_COLUMN
        ),
        "time_column": TIME_COLUMN,
        "output_semantics": {
            "anomaly_score": (
                "Higher values indicate more anomalous "
                "behavior according to Isolation Forest."
            ),
            "decision_function": (
                "Higher values indicate more normal "
                "behavior in sklearn's native convention."
            ),
        },
    }

    with SCHEMA_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            schema,
            f,
            indent=2,
        )

    metadata = {
        "milestone": "M8.1",
        "model_name": (
            "production_isolation_forest"
        ),
        "model_type": (
            "IsolationForest"
        ),
        "training_dataset": str(
            TRAIN_FILE
        ),
        "validation_dataset": str(
            VALIDATION_FILE
        ),
        "frozen_test_set_used": False,
        "training_rows": int(
            len(train)
        ),
        "validation_rows": int(
            len(validation)
        ),
        "training_time_range": [
            int(train[TIME_COLUMN].min()),
            int(train[TIME_COLUMN].max()),
        ],
        "validation_time_range": [
            int(
                validation[TIME_COLUMN].min()
            ),
            int(
                validation[TIME_COLUMN].max()
            ),
        ],
        "original_feature_count": len(
            model_features
        ),
        "transformed_feature_count": len(
            transformed_feature_names
        ),
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS,
            "max_samples": MAX_SAMPLES,
            "contamination": CONTAMINATION,
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS,
        },
        "preprocessing": {
            "imputer": "SimpleImputer",
            "strategy": "median",
            "add_indicator": True,
        },
        "score_semantics": {
            "anomaly_score": (
                "negative sklearn decision_function; "
                "higher means more anomalous"
            ),
        },
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    report = {
        "milestone": "M8.1",
        "status": "PASS",
        "model": "IsolationForest",
        "production_candidate": True,
        "training_rows": int(
            len(train)
        ),
        "validation_rows": int(
            len(validation)
        ),
        "training_time_range": [
            int(train[TIME_COLUMN].min()),
            int(train[TIME_COLUMN].max()),
        ],
        "validation_time_range": [
            int(
                validation[TIME_COLUMN].min()
            ),
            int(
                validation[TIME_COLUMN].max()
            ),
        ],
        "original_feature_count": len(
            model_features
        ),
        "transformed_feature_count": len(
            transformed_feature_names
        ),
        "frozen_test_set_used": False,
        "contamination": CONTAMINATION,
        "training_anomaly_count": (
            train_anomaly_count
        ),
        "training_anomaly_rate": float(
            train_anomaly_count
            / len(train)
        ),
        "validation_anomaly_count": (
            validation_anomaly_count
        ),
        "validation_anomaly_rate": float(
            validation_anomaly_count
            / len(validation)
        ),
        "training_score_statistics": (
            calculate_score_statistics(
                train_anomaly_score
            )
        ),
        "validation_score_statistics": (
            calculate_score_statistics(
                validation_anomaly_score
            )
        ),
        "artifacts": {
            "model": str(
                MODEL_FILE
            ),
            "schema": str(
                SCHEMA_FILE
            ),
            "metadata": str(
                METADATA_FILE
            ),
        },
        "methodology_note": (
            "Isolation Forest is used as an independent "
            "unsupervised anomaly-ranking layer. Its anomaly "
            "score is not a probability, identity attribution, "
            "or proof of illicit behavior."
        ),
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("M8.1 COMPLETE")
    print("=" * 72)

    print(
        f"Model:    {MODEL_FILE}"
    )

    print(
        f"Schema:   {SCHEMA_FILE}"
    )

    print(
        f"Metadata: {METADATA_FILE}"
    )

    print(
        f"Report:   {REPORT_FILE}"
    )

    print(
        f"\nTraining anomaly rate: "
        f"{train_anomaly_count / len(train):.4%}"
    )

    print(
        f"Validation anomaly rate: "
        f"{validation_anomaly_count / len(validation):.4%}"
    )

    print(
        "\nFrozen test set used: FALSE"
    )


if __name__ == "__main__":
    main()