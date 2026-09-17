from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data" / "evaluation" / "train_temporal_safe.parquet"
VALIDATION_FILE = ROOT / "data" / "evaluation" / "validation_temporal_safe.parquet"
MANIFEST_FILE = ROOT / "reports" / "ml" / "feature_manifest.json"

MODEL_DIR = ROOT / "models" / "production_xgboost"

MODEL_FILE = MODEL_DIR / "model.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
METADATA_FILE = MODEL_DIR / "metadata.json"

REPORT_DIR = ROOT / "reports" / "ml"
REPORT_FILE = REPORT_DIR / "m7_production_model.json"


RANDOM_STATE = 42

LABEL_COLUMN = "label"
TXID_COLUMN = "txid"
TIME_COLUMN = "time_step"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_dataset(
    df: pd.DataFrame,
    name: str,
) -> None:
    required = {
        LABEL_COLUMN,
        TXID_COLUMN,
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
            f"{name} contains null transaction IDs."
        )

    if df[TXID_COLUMN].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate transaction IDs."
        )

    if df[LABEL_COLUMN].isna().any():
        raise ValueError(
            f"{name} contains missing labels."
        )

    numeric = df.select_dtypes(include=[np.number])

    if np.isinf(numeric.to_numpy()).any():
        raise ValueError(
            f"{name} contains infinite numeric values."
        )


def load_model_features() -> tuple[list[str], list[str]]:
    manifest = load_json(MANIFEST_FILE)

    model_features = manifest.get("model_candidates")
    leakage_sensitive = manifest.get("leakage_sensitive", [])

    if not isinstance(model_features, list):
        raise ValueError(
            "feature_manifest.json does not contain "
            "a valid 'model_candidates' list."
        )

    if not isinstance(leakage_sensitive, list):
        raise ValueError(
            "feature_manifest.json does not contain "
            "a valid 'leakage_sensitive' list."
        )

    if not model_features:
        raise ValueError(
            "No model candidate features found."
        )

    return model_features, leakage_sensitive


def validate_feature_columns(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_features: list[str],
) -> None:
    missing_train = [
        col for col in model_features
        if col not in train.columns
    ]

    missing_validation = [
        col for col in model_features
        if col not in validation.columns
    ]

    if missing_train:
        raise ValueError(
            "Training dataset missing model features: "
            f"{missing_train}"
        )

    if missing_validation:
        raise ValueError(
            "Validation dataset missing model features: "
            f"{missing_validation}"
        )


def validate_temporal_order(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    train_max = int(train[TIME_COLUMN].max())
    validation_min = int(validation[TIME_COLUMN].min())

    if train_max >= validation_min:
        raise ValueError(
            "Temporal ordering violated: "
            f"train max time={train_max}, "
            f"validation min time={validation_min}."
        )


def build_training_pipeline(
    model_features: list[str],
):
    """
    Build the exact preprocessing + XGBoost pipeline used by
    the temporal-safe benchmark.

    XGBoost internally receives encoded labels:
        original 1 -> 0
        original 2 -> 1
        original 3 -> 2
    """

    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )

    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.04,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return imputer, classifier


def main() -> None:
    print("=" * 72)
    print("M7.1 PRODUCTION XGBOOST TRAINING")
    print("=" * 72)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Missing training dataset: {TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Missing validation dataset: {VALIDATION_FILE}"
        )

    train = pd.read_parquet(TRAIN_FILE)
    validation = pd.read_parquet(VALIDATION_FILE)

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")

    validate_dataset(train, "Training dataset")
    validate_dataset(validation, "Validation dataset")

    model_features, leakage_sensitive = (
        load_model_features()
    )

    print(f"Model features: {len(model_features)}")
    print(
        f"Leakage-sensitive features excluded: "
        f"{len(leakage_sensitive)}"
    )

    validate_feature_columns(
        train,
        validation,
        model_features,
    )

    validate_temporal_order(
        train,
        validation,
    )

    train_ids = set(train[TXID_COLUMN])
    validation_ids = set(validation[TXID_COLUMN])

    overlap = train_ids.intersection(validation_ids)

    if overlap:
        raise ValueError(
            "Training and validation transaction overlap detected: "
            f"{len(overlap)}"
        )

    X_train = train[model_features].copy()
    X_validation = validation[model_features].copy()

    y_train_original = train[LABEL_COLUMN].astype(int)
    y_validation_original = validation[LABEL_COLUMN].astype(int)

    valid_labels = {1, 2, 3}

    if set(y_train_original.unique()) - valid_labels:
        raise ValueError(
            "Unexpected training labels: "
            f"{sorted(set(y_train_original.unique()) - valid_labels)}"
        )

    if set(y_validation_original.unique()) - valid_labels:
        raise ValueError(
            "Unexpected validation labels: "
            f"{sorted(set(y_validation_original.unique()) - valid_labels)}"
        )

    label_mapping = {
        1: 0,
        2: 1,
        3: 2,
    }

    inverse_label_mapping = {
        0: 1,
        1: 2,
        2: 3,
    }

    y_train = y_train_original.map(label_mapping).astype(int)

    y_validation = (
        y_validation_original
        .map(label_mapping)
        .astype(int)
    )

    print("\nTraining class distribution:")
    print(y_train_original.value_counts().sort_index().to_string())

    print("\nValidation class distribution:")
    print(
        y_validation_original
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFitting preprocessing...")

    imputer, classifier = build_training_pipeline(
        model_features
    )

    X_train_transformed = imputer.fit_transform(X_train)
    X_validation_transformed = imputer.transform(
        X_validation
    )

    print(
        "Transformed training shape: "
        f"{X_train_transformed.shape}"
    )

    print(
        "Transformed validation shape: "
        f"{X_validation_transformed.shape}"
    )

    print("\nTraining XGBoost...")

    classifier.fit(
        X_train_transformed,
        y_train,
        eval_set=[
            (
                X_validation_transformed,
                y_validation,
            )
        ],
        verbose=False,
    )

    print("Training complete.")

    model_package = {
        "imputer": imputer,
        "classifier": classifier,
        "model_features": model_features,
        "label_mapping": label_mapping,
        "inverse_label_mapping": inverse_label_mapping,
        "random_state": RANDOM_STATE,
        "model_type": "XGBoost",
        "milestone": "M7.1",
    }

    joblib.dump(
        model_package,
        MODEL_FILE,
        compress=3,
    )

    transformed_feature_names = (
        imputer.get_feature_names_out(model_features)
    )

    schema = {
        "schema_version": "1.0",
        "model_type": "XGBoost",
        "original_feature_count": len(model_features),
        "transformed_feature_count": len(
            transformed_feature_names
        ),
        "model_features": model_features,
        "transformed_feature_names": (
            transformed_feature_names.tolist()
        ),
        "leakage_sensitive_features_excluded": (
            leakage_sensitive
        ),
        "required_input_columns": model_features,
        "label_column": LABEL_COLUMN,
        "transaction_id_column": TXID_COLUMN,
        "time_column": TIME_COLUMN,
        "label_mapping": {
            str(k): v
            for k, v in label_mapping.items()
        },
        "inverse_label_mapping": {
            str(k): v
            for k, v in inverse_label_mapping.items()
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
        "milestone": "M7.1",
        "model_name": "production_xgboost",
        "model_type": "XGBoostClassifier",
        "training_dataset": str(TRAIN_FILE),
        "validation_dataset": str(VALIDATION_FILE),
        "frozen_test_set_used": False,
        "training_time_range": [
            int(train[TIME_COLUMN].min()),
            int(train[TIME_COLUMN].max()),
        ],
        "validation_time_range": [
            int(validation[TIME_COLUMN].min()),
            int(validation[TIME_COLUMN].max()),
        ],
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "original_feature_count": len(model_features),
        "transformed_feature_count": int(
            len(transformed_feature_names)
        ),
        "label_mapping": {
            str(k): v
            for k, v in label_mapping.items()
        },
        "hyperparameters": {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.04,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
        },
        "preprocessing": {
            "imputer": "SimpleImputer",
            "strategy": "median",
            "add_indicator": True,
        },
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
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
        "milestone": "M7.1",
        "status": "PASS",
        "model": "XGBoost",
        "production_candidate": True,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "training_time_range": [
            int(train[TIME_COLUMN].min()),
            int(train[TIME_COLUMN].max()),
        ],
        "validation_time_range": [
            int(validation[TIME_COLUMN].min()),
            int(validation[TIME_COLUMN].max()),
        ],
        "original_feature_count": len(model_features),
        "transformed_feature_count": int(
            len(transformed_feature_names)
        ),
        "frozen_test_set_used": False,
        "label_distribution": {
            str(k): int(v)
            for k, v in (
                y_train_original
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "artifacts": {
            "model": str(MODEL_FILE),
            "schema": str(SCHEMA_FILE),
            "metadata": str(METADATA_FILE),
        },
        "label_mapping": {
            str(k): v
            for k, v in label_mapping.items()
        },
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
    print("M7.1 COMPLETE")
    print("=" * 72)

    print(f"Model:    {MODEL_FILE}")
    print(f"Schema:   {SCHEMA_FILE}")
    print(f"Metadata: {METADATA_FILE}")
    print(f"Report:   {REPORT_FILE}")

    print("\nFrozen test set used: FALSE")
    print(
        "Production model trained on temporal-safe "
        "development data only."
    )


if __name__ == "__main__":
    main()