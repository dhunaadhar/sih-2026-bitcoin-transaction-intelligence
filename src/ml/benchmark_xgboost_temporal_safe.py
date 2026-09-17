from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT / "data" / "evaluation" / "train_temporal_safe.parquet"
)
VALIDATION_PATH = (
    ROOT / "data" / "evaluation" / "validation_temporal_safe.parquet"
)
TEST_PATH = (
    ROOT / "data" / "evaluation" / "test_temporal_safe.parquet"
)

MANIFEST_PATH = (
    ROOT / "reports" / "ml" / "feature_manifest.json"
)

OUTPUT_REPORT = (
    ROOT
    / "reports"
    / "ml"
    / "xgboost_temporal_safe.json"
)

RANDOM_STATE = 42
LABEL_COLUMN = "label"


def check_source_artifacts() -> None:
    print("Checking source artifacts...")

    paths = [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
        MANIFEST_PATH,
    ]

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

        print(f"  OK: {path}")


def load_manifest() -> dict:
    print("\nLoading feature manifest...")

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    model_features = manifest["model_candidates"]
    leakage_sensitive = manifest["leakage_sensitive"]

    print(
        f"Model features: {len(model_features)}"
    )

    print(
        f"Leakage-sensitive features: "
        f"{len(leakage_sensitive)}"
    )

    if len(model_features) != 87:
        raise ValueError(
            "Expected 87 model candidates, "
            f"found {len(model_features)}"
        )

    if len(leakage_sensitive) != 4:
        raise ValueError(
            "Expected 4 leakage-sensitive features, "
            f"found {len(leakage_sensitive)}"
        )

    overlap = (
        set(model_features)
        & set(leakage_sensitive)
    )

    if overlap:
        raise ValueError(
            "Leakage-sensitive features are present "
            f"in model candidates: {sorted(overlap)}"
        )

    return manifest


def validate_dataset(
    df: pd.DataFrame,
    model_features: list[str],
    name: str,
) -> None:
    required = {
        "txid",
        "time_step",
        LABEL_COLUMN,
    }

    required.update(model_features)

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{missing}"
        )

    if df["txid"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate txids."
        )

    if df[LABEL_COLUMN].isna().any():
        raise ValueError(
            f"{name} contains missing labels."
        )

    if not pd.api.types.is_numeric_dtype(
        df[LABEL_COLUMN]
    ):
        raise ValueError(
            f"{name} label column is not numeric."
        )

    feature_matrix = df[model_features]

    numeric_matrix = feature_matrix.select_dtypes(
        include=[np.number]
    )

    if numeric_matrix.shape[1] != len(
        model_features
    ):
        non_numeric = sorted(
            set(model_features)
            - set(numeric_matrix.columns)
        )

        raise ValueError(
            f"{name} contains non-numeric "
            f"model features: {non_numeric}"
        )

    # NaN values are expected and handled by the
    # median-imputation pipeline.
    #
    # Positive/negative infinity is invalid.
    infinite_count = np.isinf(
        numeric_matrix.to_numpy(
            dtype=np.float64
        )
    ).sum()

    if infinite_count > 0:
        raise ValueError(
            f"{name} contains "
            f"{infinite_count} infinite feature values."
        )

    missing_cells = int(
        numeric_matrix.isna()
        .sum()
        .sum()
    )

    print(
        f"  {name} validation: "
        f"{missing_cells:,} NaN feature cells allowed; "
        f"{infinite_count:,} infinite values."
    )


def encode_labels(
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    dict[int, int],
    dict[int, int],
]:
    """
    Convert original labels {1,2,3} to contiguous
    XGBoost labels {0,1,2} without modifying source data.

    Returns:
        encoded train/validation/test labels,
        original -> encoded mapping,
        encoded -> original mapping.
    """

    all_labels = sorted(
        set(y_train.unique())
        | set(y_validation.unique())
        | set(y_test.unique())
    )

    expected_labels = [1, 2, 3]

    if all_labels != expected_labels:
        raise ValueError(
            "Unexpected label set. "
            f"Expected {expected_labels}, "
            f"found {all_labels}"
        )

    label_to_encoded = {
        original: encoded
        for encoded, original in enumerate(
            expected_labels
        )
    }

    encoded_to_label = {
        encoded: original
        for original, encoded in label_to_encoded.items()
    }

    y_train_encoded = y_train.map(
        label_to_encoded
    )

    y_validation_encoded = y_validation.map(
        label_to_encoded
    )

    y_test_encoded = y_test.map(
        label_to_encoded
    )

    if (
        y_train_encoded.isna().any()
        or y_validation_encoded.isna().any()
        or y_test_encoded.isna().any()
    ):
        raise ValueError(
            "Label encoding produced missing values."
        )

    return (
        y_train_encoded.astype(int),
        y_validation_encoded.astype(int),
        y_test_encoded.astype(int),
        label_to_encoded,
        encoded_to_label,
    )


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y_original: pd.Series,
    y_encoded: pd.Series,
    encoded_to_label: dict[int, int],
    split_name: str,
) -> dict:
    print(
        f"\nEvaluating {split_name.lower()} split..."
    )

    probabilities = model.predict_proba(X)

    predictions_encoded = model.predict(X)

    predictions_original = pd.Series(
        predictions_encoded,
        index=y_original.index,
    ).map(
        encoded_to_label
    )

    classes_original = [
        encoded_to_label[i]
        for i in sorted(
            encoded_to_label.keys()
        )
    ]

    accuracy = accuracy_score(
        y_original,
        predictions_original,
    )

    macro_f1 = f1_score(
        y_original,
        predictions_original,
        labels=classes_original,
        average="macro",
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_encoded,
        probabilities,
        multi_class="ovr",
        average="macro",
    )

    y_one_hot = pd.get_dummies(
        y_encoded
    )

    y_one_hot = y_one_hot.reindex(
        columns=sorted(
            encoded_to_label.keys()
        ),
        fill_value=0,
    )

    pr_auc = average_precision_score(
        y_one_hot,
        probabilities,
        average="macro",
    )

    loss = log_loss(
        y_encoded,
        probabilities,
        labels=sorted(
            encoded_to_label.keys()
        ),
    )

    class_report = classification_report(
        y_original,
        predictions_original,
        labels=classes_original,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "roc_auc_ovr_macro": float(roc_auc),
        "macro_pr_auc": float(pr_auc),
        "log_loss": float(loss),
        "classification_report": class_report,
    }


def main() -> None:
    print("=" * 70)
    print(
        "M6.17.4 — TEMPORAL-SAFE XGBOOST"
    )
    print("=" * 70)

    check_source_artifacts()

    print(
        "\nLoading temporal-safe "
        "benchmark datasets..."
    )

    train = pd.read_parquet(
        TRAIN_PATH
    )

    validation = pd.read_parquet(
        VALIDATION_PATH
    )

    test = pd.read_parquet(
        TEST_PATH
    )

    print(
        f"Train shape:       {train.shape}"
    )

    print(
        f"Validation shape:  {validation.shape}"
    )

    print(
        f"Test shape:        {test.shape}"
    )

    manifest = load_manifest()

    model_features = manifest[
        "model_candidates"
    ]

    leakage_sensitive = manifest[
        "leakage_sensitive"
    ]

    print("\nClass distributions:")

    train_distribution = (
        train[LABEL_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    validation_distribution = (
        validation[LABEL_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    test_distribution = (
        test[LABEL_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        f"Train:       {train_distribution}"
    )

    print(
        f"Validation:  {validation_distribution}"
    )

    print(
        f"Test:        {test_distribution}"
    )

    print("\nValidating datasets...")

    validate_dataset(
        train,
        model_features,
        "Train",
    )

    validate_dataset(
        validation,
        model_features,
        "Validation",
    )

    validate_dataset(
        test,
        model_features,
        "Test",
    )

    X_train = train[
        model_features
    ]

    X_validation = validation[
        model_features
    ]

    X_test = test[
        model_features
    ]

    y_train_original = train[
        LABEL_COLUMN
    ]

    y_validation_original = validation[
        LABEL_COLUMN
    ]

    y_test_original = test[
        LABEL_COLUMN
    ]

    (
        y_train,
        y_validation,
        y_test,
        label_to_encoded,
        encoded_to_label,
    ) = encode_labels(
        y_train_original,
        y_validation_original,
        y_test_original,
    )

    print(
        "\nXGBoost label encoding:"
    )

    print(
        f"  Original labels: "
        f"{sorted(label_to_encoded.keys())}"
    )

    print(
        f"  Encoded labels:  "
        f"{label_to_encoded}"
    )

    print(
        "\nBuilding XGBoost pipeline..."
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
                model_features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
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

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    print(
        "\nTraining XGBoost..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation_original,
        y_validation,
        encoded_to_label,
        "Validation",
    )

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test_original,
        y_test,
        encoded_to_label,
        "Test",
    )

    print("\n" + "=" * 70)
    print(
        "TEMPORAL-SAFE XGBOOST RESULTS"
    )
    print("=" * 70)

    print("\nValidation:")

    print(
        f"  Accuracy:   "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"  Macro F1:   "
        f"{validation_metrics['macro_f1']:.4f}"
    )

    print(
        f"  ROC-AUC:    "
        f"{validation_metrics['roc_auc_ovr_macro']:.4f}"
    )

    print(
        f"  PR-AUC:     "
        f"{validation_metrics['macro_pr_auc']:.4f}"
    )

    print(
        f"  Log Loss:   "
        f"{validation_metrics['log_loss']:.4f}"
    )

    print("\nTest:")

    print(
        f"  Accuracy:   "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"  Macro F1:   "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"  ROC-AUC:    "
        f"{test_metrics['roc_auc_ovr_macro']:.4f}"
    )

    print(
        f"  PR-AUC:     "
        f"{test_metrics['macro_pr_auc']:.4f}"
    )

    print(
        f"  Log Loss:   "
        f"{test_metrics['log_loss']:.4f}"
    )

    print(
        "\nExtracting XGBoost feature importance..."
    )

    classifier_model = model.named_steps[
        "classifier"
    ]

    preprocessor_model = model.named_steps[
        "preprocessor"
    ]

    transformed_feature_names = (
        preprocessor_model
        .get_feature_names_out(
            model_features
        )
    )

    importances = (
        classifier_model
        .feature_importances_
    )

    if len(transformed_feature_names) != len(
        importances
    ):
        raise ValueError(
            "XGBoost feature importance length "
            "does not match transformed feature count."
        )

    importance_pairs = sorted(
        zip(
            transformed_feature_names,
            importances,
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    print(
        f"Original model features: "
        f"{len(model_features)}"
    )

    print(
        f"Transformed model features: "
        f"{len(transformed_feature_names)}"
    )

    print(
        "\nTop 10 transformed features:"
    )

    top_transformed = []

    for feature_name, importance in (
        importance_pairs[:10]
    ):
        print(
            f"  {feature_name}: "
            f"{importance:.6f}"
        )

        top_transformed.append(
            {
                "feature": feature_name,
                "importance": float(
                    importance
                ),
            }
        )

    print(
        "\nSaving benchmark report..."
    )

    report = {
        "stage": "M6.17.4",
        "model": "XGBClassifier",
        "temporal_safe": True,
        "label_column": LABEL_COLUMN,
        "random_state": RANDOM_STATE,
        "feature_count": len(
            model_features
        ),
        "transformed_feature_count": len(
            transformed_feature_names
        ),
        "model_features": model_features,
        "leakage_sensitive_features_excluded": (
            leakage_sensitive
        ),
        "label_encoding": {
            "original_to_encoded": {
                str(k): int(v)
                for k, v in label_to_encoded.items()
            },
            "encoded_to_original": {
                str(k): int(v)
                for k, v in encoded_to_label.items()
            },
        },
        "missing_value_handling": {
            "strategy": "median",
            "add_indicator": True,
            "nan_allowed": True,
            "infinity_allowed": False,
        },
        "datasets": {
            "train": {
                "rows": int(
                    len(train)
                ),
                "time_min": int(
                    train["time_step"].min()
                ),
                "time_max": int(
                    train["time_step"].max()
                ),
            },
            "validation": {
                "rows": int(
                    len(validation)
                ),
                "time_min": int(
                    validation["time_step"].min()
                ),
                "time_max": int(
                    validation["time_step"].max()
                ),
            },
            "test": {
                "rows": int(
                    len(test)
                ),
                "time_min": int(
                    test["time_step"].min()
                ),
                "time_max": int(
                    test["time_step"].max()
                ),
            },
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
            "n_jobs": -1,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "feature_importance": {
            "top_10_transformed": top_transformed,
        },
    }

    OUTPUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_REPORT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("M6.17.4 COMPLETE")
    print("=" * 70)

    print(
        f"Features:       "
        f"{len(model_features)}"
    )

    print(
        f"Transformed:    "
        f"{len(transformed_feature_names)}"
    )

    print(
        f"Train:          "
        f"{len(train):,}"
    )

    print(
        f"Validation:     "
        f"{len(validation):,}"
    )

    print(
        f"Test:           "
        f"{len(test):,}"
    )

    print("\nTest metrics:")

    print(
        f"  Accuracy:     "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"  Macro F1:     "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"  ROC-AUC:      "
        f"{test_metrics['roc_auc_ovr_macro']:.4f}"
    )

    print(
        f"  Macro PR-AUC: "
        f"{test_metrics['macro_pr_auc']:.4f}"
    )

    print(
        f"  Log Loss:     "
        f"{test_metrics['log_loss']:.4f}"
    )

    print("\nReport:")
    print(
        f"  {OUTPUT_REPORT}"
    )


if __name__ == "__main__":
    main()