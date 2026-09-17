from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
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


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "evaluation" / "train_temporal_safe.parquet"
VALIDATION_PATH = (
    ROOT / "data" / "evaluation" / "validation_temporal_safe.parquet"
)
TEST_PATH = ROOT / "data" / "evaluation" / "test_temporal_safe.parquet"

MANIFEST_PATH = ROOT / "reports" / "ml" / "feature_manifest.json"
OUTPUT_REPORT = (
    ROOT
    / "reports"
    / "ml"
    / "hist_gradient_boosting_temporal_safe.json"
)

RANDOM_STATE = 42
LABEL_COLUMN = "label"


def check_source_artifacts() -> None:
    print("Checking source artifacts...")

    for path in [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
        MANIFEST_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

        print(f"  OK: {path}")


def load_manifest() -> dict:
    print("\nLoading feature manifest...")

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    model_features = manifest["model_candidates"]
    leakage_sensitive = manifest["leakage_sensitive"]

    print(f"Model features: {len(model_features)}")
    print(
        f"Leakage-sensitive features: "
        f"{len(leakage_sensitive)}"
    )

    if len(model_features) != 87:
        raise ValueError(
            f"Expected 87 model candidates, "
            f"found {len(model_features)}"
        )

    if len(leakage_sensitive) != 4:
        raise ValueError(
            f"Expected 4 leakage-sensitive features, "
            f"found {len(leakage_sensitive)}"
        )

    if set(model_features) & set(leakage_sensitive):
        raise ValueError(
            "Leakage-sensitive features are present "
            "in model candidates."
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

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
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

    if numeric_matrix.shape[1] != len(model_features):
        non_numeric = sorted(
            set(model_features) - set(numeric_matrix.columns)
        )

        raise ValueError(
            f"{name} contains non-numeric model features: "
            f"{non_numeric}"
        )

    # NaN is EXPECTED and is handled by SimpleImputer.
    # Only positive/negative infinity is invalid.
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
        numeric_matrix.isna().sum().sum()
    )

    print(
        f"  {name} validation: "
        f"{missing_cells:,} NaN feature cells allowed; "
        f"{infinite_count:,} infinite values."
    )


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
) -> dict:
    print(
        f"\nEvaluating {split_name.lower()} split..."
    )

    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    classes = model.classes_

    roc_auc = roc_auc_score(
        y,
        probabilities,
        multi_class="ovr",
        average="macro",
    )

    y_one_hot = pd.get_dummies(y)

    y_one_hot = y_one_hot.reindex(
        columns=classes,
        fill_value=0,
    )

    pr_auc = average_precision_score(
        y_one_hot,
        probabilities,
        average="macro",
    )

    loss = log_loss(
        y,
        probabilities,
        labels=classes,
    )

    class_report = classification_report(
        y,
        predictions,
        labels=classes,
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
        "M6.17.3 — TEMPORAL-SAFE "
        "HISTGRADIENTBOOSTING"
    )
    print("=" * 70)

    check_source_artifacts()

    print(
        "\nLoading temporal-safe "
        "benchmark datasets..."
    )

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(
        VALIDATION_PATH
    )
    test = pd.read_parquet(TEST_PATH)

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

    model_features = manifest["model_candidates"]
    leakage_sensitive = manifest["leakage_sensitive"]

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

    X_train = train[model_features]
    y_train = train[LABEL_COLUMN]

    X_validation = validation[model_features]
    y_validation = validation[LABEL_COLUMN]

    X_test = test[model_features]
    y_test = test[LABEL_COLUMN]

    print(
        "\nBuilding HistGradientBoosting "
        "pipeline..."
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

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        class_weight="balanced",
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
        "\nTraining HistGradientBoosting..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        "Validation",
    )

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "Test",
    )

    print("\n" + "=" * 70)
    print(
        "TEMPORAL-SAFE "
        "HISTGRADIENTBOOSTING RESULTS"
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
        "\nSaving benchmark report..."
    )

    report = {
        "stage": "M6.17.3",
        "model": (
            "HistGradientBoostingClassifier"
        ),
        "temporal_safe": True,
        "label_column": LABEL_COLUMN,
        "random_state": RANDOM_STATE,
        "feature_count": len(model_features),
        "model_features": model_features,
        "leakage_sensitive_features_excluded": (
            leakage_sensitive
        ),
        "missing_value_handling": {
            "strategy": "median",
            "add_indicator": True,
            "nan_allowed": True,
            "infinity_allowed": False,
        },
        "datasets": {
            "train": {
                "rows": int(len(train)),
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
                "rows": int(len(test)),
                "time_min": int(
                    test["time_step"].min()
                ),
                "time_max": int(
                    test["time_step"].max()
                ),
            },
        },
        "hyperparameters": {
            "learning_rate": 0.08,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
            "class_weight": "balanced",
        },
        "validation": validation_metrics,
        "test": test_metrics,
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
    print("M6.17.3 COMPLETE")
    print("=" * 70)

    print(
        f"Features:       "
        f"{len(model_features)}"
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
    print(f"  {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()