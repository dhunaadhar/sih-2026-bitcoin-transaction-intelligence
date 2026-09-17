from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "evaluation" / "train.parquet"
VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "validation.parquet"
)
TEST_PATH = PROJECT_ROOT / "data" / "evaluation" / "test.parquet"

MANIFEST_PATH = (
    PROJECT_ROOT / "reports" / "ml" / "feature_manifest.json"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "ml"
REPORT_PATH = REPORT_DIR / "xgboost.json"


def main() -> None:
    print("=" * 70)
    print("M6.9 — XGBOOST BENCHMARK")
    print("=" * 70)

    required_files = [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
        MANIFEST_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    train = pd.read_parquet(TRAIN_PATH)
    validation = pd.read_parquet(VALIDATION_PATH)
    test = pd.read_parquet(TEST_PATH)

    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    features = manifest["model_candidates"]
    target = "class"

    X_train = train[features]
    y_train = train[target]

    X_validation = validation[features]
    y_validation = validation[target]

    X_test = test[features]
    y_test = test[target]

    print(f"Features:     {len(features)}")
    print(f"Train rows:   {len(train):,}")
    print(f"Validation:   {len(validation):,}")
    print(f"Test rows:    {len(test):,}")

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    # XGBoost expects zero-based class labels.
    label_mapping = {
        1: 0,
        2: 1,
        3: 2,
    }

    inverse_mapping = {
        value: key
        for key, value in label_mapping.items()
    }

    y_train_encoded = y_train.map(label_mapping)
    y_validation_encoded = y_validation.map(label_mapping)
    y_test_encoded = y_test.map(label_mapping)

    if (
        y_train_encoded.isna().any()
        or y_validation_encoded.isna().any()
        or y_test_encoded.isna().any()
    ):
        raise ValueError(
            "Unexpected class value encountered."
        )

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "classifier",
                xgb.XGBClassifier(
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
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining XGBoost...")
    print("  Trees: 500")
    print("  Max depth: 6")
    print("  Learning rate: 0.04")
    print("  Min child weight: 3")
    print("  Subsample: 0.85")
    print("  Column sampling: 0.85")
    print("  Tree method: hist")

    model.fit(
        X_train,
        y_train_encoded,
    )

    validation_probabilities = model.predict_proba(
        X_validation
    )
    test_probabilities = model.predict_proba(X_test)

    validation_encoded_predictions = model.predict(
        X_validation
    )
    test_encoded_predictions = model.predict(X_test)

    validation_predictions = pd.Series(
        validation_encoded_predictions,
        index=y_validation.index,
    ).map(inverse_mapping)

    test_predictions = pd.Series(
        test_encoded_predictions,
        index=y_test.index,
    ).map(inverse_mapping)

    classes = np.array([1, 2, 3])

    validation_roc_auc = roc_auc_score(
        y_validation,
        validation_probabilities,
        multi_class="ovr",
        average="macro",
    )

    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
        multi_class="ovr",
        average="macro",
    )

    validation_one_hot = pd.get_dummies(
        y_validation
    ).reindex(columns=classes, fill_value=0)

    test_one_hot = pd.get_dummies(
        y_test
    ).reindex(columns=classes, fill_value=0)

    validation_pr_auc = average_precision_score(
        validation_one_hot,
        validation_probabilities,
        average="macro",
    )

    test_pr_auc = average_precision_score(
        test_one_hot,
        test_probabilities,
        average="macro",
    )

    report = {
        "model": "XGBClassifier",
        "task": "three_class_classification",
        "features": features,
        "label_mapping": label_mapping,
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
            "random_state": 42,
        },
        "preprocessing": {
            "imputation": "median",
            "missingness_indicators": True,
            "fit_only_on_training_data": True,
        },
        "validation": {
            "accuracy": float(
                accuracy_score(
                    y_validation,
                    validation_predictions,
                )
            ),
            "macro_precision": float(
                precision_score(
                    y_validation,
                    validation_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_recall": float(
                recall_score(
                    y_validation,
                    validation_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_validation,
                    validation_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "roc_auc_ovr_macro": float(
                validation_roc_auc
            ),
            "pr_auc_macro": float(
                validation_pr_auc
            ),
        },
        "test": {
            "accuracy": float(
                accuracy_score(
                    y_test,
                    test_predictions,
                )
            ),
            "macro_precision": float(
                precision_score(
                    y_test,
                    test_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_recall": float(
                recall_score(
                    y_test,
                    test_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_test,
                    test_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "roc_auc_ovr_macro": float(
                test_roc_auc
            ),
            "pr_auc_macro": float(
                test_pr_auc
            ),
            "classification_report": classification_report(
                y_test,
                test_predictions,
                output_dict=True,
                zero_division=0,
            ),
        },
        "status": "PASS",
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nValidation:")
    print(
        f"  Accuracy:   "
        f"{report['validation']['accuracy']:.4f}"
    )
    print(
        f"  Macro F1:   "
        f"{report['validation']['macro_f1']:.4f}"
    )
    print(
        f"  ROC-AUC:    "
        f"{report['validation']['roc_auc_ovr_macro']:.4f}"
    )
    print(
        f"  PR-AUC:     "
        f"{report['validation']['pr_auc_macro']:.4f}"
    )

    print("\nTest:")
    print(
        f"  Accuracy:   "
        f"{report['test']['accuracy']:.4f}"
    )
    print(
        f"  Macro F1:   "
        f"{report['test']['macro_f1']:.4f}"
    )
    print(
        f"  ROC-AUC:    "
        f"{report['test']['roc_auc_ovr_macro']:.4f}"
    )
    print(
        f"  PR-AUC:     "
        f"{report['test']['pr_auc_macro']:.4f}"
    )

    print(f"\nReport: {REPORT_PATH}")

    print("=" * 70)


if __name__ == "__main__":
    main()