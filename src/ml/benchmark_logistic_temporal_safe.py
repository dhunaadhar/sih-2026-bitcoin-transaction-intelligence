"""
M6.17.1 — TEMPORAL-SAFE LOGISTIC REGRESSION BENCHMARK

Runs the M6.6 Logistic Regression benchmark on the M6.16
temporal-safe benchmark datasets.

This stage intentionally preserves the original M6.6 model
configuration so the global-vs-temporal-safe comparison remains
methodologically comparable.

Input:
    data/evaluation/train_temporal_safe.parquet
    data/evaluation/validation_temporal_safe.parquet
    data/evaluation/test_temporal_safe.parquet
    reports/ml/feature_manifest.json

Output:
    reports/ml/logistic_regression_temporal_safe.json
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "train_temporal_safe.parquet"
)

VALIDATION_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "validation_temporal_safe.parquet"
)

TEST_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "test_temporal_safe.parquet"
)

FEATURE_MANIFEST_PATH = (
    ROOT
    / "reports"
    / "ml"
    / "feature_manifest.json"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "ml"
    / "logistic_regression_temporal_safe.json"
)


# =====================================================================
# HELPERS
# =====================================================================

def multiclass_pr_auc(
    y_true,
    probabilities,
    classes,
):
    """
    Compute macro one-vs-rest Average Precision.

    This is the same conceptual multiclass PR-AUC treatment used
    in the original benchmark.
    """

    scores = []

    y_true_array = np.asarray(
        y_true
    )

    for index, class_value in enumerate(classes):

        binary_true = (
            y_true_array
            == class_value
        ).astype(int)

        score = average_precision_score(
            binary_true,
            probabilities[:, index],
        )

        scores.append(score)

    return float(
        np.mean(scores)
    )


def multiclass_roc_auc(
    y_true,
    probabilities,
):
    """
    Compute macro one-vs-rest ROC-AUC.
    """

    return float(
        roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print(
        "M6.17.1 — TEMPORAL-SAFE LOGISTIC REGRESSION"
    )
    print("=" * 70)

    # -------------------------------------------------------------
    # Check artifacts.
    # -------------------------------------------------------------

    print()
    print(
        "Checking source artifacts..."
    )

    for path in [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
        FEATURE_MANIFEST_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required artifact not found:\n{path}"
            )

        print(
            f"  OK: {path}"
        )

    # -------------------------------------------------------------
    # Load benchmark partitions.
    # -------------------------------------------------------------

    print()
    print(
        "Loading temporal-safe benchmark datasets..."
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

    model_features = list(
        manifest["model_candidates"]
    )

    leakage_sensitive = list(
        manifest["leakage_sensitive"]
    )

    print(
        f"Model features: "
        f"{len(model_features):,}"
    )

    print(
        f"Leakage-sensitive features: "
        f"{len(leakage_sensitive):,}"
    )

    # -------------------------------------------------------------
    # Validate model feature separation.
    # -------------------------------------------------------------

    overlap = (
        set(model_features)
        & set(leakage_sensitive)
    )

    if overlap:

        raise ValueError(
            "Leakage-sensitive features present in model features:\n"
            + "\n".join(
                sorted(overlap)
            )
        )

    # -------------------------------------------------------------
    # Validate feature availability.
    # -------------------------------------------------------------

    for split_name, dataframe in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:

        missing = [
            feature
            for feature in model_features
            if feature not in dataframe.columns
        ]

        if missing:

            raise ValueError(
                f"{split_name} is missing model features:\n"
                + "\n".join(missing)
            )

        if "label" not in dataframe.columns:

            raise ValueError(
                f"{split_name} is missing label."
            )

    # -------------------------------------------------------------
    # Build X/y.
    # -------------------------------------------------------------

    X_train = train[
        model_features
    ]

    y_train = train[
        "label"
    ]

    X_validation = validation[
        model_features
    ]

    y_validation = validation[
        "label"
    ]

    X_test = test[
        model_features
    ]

    y_test = test[
        "label"
    ]

    print()
    print(
        "Class distributions:"
    )

    print(
        f"Train:       "
        f"{y_train.value_counts().sort_index().to_dict()}"
    )

    print(
        f"Validation:  "
        f"{y_validation.value_counts().sort_index().to_dict()}"
    )

    print(
        f"Test:        "
        f"{y_test.value_counts().sort_index().to_dict()}"
    )

    # -------------------------------------------------------------
    # Logistic Regression pipeline.
    #
    # Same configuration as M6.6.
    # -------------------------------------------------------------

    print()
    print(
        "Building Logistic Regression pipeline..."
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
                "scaler",
                StandardScaler(),
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    # -------------------------------------------------------------
    # Train.
    # -------------------------------------------------------------

    print()
    print(
        "Training Logistic Regression..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # -------------------------------------------------------------
    # Predict validation.
    # -------------------------------------------------------------

    print()
    print(
        "Evaluating validation split..."
    )

    validation_predictions = model.predict(
        X_validation
    )

    validation_probabilities = model.predict_proba(
        X_validation
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions,
    )

    validation_macro_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="macro",
    )

    validation_roc_auc = multiclass_roc_auc(
        y_validation,
        validation_probabilities,
    )

    validation_pr_auc = multiclass_pr_auc(
        y_validation,
        validation_probabilities,
        model.named_steps[
            "classifier"
        ].classes_,
    )

    validation_log_loss = log_loss(
        y_validation,
        validation_probabilities,
        labels=model.named_steps[
            "classifier"
        ].classes_,
    )

    # -------------------------------------------------------------
    # Predict test.
    # -------------------------------------------------------------

    print(
        "Evaluating test split..."
    )

    test_predictions = model.predict(
        X_test
    )

    test_probabilities = model.predict_proba(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
    )

    test_roc_auc = multiclass_roc_auc(
        y_test,
        test_probabilities,
    )

    test_pr_auc = multiclass_pr_auc(
        y_test,
        test_probabilities,
        model.named_steps[
            "classifier"
        ].classes_,
    )

    test_log_loss = log_loss(
        y_test,
        test_probabilities,
        labels=model.named_steps[
            "classifier"
        ].classes_,
    )

    # -------------------------------------------------------------
    # Print metrics.
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TEMPORAL-SAFE LOGISTIC REGRESSION RESULTS"
    )
    print("=" * 70)

    print()
    print(
        "Validation:"
    )

    print(
        f"  Accuracy:   {validation_accuracy:.4f}"
    )

    print(
        f"  Macro F1:   {validation_macro_f1:.4f}"
    )

    print(
        f"  ROC-AUC:    {validation_roc_auc:.4f}"
    )

    print(
        f"  PR-AUC:     {validation_pr_auc:.4f}"
    )

    print(
        f"  Log Loss:   {validation_log_loss:.4f}"
    )

    print()
    print(
        "Test:"
    )

    print(
        f"  Accuracy:   {test_accuracy:.4f}"
    )

    print(
        f"  Macro F1:   {test_macro_f1:.4f}"
    )

    print(
        f"  ROC-AUC:    {test_roc_auc:.4f}"
    )

    print(
        f"  PR-AUC:     {test_pr_auc:.4f}"
    )

    print(
        f"  Log Loss:   {test_log_loss:.4f}"
    )

    # -------------------------------------------------------------
    # Per-class test metrics.
    # -------------------------------------------------------------

    classes = (
        model.named_steps[
            "classifier"
        ].classes_
    )

    per_class = {}

    for index, class_value in enumerate(classes):

        binary_true = (
            y_test.to_numpy()
            == class_value
        ).astype(int)

        binary_pred = (
            test_predictions
            == class_value
        ).astype(int)

        per_class[
            str(int(class_value))
        ] = {
            "precision": float(
                __import__(
                    "sklearn.metrics",
                    fromlist=["precision_score"],
                ).precision_score(
                    binary_true,
                    binary_pred,
                    zero_division=0,
                )
            ),

            "recall": float(
                __import__(
                    "sklearn.metrics",
                    fromlist=["recall_score"],
                ).recall_score(
                    binary_true,
                    binary_pred,
                    zero_division=0,
                )
            ),

            "f1": float(
                __import__(
                    "sklearn.metrics",
                    fromlist=["f1_score"],
                ).f1_score(
                    binary_true,
                    binary_pred,
                    zero_division=0,
                )
            ),

            "average_precision": float(
                average_precision_score(
                    binary_true,
                    test_probabilities[
                        :,
                        index,
                    ],
                )
            ),
        }

    # -------------------------------------------------------------
    # Report.
    # -------------------------------------------------------------

    report = {
        "stage": "M6.17.1",

        "model": (
            "LogisticRegression"
        ),

        "status": "PASS",

        "methodology": {
            "feature_source": str(
                ROOT
                / "data"
                / "derived"
                / "unified_features_temporal_safe.parquet"
            ),

            "benchmark_source": (
                "M6.16 temporal-safe benchmark datasets"
            ),

            "split_type": "chronological",

            "train_time_steps": "1-29",

            "validation_time_steps": "30-39",

            "test_time_steps": "40-49",

            "future_information_used": False,

            "same_time_step_graph_information_used": False,

            "random_split": False,

            "resampling": False,

            "class_balancing": "model class_weight=balanced",
        },

        "features": {
            "count": int(
                len(model_features)
            ),

            "leakage_sensitive_excluded": int(
                len(leakage_sensitive)
            ),

            "feature_names": model_features,
        },

        "dataset": {
            "train_rows": int(
                len(train)
            ),

            "validation_rows": int(
                len(validation)
            ),

            "test_rows": int(
                len(test)
            ),

            "train_class_distribution": {
                str(int(key)): int(value)
                for key, value
                in y_train.value_counts()
                .sort_index()
                .items()
            },

            "validation_class_distribution": {
                str(int(key)): int(value)
                for key, value
                in y_validation.value_counts()
                .sort_index()
                .items()
            },

            "test_class_distribution": {
                str(int(key)): int(value)
                for key, value
                in y_test.value_counts()
                .sort_index()
                .items()
            },
        },

        "model_parameters": {
            "max_iter": 1000,

            "class_weight": "balanced",

            "solver": "lbfgs",

            "random_state": 42,

            "imputation": {
                "strategy": "median",
                "add_indicator": True,
            },

            "scaling": "StandardScaler",
        },

        "validation_metrics": {
            "accuracy": float(
                validation_accuracy
            ),

            "macro_f1": float(
                validation_macro_f1
            ),

            "roc_auc_ovr_macro": float(
                validation_roc_auc
            ),

            "macro_pr_auc": float(
                validation_pr_auc
            ),

            "log_loss": float(
                validation_log_loss
            ),
        },

        "test_metrics": {
            "accuracy": float(
                test_accuracy
            ),

            "macro_f1": float(
                test_macro_f1
            ),

            "roc_auc_ovr_macro": float(
                test_roc_auc
            ),

            "macro_pr_auc": float(
                test_pr_auc
            ),

            "log_loss": float(
                test_log_loss
            ),
        },

        "test_per_class": per_class,

        "comparison_reference": {
            "global_m6_6_validation": {
                "accuracy": 0.3638,
                "macro_f1": 0.3623,
                "roc_auc_ovr_macro": 0.7567,
                "macro_pr_auc": 0.5404,
            },

            "global_m6_6_test": {
                "accuracy": 0.2696,
                "macro_f1": 0.2685,
                "roc_auc_ovr_macro": 0.6947,
                "macro_pr_auc": 0.4529,
            },
        },

        "next_step": (
            "Run temporal-safe Random Forest, "
            "HistGradientBoosting, and XGBoost using "
            "the identical M6 benchmark methodology."
        ),
    }

    # -------------------------------------------------------------
    # Save report.
    # -------------------------------------------------------------

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
        "M6.17.1 COMPLETE"
    )
    print("=" * 70)

    print(
        f"Features:       {len(model_features):,}"
    )

    print(
        f"Train:          {len(train):,}"
    )

    print(
        f"Validation:     {len(validation):,}"
    )

    print(
        f"Test:           {len(test):,}"
    )

    print()
    print(
        "Test metrics:"
    )

    print(
        f"  Accuracy:     {test_accuracy:.4f}"
    )

    print(
        f"  Macro F1:     {test_macro_f1:.4f}"
    )

    print(
        f"  ROC-AUC:      {test_roc_auc:.4f}"
    )

    print(
        f"  Macro PR-AUC: {test_pr_auc:.4f}"
    )

    print(
        f"  Log Loss:     {test_log_loss:.4f}"
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