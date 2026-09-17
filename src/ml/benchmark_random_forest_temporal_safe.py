"""
M6.17.2 — TEMPORAL-SAFE RANDOM FOREST BENCHMARK

Runs the M6.7 Random Forest benchmark on the M6.16
temporal-safe benchmark datasets.

The model configuration is intentionally kept identical to M6.7
for a valid global-vs-temporal-safe comparison.

Input:
    data/evaluation/train_temporal_safe.parquet
    data/evaluation/validation_temporal_safe.parquet
    data/evaluation/test_temporal_safe.parquet
    reports/ml/feature_manifest.json

Output:
    reports/ml/random_forest_temporal_safe.json
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


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
    / "random_forest_temporal_safe.json"
)


# =====================================================================
# METRIC HELPERS
# =====================================================================

def multiclass_pr_auc(
    y_true,
    probabilities,
    classes,
):
    """
    Compute macro one-vs-rest Average Precision.
    """

    y_true_array = np.asarray(
        y_true
    )

    scores = []

    for index, class_value in enumerate(classes):

        binary_true = (
            y_true_array == class_value
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


def calculate_metrics(
    y_true,
    predictions,
    probabilities,
    classes,
):
    """
    Calculate the complete benchmark metric set.
    """

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
            )
        ),

        "roc_auc_ovr_macro": float(
            multiclass_roc_auc(
                y_true,
                probabilities,
            )
        ),

        "macro_pr_auc": float(
            multiclass_pr_auc(
                y_true,
                probabilities,
                classes,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=classes,
            )
        ),
    }


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================

def extract_feature_importance(
    model,
    model_features,
):
    """
    Extract Random Forest feature importance after the imputation
    stage.

    Because SimpleImputer(add_indicator=True) adds missingness
    indicator columns, the fitted Random Forest has more features
    than the original 87 model candidates.

    The transformed feature names are therefore obtained directly
    from the fitted imputer rather than assuming a one-to-one
    mapping.
    """

    imputer = model.named_steps[
        "imputer"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    transformed_feature_names = (
        imputer.get_feature_names_out(
            model_features
        )
    )

    importance_values = (
        classifier.feature_importances_
    )

    if len(transformed_feature_names) != len(
        importance_values
    ):

        raise ValueError(
            "Transformed feature-name count does not match "
            "Random Forest feature-importance count.\n"
            f"Names: {len(transformed_feature_names)}\n"
            f"Importances: {len(importance_values)}"
        )

    feature_importance = []

    for feature, importance in zip(
        transformed_feature_names,
        importance_values,
    ):

        feature_importance.append(
            {
                "feature": str(
                    feature
                ),

                "importance": float(
                    importance
                ),
            }
        )

    feature_importance.sort(
        key=lambda item: item[
            "importance"
        ],
        reverse=True,
    )

    return feature_importance


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print(
        "M6.17.2 — TEMPORAL-SAFE RANDOM FOREST"
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
    # Load benchmark datasets.
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
    # Validate feature separation.
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
    # Random Forest pipeline.
    #
    # EXACT M6.7 CONFIGURATION
    # -------------------------------------------------------------

    print()
    print(
        "Building Random Forest pipeline..."
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
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
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
        "Training Random Forest..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # -------------------------------------------------------------
    # Validation predictions.
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

    classes = (
        model.named_steps[
            "classifier"
        ].classes_
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
        validation_probabilities,
        classes,
    )

    # -------------------------------------------------------------
    # Test predictions.
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

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_probabilities,
        classes,
    )

    # -------------------------------------------------------------
    # Print results.
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TEMPORAL-SAFE RANDOM FOREST RESULTS"
    )
    print("=" * 70)

    print()
    print(
        "Validation:"
    )

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

    print()
    print(
        "Test:"
    )

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

    # -------------------------------------------------------------
    # Per-class test metrics.
    # -------------------------------------------------------------

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
                precision_score(
                    binary_true,
                    binary_pred,
                    zero_division=0,
                )
            ),

            "recall": float(
                recall_score(
                    binary_true,
                    binary_pred,
                    zero_division=0,
                )
            ),

            "f1": float(
                f1_score(
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
    # Feature importance.
    # -------------------------------------------------------------

    print()
    print(
        "Extracting Random Forest feature importance..."
    )

    feature_importance = extract_feature_importance(
        model,
        model_features,
    )

    transformed_feature_count = len(
        feature_importance
    )

    print(
        f"Original model features: "
        f"{len(model_features):,}"
    )

    print(
        f"Transformed model features: "
        f"{transformed_feature_count:,}"
    )

    print()
    print(
        "Top 10 transformed features:"
    )

    for item in feature_importance[:10]:

        print(
            f"  {item['feature']}: "
            f"{item['importance']:.6f}"
        )

    # -------------------------------------------------------------
    # Aggregate importance back to original features.
    #
    # Missingness indicators are retained separately in the
    # transformed ranking, while the original feature importance
    # is also computed by summing an original feature's own
    # importance with its corresponding missingness indicator.
    # -------------------------------------------------------------

    aggregated_importance = {}

    for feature in model_features:

        aggregated_importance[
            feature
        ] = 0.0

    for item in feature_importance:

        transformed_name = item[
            "feature"
        ]

        importance = item[
            "importance"
        ]

        if transformed_name.startswith(
            "missingindicator_"
        ):

            original_name = (
                transformed_name[
                    len("missingindicator_"):
                ]
            )

            if original_name in aggregated_importance:

                aggregated_importance[
                    original_name
                ] += importance

        elif transformed_name.startswith(
            "missing_indicator_"
        ):

            original_name = (
                transformed_name[
                    len("missing_indicator_"):
                ]
            )

            if original_name in aggregated_importance:

                aggregated_importance[
                    original_name
                ] += importance

        else:

            if transformed_name in aggregated_importance:

                aggregated_importance[
                    transformed_name
                ] += importance

    aggregated_feature_importance = [
        {
            "feature": feature,
            "importance": float(
                importance
            ),
        }

        for feature, importance
        in aggregated_importance.items()
    ]

    aggregated_feature_importance.sort(
        key=lambda item: item[
            "importance"
        ],
        reverse=True,
    )

    # -------------------------------------------------------------
    # Report.
    # -------------------------------------------------------------

    report = {
        "stage": "M6.17.2",

        "model": (
            "RandomForestClassifier"
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

            "class_balancing": (
                "RandomForest "
                "class_weight=balanced_subsample"
            ),
        },

        "features": {
            "original_model_feature_count": int(
                len(model_features)
            ),

            "transformed_feature_count": int(
                transformed_feature_count
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
            "n_estimators": 300,

            "max_depth": None,

            "min_samples_leaf": 2,

            "class_weight": (
                "balanced_subsample"
            ),

            "n_jobs": -1,

            "random_state": 42,

            "imputation": {
                "strategy": "median",
                "add_indicator": True,
            },
        },

        "validation_metrics": validation_metrics,

        "test_metrics": test_metrics,

        "test_per_class": per_class,

        "feature_importance": {
            "transformed_top_25": (
                feature_importance[:25]
            ),

            "aggregated_original_top_25": (
                aggregated_feature_importance[:25]
            ),
        },

        "comparison_reference": {
            "global_m6_7_validation": {
                "accuracy": 0.8160,
                "macro_f1": 0.5803,
                "roc_auc_ovr_macro": 0.8362,
                "macro_pr_auc": 0.6299,
            },

            "global_m6_7_test": {
                "accuracy": 0.8115,
                "macro_f1": 0.4562,
                "roc_auc_ovr_macro": 0.7992,
                "macro_pr_auc": 0.5232,
            },
        },

        "next_step": (
            "Run temporal-safe "
            "HistGradientBoosting and XGBoost "
            "using the identical M6 benchmark methodology."
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
        "M6.17.2 COMPLETE"
    )
    print("=" * 70)

    print(
        f"Features:       "
        f"{len(model_features):,}"
    )

    print(
        f"Transformed:    "
        f"{transformed_feature_count:,}"
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

    print()
    print(
        "Test metrics:"
    )

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