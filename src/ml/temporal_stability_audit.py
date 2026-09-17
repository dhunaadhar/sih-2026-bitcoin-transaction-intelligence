"""
M6.11 — Temporal Stability + Graph Leakage Audit

Purpose
-------
1. Evaluate XGBoost stability across early and late portions of the
   untouched test period.
2. Measure per-class ROC-AUC and PR-AUC.
3. Measure Precision@K / Recall@K for each class.
4. Identify graph/entity features that may contain future information.
5. Produce a machine-readable audit report.

IMPORTANT
---------
This script does NOT modify the existing benchmark dataset or model.
It is an audit gate before final model selection.

Test period:
    Early = time steps 40–44
    Late  = time steps 45–49

Class IDs remain numeric: 1, 2, 3.
No semantic interpretation is assigned to the classes.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "evaluation" / "train.parquet"
TEST_PATH = ROOT / "data" / "evaluation" / "test.parquet"

MANIFEST_PATH = ROOT / "reports" / "ml" / "feature_manifest.json"

OUTPUT_PATH = ROOT / "reports" / "ml" / "temporal_stability_audit.json"


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

RANDOM_STATE = 42

EARLY_TEST_START = 40
EARLY_TEST_END = 44

LATE_TEST_START = 45
LATE_TEST_END = 49

TOP_K_VALUES = [50, 100, 250, 500, 1000]

CLASSES = np.array([1, 2, 3])

XGB_PARAMS = {
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
    "random_state": RANDOM_STATE,
}


# ---------------------------------------------------------------------
# GRAPH FEATURE DEFINITIONS
# ---------------------------------------------------------------------

GRAPH_KEYWORDS = [
    "cosponsor",
    "cluster",
    "repeated",
    "shared",
    "entity",
    "graph",
]

GRAPH_FEATURES_EXPLICIT = [
    "input_repeated_cosponsor_tx_count",
    "input_repeated_cosponsor_tx_ratio",
    "output_repeated_cosponsor_tx_count",
    "output_repeated_cosponsor_tx_ratio",
    "input_max_repeated_cosponsor_degree",
    "output_max_repeated_cosponsor_degree",
    "input_max_shared_transaction_count",
    "output_max_shared_transaction_count",
    "input_distinct_cosponsor_cluster_t1",
    "input_distinct_cosponsor_cluster_t2",
    "input_distinct_cosponsor_cluster_t3",
    "output_distinct_cosponsor_cluster_t1",
    "output_distinct_cosponsor_cluster_t2",
    "output_distinct_cosponsor_cluster_t3",
    "input_max_cosponsor_cluster_size_t1",
    "input_max_cosponsor_cluster_size_t2",
    "input_max_cosponsor_cluster_size_t3",
    "output_max_cosponsor_cluster_size_t1",
    "output_max_cosponsor_cluster_size_t2",
    "output_max_cosponsor_cluster_size_t3",
]


# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------

def load_manifest():
    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_data():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    if "class" not in train.columns:
        raise ValueError("Training dataset has no class column.")

    if "class" not in test.columns:
        raise ValueError("Test dataset has no class column.")

    if "time_step" not in train.columns:
        raise ValueError("Training dataset has no time_step.")

    if "time_step" not in test.columns:
        raise ValueError("Test dataset has no time_step.")

    return train, test


# ---------------------------------------------------------------------
# FEATURE PREPARATION
# ---------------------------------------------------------------------

def prepare_features(
    train,
    test,
    feature_names,
):
    missing = [
        feature
        for feature in feature_names
        if feature not in train.columns
        or feature not in test.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X_train = train[feature_names].copy()
    X_test = test[feature_names].copy()

    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    return X_train, X_test


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true,
    probabilities,
):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    predictions = (
        np.argmax(probabilities, axis=1) + 1
    )

    metrics = {
        "rows": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
            )
        ),
    }

    # -------------------------------------------------------------
    # Macro ROC-AUC
    # -------------------------------------------------------------

    try:
        metrics["macro_roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=CLASSES,
            )
        )
    except ValueError:
        metrics["macro_roc_auc"] = None

    # -------------------------------------------------------------
    # Macro PR-AUC
    # -------------------------------------------------------------

    y_onehot = np.column_stack([
        (y_true == class_id).astype(int)
        for class_id in CLASSES
    ])

    per_class_pr_auc = {}

    for index, class_id in enumerate(CLASSES):

        positives = int(
            np.sum(y_true == class_id)
        )

        negatives = int(
            np.sum(y_true != class_id)
        )

        if positives == 0 or negatives == 0:
            per_class_pr_auc[str(int(class_id))] = None
            continue

        per_class_pr_auc[str(int(class_id))] = float(
            average_precision_score(
                y_onehot[:, index],
                probabilities[:, index],
            )
        )

    valid_pr = [
        value
        for value in per_class_pr_auc.values()
        if value is not None
    ]

    metrics["macro_pr_auc"] = (
        float(np.mean(valid_pr))
        if valid_pr
        else None
    )

    metrics["per_class_pr_auc"] = per_class_pr_auc

    # -------------------------------------------------------------
    # Per-class ROC-AUC
    # -------------------------------------------------------------

    per_class_roc_auc = {}

    for index, class_id in enumerate(CLASSES):

        positives = int(
            np.sum(y_true == class_id)
        )

        negatives = int(
            np.sum(y_true != class_id)
        )

        if positives == 0 or negatives == 0:
            per_class_roc_auc[str(int(class_id))] = None
            continue

        per_class_roc_auc[str(int(class_id))] = float(
            roc_auc_score(
                y_onehot[:, index],
                probabilities[:, index],
            )
        )

    metrics["per_class_roc_auc"] = per_class_roc_auc

    return metrics


def ranking_metrics(
    y_true,
    probabilities,
):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    output = {}

    for class_index, class_id in enumerate(CLASSES):

        scores = probabilities[:, class_index]

        order = np.argsort(-scores)

        total_positives = int(
            np.sum(y_true == class_id)
        )

        class_result = {
            "total_positives": total_positives,
            "top_k": {},
        }

        for k in TOP_K_VALUES:

            actual_k = min(
                k,
                len(y_true),
            )

            selected = order[:actual_k]

            hits = int(
                np.sum(
                    y_true[selected] == class_id
                )
            )

            precision = (
                hits / actual_k
                if actual_k > 0
                else 0.0
            )

            recall = (
                hits / total_positives
                if total_positives > 0
                else 0.0
            )

            class_result["top_k"][str(k)] = {
                "hits": hits,
                "precision": float(precision),
                "recall": float(recall),
            }

        output[str(int(class_id))] = class_result

    return output


# ---------------------------------------------------------------------
# TEMPORAL WINDOWS
# ---------------------------------------------------------------------

def evaluate_window(
    name,
    dataframe,
    probabilities,
):
    y_true = dataframe["class"].to_numpy()

    metrics = calculate_metrics(
        y_true,
        probabilities,
    )

    ranking = ranking_metrics(
        y_true,
        probabilities,
    )

    time_min = int(
        dataframe["time_step"].min()
    )

    time_max = int(
        dataframe["time_step"].max()
    )

    return {
        "name": name,
        "time_step_min": time_min,
        "time_step_max": time_max,
        "metrics": metrics,
        "ranking": ranking,
    }


# ---------------------------------------------------------------------
# GRAPH FEATURE AUDIT
# ---------------------------------------------------------------------

def identify_graph_features(
    feature_names,
    test,
):
    """
    Identify graph/entity-related features using both explicit names
    and conservative keyword matching.

    This does NOT claim that a feature is actually temporally leaky.
    It only identifies features requiring provenance review.
    """

    identified = []

    for feature in feature_names:

        lower = feature.lower()

        explicit_match = (
            feature in GRAPH_FEATURES_EXPLICIT
        )

        keyword_match = any(
            keyword in lower
            for keyword in GRAPH_KEYWORDS
        )

        if explicit_match or keyword_match:

            identified.append(feature)

    identified = sorted(
        set(identified)
    )

    audit = []

    for feature in identified:

        series = test[feature]

        non_null = series.notna()

        non_null_count = int(
            non_null.sum()
        )

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        if non_null_count > 0:

            first_non_null_time = int(
                test.loc[
                    non_null,
                    "time_step"
                ].min()
            )

            last_non_null_time = int(
                test.loc[
                    non_null,
                    "time_step"
                ].max()
            )

        else:

            first_non_null_time = None
            last_non_null_time = None

        audit.append(
            {
                "feature": feature,
                "non_null_count": non_null_count,
                "unique_count": unique_count,
                "first_non_null_test_time": (
                    first_non_null_time
                ),
                "last_non_null_test_time": (
                    last_non_null_time
                ),
                "requires_temporal_provenance_review": True,
            }
        )

    return audit


# ---------------------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------------------

def get_graph_feature_importance(
    model,
    feature_names,
):
    """
    XGBoost tree importance for graph-related candidate features.

    Importance is reported for audit purposes only.
    """

    importance = model.feature_importances_

    rows = []

    for feature, value in zip(
        feature_names,
        importance,
    ):

        lower = feature.lower()

        if (
            feature in GRAPH_FEATURES_EXPLICIT
            or any(
                keyword in lower
                for keyword in GRAPH_KEYWORDS
            )
        ):
            rows.append(
                {
                    "feature": feature,
                    "importance": float(value),
                }
            )

    rows.sort(
        key=lambda row: row["importance"],
        reverse=True,
    )

    return rows


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("M6.11 — TEMPORAL STABILITY + GRAPH LEAKAGE AUDIT")
    print("=" * 70)

    # -------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------

    train, test = load_data()
    manifest = load_manifest()

    feature_names = manifest["model_candidates"]

    print(
        f"Features:     {len(feature_names)}"
    )

    print(
        f"Train rows:   {len(train):,}"
    )

    print(
        f"Test rows:    {len(test):,}"
    )

    # -------------------------------------------------------------
    # PREPARE
    # -------------------------------------------------------------

    X_train, X_test = prepare_features(
        train,
        test,
        feature_names,
    )

    y_train = train["class"].to_numpy()

    # -------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------

    print()
    print("Training XGBoost...")

    model = xgb.XGBClassifier(
        **XGB_PARAMS
    )

    model.fit(
        X_train,
        y_train - 1,
    )

    # -------------------------------------------------------------
    # TEST PREDICTIONS
    # -------------------------------------------------------------

    print(
        "Generating test probabilities..."
    )

    test_probabilities = model.predict_proba(
        X_test
    )

    # -------------------------------------------------------------
    # WINDOWS
    # -------------------------------------------------------------

    early_mask = (
        (test["time_step"] >= EARLY_TEST_START)
        & (test["time_step"] <= EARLY_TEST_END)
    )

    late_mask = (
        (test["time_step"] >= LATE_TEST_START)
        & (test["time_step"] <= LATE_TEST_END)
    )

    early_test = test.loc[
        early_mask
    ].copy()

    late_test = test.loc[
        late_mask
    ].copy()

    early_probabilities = test_probabilities[
        early_mask.to_numpy()
    ]

    late_probabilities = test_probabilities[
        late_mask.to_numpy()
    ]

    print()
    print(
        f"Early test rows: {len(early_test):,}"
    )

    print(
        f"Late test rows:  {len(late_test):,}"
    )

    # -------------------------------------------------------------
    # EVALUATE
    # -------------------------------------------------------------

    early_results = evaluate_window(
        "early_test",
        early_test,
        early_probabilities,
    )

    late_results = evaluate_window(
        "late_test",
        late_test,
        late_probabilities,
    )

    full_results = evaluate_window(
        "full_test",
        test,
        test_probabilities,
    )

    # -------------------------------------------------------------
    # TEMPORAL DELTAS
    # -------------------------------------------------------------

    def delta(metric):
        early_value = (
            early_results["metrics"].get(metric)
        )

        late_value = (
            late_results["metrics"].get(metric)
        )

        if (
            early_value is None
            or late_value is None
        ):
            return None

        return float(
            late_value - early_value
        )

    temporal_delta = {
        "accuracy_late_minus_early": delta(
            "accuracy"
        ),
        "macro_f1_late_minus_early": delta(
            "macro_f1"
        ),
        "macro_roc_auc_late_minus_early": delta(
            "macro_roc_auc"
        ),
        "macro_pr_auc_late_minus_early": delta(
            "macro_pr_auc"
        ),
    }

    # -------------------------------------------------------------
    # GRAPH AUDIT
    # -------------------------------------------------------------

    print()
    print(
        "Auditing graph/entity features..."
    )

    graph_features = identify_graph_features(
        feature_names,
        test,
    )

    graph_importance = get_graph_feature_importance(
        model,
        feature_names,
    )

    # -------------------------------------------------------------
    # PRINT
    # -------------------------------------------------------------

    print()
    print("-" * 70)
    print("FULL TEST")
    print("-" * 70)

    print(
        f"Accuracy:     "
        f"{full_results['metrics']['accuracy']:.4f}"
    )

    print(
        f"Macro F1:     "
        f"{full_results['metrics']['macro_f1']:.4f}"
    )

    print(
        f"Macro ROC-AUC:"
        f" {full_results['metrics']['macro_roc_auc']:.4f}"
    )

    print(
        f"Macro PR-AUC: "
        f"{full_results['metrics']['macro_pr_auc']:.4f}"
    )

    print()
    print("-" * 70)
    print("EARLY TEST — TIME 40–44")
    print("-" * 70)

    print(
        f"Rows:         "
        f"{early_results['metrics']['rows']:,}"
    )

    print(
        f"Accuracy:     "
        f"{early_results['metrics']['accuracy']:.4f}"
    )

    print(
        f"Macro F1:     "
        f"{early_results['metrics']['macro_f1']:.4f}"
    )

    print(
        f"Macro ROC-AUC:"
        f" {early_results['metrics']['macro_roc_auc']:.4f}"
    )

    print(
        f"Macro PR-AUC: "
        f"{early_results['metrics']['macro_pr_auc']:.4f}"
    )

    print()
    print("-" * 70)
    print("LATE TEST — TIME 45–49")
    print("-" * 70)

    print(
        f"Rows:         "
        f"{late_results['metrics']['rows']:,}"
    )

    print(
        f"Accuracy:     "
        f"{late_results['metrics']['accuracy']:.4f}"
    )

    print(
        f"Macro F1:     "
        f"{late_results['metrics']['macro_f1']:.4f}"
    )

    print(
        f"Macro ROC-AUC:"
        f" {late_results['metrics']['macro_roc_auc']:.4f}"
    )

    print(
        f"Macro PR-AUC: "
        f"{late_results['metrics']['macro_pr_auc']:.4f}"
    )

    print()
    print("-" * 70)
    print("TEMPORAL DELTA — LATE MINUS EARLY")
    print("-" * 70)

    for key, value in temporal_delta.items():

        if value is None:
            print(
                f"{key}: unavailable"
            )
        else:
            print(
                f"{key}: {value:+.4f}"
            )

    print()
    print("-" * 70)
    print("GRAPH / ENTITY FEATURES")
    print("-" * 70)

    print(
        f"Candidate graph/entity features: "
        f"{len(graph_features)}"
    )

    for row in graph_importance[:15]:

        print(
            f"  {row['feature']}: "
            f"{row['importance']:.6f}"
        )

    # -------------------------------------------------------------
    # SAVE REPORT
    # -------------------------------------------------------------

    report = {
        "stage": "M6.11",
        "task": (
            "temporal_stability_and_graph_leakage_audit"
        ),

        "classes": [
            int(c)
            for c in CLASSES
        ],

        "features": {
            "count": len(feature_names),
            "names": feature_names,
        },

        "dataset": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_time_min": int(
                test["time_step"].min()
            ),
            "test_time_max": int(
                test["time_step"].max()
            ),
        },

        "temporal_windows": {
            "early": {
                "time_start": EARLY_TEST_START,
                "time_end": EARLY_TEST_END,
            },
            "late": {
                "time_start": LATE_TEST_START,
                "time_end": LATE_TEST_END,
            },
        },

        "results": {
            "full_test": full_results,
            "early_test": early_results,
            "late_test": late_results,
            "late_minus_early": temporal_delta,
        },

        "graph_leakage_audit": {
            "status": (
                "requires_provenance_review"
                if graph_features
                else "no_graph_features_detected"
            ),
            "candidate_features": graph_features,
            "xgboost_importance": graph_importance,
            "interpretation": (
                "Feature identification does not establish "
                "temporal leakage. Graph/entity features "
                "constructed using future observations must "
                "be rebuilt causally before production "
                "evaluation."
            ),
        },

        "methodology": {
            "temporal_split": True,
            "test_used_for_training": False,
            "early_window": "40-44",
            "late_window": "45-49",
            "class_semantics": (
                "numeric IDs retained; no semantic "
                "interpretation assigned"
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print(
        f"Report: {OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()