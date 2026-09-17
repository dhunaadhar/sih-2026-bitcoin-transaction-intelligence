"""
M6.10 — Ranking and Calibration Evaluation

Purpose:
1. Train the M6.9 XGBoost benchmark model.
2. Evaluate investigative ranking quality using Precision@K / Recall@K.
3. Evaluate per-class ranking.
4. Fit temperature scaling on VALIDATION ONLY.
5. Evaluate calibrated probabilities on untouched TEST data.
6. Report log loss and multiclass Brier score.

Important:
- Temporal train/validation/test partitions are reused exactly.
- The test set is never used to fit calibration parameters.
- Class IDs remain numeric: 1, 2, 3.
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
    log_loss,
)

from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "evaluation" / "train.parquet"
VAL_PATH = ROOT / "data" / "evaluation" / "validation.parquet"
TEST_PATH = ROOT / "data" / "evaluation" / "test.parquet"

MANIFEST_PATH = ROOT / "reports" / "ml" / "feature_manifest.json"

OUTPUT_PATH = ROOT / "reports" / "ml" / "ranking_calibration.json"


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

RANDOM_STATE = 42

TOP_K_VALUES = [50, 100, 250, 500, 1000]

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
# HELPERS
# ---------------------------------------------------------------------

def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(path):
    df = pd.read_parquet(path)

    if "txid" not in df.columns:
        raise ValueError(f"{path} does not contain txid")

    if "class" not in df.columns:
        raise ValueError(f"{path} does not contain class")

    return df


def prepare_features(train, val, test, feature_names):
    missing = [
        feature
        for feature in feature_names
        if feature not in train.columns
        or feature not in val.columns
        or feature not in test.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X_train = train[feature_names].copy()
    X_val = val[feature_names].copy()
    X_test = test[feature_names].copy()

    # Training-only fitting of imputer.
    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    return X_train, X_val, X_test


def multiclass_brier_score(y_true, probabilities, classes):
    """
    Multiclass Brier score:

        mean(sum_c (p_c - y_c)^2)

    Lower is better.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    target = np.zeros_like(probabilities, dtype=float)

    for index, class_id in enumerate(classes):
        target[:, index] = (y_true == class_id).astype(float)

    return float(
        np.mean(np.sum((probabilities - target) ** 2, axis=1))
    )


def ranking_metrics(y_true, probabilities, classes):
    """
    Calculate per-class Precision@K and Recall@K.

    For each class:
      - Rank transactions by P(class).
      - Take top K.
      - Precision@K = positives in top K / K
      - Recall@K = positives in top K / total positives
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    results = {}

    for class_index, class_id in enumerate(classes):

        scores = probabilities[:, class_index]

        order = np.argsort(-scores)

        total_positives = int(
            np.sum(y_true == class_id)
        )

        class_results = {}

        for k in TOP_K_VALUES:

            k_actual = min(k, len(y_true))

            top_indices = order[:k_actual]

            hits = int(
                np.sum(y_true[top_indices] == class_id)
            )

            precision = hits / k_actual if k_actual else 0.0

            recall = (
                hits / total_positives
                if total_positives > 0
                else 0.0
            )

            class_results[str(k)] = {
                "hits": hits,
                "precision_at_k": float(precision),
                "recall_at_k": float(recall),
            }

        results[str(int(class_id))] = {
            "total_test_instances": total_positives,
            "ranking": class_results,
        }

    return results


def fit_temperature(logits, y_true, classes):
    """
    Fit a single temperature parameter on VALIDATION ONLY.

    p_i = softmax(logit_i / T)

    T > 1 generally softens probabilities.
    T < 1 generally sharpens probabilities.
    """

    y_true = np.asarray(y_true)
    logits = np.asarray(logits)

    class_to_index = {
        int(class_id): index
        for index, class_id in enumerate(classes)
    }

    y_indices = np.array([
        class_to_index[int(value)]
        for value in y_true
    ])

    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def objective(log_temperature):
        temperature = np.exp(log_temperature)

        probabilities = softmax(
            logits / temperature
        )

        probabilities = np.clip(
            probabilities,
            1e-12,
            1.0,
        )

        loss = -np.mean(
            np.log(
                probabilities[
                    np.arange(len(y_indices)),
                    y_indices,
                ]
            )
        )

        return loss

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-5},
    )

    temperature = float(np.exp(result.x))

    return temperature


def softmax(logits):
    logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exp_logits = np.exp(logits)

    return exp_logits / np.sum(
        exp_logits,
        axis=1,
        keepdims=True,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("M6.10 — RANKING + CALIBRATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------

    train = load_dataset(TRAIN_PATH)
    val = load_dataset(VAL_PATH)
    test = load_dataset(TEST_PATH)

    manifest = load_manifest()

    feature_names = manifest["model_candidates"]

    classes = np.array([1, 2, 3])

    print(f"Features:     {len(feature_names)}")
    print(f"Train rows:   {len(train):,}")
    print(f"Validation:   {len(val):,}")
    print(f"Test rows:    {len(test):,}")

    # -------------------------------------------------------------
    # PREPARE FEATURES
    # -------------------------------------------------------------

    X_train, X_val, X_test = prepare_features(
        train,
        val,
        test,
        feature_names,
    )

    y_train = train["class"].to_numpy()
    y_val = val["class"].to_numpy()
    y_test = test["class"].to_numpy()

    # -------------------------------------------------------------
    # TRAIN XGBOOST
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
    # RAW PROBABILITIES
    # -------------------------------------------------------------

    print("Generating probabilities...")

    val_prob = model.predict_proba(X_val)
    test_prob = model.predict_proba(X_test)

    val_pred = np.argmax(val_prob, axis=1) + 1
    test_pred = np.argmax(test_prob, axis=1) + 1

    # -------------------------------------------------------------
    # BASELINE TEST METRICS
    # -------------------------------------------------------------

    baseline_accuracy = accuracy_score(
        y_test,
        test_pred,
    )

    baseline_macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
    )

    baseline_log_loss = log_loss(
        y_test,
        test_prob,
        labels=classes,
    )

    baseline_brier = multiclass_brier_score(
        y_test,
        test_prob,
        classes,
    )

    # -------------------------------------------------------------
    # RANKING
    # -------------------------------------------------------------

    print()
    print("Ranking evaluation...")

    ranking = ranking_metrics(
        y_test,
        test_prob,
        classes,
    )

    # -------------------------------------------------------------
    # TEMPERATURE CALIBRATION
    # -------------------------------------------------------------

    print()
    print("Fitting temperature calibration on validation set...")

    # XGBoost probabilities are converted back to logits.
    val_prob_clipped = np.clip(
        val_prob,
        1e-12,
        1.0,
    )

    test_prob_clipped = np.clip(
        test_prob,
        1e-12,
        1.0,
    )

    val_logits = np.log(val_prob_clipped)
    test_logits = np.log(test_prob_clipped)

    temperature = fit_temperature(
        val_logits,
        y_val,
        classes,
    )

    print(
        f"  Optimal temperature: {temperature:.6f}"
    )

    calibrated_val_prob = softmax(
        val_logits / temperature
    )

    calibrated_test_prob = softmax(
        test_logits / temperature
    )

    # -------------------------------------------------------------
    # CALIBRATED TEST METRICS
    # -------------------------------------------------------------

    calibrated_pred = (
        np.argmax(calibrated_test_prob, axis=1) + 1
    )

    calibrated_accuracy = accuracy_score(
        y_test,
        calibrated_pred,
    )

    calibrated_macro_f1 = f1_score(
        y_test,
        calibrated_pred,
        average="macro",
    )

    calibrated_log_loss = log_loss(
        y_test,
        calibrated_test_prob,
        labels=classes,
    )

    calibrated_brier = multiclass_brier_score(
        y_test,
        calibrated_test_prob,
        classes,
    )

    # -------------------------------------------------------------
    # CALIBRATED RANKING
    # -------------------------------------------------------------

    calibrated_ranking = ranking_metrics(
        y_test,
        calibrated_test_prob,
        classes,
    )

    # -------------------------------------------------------------
    # VALIDATION CALIBRATION DIAGNOSTICS
    # -------------------------------------------------------------

    validation_log_loss_before = log_loss(
        y_val,
        val_prob,
        labels=classes,
    )

    validation_log_loss_after = log_loss(
        y_val,
        calibrated_val_prob,
        labels=classes,
    )

    validation_brier_before = multiclass_brier_score(
        y_val,
        val_prob,
        classes,
    )

    validation_brier_after = multiclass_brier_score(
        y_val,
        calibrated_val_prob,
        classes,
    )

    # -------------------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------------------

    print()
    print("-" * 70)
    print("TEST BASELINE")
    print("-" * 70)

    print(
        f"Accuracy:       {baseline_accuracy:.4f}"
    )
    print(
        f"Macro F1:       {baseline_macro_f1:.4f}"
    )
    print(
        f"Log Loss:       {baseline_log_loss:.4f}"
    )
    print(
        f"Brier Score:    {baseline_brier:.4f}"
    )

    print()
    print("-" * 70)
    print("CALIBRATED TEST")
    print("-" * 70)

    print(
        f"Accuracy:       {calibrated_accuracy:.4f}"
    )
    print(
        f"Macro F1:       {calibrated_macro_f1:.4f}"
    )
    print(
        f"Log Loss:       {calibrated_log_loss:.4f}"
    )
    print(
        f"Brier Score:    {calibrated_brier:.4f}"
    )

    print()
    print("-" * 70)
    print("VALIDATION CALIBRATION")
    print("-" * 70)

    print(
        f"Log Loss:  {validation_log_loss_before:.4f}"
        f" -> {validation_log_loss_after:.4f}"
    )

    print(
        f"Brier:     {validation_brier_before:.4f}"
        f" -> {validation_brier_after:.4f}"
    )

    print()
    print("-" * 70)
    print("PRECISION@K / RECALL@K — TEST")
    print("-" * 70)

    for class_id in classes:

        print()
        print(f"Class {class_id}")

        class_data = calibrated_ranking[
            str(int(class_id))
        ]

        for k in TOP_K_VALUES:

            metrics = class_data["ranking"][str(k)]

            print(
                f"  K={k:<4} "
                f"Precision={metrics['precision_at_k']:.4f} "
                f"Recall={metrics['recall_at_k']:.4f} "
                f"Hits={metrics['hits']}"
            )

    # -------------------------------------------------------------
    # SAVE REPORT
    # -------------------------------------------------------------

    report = {
        "stage": "M6.10",
        "task": "ranking_and_calibration",

        "classes": [int(c) for c in classes],

        "features": {
            "count": len(feature_names),
            "names": feature_names,
        },

        "dataset": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(val)),
            "test_rows": int(len(test)),
        },

        "model": {
            "type": "XGBoost",
            "parameters": XGB_PARAMS,
        },

        "ranking": {
            "top_k_values": TOP_K_VALUES,
            "test_raw_probabilities": ranking,
            "test_calibrated_probabilities": calibrated_ranking,
        },

        "calibration": {
            "method": "temperature_scaling",
            "fit_on": "validation_only",
            "temperature": temperature,

            "validation": {
                "log_loss_before": float(
                    validation_log_loss_before
                ),
                "log_loss_after": float(
                    validation_log_loss_after
                ),
                "brier_before": float(
                    validation_brier_before
                ),
                "brier_after": float(
                    validation_brier_after
                ),
            },

            "test": {
                "accuracy_before": float(
                    baseline_accuracy
                ),
                "accuracy_after": float(
                    calibrated_accuracy
                ),
                "macro_f1_before": float(
                    baseline_macro_f1
                ),
                "macro_f1_after": float(
                    calibrated_macro_f1
                ),
                "log_loss_before": float(
                    baseline_log_loss
                ),
                "log_loss_after": float(
                    calibrated_log_loss
                ),
                "brier_before": float(
                    baseline_brier
                ),
                "brier_after": float(
                    calibrated_brier
                ),
            },
        },

        "methodology": {
            "temporal_split": True,
            "test_used_for_calibration": False,
            "ranking_primary_use": (
                "prioritize transactions for investigation"
            ),
            "class_semantics": (
                "numeric class IDs retained; "
                "no semantic interpretation assigned"
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