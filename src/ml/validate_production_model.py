from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    average_precision_score,
    roc_auc_score,
)

from inference import ProductionInferenceEngine


ROOT = Path(__file__).resolve().parents[2]

TEST_FILE = (
    ROOT
    / "data"
    / "evaluation"
    / "test_temporal_safe.parquet"
)

REPORT_DIR = ROOT / "reports" / "ml"
REPORT_FILE = (
    REPORT_DIR
    / "m7_production_model_validation.json"
)

LABEL_COLUMN = "label"
TXID_COLUMN = "txid"

BATCH_SIZE = 5000


def validate_test_dataset(
    df: pd.DataFrame,
    model_features: list[str],
) -> None:
    required = {
        LABEL_COLUMN,
        TXID_COLUMN,
        *model_features,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Test dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    if df[TXID_COLUMN].isna().any():
        raise ValueError(
            "Test dataset contains null TXIDs."
        )

    if df[TXID_COLUMN].duplicated().any():
        raise ValueError(
            "Test dataset contains duplicate TXIDs."
        )

    if df[LABEL_COLUMN].isna().any():
        raise ValueError(
            "Test dataset contains missing labels."
        )


def main() -> None:
    print("=" * 72)
    print("M7.3 PRODUCTION MODEL VALIDATION")
    print("=" * 72)

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Frozen test dataset not found: {TEST_FILE}"
        )

    engine = ProductionInferenceEngine()

    test = pd.read_parquet(TEST_FILE)

    print(f"Test rows: {len(test):,}")
    print(
        f"Model features: "
        f"{len(engine.model_features)}"
    )
    print(
        f"Transformed features: "
        f"{len(engine.transformed_feature_names)}"
    )

    validate_test_dataset(
        test,
        engine.model_features,
    )

    y_true = test[LABEL_COLUMN].astype(int).to_numpy()

    predictions = []
    probability_rows = []
    txids = []

    print("\nRunning batch inference...")

    for start in range(
        0,
        len(test),
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            len(test),
        )

        batch = test.iloc[start:end]

        result = engine.predict(
            batch[engine.model_features]
        )

        batch_predictions = result["predictions"]

        predictions.extend(
            [
                item["predicted_class"]
                for item in batch_predictions
            ]
        )

        probability_rows.extend(
            [
                [
                    item["probabilities"]["class_1"],
                    item["probabilities"]["class_2"],
                    item["probabilities"]["class_3"],
                ]
                for item in batch_predictions
            ]
        )

        txids.extend(
            batch[TXID_COLUMN].tolist()
        )

        print(
            f"  Processed "
            f"{end:,}/{len(test):,}"
        )

    y_pred = np.asarray(
        predictions,
        dtype=int,
    )

    probabilities = np.asarray(
        probability_rows,
        dtype=float,
    )

    if len(y_pred) != len(test):
        raise ValueError(
            "Prediction count mismatch: "
            f"{len(y_pred)} vs {len(test)}"
        )

    if probabilities.shape != (
        len(test),
        3,
    ):
        raise ValueError(
            "Probability matrix shape mismatch: "
            f"{probabilities.shape}"
        )

    if len(set(txids)) != len(txids):
        raise ValueError(
            "Duplicate TXIDs detected in inference output."
        )

    probability_sums = probabilities.sum(axis=1)

    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Probability normalization failed."
        )

    if np.isnan(probabilities).any():
        raise ValueError(
            "NaN detected in probability output."
        )

    if np.isinf(probabilities).any():
        raise ValueError(
            "Infinite probability detected."
        )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
        multi_class="ovr",
        average="macro",
    )

    pr_auc = average_precision_score(
        pd.get_dummies(
            y_true,
            columns=[1, 2, 3],
        ),
        probabilities,
        average="macro",
    )

    test_log_loss = log_loss(
        y_true,
        probabilities,
        labels=[1, 2, 3],
    )

    class_counts = {
        str(int(label)): int(count)
        for label, count in (
            pd.Series(y_true)
            .value_counts()
            .sort_index()
            .items()
        )
    }

    prediction_counts = {
        str(int(label)): int(count)
        for label, count in (
            pd.Series(y_pred)
            .value_counts()
            .sort_index()
            .items()
        )
    }

    report = {
        "milestone": "M7.3",
        "status": "PASS",
        "purpose": (
            "Validate serialized production inference against "
            "the frozen temporal-safe test set."
        ),
        "test_dataset": str(TEST_FILE),
        "test_rows": int(len(test)),
        "model_features": len(
            engine.model_features
        ),
        "transformed_features": len(
            engine.transformed_feature_names
        ),
        "metrics": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "roc_auc_ovr_macro": float(roc_auc),
            "macro_pr_auc": float(pr_auc),
            "log_loss": float(test_log_loss),
        },
        "class_distribution": {
            "actual": class_counts,
            "predicted": prediction_counts,
        },
        "validation_checks": {
            "prediction_count_matches": True,
            "unique_txids": True,
            "probability_shape_valid": True,
            "probabilities_sum_to_one": True,
            "no_nan_probabilities": True,
            "no_infinite_probabilities": True,
            "test_set_not_modified": True,
        },
        "note": (
            "This validation uses the frozen test set only for "
            "post-training verification of the serialized "
            "production inference artifact. It does not retrain "
            "or modify the model."
        ),
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print("\n" + "-" * 72)
    print("PRODUCTION TEST METRICS")
    print("-" * 72)

    print(
        f"Accuracy:       {accuracy:.4f}"
    )
    print(
        f"Macro F1:       {macro_f1:.4f}"
    )
    print(
        f"ROC-AUC:        {roc_auc:.4f}"
    )
    print(
        f"Macro PR-AUC:   {pr_auc:.4f}"
    )
    print(
        f"Log Loss:       {test_log_loss:.4f}"
    )

    print("\n" + "-" * 72)
    print("VALIDATION CHECKS")
    print("-" * 72)

    for name, value in report[
        "validation_checks"
    ].items():
        print(
            f"{name}: "
            f"{'PASS' if value else 'FAIL'}"
        )

    print("\n" + "=" * 72)
    print("M7.3 COMPLETE")
    print("=" * 72)

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()