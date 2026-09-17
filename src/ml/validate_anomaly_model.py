from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_inference import ProductionAnomalyEngine


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
    / "m8_anomaly_validation.json"
)

TXID_COLUMN = "txid"
LABEL_COLUMN = "label"
TIME_COLUMN = "time_step"

BATCH_SIZE = 5000


def validate_test_dataset(
    df: pd.DataFrame,
    model_features: list[str],
) -> None:
    required = {
        TXID_COLUMN,
        LABEL_COLUMN,
        TIME_COLUMN,
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


def describe_scores(
    scores: np.ndarray,
) -> dict[str, float | int]:
    percentiles = {
        "p01": 1,
        "p05": 5,
        "p10": 10,
        "p25": 25,
        "p50": 50,
        "p75": 75,
        "p90": 90,
        "p95": 95,
        "p99": 99,
    }

    result: dict[str, float | int] = {
        "count": int(len(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
    }

    for name, percentile in percentiles.items():
        result[name] = float(
            np.percentile(
                scores,
                percentile,
            )
        )

    return result


def describe_by_label(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    result = {}

    for label in sorted(
        np.unique(labels)
    ):
        mask = labels == label

        label_scores = scores[mask]

        result[str(int(label))] = {
            "count": int(
                len(label_scores)
            ),
            "mean_anomaly_score": float(
                np.mean(label_scores)
            ),
            "median_anomaly_score": float(
                np.median(label_scores)
            ),
            "p90_anomaly_score": float(
                np.percentile(
                    label_scores,
                    90,
                )
            ),
            "p95_anomaly_score": float(
                np.percentile(
                    label_scores,
                    95,
                )
            ),
            "p99_anomaly_score": float(
                np.percentile(
                    label_scores,
                    99,
                )
            ),
        }

    return result


def describe_by_time(
    time_steps: np.ndarray,
    scores: np.ndarray,
    flags: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    result = {}

    for time_step in sorted(
        np.unique(time_steps)
    ):
        mask = time_steps == time_step

        step_scores = scores[mask]
        step_flags = flags[mask]

        result[str(int(time_step))] = {
            "count": int(
                len(step_scores)
            ),
            "mean_anomaly_score": float(
                np.mean(step_scores)
            ),
            "median_anomaly_score": float(
                np.median(step_scores)
            ),
            "anomaly_flag_count": int(
                np.sum(step_flags == -1)
            ),
            "anomaly_flag_rate": float(
                np.mean(step_flags == -1)
            ),
        }

    return result


def top_ranked_transactions(
    test: pd.DataFrame,
    scores: np.ndarray,
    flags: np.ndarray,
    top_k: int = 100,
) -> list[dict]:
    ranking = pd.DataFrame(
        {
            "txid": test[TXID_COLUMN].tolist(),
            "time_step": test[
                TIME_COLUMN
            ].astype(int).tolist(),
            "label": test[
                LABEL_COLUMN
            ].astype(int).tolist(),
            "anomaly_score": scores,
            "anomaly_flag": flags,
        }
    )

    ranking = ranking.sort_values(
        "anomaly_score",
        ascending=False,
    )

    top = ranking.head(
        top_k
    )

    records = []

    for _, row in top.iterrows():
        records.append(
            {
                "txid": row["txid"],
                "time_step": int(
                    row["time_step"]
                ),
                "label": int(
                    row["label"]
                ),
                "anomaly_score": float(
                    row["anomaly_score"]
                ),
                "anomaly_flag": int(
                    row["anomaly_flag"]
                ),
            }
        )

    return records


def main() -> None:
    print("=" * 72)
    print("M8.3 PRODUCTION ANOMALY MODEL VALIDATION")
    print("=" * 72)

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Frozen test dataset not found: "
            f"{TEST_FILE}"
        )

    engine = ProductionAnomalyEngine()

    test = pd.read_parquet(
        TEST_FILE
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

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

    all_scores = []
    all_flags = []
    all_decision_functions = []

    txids = test[
        TXID_COLUMN
    ].tolist()

    labels = (
        test[LABEL_COLUMN]
        .astype(int)
        .to_numpy()
    )

    time_steps = (
        test[TIME_COLUMN]
        .astype(int)
        .to_numpy()
    )

    print("\nRunning batch anomaly inference...")

    for start in range(
        0,
        len(test),
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            len(test),
        )

        batch = test.iloc[
            start:end
        ]

        result = engine.score(
            batch[
                engine.model_features
            ]
        )

        batch_results = result[
            "scores"
        ]

        all_scores.extend(
            [
                item["anomaly_score"]
                for item in batch_results
            ]
        )

        all_flags.extend(
            [
                item["anomaly_flag"]
                for item in batch_results
            ]
        )

        all_decision_functions.extend(
            [
                item["decision_function"]
                for item in batch_results
            ]
        )

        print(
            f"  Processed "
            f"{end:,}/{len(test):,}"
        )

    anomaly_scores = np.asarray(
        all_scores,
        dtype=float,
    )

    anomaly_flags = np.asarray(
        all_flags,
        dtype=int,
    )

    decision_functions = np.asarray(
        all_decision_functions,
        dtype=float,
    )

    if len(anomaly_scores) != len(test):
        raise ValueError(
            "Anomaly score count mismatch: "
            f"{len(anomaly_scores)} vs "
            f"{len(test)}"
        )

    if len(anomaly_flags) != len(test):
        raise ValueError(
            "Anomaly flag count mismatch."
        )

    if len(decision_functions) != len(test):
        raise ValueError(
            "Decision-function count mismatch."
        )

    if not np.isfinite(
        anomaly_scores
    ).all():
        raise ValueError(
            "Non-finite anomaly scores detected."
        )

    if not np.isfinite(
        decision_functions
    ).all():
        raise ValueError(
            "Non-finite decision functions detected."
        )

    if not np.allclose(
        anomaly_scores,
        -decision_functions,
        atol=1e-12,
    ):
        raise ValueError(
            "Anomaly-score transformation "
            "is inconsistent."
        )

    if not np.isin(
        anomaly_flags,
        [-1, 1],
    ).all():
        raise ValueError(
            "Unexpected anomaly flag values detected."
        )

    if len(set(txids)) != len(txids):
        raise ValueError(
            "Duplicate TXIDs detected."
        )

    anomaly_count = int(
        np.sum(
            anomaly_flags == -1
        )
    )

    anomaly_rate = float(
        anomaly_count / len(test)
    )

    print("\n" + "-" * 72)
    print("ANOMALY SUMMARY")
    print("-" * 72)

    print(
        f"Anomaly flags: "
        f"{anomaly_count:,}/{len(test):,}"
    )

    print(
        f"Anomaly flag rate: "
        f"{anomaly_rate:.4%}"
    )

    print(
        f"Minimum score: "
        f"{np.min(anomaly_scores):.6f}"
    )

    print(
        f"Maximum score: "
        f"{np.max(anomaly_scores):.6f}"
    )

    print(
        f"Mean score: "
        f"{np.mean(anomaly_scores):.6f}"
    )

    print(
        f"Median score: "
        f"{np.median(anomaly_scores):.6f}"
    )

    print("\nClass-conditioned anomaly statistics:")

    by_label = describe_by_label(
        labels,
        anomaly_scores,
    )

    for label, stats in by_label.items():
        print(
            f"  Class {label}: "
            f"count={stats['count']:,}, "
            f"mean={stats['mean_anomaly_score']:.6f}, "
            f"p95={stats['p95_anomaly_score']:.6f}"
        )

    print("\nTop 10 anomaly-ranked transactions:")

    top_transactions = (
        top_ranked_transactions(
            test,
            anomaly_scores,
            anomaly_flags,
            top_k=10,
        )
    )

    for rank, record in enumerate(
        top_transactions,
        start=1,
    ):
        print(
            f"  {rank:02d}. "
            f"TXID={record['txid']}, "
            f"time={record['time_step']}, "
            f"label={record['label']}, "
            f"score={record['anomaly_score']:.6f}, "
            f"flag={record['anomaly_flag']}"
        )

    report = {
        "milestone": "M8.3",
        "status": "PASS",
        "purpose": (
            "Validate the serialized temporal-safe "
            "Isolation Forest on the frozen test set "
            "and characterize its anomaly signal."
        ),
        "test_dataset": str(
            TEST_FILE
        ),
        "test_rows": int(
            len(test)
        ),
        "model_features": len(
            engine.model_features
        ),
        "transformed_features": len(
            engine.transformed_feature_names
        ),
        "anomaly_flag_definition": {
            "-1": "anomalous according to Isolation Forest",
            "1": "normal according to Isolation Forest",
        },
        "score_definition": (
            "anomaly_score = "
            "-decision_function; "
            "higher values indicate more anomalous "
            "behavior."
        ),
        "overall": {
            "anomaly_flag_count": anomaly_count,
            "anomaly_flag_rate": anomaly_rate,
            "score_statistics": describe_scores(
                anomaly_scores
            ),
            "decision_function_statistics": (
                describe_scores(
                    decision_functions
                )
            ),
        },
        "by_label": by_label,
        "by_time_step": describe_by_time(
            time_steps,
            anomaly_scores,
            anomaly_flags,
        ),
        "top_100_anomaly_ranked": (
            top_ranked_transactions(
                test,
                anomaly_scores,
                anomaly_flags,
                top_k=100,
            )
        ),
        "validation_checks": {
            "score_count_matches": True,
            "flag_count_matches": True,
            "decision_function_count_matches": True,
            "all_scores_finite": True,
            "all_decision_functions_finite": True,
            "score_transformation_valid": True,
            "flags_valid": True,
            "unique_txids": True,
            "test_set_not_modified": True,
        },
        "interpretation_note": (
            "Isolation Forest provides an unsupervised "
            "behavioral anomaly signal. Anomaly scores "
            "are not probabilities and do not establish "
            "illicit activity, identity, ownership, or "
            "guilt."
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

    print("\n" + "=" * 72)
    print("M8.3 COMPLETE")
    print("=" * 72)

    print(
        f"Report: {REPORT_FILE}"
    )

    print(
        "\nFrozen test set used only for "
        "post-training validation."
    )

    print(
        "No model parameters were changed."
    )


if __name__ == "__main__":
    main()