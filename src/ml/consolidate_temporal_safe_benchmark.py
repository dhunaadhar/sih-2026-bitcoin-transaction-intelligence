from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = ROOT / "reports" / "ml"

OUTPUT_FILE = REPORT_DIR / "m6_18_consolidated_temporal_safe_benchmark.json"


MODEL_REPORTS = {
    "Logistic Regression": {
        "temporal_safe": REPORT_DIR / "logistic_regression_temporal_safe.json",
        "global": REPORT_DIR / "logistic_regression.json",
    },
    "Random Forest": {
        "temporal_safe": REPORT_DIR / "random_forest_temporal_safe.json",
        "global": REPORT_DIR / "random_forest.json",
    },
    "HistGradientBoosting": {
        "temporal_safe": REPORT_DIR / "hist_gradient_boosting_temporal_safe.json",
        "global": REPORT_DIR / "hist_gradient_boosting.json",
    },
    "XGBoost": {
        "temporal_safe": REPORT_DIR / "xgboost_temporal_safe.json",
        "global": REPORT_DIR / "xgboost.json",
    },
}


REQUIRED_METRICS = [
    "accuracy",
    "macro_f1",
    "roc_auc_macro",
    "pr_auc_macro",
]

OPTIONAL_METRICS = [
    "log_loss",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def locate_test_section(report: dict[str, Any]) -> dict[str, Any]:
    """
    Support the different report schemas produced by the benchmark scripts.
    """

    candidates = [
        report.get("test"),
        (
            report.get("metrics", {}).get("test")
            if isinstance(report.get("metrics"), dict)
            else None
        ),
        report.get("test_metrics"),
    ]

    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate

    # Some reports store test metrics directly at the top level.
    direct_metric_keys = {
        "accuracy",
        "test_accuracy",
        "macro_f1",
        "test_macro_f1",
        "roc_auc_macro",
        "roc_auc_ovr_macro",
        "test_roc_auc_macro",
        "test_roc_auc_ovr_macro",
        "pr_auc_macro",
        "macro_pr_auc",
        "test_pr_auc_macro",
        "test_macro_pr_auc",
        "log_loss",
        "test_log_loss",
    }

    if any(key in report for key in direct_metric_keys):
        return report

    raise ValueError(
        "Model report does not contain a recognizable test section. "
        f"Available top-level keys: {list(report.keys())}"
    )


def normalize_metric_keys(test_section: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize metric naming differences across benchmark reports.

    Required:
        accuracy
        macro_f1
        roc_auc_macro
        pr_auc_macro

    Optional:
        log_loss
    """

    aliases = {
        "accuracy": [
            "accuracy",
            "test_accuracy",
        ],
        "macro_f1": [
            "macro_f1",
            "f1_macro",
            "test_macro_f1",
        ],
        "roc_auc_macro": [
            "roc_auc_macro",
            "roc_auc_ovr_macro",
            "roc_auc",
            "test_roc_auc_macro",
            "test_roc_auc_ovr_macro",
        ],
        "pr_auc_macro": [
            "pr_auc_macro",
            "macro_pr_auc",
            "pr_auc",
            "average_precision_macro",
            "test_pr_auc_macro",
            "test_macro_pr_auc",
        ],
        "log_loss": [
            "log_loss",
            "test_log_loss",
        ],
    }

    normalized: dict[str, Any] = {}

    for canonical_name, possible_keys in aliases.items():
        found = False

        for key in possible_keys:
            if key in test_section:
                normalized[canonical_name] = test_section[key]
                found = True
                break

        if not found:
            if canonical_name in OPTIONAL_METRICS:
                normalized[canonical_name] = None
            else:
                raise ValueError(
                    f"Missing required metric '{canonical_name}'. "
                    f"Available metric keys: {list(test_section.keys())}"
                )

    return normalized


def extract_test_metrics(path: Path) -> dict[str, Any]:
    report = load_json(path)

    test_section = locate_test_section(report)

    metrics = normalize_metric_keys(test_section)

    return {
        "metrics": metrics,
        "report_top_level_keys": list(report.keys()),
        "test_section_keys": list(test_section.keys()),
    }


def build_model_table() -> pd.DataFrame:
    rows = []

    for model_name, paths in MODEL_REPORTS.items():
        temporal = extract_test_metrics(paths["temporal_safe"])
        global_result = extract_test_metrics(paths["global"])

        temporal_metrics = temporal["metrics"]
        global_metrics = global_result["metrics"]

        row = {
            "model": model_name,
        }

        for metric in REQUIRED_METRICS + OPTIONAL_METRICS:
            temporal_value = temporal_metrics.get(metric)
            global_value = global_metrics.get(metric)

            row[f"temporal_safe_{metric}"] = temporal_value
            row[f"global_{metric}"] = global_value

            if (
                temporal_value is not None
                and global_value is not None
            ):
                row[f"delta_{metric}"] = (
                    temporal_value - global_value
                )
            else:
                row[f"delta_{metric}"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def dataframe_to_records(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")

    cleaned = []

    for record in records:
        cleaned_record = {}

        for key, value in record.items():
            if pd.isna(value):
                cleaned_record[key] = None
            elif hasattr(value, "item"):
                cleaned_record[key] = value.item()
            else:
                cleaned_record[key] = value

        cleaned.append(cleaned_record)

    return cleaned


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("M6.18 TEMPORAL-SAFE BENCHMARK CONSOLIDATION")
    print("=" * 72)

    for model_name, paths in MODEL_REPORTS.items():
        print(f"\n[{model_name}]")

        for label, path in paths.items():
            print(f"  {label}: {path}")

            if not path.exists():
                raise FileNotFoundError(
                    f"Required report does not exist: {path}"
                )

    table = build_model_table()

    print("\n" + "-" * 72)
    print("TEMPORAL-SAFE TEST BENCHMARK")
    print("-" * 72)

    display_columns = [
        "model",
        "temporal_safe_accuracy",
        "temporal_safe_macro_f1",
        "temporal_safe_roc_auc_macro",
        "temporal_safe_pr_auc_macro",
        "temporal_safe_log_loss",
    ]

    print(
        table[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "-" * 72)
    print("TEMPORAL-SAFE MINUS GLOBAL")
    print("-" * 72)

    delta_columns = [
        "model",
        "delta_accuracy",
        "delta_macro_f1",
        "delta_roc_auc_macro",
        "delta_pr_auc_macro",
        "delta_log_loss",
    ]

    print(
        table[delta_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # Select a metric-based candidate for downstream evaluation.
    # This is not a universal model-quality ranking.
    valid_pr_auc = table.dropna(
        subset=["temporal_safe_pr_auc_macro"]
    )

    if valid_pr_auc.empty:
        downstream_candidate = None
    else:
        candidate_row = valid_pr_auc.loc[
            valid_pr_auc["temporal_safe_pr_auc_macro"].idxmax()
        ]

        downstream_candidate = {
            "model": candidate_row["model"],
            "selection_metric": (
                "temporal_safe_test_pr_auc_macro"
            ),
            "value": float(
                candidate_row["temporal_safe_pr_auc_macro"]
            ),
            "selection_note": (
                "Metric-based candidate for downstream evaluation. "
                "Final production selection should also consider "
                "calibration, temporal stability, explainability, "
                "latency, and operational requirements."
            ),
        }

    payload = {
        "milestone": "M6.18",
        "title": "Consolidated Temporal-Safe ML Benchmark",
        "status": "PASS",
        "purpose": (
            "Consolidate causal temporal-safe benchmark results "
            "and compare them with the earlier globally constructed "
            "benchmark."
        ),
        "methodology": {
            "split": {
                "train": "time_step 1-29",
                "validation": "time_step 30-39",
                "test": "time_step 40-49",
            },
            "graph_features": (
                "Temporal-safe graph/entity features constructed "
                "using only information available before each "
                "transaction time step."
            ),
            "leakage_sensitive_features": 4,
            "original_model_candidate_features": 87,
            "comparison_note": (
                "The global benchmark is retained for methodological "
                "comparison only. The temporal-safe benchmark is the "
                "causal benchmark for downstream model development."
            ),
        },
        "metrics": {
            "required": REQUIRED_METRICS,
            "optional": OPTIONAL_METRICS,
            "aliases_supported": {
                "roc_auc_macro": [
                    "roc_auc_macro",
                    "roc_auc_ovr_macro",
                    "roc_auc",
                ],
                "pr_auc_macro": [
                    "pr_auc_macro",
                    "macro_pr_auc",
                    "pr_auc",
                    "average_precision_macro",
                ],
            },
            "optional_metric_policy": (
                "Missing optional metrics are recorded as null "
                "and do not invalidate benchmark consolidation."
            ),
        },
        "benchmark_table": dataframe_to_records(table),
        "current_downstream_candidate": downstream_candidate,
        "model_reports": {
            model_name: {
                "temporal_safe": str(
                    paths["temporal_safe"]
                ),
                "global": str(paths["global"]),
            }
            for model_name, paths in MODEL_REPORTS.items()
        },
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 72)
    print("M6.18 COMPLETE")
    print("=" * 72)
    print(f"Output: {OUTPUT_FILE}")

    if downstream_candidate:
        print(
            "Current downstream candidate: "
            f"{downstream_candidate['model']} "
            f"(PR-AUC="
            f"{downstream_candidate['value']:.4f})"
        )
    else:
        print("Current downstream candidate: NONE")


if __name__ == "__main__":
    main()