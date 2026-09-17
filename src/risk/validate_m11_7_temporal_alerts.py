from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RANKED_ALERTS_PATH = ROOT / "data" / "derived" / "ranked_alerts.parquet"
RISK_SCORES_PATH = ROOT / "data" / "derived" / "unified_risk_scores.parquet"

REPORT_DIR = ROOT / "reports" / "risk"
REPORT_PATH = REPORT_DIR / "m11_7_temporal_alert_validation.json"

EXPECTED_ROWS = 203_769
EXPECTED_TIME_MIN = 1
EXPECTED_TIME_MAX = 49
TOP_QUEUE_SIZE = 1_000

PRIORITY_ORDER = {
    "LOW": 0,
    "GUARDED": 1,
    "MODERATE": 2,
    "HIGH": 3,
    "VERY_HIGH": 4,
}


def normalize_txid_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.isna().any():
        raise ValueError("TXID column contains missing or non-numeric values.")

    values: list[str] = []

    for value in numeric:
        if not np.isfinite(value):
            raise ValueError("TXID column contains non-finite values.")

        if float(value).is_integer():
            values.append(str(int(value)))
        else:
            values.append(str(value))

    return pd.Series(values, index=series.index, dtype="string")


def priority_from_score(score: float) -> str:
    if score >= 80.0:
        return "VERY_HIGH"
    if score >= 60.0:
        return "HIGH"
    if score >= 40.0:
        return "MODERATE"
    if score >= 20.0:
        return "GUARDED"
    return "LOW"


def check(
    name: str,
    condition: bool,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "check": name,
        "status": "PASS" if condition else "FAIL",
    }

    if details is not None:
        result["details"] = details

    return result


def main() -> None:
    print("=" * 65)
    print("M11.7 TEMPORAL ALERT VALIDATION")
    print("=" * 65)

    if not RANKED_ALERTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing ranked-alert artifact: {RANKED_ALERTS_PATH}"
        )

    if not RISK_SCORES_PATH.exists():
        raise FileNotFoundError(
            f"Missing M11.4 risk-score artifact: {RISK_SCORES_PATH}"
        )

    print(f"Ranked alerts: {RANKED_ALERTS_PATH}")
    print(f"M11.4 risk scores: {RISK_SCORES_PATH}")
    print()

    alerts = pd.read_parquet(RANKED_ALERTS_PATH)
    risk = pd.read_parquet(RISK_SCORES_PATH)

    print(f"Ranked alert rows: {len(alerts):,}")
    print(f"M11.4 risk-score rows: {len(risk):,}")
    print()

    checks: list[dict[str, Any]] = []

    required_alert_columns = {
        "txid",
        "time_step",
        "risk_score",
        "alert_priority",
        "investigation_rank",
        "effective_evidence_channel_count",
    }

    required_risk_columns = {
        "txid",
        "time_step",
        "risk_score",
    }

    missing_alert_columns = sorted(
        required_alert_columns - set(alerts.columns)
    )

    missing_risk_columns = sorted(
        required_risk_columns - set(risk.columns)
    )

    checks.append(
        check(
            "ranked_alert_schema",
            len(missing_alert_columns) == 0,
            {"missing_columns": missing_alert_columns},
        )
    )

    checks.append(
        check(
            "risk_score_schema",
            len(missing_risk_columns) == 0,
            {"missing_columns": missing_risk_columns},
        )
    )

    if missing_alert_columns or missing_risk_columns:
        raise RuntimeError(
            "Required columns are missing."
        )

    alerts = alerts.copy()
    risk = risk.copy()

    alerts["txid_normalized"] = normalize_txid_series(alerts["txid"])
    risk["txid_normalized"] = normalize_txid_series(risk["txid"])

    # ---------------------------------------------------------------
    # 1. Dataset integrity
    # ---------------------------------------------------------------

    checks.append(
        check(
            "expected_row_count",
            len(alerts) == EXPECTED_ROWS,
            {
                "actual": int(len(alerts)),
                "expected": EXPECTED_ROWS,
            },
        )
    )

    checks.append(
        check(
            "unique_txids",
            alerts["txid_normalized"].nunique() == len(alerts),
            {
                "rows": int(len(alerts)),
                "unique_txids": int(
                    alerts["txid_normalized"].nunique()
                ),
            },
        )
    )

    checks.append(
        check(
            "risk_score_txid_coverage",
            set(alerts["txid_normalized"])
            == set(risk["txid_normalized"]),
            {
                "alert_only": int(
                    len(
                        set(alerts["txid_normalized"])
                        - set(risk["txid_normalized"])
                    )
                ),
                "risk_only": int(
                    len(
                        set(risk["txid_normalized"])
                        - set(alerts["txid_normalized"])
                    )
                ),
            },
        )
    )

    # ---------------------------------------------------------------
    # 2. Time-step integrity
    # ---------------------------------------------------------------

    time_values = pd.to_numeric(
        alerts["time_step"],
        errors="coerce",
    )

    time_valid = bool(
        time_values.notna().all()
        and time_values.between(
            EXPECTED_TIME_MIN,
            EXPECTED_TIME_MAX,
        ).all()
    )

    checks.append(
        check(
            "time_step_range",
            time_valid,
            {
                "min": (
                    int(time_values.min())
                    if time_values.notna().any()
                    else None
                ),
                "max": (
                    int(time_values.max())
                    if time_values.notna().any()
                    else None
                ),
                "expected_min": EXPECTED_TIME_MIN,
                "expected_max": EXPECTED_TIME_MAX,
            },
        )
    )

    observed_steps = sorted(
        int(x)
        for x in time_values.dropna().unique()
    )

    expected_steps = list(
        range(
            EXPECTED_TIME_MIN,
            EXPECTED_TIME_MAX + 1,
        )
    )

    checks.append(
        check(
            "all_expected_time_steps_present",
            observed_steps == expected_steps,
            {
                "observed_steps": observed_steps,
                "expected_step_count": len(expected_steps),
            },
        )
    )

    # ---------------------------------------------------------------
    # 3. Numeric integrity
    # ---------------------------------------------------------------

    numeric_columns = [
        "time_step",
        "risk_score",
        "investigation_rank",
        "effective_evidence_channel_count",
    ]

    numeric_integrity = True
    numeric_details: dict[str, Any] = {}

    for column in numeric_columns:
        values = pd.to_numeric(
            alerts[column],
            errors="coerce",
        )

        missing = int(values.isna().sum())
        infinite = int(
            np.isinf(
                values.to_numpy(dtype=float)
            ).sum()
        )

        numeric_details[column] = {
            "missing": missing,
            "infinite": infinite,
        }

        if missing > 0 or infinite > 0:
            numeric_integrity = False

    checks.append(
        check(
            "numeric_integrity",
            numeric_integrity,
            numeric_details,
        )
    )

    # ---------------------------------------------------------------
    # 4. Risk and priority consistency
    # ---------------------------------------------------------------

    risk_min = float(alerts["risk_score"].min())
    risk_max = float(alerts["risk_score"].max())

    checks.append(
        check(
            "risk_score_range",
            risk_min >= 0.0 and risk_max <= 100.0,
            {
                "min": risk_min,
                "max": risk_max,
            },
        )
    )

    expected_priority = alerts["risk_score"].apply(
        priority_from_score
    )

    priority_match = (
        expected_priority.to_numpy()
        == alerts["alert_priority"]
        .astype(str)
        .to_numpy()
    )

    checks.append(
        check(
            "priority_consistency",
            bool(priority_match.all()),
            {
                "mismatches": int(
                    (~priority_match).sum()
                ),
            },
        )
    )

    # ---------------------------------------------------------------
    # 5. M11.4 time/risk consistency
    # ---------------------------------------------------------------

    risk_lookup = risk[
        [
            "txid_normalized",
            "time_step",
            "risk_score",
        ]
    ].rename(
        columns={
            "time_step": "m11_4_time_step",
            "risk_score": "m11_4_risk_score",
        }
    )

    merged = alerts.merge(
        risk_lookup,
        on="txid_normalized",
        how="left",
        validate="one_to_one",
    )

    time_match = (
        pd.to_numeric(
            merged["time_step"],
            errors="coerce",
        ).to_numpy()
        == pd.to_numeric(
            merged["m11_4_time_step"],
            errors="coerce",
        ).to_numpy()
    )

    score_match = np.isclose(
        pd.to_numeric(
            merged["risk_score"],
            errors="coerce",
        ).to_numpy(dtype=float),
        pd.to_numeric(
            merged["m11_4_risk_score"],
            errors="coerce",
        ).to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    )

    checks.append(
        check(
            "m11_4_time_alignment",
            bool(time_match.all()),
            {
                "mismatches": int(
                    (~time_match).sum()
                ),
            },
        )
    )

    checks.append(
        check(
            "m11_4_risk_alignment",
            bool(score_match.all()),
            {
                "mismatches": int(
                    (~score_match).sum()
                ),
            },
        )
    )

    # ---------------------------------------------------------------
    # 6. Temporal distribution
    # ---------------------------------------------------------------

    temporal_rows: list[dict[str, Any]] = []

    for step in expected_steps:
        subset = alerts[
            alerts["time_step"] == step
        ]

        count = len(subset)

        if count == 0:
            temporal_rows.append(
                {
                    "time_step": step,
                    "transaction_count": 0,
                    "mean_risk": None,
                    "median_risk": None,
                    "high_count": 0,
                    "very_high_count": 0,
                    "high_rate": 0.0,
                    "very_high_rate": 0.0,
                    "top_1000_count": 0,
                }
            )
            continue

        priority_values = subset[
            "alert_priority"
        ].astype(str)

        high_mask = priority_values == "HIGH"
        very_high_mask = priority_values == "VERY_HIGH"

        temporal_rows.append(
            {
                "time_step": step,
                "transaction_count": int(count),
                "mean_risk": float(
                    subset["risk_score"].mean()
                ),
                "median_risk": float(
                    subset["risk_score"].median()
                ),
                "high_count": int(
                    high_mask.sum()
                ),
                "very_high_count": int(
                    very_high_mask.sum()
                ),
                "high_rate": float(
                    high_mask.mean()
                ),
                "very_high_rate": float(
                    very_high_mask.mean()
                ),
                "top_1000_count": int(
                    (
                        subset["investigation_rank"]
                        <= TOP_QUEUE_SIZE
                    ).sum()
                ),
            }
        )

    temporal_df = pd.DataFrame(temporal_rows)

    checks.append(
        check(
            "temporal_diagnostics_complete",
            len(temporal_df) == len(expected_steps)
            and bool(
                temporal_df["transaction_count"]
                .gt(0)
                .all()
            ),
            {
                "time_steps": int(len(temporal_df)),
                "zero_transaction_steps": int(
                    (
                        temporal_df["transaction_count"]
                        == 0
                    ).sum()
                ),
            },
        )
    )

    # ---------------------------------------------------------------
    # 7. Alert-rate temporal reproducibility
    # ---------------------------------------------------------------

    temporal_rates_finite = bool(
        np.isfinite(
            temporal_df[
                [
                    "mean_risk",
                    "median_risk",
                    "high_rate",
                    "very_high_rate",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )

    checks.append(
        check(
            "temporal_rate_integrity",
            temporal_rates_finite,
            {},
        )
    )

    # ---------------------------------------------------------------
    # 8. Rank distribution by time
    # ---------------------------------------------------------------

    top_counts_sum = int(
        temporal_df["top_1000_count"].sum()
    )

    checks.append(
        check(
            "top_queue_temporal_partition",
            top_counts_sum == TOP_QUEUE_SIZE,
            {
                "sum_of_top_queue_counts": top_counts_sum,
                "expected": TOP_QUEUE_SIZE,
            },
        )
    )

    # ---------------------------------------------------------------
    # 9. Chronological ordering of transaction data
    # ---------------------------------------------------------------

    time_counts = (
        alerts.groupby("time_step")
        .size()
        .sort_index()
    )

    chronological_coverage = (
        list(time_counts.index.astype(int))
        == expected_steps
    )

    checks.append(
        check(
            "chronological_time_coverage",
            chronological_coverage,
            {
                "first_time_step": int(
                    time_counts.index.min()
                ),
                "last_time_step": int(
                    time_counts.index.max()
                ),
                "time_step_count": int(
                    len(time_counts)
                ),
            },
        )
    )

    # ---------------------------------------------------------------
    # 10. Temporal concentration diagnostics
    # ---------------------------------------------------------------

    highest_mean_row = temporal_df.loc[
        temporal_df["mean_risk"].idxmax()
    ]

    lowest_mean_row = temporal_df.loc[
        temporal_df["mean_risk"].idxmin()
    ]

    highest_high_rate_row = temporal_df.loc[
        temporal_df["high_rate"].idxmax()
    ]

    highest_very_high_rate_row = temporal_df.loc[
        temporal_df["very_high_rate"].idxmax()
    ]

    highest_top_queue_row = temporal_df.loc[
        temporal_df["top_1000_count"].idxmax()
    ]

    temporal_summary = {
        "highest_mean_risk_time_step": int(
            highest_mean_row["time_step"]
        ),
        "highest_mean_risk": float(
            highest_mean_row["mean_risk"]
        ),
        "lowest_mean_risk_time_step": int(
            lowest_mean_row["time_step"]
        ),
        "lowest_mean_risk": float(
            lowest_mean_row["mean_risk"]
        ),
        "highest_high_rate_time_step": int(
            highest_high_rate_row["time_step"]
        ),
        "highest_high_rate": float(
            highest_high_rate_row["high_rate"]
        ),
        "highest_very_high_rate_time_step": int(
            highest_very_high_rate_row["time_step"]
        ),
        "highest_very_high_rate": float(
            highest_very_high_rate_row["very_high_rate"]
        ),
        "highest_top_queue_time_step": int(
            highest_top_queue_row["time_step"]
        ),
        "highest_top_queue_count": int(
            highest_top_queue_row["top_1000_count"]
        ),
    }

    # ---------------------------------------------------------------
    # 11. Top queue temporal range
    # ---------------------------------------------------------------

    top_queue = alerts[
        alerts["investigation_rank"] <= TOP_QUEUE_SIZE
    ]

    top_queue_time_min = int(
        top_queue["time_step"].min()
    )

    top_queue_time_max = int(
        top_queue["time_step"].max()
    )

    checks.append(
        check(
            "top_queue_time_range",
            (
                top_queue_time_min
                >= EXPECTED_TIME_MIN
                and top_queue_time_max
                <= EXPECTED_TIME_MAX
            ),
            {
                "min_time_step": top_queue_time_min,
                "max_time_step": top_queue_time_max,
            },
        )
    )

    # ---------------------------------------------------------------
    # 12. Final status
    # ---------------------------------------------------------------

    failed_checks = [
        item["check"]
        for item in checks
        if item["status"] != "PASS"
    ]

    status = "PASS" if not failed_checks else "FAIL"

    priority_distribution = {
        str(k): int(v)
        for k, v in alerts["alert_priority"]
        .value_counts()
        .to_dict()
        .items()
    }

    report = {
        "milestone": "M11.7",
        "title": "Temporal Alert Validation",
        "status": status,
        "artifacts": {
            "ranked_alerts": str(
                RANKED_ALERTS_PATH.relative_to(ROOT)
            ),
            "m11_4_risk_scores": str(
                RISK_SCORES_PATH.relative_to(ROOT)
            ),
            "report": str(
                REPORT_PATH.relative_to(ROOT)
            ),
        },
        "dataset": {
            "rows": int(len(alerts)),
            "unique_txids": int(
                alerts["txid_normalized"].nunique()
            ),
            "time_step_min": int(time_values.min()),
            "time_step_max": int(time_values.max()),
            "time_step_count": int(
                time_values.nunique()
            ),
        },
        "risk": {
            "mean": float(
                alerts["risk_score"].mean()
            ),
            "median": float(
                alerts["risk_score"].median()
            ),
            "min": risk_min,
            "max": risk_max,
        },
        "alerts": {
            "priority_distribution": priority_distribution,
            "top_queue_size": TOP_QUEUE_SIZE,
            "top_queue_mean_risk": float(
                top_queue["risk_score"].mean()
            ),
            "top_queue_time_min": top_queue_time_min,
            "top_queue_time_max": top_queue_time_max,
        },
        "temporal_summary": temporal_summary,
        "temporal_diagnostics": temporal_rows,
        "checks": checks,
        "failed_checks": failed_checks,
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            allow_nan=False,
        )

    print("M11.7 TEMPORAL VALIDATION RESULT")
    print("-" * 65)

    for item in checks:
        marker = (
            "PASS"
            if item["status"] == "PASS"
            else "FAIL"
        )
        print(
            f"{marker:<6} {item['check']}"
        )

    print()
    print(f"ROWS: {len(alerts):,}")
    print(
        f"TIME RANGE: "
        f"{int(time_values.min())} - "
        f"{int(time_values.max())}"
    )
    print(
        f"TIME STEPS: "
        f"{int(time_values.nunique())}"
    )
    print(
        f"MEAN RISK: "
        f"{float(alerts['risk_score'].mean()):.6f}"
    )
    print(
        f"MEDIAN RISK: "
        f"{float(alerts['risk_score'].median()):.6f}"
    )
    print(
        f"TOP-1000 MEAN RISK: "
        f"{float(top_queue['risk_score'].mean()):.6f}"
    )

    print()
    print(
        "HIGHEST MEAN-RISK TIME STEP: "
        f"{temporal_summary['highest_mean_risk_time_step']}"
    )

    print(
        "LOWEST MEAN-RISK TIME STEP: "
        f"{temporal_summary['lowest_mean_risk_time_step']}"
    )

    print(
        "HIGHEST HIGH-ALERT RATE TIME STEP: "
        f"{temporal_summary['highest_high_rate_time_step']}"
    )

    print(
        "HIGHEST VERY-HIGH RATE TIME STEP: "
        f"{temporal_summary['highest_very_high_rate_time_step']}"
    )

    print(
        "MOST TOP-1000 ALERTS IN TIME STEP: "
        f"{temporal_summary['highest_top_queue_time_step']}"
    )

    print()
    print(f"STATUS: {status}")
    print(f"REPORT: {REPORT_PATH}")

    if failed_checks:
        print()
        print("FAILED CHECKS:")

        for name in failed_checks:
            print(f"  - {name}")

        raise RuntimeError(
            f"M11.7 validation failed: {failed_checks}"
        )

    print()
    print("M11.7 COMPLETE")


if __name__ == "__main__":
    main()