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
REPORT_PATH = REPORT_DIR / "m11_6_ranked_alert_validation.json"

EXPECTED_ROWS = 203_769
TOP_QUEUE_SIZE = 1_000

# M11 has five possible evidence channels:
#   1. supervised ML
#   2. anomaly
#   3. behavioral
#   4. entity / graph
#   5. network
MAX_EVIDENCE_CHANNELS = 5


def normalize_txid_series(series: pd.Series) -> pd.Series:
    """
    Normalize TXIDs so integer-like floating representations such as
    4304541.0 become the canonical string 4304541.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.isna().any():
        raise ValueError("TXID column contains non-numeric or missing values.")

    values = []

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


def deterministic_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct the deterministic M11.5 ranking order.

    Ranking precedence:
      1. risk_score descending
      2. effective_evidence_channel_count descending
      3. behavioral_signal descending
      4. entity_signal descending
      5. anomaly_signal descending
      6. network_signal descending
      7. time_step ascending
      8. txid ascending
    """
    ranked = df.copy()

    ranked["_txid_sort"] = ranked["txid_normalized"].astype(str)

    sort_columns = [
        "risk_score",
        "effective_evidence_channel_count",
        "behavioral_signal",
        "entity_signal",
        "anomaly_signal",
        "network_signal",
        "time_step",
        "_txid_sort",
    ]

    ascending = [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]

    ranked = ranked.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    ).reset_index(drop=True)

    return ranked.drop(columns=["_txid_sort"])


def check(
    name: str,
    condition: bool,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "check": name,
        "status": "PASS" if condition else "FAIL",
    }

    if details:
        result["details"] = details

    return result


def main() -> None:
    print("=" * 65)
    print("M11.6 RANKED ALERT VALIDATION")
    print("=" * 65)

    if not RANKED_ALERTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing ranked alert artifact: {RANKED_ALERTS_PATH}"
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
        "behavioral_signal",
        "entity_signal",
        "anomaly_signal",
        "network_signal",
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
            "Required columns are missing. Validation cannot continue."
        )

    alerts = alerts.copy()
    risk = risk.copy()

    alerts["txid_normalized"] = normalize_txid_series(alerts["txid"])
    risk["txid_normalized"] = normalize_txid_series(risk["txid"])

    # ------------------------------------------------------------------
    # 1. Row and TXID integrity
    # ------------------------------------------------------------------

    alert_unique = alerts["txid_normalized"].nunique()
    risk_unique = risk["txid_normalized"].nunique()

    checks.append(
        check(
            "ranked_alert_row_count",
            len(alerts) == EXPECTED_ROWS,
            {
                "actual": int(len(alerts)),
                "expected": EXPECTED_ROWS,
            },
        )
    )

    checks.append(
        check(
            "ranked_alert_unique_txids",
            alert_unique == len(alerts),
            {
                "rows": int(len(alerts)),
                "unique_txids": int(alert_unique),
            },
        )
    )

    checks.append(
        check(
            "risk_score_unique_txids",
            risk_unique == len(risk),
            {
                "rows": int(len(risk)),
                "unique_txids": int(risk_unique),
            },
        )
    )

    alert_txids = set(alerts["txid_normalized"])
    risk_txids = set(risk["txid_normalized"])

    checks.append(
        check(
            "ranked_alert_txid_coverage",
            alert_txids == risk_txids,
            {
                "alert_only": int(len(alert_txids - risk_txids)),
                "risk_only": int(len(risk_txids - alert_txids)),
            },
        )
    )

    # ------------------------------------------------------------------
    # 2. Numeric integrity
    # ------------------------------------------------------------------

    numeric_columns = [
        "time_step",
        "risk_score",
        "investigation_rank",
        "effective_evidence_channel_count",
        "behavioral_signal",
        "entity_signal",
        "anomaly_signal",
        "network_signal",
    ]

    numeric_integrity = True
    numeric_details: dict[str, Any] = {}

    for column in numeric_columns:
        values = pd.to_numeric(alerts[column], errors="coerce")

        missing = int(values.isna().sum())
        infinite = int(
            np.isinf(values.to_numpy(dtype=float)).sum()
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

    # ------------------------------------------------------------------
    # 3. Risk-score range
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 4. Priority-band reproducibility
    # ------------------------------------------------------------------

    expected_priorities = alerts["risk_score"].apply(priority_from_score)

    priority_match = (
        expected_priorities.to_numpy()
        == alerts["alert_priority"].astype(str).to_numpy()
    )

    checks.append(
        check(
            "priority_band_consistency",
            bool(priority_match.all()),
            {
                "mismatches": int((~priority_match).sum()),
                "distribution": {
                    str(k): int(v)
                    for k, v in alerts["alert_priority"]
                    .value_counts()
                    .to_dict()
                    .items()
                },
            },
        )
    )

    # ------------------------------------------------------------------
    # 5. Investigation rank integrity
    # ------------------------------------------------------------------

    ranks = pd.to_numeric(
        alerts["investigation_rank"],
        errors="coerce",
    )

    expected_ranks = np.arange(1, len(alerts) + 1)

    rank_sequence_ok = (
        len(ranks) == len(expected_ranks)
        and ranks.notna().all()
        and np.array_equal(
            ranks.to_numpy(dtype=np.int64),
            expected_ranks,
        )
    )

    checks.append(
        check(
            "investigation_rank_sequence",
            bool(rank_sequence_ok),
            {
                "first_rank": (
                    int(ranks.min())
                    if ranks.notna().any()
                    else None
                ),
                "last_rank": (
                    int(ranks.max())
                    if ranks.notna().any()
                    else None
                ),
                "expected_last_rank": int(len(alerts)),
                "unique_ranks": int(ranks.nunique()),
            },
        )
    )

    # ------------------------------------------------------------------
    # 6. Deterministic ranking reconstruction
    # ------------------------------------------------------------------

    reconstructed = deterministic_rank(alerts)

    stored_order = alerts["txid_normalized"].tolist()
    reconstructed_order = reconstructed["txid_normalized"].tolist()

    ranking_reproducible = stored_order == reconstructed_order

    checks.append(
        check(
            "deterministic_ranking_reproducibility",
            ranking_reproducible,
            {
                "mismatched_positions": int(
                    sum(
                        a != b
                        for a, b in zip(
                            stored_order,
                            reconstructed_order,
                        )
                    )
                ),
            },
        )
    )

    # ------------------------------------------------------------------
    # 7. Monotonic risk ordering
    # ------------------------------------------------------------------

    ordered_risk = alerts["risk_score"].to_numpy(dtype=float)

    risk_monotonic = bool(
        np.all(
            ordered_risk[:-1] >= ordered_risk[1:]
        )
    )

    checks.append(
        check(
            "risk_score_monotonic_descending",
            risk_monotonic,
            {
                "violations": int(
                    np.sum(
                        ordered_risk[:-1] < ordered_risk[1:]
                    )
                )
            },
        )
    )

    # ------------------------------------------------------------------
    # 8. Top-1000 queue integrity
    # ------------------------------------------------------------------

    top_queue = alerts.head(TOP_QUEUE_SIZE)

    top_queue_ok = len(top_queue) == TOP_QUEUE_SIZE

    if top_queue_ok:
        top_queue_ranks = pd.to_numeric(
            top_queue["investigation_rank"],
            errors="coerce",
        )

        top_queue_ok = bool(
            np.array_equal(
                top_queue_ranks.to_numpy(dtype=np.int64),
                np.arange(1, TOP_QUEUE_SIZE + 1),
            )
        )

    checks.append(
        check(
            "top_1000_queue_integrity",
            top_queue_ok,
            {
                "queue_size": int(len(top_queue)),
                "expected_size": TOP_QUEUE_SIZE,
            },
        )
    )

    # ------------------------------------------------------------------
    # 9. Top-1000 risk boundary
    # ------------------------------------------------------------------

    top_boundary_ok = True
    boundary_details: dict[str, Any] = {}

    if len(alerts) >= TOP_QUEUE_SIZE:
        top_min = float(
            alerts.iloc[TOP_QUEUE_SIZE - 1]["risk_score"]
        )
        remainder = alerts.iloc[TOP_QUEUE_SIZE:]["risk_score"]

        if len(remainder) > 0:
            remainder_max = float(remainder.max())

            top_boundary_ok = top_min >= remainder_max

            boundary_details = {
                "top_1000_min_risk": top_min,
                "remaining_max_risk": remainder_max,
            }
        else:
            boundary_details = {
                "top_1000_min_risk": top_min,
                "remaining_max_risk": None,
            }

    checks.append(
        check(
            "top_1000_high_risk_boundary",
            top_boundary_ok,
            boundary_details,
        )
    )

    # ------------------------------------------------------------------
    # 10. Evidence-channel integrity
    # ------------------------------------------------------------------

    evidence_columns = [
        "behavioral_signal",
        "entity_signal",
        "anomaly_signal",
        "network_signal",
    ]

    evidence_range_ok = True
    evidence_details: dict[str, Any] = {}

    for column in evidence_columns:
        values = pd.to_numeric(
            alerts[column],
            errors="coerce",
        )

        column_min = float(values.min())
        column_max = float(values.max())

        evidence_details[column] = {
            "min": column_min,
            "max": column_max,
        }

        if column_min < 0.0 or column_max > 1.0:
            evidence_range_ok = False

    channel_counts = pd.to_numeric(
        alerts["effective_evidence_channel_count"],
        errors="coerce",
    )

    # M11.4 explicitly defines the valid range as 0 through 5.
    channel_count_ok = bool(
        channel_counts.between(
            0,
            MAX_EVIDENCE_CHANNELS,
            inclusive="both",
        ).all()
    )

    checks.append(
        check(
            "evidence_signal_ranges",
            evidence_range_ok,
            evidence_details,
        )
    )

    checks.append(
        check(
            "evidence_channel_count_range",
            channel_count_ok,
            {
                "min": int(channel_counts.min()),
                "max": int(channel_counts.max()),
                "allowed_min": 0,
                "allowed_max": MAX_EVIDENCE_CHANNELS,
                "unique": sorted(
                    int(x)
                    for x in channel_counts.unique()
                ),
            },
        )
    )

    # ------------------------------------------------------------------
    # 11. M11.4 score consistency
    # ------------------------------------------------------------------

    risk_lookup = risk[
        [
            "txid_normalized",
            "time_step",
            "risk_score",
        ]
    ].copy()

    risk_lookup = risk_lookup.rename(
        columns={
            "risk_score": "risk_score_m11_4",
            "time_step": "time_step_m11_4",
        }
    )

    merged = alerts.merge(
        risk_lookup,
        on="txid_normalized",
        how="left",
        validate="one_to_one",
    )

    score_match = np.isclose(
        merged["risk_score"].to_numpy(dtype=float),
        merged["risk_score_m11_4"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    )

    time_match = (
        pd.to_numeric(
            merged["time_step"],
            errors="coerce",
        ).to_numpy()
        == pd.to_numeric(
            merged["time_step_m11_4"],
            errors="coerce",
        ).to_numpy()
    )

    checks.append(
        check(
            "m11_4_risk_score_consistency",
            bool(score_match.all()),
            {
                "mismatches": int((~score_match).sum()),
            },
        )
    )

    checks.append(
        check(
            "m11_4_time_step_consistency",
            bool(time_match.all()),
            {
                "mismatches": int((~time_match).sum()),
            },
        )
    )

    # ------------------------------------------------------------------
    # 12. Top-1000 class distribution
    # ------------------------------------------------------------------

    class_distribution: dict[str, int] = {}

    if "ml_predicted_class" in top_queue.columns:
        class_distribution = {
            str(k): int(v)
            for k, v in top_queue["ml_predicted_class"]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        }

    # ------------------------------------------------------------------
    # 13. Final summary
    # ------------------------------------------------------------------

    failed_checks = [
        item["check"]
        for item in checks
        if item["status"] != "PASS"
    ]

    status = "PASS" if not failed_checks else "FAIL"

    report = {
        "milestone": "M11.6",
        "title": "Ranked Alert Validation",
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
            "unique_txids": int(alert_unique),
            "risk_score_min": risk_min,
            "risk_score_max": risk_max,
        },
        "alerts": {
            "active_alerts": int(
                alerts["alert_priority"].isin(
                    ["VERY_HIGH", "HIGH", "MODERATE"]
                ).sum()
            ),
            "priority_distribution": {
                str(k): int(v)
                for k, v in alerts["alert_priority"]
                .value_counts()
                .to_dict()
                .items()
            },
            "top_queue_size": TOP_QUEUE_SIZE,
            "top_queue_class_distribution": class_distribution,
        },
        "ranking": {
            "deterministic_reproducible": ranking_reproducible,
            "risk_monotonic_descending": risk_monotonic,
            "top_1000_boundary_valid": top_boundary_ok,
            "first_rank": int(ranks.min()),
            "last_rank": int(ranks.max()),
        },
        "evidence": {
            "channel_count_min": int(channel_counts.min()),
            "channel_count_max": int(channel_counts.max()),
            "allowed_channel_count_max": MAX_EVIDENCE_CHANNELS,
            "signal_ranges_valid": evidence_range_ok,
        },
        "checks": checks,
        "failed_checks": failed_checks,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

    print("M11.6 VALIDATION RESULT")
    print("-" * 65)

    for item in checks:
        marker = "PASS" if item["status"] == "PASS" else "FAIL"
        print(f"{marker:<6} {item['check']}")

    print()
    print(f"ROWS: {len(alerts):,}")
    print(f"UNIQUE TXIDS: {alert_unique:,}")
    print(f"RISK RANGE: {risk_min:.6f} - {risk_max:.6f}")
    print(f"TOP QUEUE: {TOP_QUEUE_SIZE:,}")
    print(
        "TOP-1000 MEAN RISK: "
        f"{float(top_queue['risk_score'].mean()):.6f}"
    )
    print(
        "OVERALL MEAN RISK: "
        f"{float(alerts['risk_score'].mean()):.6f}"
    )
    print(
        "EVIDENCE CHANNEL COUNT RANGE: "
        f"{int(channel_counts.min())} - "
        f"{int(channel_counts.max())} "
        f"(allowed 0 - {MAX_EVIDENCE_CHANNELS})"
    )

    if class_distribution:
        print(
            "TOP-1000 CLASS DISTRIBUTION: "
            f"{class_distribution}"
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
            f"M11.6 validation failed: {failed_checks}"
        )

    print()
    print("M11.6 COMPLETE")


if __name__ == "__main__":
    main()