from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

BEHAVIORAL_FILE = (
    ROOT
    / "data"
    / "derived"
    / "behavioral_evidence.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "behavior"
    / "m9_4_behavioral_temporal_validation.json"
)

TXID_COLUMN = "txid"

PEELING_WEIGHT = 0.50
MIXING_WEIGHT = 0.50

TIME_MIN = 1
TIME_MAX = 49

LOW_MAX = 0.20
MODERATE_MAX = 0.50
HIGH_MAX = 0.75


def normalize_txid(value) -> str:

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    try:
        numeric = float(text)

        if not np.isfinite(numeric):
            return ""

        if numeric.is_integer():
            return str(int(numeric))

        return format(
            numeric,
            ".15g",
        )

    except (
        ValueError,
        TypeError,
    ):
        return text


def require_file(
    path: Path,
) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )


def load_behavioral_data() -> pd.DataFrame:

    require_file(BEHAVIORAL_FILE)

    frame = pd.read_parquet(
        BEHAVIORAL_FILE
    )

    return frame


def validate_schema(
    frame: pd.DataFrame,
) -> dict:

    required_columns = {
        "txid",
        "time_step",
        "peeling_chain_candidate",
        "mixing_pattern_candidate",
        "peeling_evidence_score",
        "mixing_evidence_score",
        "behavioral_evidence_score",
        "behavioral_signal_count",
        "multiple_behavioral_signals",
        "behavioral_signal_agreement",
        "peeling_mixing_interaction",
        "behavioral_evidence_level",
    }

    missing = (
        required_columns
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    return {
        "schema_pass": True,
        "required_columns": int(
            len(required_columns)
        ),
        "actual_columns": int(
            len(frame.columns)
        ),
    }


def validate_row_integrity(
    frame: pd.DataFrame,
) -> dict:

    if frame.empty:
        raise ValueError(
            "Behavioral evidence artifact is empty."
        )

    if frame[
        TXID_COLUMN
    ].isna().any():

        raise ValueError(
            "Null TXIDs detected."
        )

    normalized_txids = (
        frame[
            TXID_COLUMN
        ].map(normalize_txid)
    )

    invalid_txids = int(
        (normalized_txids == "").sum()
    )

    duplicate_txids = int(
        normalized_txids.duplicated().sum()
    )

    if invalid_txids:
        raise ValueError(
            f"Invalid TXIDs: {invalid_txids}"
        )

    if duplicate_txids:
        raise ValueError(
            f"Duplicate TXIDs: {duplicate_txids}"
        )

    return {
        "row_integrity_pass": True,
        "rows": int(
            len(frame)
        ),
        "unique_txids": int(
            normalized_txids.nunique()
        ),
        "invalid_txids": invalid_txids,
        "duplicate_txids": duplicate_txids,
    }


def validate_time_steps(
    frame: pd.DataFrame,
) -> dict:

    values = frame[
        "time_step"
    ].to_numpy(
        dtype=float
    )

    finite = bool(
        np.isfinite(values).all()
    )

    integer_like = bool(
        np.isclose(
            values,
            np.round(values),
        ).all()
    )

    in_range = bool(
        (
            (values >= TIME_MIN)
            & (values <= TIME_MAX)
        ).all()
    )

    if not finite:
        raise ValueError(
            "Non-finite time-step values."
        )

    if not integer_like:
        raise ValueError(
            "Non-integer time-step values."
        )

    if not in_range:
        raise ValueError(
            "Time-step values outside "
            f"{TIME_MIN}-{TIME_MAX}."
        )

    unique_steps = np.unique(
        values.astype(int)
    )

    expected_steps = np.arange(
        TIME_MIN,
        TIME_MAX + 1,
    )

    complete = bool(
        np.array_equal(
            unique_steps,
            expected_steps,
        )
    )

    if not complete:
        raise ValueError(
            "Expected every time step from "
            f"{TIME_MIN} to {TIME_MAX}."
        )

    return {
        "time_step_pass": True,
        "minimum_time_step": int(
            unique_steps.min()
        ),
        "maximum_time_step": int(
            unique_steps.max()
        ),
        "unique_time_steps": int(
            unique_steps.size
        ),
        "all_expected_time_steps_present": complete,
    }


def validate_numeric_integrity(
    frame: pd.DataFrame,
) -> dict:

    numeric_columns = [
        "peeling_evidence_score",
        "mixing_evidence_score",
        "behavioral_evidence_score",
        "peeling_mixing_interaction",
        "behavioral_signal_count",
    ]

    finite_checks = {}
    range_checks = {}

    for column in numeric_columns:

        values = frame[
            column
        ].to_numpy(
            dtype=float
        )

        finite = bool(
            np.isfinite(values).all()
        )

        finite_checks[
            f"{column}_finite"
        ] = finite

        if not finite:
            raise ValueError(
                f"Non-finite values detected in "
                f"{column}."
            )

    bounded_columns = [
        "peeling_evidence_score",
        "mixing_evidence_score",
        "behavioral_evidence_score",
        "peeling_mixing_interaction",
    ]

    for column in bounded_columns:

        values = frame[
            column
        ].to_numpy(
            dtype=float
        )

        valid = bool(
            (
                (values >= 0.0)
                & (values <= 1.0)
            ).all()
        )

        range_checks[
            f"{column}_in_range"
        ] = valid

        if not valid:
            raise ValueError(
                f"{column} contains values outside [0,1]."
            )

    return {
        "numeric_integrity_pass": True,
        "finite_checks": finite_checks,
        "range_checks": range_checks,
    }


def validate_candidate_evidence_consistency(
    frame: pd.DataFrame,
) -> dict:

    peeling_candidate = (
        frame[
            "peeling_chain_candidate"
        ].astype(bool)
    )

    mixing_candidate = (
        frame[
            "mixing_pattern_candidate"
        ].astype(bool)
    )

    peeling_score = frame[
        "peeling_evidence_score"
    ].to_numpy(
        dtype=float
    )

    mixing_score = frame[
        "mixing_evidence_score"
    ].to_numpy(
        dtype=float
    )

    peeling_without_evidence = int(
        (
            peeling_candidate.to_numpy()
            & (peeling_score <= 0)
        ).sum()
    )

    mixing_without_evidence = int(
        (
            mixing_candidate.to_numpy()
            & (mixing_score <= 0)
        ).sum()
    )

    evidence_without_candidate_peeling = int(
        (
            (~peeling_candidate.to_numpy())
            & (peeling_score > 0)
        ).sum()
    )

    evidence_without_candidate_mixing = int(
        (
            (~mixing_candidate.to_numpy())
            & (mixing_score > 0)
        ).sum()
    )

    if peeling_without_evidence:
        raise ValueError(
            "Peeling candidates without positive "
            "peeling evidence detected."
        )

    if mixing_without_evidence:
        raise ValueError(
            "Mixing candidates without positive "
            "mixing evidence detected."
        )

    return {
        "candidate_evidence_pass": True,
        "peeling_candidates_without_positive_evidence": (
            peeling_without_evidence
        ),
        "mixing_candidates_without_positive_evidence": (
            mixing_without_evidence
        ),
        "peeling_positive_evidence_without_candidate": (
            evidence_without_candidate_peeling
        ),
        "mixing_positive_evidence_without_candidate": (
            evidence_without_candidate_mixing
        ),
    }


def validate_fusion(
    frame: pd.DataFrame,
) -> dict:

    expected = (
        PEELING_WEIGHT
        * frame[
            "peeling_evidence_score"
        ].to_numpy(
            dtype=float
        )
        +
        MIXING_WEIGHT
        * frame[
            "mixing_evidence_score"
        ].to_numpy(
            dtype=float
        )
    )

    actual = frame[
        "behavioral_evidence_score"
    ].to_numpy(
        dtype=float
    )

    matches = np.isclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-12,
    )

    mismatch_count = int(
        (~matches).sum()
    )

    if mismatch_count:
        raise ValueError(
            "Behavioral fusion mismatch count: "
            f"{mismatch_count}"
        )

    return {
        "fusion_pass": True,
        "peeling_weight": float(
            PEELING_WEIGHT
        ),
        "mixing_weight": float(
            MIXING_WEIGHT
        ),
        "weight_sum": float(
            PEELING_WEIGHT
            + MIXING_WEIGHT
        ),
        "formula_mismatches": mismatch_count,
    }


def validate_signal_counts(
    frame: pd.DataFrame,
) -> dict:

    peeling = (
        frame[
            "peeling_chain_candidate"
        ].astype(bool)
    )

    mixing = (
        frame[
            "mixing_pattern_candidate"
        ].astype(bool)
    )

    expected_count = (
        peeling.astype(int)
        + mixing.astype(int)
    )

    actual_count = frame[
        "behavioral_signal_count"
    ].astype(int)

    count_match = (
        expected_count
        == actual_count
    )

    expected_multiple = (
        expected_count >= 2
    )

    actual_multiple = frame[
        "multiple_behavioral_signals"
    ].astype(bool)

    multiple_match = (
        expected_multiple
        == actual_multiple
    )

    expected_agreement = (
        (
            frame[
                "peeling_evidence_score"
            ].to_numpy(
                dtype=float
            )
            > 0
        )
        &
        (
            frame[
                "mixing_evidence_score"
            ].to_numpy(
                dtype=float
            )
            > 0
        )
    )

    actual_agreement = frame[
        "behavioral_signal_agreement"
    ].astype(bool).to_numpy()

    agreement_match = (
        expected_agreement
        == actual_agreement
    )

    if not bool(
        count_match.all()
    ):
        raise ValueError(
            "Behavioral signal count mismatch."
        )

    if not bool(
        multiple_match.all()
    ):
        raise ValueError(
            "Multiple-signal flag mismatch."
        )

    if not bool(
        agreement_match.all()
    ):
        raise ValueError(
            "Signal-agreement flag mismatch."
        )

    return {
        "signal_logic_pass": True,
        "signal_count_mismatches": int(
            (~count_match).sum()
        ),
        "multiple_signal_mismatches": int(
            (~multiple_match).sum()
        ),
        "agreement_mismatches": int(
            (~agreement_match).sum()
        ),
        "zero_signal_transactions": int(
            (expected_count == 0).sum()
        ),
        "single_signal_transactions": int(
            (expected_count == 1).sum()
        ),
        "dual_signal_transactions": int(
            (expected_count == 2).sum()
        ),
    }


def validate_interaction(
    frame: pd.DataFrame,
) -> dict:

    expected = (
        frame[
            "peeling_evidence_score"
        ].to_numpy(
            dtype=float
        )
        *
        frame[
            "mixing_evidence_score"
        ].to_numpy(
            dtype=float
        )
    )

    actual = frame[
        "peeling_mixing_interaction"
    ].to_numpy(
        dtype=float
    )

    matches = np.isclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-12,
    )

    mismatch_count = int(
        (~matches).sum()
    )

    if mismatch_count:
        raise ValueError(
            "Interaction mismatch count: "
            f"{mismatch_count}"
        )

    return {
        "interaction_pass": True,
        "interaction_mismatches": mismatch_count,
        "mean_interaction": float(
            actual.mean()
        ),
        "maximum_interaction": float(
            actual.max()
        ),
    }


def expected_level(
    score: float,
) -> str:

    if score < LOW_MAX:
        return "LOW"

    if score < MODERATE_MAX:
        return "MODERATE"

    if score < HIGH_MAX:
        return "HIGH"

    return "VERY_HIGH"


def validate_evidence_levels(
    frame: pd.DataFrame,
) -> dict:

    scores = frame[
        "behavioral_evidence_score"
    ].to_numpy(
        dtype=float
    )

    actual = (
        frame[
            "behavioral_evidence_level"
        ]
        .astype(str)
        .to_numpy()
    )

    expected = np.array(
        [
            expected_level(
                float(score)
            )
            for score in scores
        ],
        dtype=object,
    )

    matches = (
        actual == expected
    )

    mismatch_count = int(
        (~matches).sum()
    )

    if mismatch_count:
        raise ValueError(
            "Evidence-level mismatch count: "
            f"{mismatch_count}"
        )

    distribution = {
        str(key): int(value)
        for key, value in (
            frame[
                "behavioral_evidence_level"
            ]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    return {
        "evidence_level_pass": True,
        "level_mismatches": mismatch_count,
        "distribution": distribution,
        "thresholds": {
            "LOW": f"< {LOW_MAX}",
            "MODERATE": (
                f">= {LOW_MAX} and < {MODERATE_MAX}"
            ),
            "HIGH": (
                f">= {MODERATE_MAX} and < {HIGH_MAX}"
            ),
            "VERY_HIGH": (
                f">= {HIGH_MAX}"
            ),
        },
    }


def build_temporal_diagnostics(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict,
]:

    grouped = (
        frame
        .groupby(
            "time_step",
            sort=True,
        )
        .agg(
            transactions=(
                TXID_COLUMN,
                "count",
            ),
            peeling_candidates=(
                "peeling_chain_candidate",
                "sum",
            ),
            mixing_candidates=(
                "mixing_pattern_candidate",
                "sum",
            ),
            dual_signal_transactions=(
                "multiple_behavioral_signals",
                "sum",
            ),
            mean_peeling_evidence=(
                "peeling_evidence_score",
                "mean",
            ),
            mean_mixing_evidence=(
                "mixing_evidence_score",
                "mean",
            ),
            mean_behavioral_evidence=(
                "behavioral_evidence_score",
                "mean",
            ),
            median_behavioral_evidence=(
                "behavioral_evidence_score",
                "median",
            ),
            p95_behavioral_evidence=(
                "behavioral_evidence_score",
                lambda x: float(
                    np.quantile(
                        x,
                        0.95,
                    )
                ),
            ),
            max_behavioral_evidence=(
                "behavioral_evidence_score",
                "max",
            ),
            mean_interaction=(
                "peeling_mixing_interaction",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped[
        "peeling_candidate_rate"
    ] = (
        grouped[
            "peeling_candidates"
        ]
        / grouped[
            "transactions"
        ]
    )

    grouped[
        "mixing_candidate_rate"
    ] = (
        grouped[
            "mixing_candidates"
        ]
        / grouped[
            "transactions"
        ]
    )

    grouped[
        "dual_signal_rate"
    ] = (
        grouped[
            "dual_signal_transactions"
        ]
        / grouped[
            "transactions"
        ]
    )

    overall_mean = float(
        frame[
            "behavioral_evidence_score"
        ].mean()
    )

    overall_std = float(
        frame[
            "behavioral_evidence_score"
        ].std(
            ddof=0
        )
    )

    if overall_std > 0:
        grouped[
            "behavioral_mean_zscore"
        ] = (
            grouped[
                "mean_behavioral_evidence"
            ]
            - overall_mean
        ) / overall_std

    else:
        grouped[
            "behavioral_mean_zscore"
        ] = 0.0

    temporal_summary = {
        "time_steps": int(
            len(grouped)
        ),
        "overall_mean_behavioral_evidence": (
            overall_mean
        ),
        "overall_std_behavioral_evidence": (
            overall_std
        ),
        "highest_mean_behavioral_time_step": int(
            grouped.loc[
                grouped[
                    "mean_behavioral_evidence"
                ].idxmax(),
                "time_step",
            ]
        ),
        "lowest_mean_behavioral_time_step": int(
            grouped.loc[
                grouped[
                    "mean_behavioral_evidence"
                ].idxmin(),
                "time_step",
            ]
        ),
        "highest_peeling_rate_time_step": int(
            grouped.loc[
                grouped[
                    "peeling_candidate_rate"
                ].idxmax(),
                "time_step",
            ]
        ),
        "highest_mixing_rate_time_step": int(
            grouped.loc[
                grouped[
                    "mixing_candidate_rate"
                ].idxmax(),
                "time_step",
            ]
        ),
        "highest_dual_signal_rate_time_step": int(
            grouped.loc[
                grouped[
                    "dual_signal_rate"
                ].idxmax(),
                "time_step",
            ]
        ),
        "maximum_mean_behavioral_zscore": float(
            grouped[
                "behavioral_mean_zscore"
            ].max()
        ),
        "minimum_mean_behavioral_zscore": float(
            grouped[
                "behavioral_mean_zscore"
            ].min()
        ),
    }

    return grouped, temporal_summary


def validate_temporal_diagnostics(
    temporal: pd.DataFrame,
) -> dict:

    required = [
        "time_step",
        "transactions",
        "mean_behavioral_evidence",
        "peeling_candidate_rate",
        "mixing_candidate_rate",
        "dual_signal_rate",
    ]

    missing = (
        set(required)
        - set(temporal.columns)
    )

    if missing:
        raise ValueError(
            "Temporal diagnostic columns missing: "
            f"{sorted(missing)}"
        )

    if temporal[
        "transactions"
    ].le(0).any():

        raise ValueError(
            "A time step contains zero transactions."
        )

    rate_columns = [
        "peeling_candidate_rate",
        "mixing_candidate_rate",
        "dual_signal_rate",
    ]

    rate_checks = {}

    for column in rate_columns:

        values = temporal[
            column
        ].to_numpy(
            dtype=float
        )

        finite = bool(
            np.isfinite(values).all()
        )

        bounded = bool(
            (
                (values >= 0)
                & (values <= 1)
            ).all()
        )

        rate_checks[
            f"{column}_finite"
        ] = finite

        rate_checks[
            f"{column}_in_range"
        ] = bounded

        if not finite or not bounded:
            raise ValueError(
                f"Invalid temporal rate values in {column}."
            )

    behavioral_means = temporal[
        "mean_behavioral_evidence"
    ].to_numpy(
        dtype=float
    )

    behavioral_valid = bool(
        (
            np.isfinite(
                behavioral_means
            ).all()
            and
            (
                (behavioral_means >= 0)
                & (behavioral_means <= 1)
            ).all()
        )
    )

    if not behavioral_valid:
        raise ValueError(
            "Invalid temporal behavioral means."
        )

    return {
        "temporal_diagnostics_pass": True,
        "rate_checks": rate_checks,
        "behavioral_mean_checks_pass": True,
    }


def build_concentration_diagnostics(
    frame: pd.DataFrame,
) -> dict:

    behavioral = frame[
        "behavioral_evidence_score"
    ].to_numpy(
        dtype=float
    )

    peeling = frame[
        "peeling_evidence_score"
    ].to_numpy(
        dtype=float
    )

    mixing = frame[
        "mixing_evidence_score"
    ].to_numpy(
        dtype=float
    )

    top_percentages = {
        "top_0_1_percent": max(
            1,
            int(
                np.ceil(
                    len(frame) * 0.001
                )
            ),
        ),
        "top_1_percent": max(
            1,
            int(
                np.ceil(
                    len(frame) * 0.01
                )
            ),
        ),
        "top_5_percent": max(
            1,
            int(
                np.ceil(
                    len(frame) * 0.05
                )
            ),
        ),
    }

    sorted_behavioral = np.sort(
        behavioral
    )[::-1]

    concentration = {}

    for label, count in (
        top_percentages.items()
    ):

        values = sorted_behavioral[
            :count
        ]

        concentration[
            label
        ] = {
            "transactions": int(
                count
            ),
            "mean_behavioral_evidence": float(
                values.mean()
            ),
            "sum_behavioral_evidence": float(
                values.sum()
            ),
        }

    positive_behavioral = int(
        (behavioral > 0).sum()
    )

    positive_peeling = int(
        (peeling > 0).sum()
    )

    positive_mixing = int(
        (mixing > 0).sum()
    )

    return {
        "positive_behavioral_evidence_transactions": (
            positive_behavioral
        ),
        "positive_peeling_evidence_transactions": (
            positive_peeling
        ),
        "positive_mixing_evidence_transactions": (
            positive_mixing
        ),
        "positive_behavioral_rate": float(
            positive_behavioral
            / len(frame)
        ),
        "concentration": concentration,
    }


def build_behavioral_statistics(
    frame: pd.DataFrame,
) -> dict:

    score_columns = [
        "peeling_evidence_score",
        "mixing_evidence_score",
        "behavioral_evidence_score",
        "peeling_mixing_interaction",
    ]

    statistics = {}

    for column in score_columns:

        values = frame[
            column
        ].to_numpy(
            dtype=float
        )

        statistics[
            column
        ] = {
            "mean": float(
                np.mean(values)
            ),
            "median": float(
                np.median(values)
            ),
            "std": float(
                np.std(
                    values
                )
            ),
            "minimum": float(
                np.min(values)
            ),
            "maximum": float(
                np.max(values)
            ),
            "p95": float(
                np.quantile(
                    values,
                    0.95,
                )
            ),
            "p99": float(
                np.quantile(
                    values,
                    0.99,
                )
            ),
        }

    return statistics


def build_reproducibility_fingerprint(
    frame: pd.DataFrame,
) -> dict:

    canonical = frame[
        [
            TXID_COLUMN,
            "time_step",
            "peeling_chain_candidate",
            "mixing_pattern_candidate",
            "peeling_evidence_score",
            "mixing_evidence_score",
            "behavioral_evidence_score",
            "behavioral_signal_count",
            "multiple_behavioral_signals",
            "behavioral_signal_agreement",
            "peeling_mixing_interaction",
            "behavioral_evidence_level",
        ]
    ].copy()

    canonical[
        TXID_COLUMN
    ] = (
        canonical[
            TXID_COLUMN
        ].map(normalize_txid)
    )

    canonical = canonical.sort_values(
        TXID_COLUMN
    )

    payload = (
        canonical
        .to_json(
            orient="records",
            date_format="iso",
            double_precision=15,
        )
        .encode(
            "utf-8"
        )
    )

    import hashlib

    fingerprint = hashlib.sha256(
        payload
    ).hexdigest()

    return {
        "fingerprint_algorithm": "SHA256",
        "behavioral_artifact_fingerprint": (
            fingerprint
        ),
    }


def validate_reproducibility(
    frame: pd.DataFrame,
) -> dict:

    first = (
        build_reproducibility_fingerprint(
            frame
        )[
            "behavioral_artifact_fingerprint"
        ]
    )

    second = (
        build_reproducibility_fingerprint(
            frame
        )[
            "behavioral_artifact_fingerprint"
        ]
    )

    reproducible = bool(
        first == second
    )

    if not reproducible:
        raise ValueError(
            "Diagnostic fingerprint was not reproducible."
        )

    return {
        "reproducibility_pass": reproducible,
        "fingerprint_algorithm": "SHA256",
        "fingerprint": first,
    }


def write_report(
    report: dict,
) -> None:

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            allow_nan=False,
        )


def main() -> None:

    print("=" * 72)
    print(
        "M9.4 BEHAVIORAL TEMPORAL VALIDATION"
    )
    print("=" * 72)

    frame = load_behavioral_data()

    print(
        f"Behavioral rows: "
        f"{len(frame):,}"
    )

    print(
        "\n[1/9] Schema validation..."
    )

    schema = validate_schema(
        frame
    )

    print(
        "PASS"
    )

    print(
        "\n[2/9] Row and TXID integrity..."
    )

    integrity = validate_row_integrity(
        frame
    )

    print(
        f"PASS — "
        f"{integrity['unique_txids']:,} unique TXIDs"
    )

    print(
        "\n[3/9] Time-step validation..."
    )

    time_validation = validate_time_steps(
        frame
    )

    print(
        f"PASS — "
        f"{time_validation['minimum_time_step']}-"
        f"{time_validation['maximum_time_step']}"
    )

    print(
        "\n[4/9] Numeric integrity and score ranges..."
    )

    numeric = validate_numeric_integrity(
        frame
    )

    print(
        "PASS"
    )

    print(
        "\n[5/9] Candidate/evidence consistency..."
    )

    candidate_consistency = (
        validate_candidate_evidence_consistency(
            frame
        )
    )

    print(
        "PASS"
    )

    print(
        "\n[6/9] Fusion, signal and interaction validation..."
    )

    fusion = validate_fusion(
        frame
    )

    signals = validate_signal_counts(
        frame
    )

    interaction = validate_interaction(
        frame
    )

    print(
        "PASS"
    )

    print(
        "\n[7/9] Evidence-level validation..."
    )

    levels = validate_evidence_levels(
        frame
    )

    print(
        "PASS"
    )

    print(
        "\n[8/9] Temporal diagnostics..."
    )

    temporal_frame, temporal_summary = (
        build_temporal_diagnostics(
            frame
        )
    )

    temporal_validation = (
        validate_temporal_diagnostics(
            temporal_frame
        )
    )

    print(
        f"PASS — "
        f"{temporal_summary['time_steps']} time steps"
    )

    print(
        "\n[9/9] Concentration and reproducibility..."
    )

    concentration = (
        build_concentration_diagnostics(
            frame
        )
    )

    statistics = (
        build_behavioral_statistics(
            frame
        )
    )

    reproducibility = (
        validate_reproducibility(
            frame
        )
    )

    print(
        "PASS"
    )

    report = {
        "milestone": "M9.4",
        "status": "PASS",
        "purpose": (
            "Temporal and diagnostic validation of the "
            "completed M9 behavioral intelligence layer."
        ),
        "checks": {
            "schema": schema,
            "row_integrity": integrity,
            "time_validation": time_validation,
            "numeric_integrity": numeric,
            "candidate_consistency": candidate_consistency,
            "fusion": fusion,
            "signals": signals,
            "interaction": interaction,
            "evidence_levels": levels,
            "temporal_validation": temporal_validation,
            "reproducibility": reproducibility,
        },
        "behavioral_statistics": statistics,
        "temporal_summary": temporal_summary,
        "temporal_distribution": (
            temporal_frame
            .to_dict(
                orient="records"
            )
        ),
        "concentration_diagnostics": concentration,
        "final_summary": {
            "transactions": int(
                len(frame)
            ),
            "peeling_candidates": int(
                frame[
                    "peeling_chain_candidate"
                ].astype(bool).sum()
            ),
            "mixing_candidates": int(
                frame[
                    "mixing_pattern_candidate"
                ].astype(bool).sum()
            ),
            "dual_signal_transactions": int(
                signals[
                    "dual_signal_transactions"
                ]
            ),
            "mean_behavioral_evidence": float(
                frame[
                    "behavioral_evidence_score"
                ].mean()
            ),
            "median_behavioral_evidence": float(
                frame[
                    "behavioral_evidence_score"
                ].median()
            ),
            "maximum_behavioral_evidence": float(
                frame[
                    "behavioral_evidence_score"
                ].max()
            ),
        },
        "interpretation": (
            "M9.4 validates behavioral evidence consistency "
            "and temporal distribution. These diagnostics "
            "measure structural behavioral evidence for "
            "investigative prioritization. They do not establish "
            "identity, ownership, intent, mixer usage, illicit "
            "activity, or guilt."
        ),
        "limitations": {
            "behavioral_layer": (
                "Peeling-chain and mixing indicators are "
                "behavioral/structural signals rather than "
                "identity attribution."
            ),
            "temporal_diagnostics": (
                "Temporal concentration indicates when the "
                "behavioral signals are more or less prevalent "
                "in the supplied dataset; it is not evidence "
                "of real-world criminal activity."
            ),
            "offline_scope": (
                "Validation is performed entirely offline "
                "against the generated behavioral artifact."
            ),
        },
        "artifacts": {
            "input": str(
                BEHAVIORAL_FILE
            ),
            "report": str(
                REPORT_FILE
            ),
        },
        "offline_processing": True,
    }

    write_report(
        report
    )

    print(
        "\n" + "-" * 72
    )
    print(
        "M9.4 DIAGNOSTIC SUMMARY"
    )
    print(
        "-" * 72
    )

    print(
        f"Transactions: "
        f"{len(frame):,}"
    )

    print(
        f"Peeling candidates: "
        f"{int(frame['peeling_chain_candidate'].astype(bool).sum()):,}"
    )

    print(
        f"Mixing candidates: "
        f"{int(frame['mixing_pattern_candidate'].astype(bool).sum()):,}"
    )

    print(
        f"Dual-signal transactions: "
        f"{signals['dual_signal_transactions']:,}"
    )

    print(
        f"Mean behavioral evidence: "
        f"{frame['behavioral_evidence_score'].mean():.4f}"
    )

    print(
        f"Median behavioral evidence: "
        f"{frame['behavioral_evidence_score'].median():.4f}"
    )

    print(
        f"Maximum behavioral evidence: "
        f"{frame['behavioral_evidence_score'].max():.4f}"
    )

    print(
        f"Highest mean behavioral time step: "
        f"{temporal_summary['highest_mean_behavioral_time_step']}"
    )

    print(
        f"Lowest mean behavioral time step: "
        f"{temporal_summary['lowest_mean_behavioral_time_step']}"
    )

    print(
        f"Highest peeling-rate time step: "
        f"{temporal_summary['highest_peeling_rate_time_step']}"
    )

    print(
        f"Highest mixing-rate time step: "
        f"{temporal_summary['highest_mixing_rate_time_step']}"
    )

    print(
        f"Highest dual-signal-rate time step: "
        f"{temporal_summary['highest_dual_signal_rate_time_step']}"
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "M9.4 BEHAVIORAL TEMPORAL VALIDATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "Status: PASS"
    )

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()