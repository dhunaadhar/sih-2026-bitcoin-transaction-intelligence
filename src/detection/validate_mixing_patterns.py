from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CANONICAL_FILE = (
    ROOT
    / "data"
    / "canonical"
    / "canonical_transactions.parquet"
)

INDICATORS_FILE = (
    ROOT
    / "data"
    / "derived"
    / "mixing_pattern_indicators.parquet"
)

REPORT_FILE = (
    ROOT
    / "reports"
    / "behavior"
    / "m9_2_mixing_validation.json"
)

TXID_COLUMN = "txid"

FAN_IN_THRESHOLD = 5
FAN_OUT_THRESHOLD = 5
HIGH_PARTICIPANT_THRESHOLD = 10
EXTREME_PARTICIPANT_THRESHOLD = 25

BALANCED_RATIO_MIN = 0.40
BALANCED_RATIO_MAX = 2.50


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


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    require_file(CANONICAL_FILE)
    require_file(INDICATORS_FILE)

    canonical = pd.read_parquet(
        CANONICAL_FILE
    )

    indicators = pd.read_parquet(
        INDICATORS_FILE
    )

    return (
        canonical,
        indicators,
    )


def validate_schema(
    canonical: pd.DataFrame,
    indicators: pd.DataFrame,
) -> None:

    canonical_required = {
        "txid",
        "time_step",
    }

    indicator_required = {
        "txid",
        "time_step",
        "input_address_count",
        "output_address_count",
        "total_participant_count",
        "fan_in",
        "fan_out",
        "pure_fan_in",
        "pure_fan_out",
        "balanced_fan_pattern",
        "high_participant_count",
        "extreme_participant_count",
        "input_output_ratio",
        "mixing_pattern_candidate",
        "mixing_evidence_score",
    }

    missing_canonical = (
        canonical_required
        - set(canonical.columns)
    )

    if missing_canonical:
        raise ValueError(
            "Canonical dataset missing columns: "
            f"{sorted(missing_canonical)}"
        )

    missing_indicators = (
        indicator_required
        - set(indicators.columns)
    )

    if missing_indicators:
        raise ValueError(
            "Mixing indicators missing columns: "
            f"{sorted(missing_indicators)}"
        )


def validate_row_and_txid_integrity(
    canonical: pd.DataFrame,
    indicators: pd.DataFrame,
) -> dict:

    canonical_txids = (
        canonical[
            "txid"
        ]
        .map(normalize_txid)
    )

    indicator_txids = (
        indicators[
            "txid"
        ]
        .map(normalize_txid)
    )

    checks = {
        "canonical_row_count": bool(
            len(canonical) == 203769
        ),
        "indicator_row_count": bool(
            len(indicators) == len(canonical)
        ),
        "canonical_txids_unique": bool(
            not canonical_txids.duplicated().any()
        ),
        "indicator_txids_unique": bool(
            not indicator_txids.duplicated().any()
        ),
        "exact_txid_coverage": bool(
            set(canonical_txids)
            == set(indicator_txids)
        ),
    }

    if not all(checks.values()):
        failed = [
            key
            for key, value in checks.items()
            if not value
        ]

        raise ValueError(
            "TXID/row integrity validation failed: "
            f"{failed}"
        )

    return {
        **checks,
        "canonical_rows": int(
            len(canonical)
        ),
        "indicator_rows": int(
            len(indicators)
        ),
        "canonical_unique_txids": int(
            canonical_txids.nunique()
        ),
        "indicator_unique_txids": int(
            indicator_txids.nunique()
        ),
    }


def validate_participant_counts(
    indicators: pd.DataFrame,
) -> dict:

    input_counts = indicators[
        "input_address_count"
    ].to_numpy(
        dtype=float
    )

    output_counts = indicators[
        "output_address_count"
    ].to_numpy(
        dtype=float
    )

    participant_counts = indicators[
        "total_participant_count"
    ].to_numpy(
        dtype=float
    )

    checks = {
        "input_counts_finite": bool(
            np.isfinite(input_counts).all()
        ),
        "output_counts_finite": bool(
            np.isfinite(output_counts).all()
        ),
        "participant_counts_finite": bool(
            np.isfinite(
                participant_counts
            ).all()
        ),
        "input_counts_nonnegative": bool(
            (input_counts >= 0).all()
        ),
        "output_counts_nonnegative": bool(
            (output_counts >= 0).all()
        ),
        "participant_counts_nonnegative": bool(
            (participant_counts >= 0).all()
        ),
        "participant_sum_consistent": bool(
            np.array_equal(
                participant_counts,
                input_counts
                + output_counts,
            )
        ),
    }

    if not all(checks.values()):
        failed = [
            key
            for key, value in checks.items()
            if not value
        ]

        raise ValueError(
            "Participant-count validation failed: "
            f"{failed}"
        )

    return {
        **checks,
        "minimum_input_count": int(
            input_counts.min()
        ),
        "maximum_input_count": int(
            input_counts.max()
        ),
        "minimum_output_count": int(
            output_counts.min()
        ),
        "maximum_output_count": int(
            output_counts.max()
        ),
        "maximum_participant_count": int(
            participant_counts.max()
        ),
    }


def validate_structural_flags(
    indicators: pd.DataFrame,
) -> dict:

    input_counts = indicators[
        "input_address_count"
    ]

    output_counts = indicators[
        "output_address_count"
    ]

    participants = indicators[
        "total_participant_count"
    ]

    expected_fan_in = (
        input_counts
        >= FAN_IN_THRESHOLD
    )

    expected_fan_out = (
        output_counts
        >= FAN_OUT_THRESHOLD
    )

    expected_high = (
        participants
        >= HIGH_PARTICIPANT_THRESHOLD
    )

    expected_extreme = (
        participants
        >= EXTREME_PARTICIPANT_THRESHOLD
    )

    expected_pure_in = (
        expected_fan_in
        & ~expected_fan_out
    )

    expected_pure_out = (
        expected_fan_out
        & ~expected_fan_in
    )

    ratio = indicators[
        "input_output_ratio"
    ].to_numpy(
        dtype=float
    )

    expected_balanced = (
        (input_counts > 0)
        & (output_counts > 0)
        & (
            ratio
            >= BALANCED_RATIO_MIN
        )
        & (
            ratio
            <= BALANCED_RATIO_MAX
        )
    )

    expected_balanced_fan = (
        expected_fan_in
        & expected_fan_out
        & expected_balanced
    )

    checks = {
        "fan_in_flag_correct": bool(
            (
                indicators[
                    "fan_in"
                ]
                == expected_fan_in
            ).all()
        ),
        "fan_out_flag_correct": bool(
            (
                indicators[
                    "fan_out"
                ]
                == expected_fan_out
            ).all()
        ),
        "pure_fan_in_flag_correct": bool(
            (
                indicators[
                    "pure_fan_in"
                ]
                == expected_pure_in
            ).all()
        ),
        "pure_fan_out_flag_correct": bool(
            (
                indicators[
                    "pure_fan_out"
                ]
                == expected_pure_out
            ).all()
        ),
        "balanced_flag_correct": bool(
            (
                indicators[
                    "balanced_fan_pattern"
                ]
                == expected_balanced_fan
            ).all()
        ),
        "high_participant_flag_correct": bool(
            (
                indicators[
                    "high_participant_count"
                ]
                == expected_high
            ).all()
        ),
        "extreme_participant_flag_correct": bool(
            (
                indicators[
                    "extreme_participant_count"
                ]
                == expected_extreme
            ).all()
        ),
    }

    if not all(checks.values()):
        failed = [
            key
            for key, value in checks.items()
            if not value
        ]

        raise ValueError(
            "Structural flag validation failed: "
            f"{failed}"
        )

    return {
        **checks,
        "fan_in_count": int(
            expected_fan_in.sum()
        ),
        "fan_out_count": int(
            expected_fan_out.sum()
        ),
        "balanced_fan_pattern_count": int(
            expected_balanced_fan.sum()
        ),
        "high_participant_count": int(
            expected_high.sum()
        ),
        "extreme_participant_count": int(
            expected_extreme.sum()
        ),
    }


def validate_candidate_definition(
    indicators: pd.DataFrame,
) -> dict:

    expected_candidate = (
        indicators[
            "fan_in"
        ]
        & indicators[
            "fan_out"
        ]
        & indicators[
            "balanced_fan_pattern"
        ]
        & indicators[
            "high_participant_count"
        ]
    )

    actual_candidate = (
        indicators[
            "mixing_pattern_candidate"
        ]
    )

    candidate_definition_correct = bool(
        (
            actual_candidate
            == expected_candidate
        ).all()
    )

    if not candidate_definition_correct:
        raise ValueError(
            "Mixing candidate definition is "
            "inconsistent with the hardened "
            "M9.2 definition."
        )

    candidate_mask = (
        actual_candidate
    )

    checks = {
        "candidate_definition_correct": bool(
            candidate_definition_correct
        ),
        "all_candidates_have_fan_in": bool(
            (
                indicators.loc[
                    candidate_mask,
                    "input_address_count",
                ]
                >= FAN_IN_THRESHOLD
            ).all()
        ),
        "all_candidates_have_fan_out": bool(
            (
                indicators.loc[
                    candidate_mask,
                    "output_address_count",
                ]
                >= FAN_OUT_THRESHOLD
            ).all()
        ),
        "all_candidates_have_high_participant_count": bool(
            (
                indicators.loc[
                    candidate_mask,
                    "total_participant_count",
                ]
                >= HIGH_PARTICIPANT_THRESHOLD
            ).all()
        ),
        "all_candidates_are_balanced": bool(
            indicators.loc[
                candidate_mask,
                "balanced_fan_pattern",
            ].all()
        ),
    }

    if not all(checks.values()):
        failed = [
            key
            for key, value in checks.items()
            if not value
        ]

        raise ValueError(
            "Candidate-definition validation failed: "
            f"{failed}"
        )

    return {
        **checks,
        "candidate_count": int(
            candidate_mask.sum()
        ),
        "candidate_rate": float(
            candidate_mask.mean()
        ),
    }


def validate_ratios_and_scores(
    indicators: pd.DataFrame,
) -> dict:

    ratios = indicators[
        "input_output_ratio"
    ].to_numpy(
        dtype=float
    )

    scores = indicators[
        "mixing_evidence_score"
    ].to_numpy(
        dtype=float
    )

    checks = {
        "ratios_finite": bool(
            np.isfinite(
                ratios
            ).all()
        ),
        "ratios_nonnegative": bool(
            (ratios >= 0).all()
        ),
        "scores_finite": bool(
            np.isfinite(
                scores
            ).all()
        ),
        "scores_in_range": bool(
            (
                (scores >= 0)
                & (scores <= 1)
            ).all()
        ),
    }

    if not all(checks.values()):
        failed = [
            key
            for key, value in checks.items()
            if not value
        ]

        raise ValueError(
            "Ratio/score validation failed: "
            f"{failed}"
        )

    return {
        **checks,
        "minimum_ratio": float(
            ratios.min()
        ),
        "maximum_ratio": float(
            ratios.max()
        ),
        "mean_evidence_score": float(
            scores.mean()
        ),
        "maximum_evidence_score": float(
            scores.max()
        ),
    }


def validate_reproducibility(
    indicators: pd.DataFrame,
) -> dict:

    input_counts = indicators[
        "input_address_count"
    ]

    output_counts = indicators[
        "output_address_count"
    ]

    participants = (
        input_counts
        + output_counts
    )

    ratio = indicators[
        "input_output_ratio"
    ]

    recomputed_candidate = (
        (input_counts >= FAN_IN_THRESHOLD)
        & (
            output_counts
            >= FAN_OUT_THRESHOLD
        )
        & (
            participants
            >= HIGH_PARTICIPANT_THRESHOLD
        )
        & (
            ratio
            >= BALANCED_RATIO_MIN
        )
        & (
            ratio
            <= BALANCED_RATIO_MAX
        )
    )

    stored_candidate = indicators[
        "mixing_pattern_candidate"
    ]

    matches = (
        recomputed_candidate
        == stored_candidate
    )

    candidate_flags_identical = bool(
        matches.all()
    )

    if not candidate_flags_identical:
        raise ValueError(
            "Reproducibility check failed."
        )

    return {
        "recomputed_candidate_count": int(
            recomputed_candidate.sum()
        ),
        "stored_candidate_count": int(
            stored_candidate.sum()
        ),
        "candidate_flags_identical": (
            candidate_flags_identical
        ),
    }


def build_report(
    integrity: dict,
    participants: dict,
    structural: dict,
    candidate: dict,
    ratios_scores: dict,
    reproducibility: dict,
) -> dict:

    checks = {}

    sections = (
        integrity,
        participants,
        structural,
        candidate,
        ratios_scores,
        reproducibility,
    )

    for section in sections:
        for key, value in section.items():

            if isinstance(
                value,
                (
                    bool,
                    np.bool_,
                ),
            ):
                checks[key] = bool(value)

    overall_pass = bool(
        all(checks.values())
    )

    return {
        "milestone": "M9.2",
        "validation_type": (
            "independent_mixing_pattern_validation"
        ),
        "status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
        "hardened_candidate_definition": {
            "fan_in_min": int(
                FAN_IN_THRESHOLD
            ),
            "fan_out_min": int(
                FAN_OUT_THRESHOLD
            ),
            "participant_min": int(
                HIGH_PARTICIPANT_THRESHOLD
            ),
            "balanced_ratio_min": float(
                BALANCED_RATIO_MIN
            ),
            "balanced_ratio_max": float(
                BALANCED_RATIO_MAX
            ),
        },
        "integrity": integrity,
        "participant_validation": participants,
        "structural_validation": structural,
        "candidate_validation": candidate,
        "ratio_score_validation": ratios_scores,
        "reproducibility": reproducibility,
        "all_boolean_checks_pass": (
            bool(overall_pass)
        ),
        "interpretation": (
            "The detector identifies transactions whose "
            "observable fan-in/fan-out structure is "
            "compatible with a mixing-like pattern. "
            "This is structural investigative evidence "
            "and does not establish mixer usage, illicit "
            "activity, ownership, identity, intent, or guilt."
        ),
        "source_limitation": (
            "AddrTx and TxAddr provide transaction/address "
            "relationships but do not provide individual "
            "address-level BTC amounts. Therefore this "
            "validator does not claim address-level value "
            "distribution or equal-denomination outputs."
        ),
        "artifacts": {
            "indicators": str(
                INDICATORS_FILE
            ),
            "validation_report": str(
                REPORT_FILE
            ),
        },
    }


def main() -> None:

    print("=" * 72)
    print(
        "M9.2 INDEPENDENT MIXING VALIDATION"
    )
    print("=" * 72)

    canonical, indicators = load_data()

    print(
        f"Canonical rows: "
        f"{len(canonical):,}"
    )

    print(
        f"Indicator rows: "
        f"{len(indicators):,}"
    )

    print(
        "\n[1/6] Validating schemas..."
    )

    validate_schema(
        canonical,
        indicators,
    )

    print(
        "PASS"
    )

    print(
        "\n[2/6] Validating row and TXID integrity..."
    )

    integrity = (
        validate_row_and_txid_integrity(
            canonical,
            indicators,
        )
    )

    print(
        f"PASS — "
        f"{integrity['canonical_unique_txids']:,} "
        f"canonical TXIDs"
    )

    print(
        "\n[3/6] Validating participant counts..."
    )

    participants = (
        validate_participant_counts(
            indicators,
        )
    )

    print(
        "PASS"
    )

    print(
        "\n[4/6] Validating structural flags..."
    )

    structural = (
        validate_structural_flags(
            indicators,
        )
    )

    print(
        f"PASS — "
        f"fan-in {structural['fan_in_count']:,}, "
        f"fan-out {structural['fan_out_count']:,}, "
        f"balanced "
        f"{structural['balanced_fan_pattern_count']:,}"
    )

    print(
        "\n[5/6] Validating hardened candidate definition..."
    )

    candidate = (
        validate_candidate_definition(
            indicators,
        )
    )

    print(
        f"PASS — "
        f"{candidate['candidate_count']:,} "
        f"candidates "
        f"({candidate['candidate_rate']:.4%})"
    )

    print(
        "\n[6/6] Validating ratios, scores, "
        "and reproducibility..."
    )

    ratios_scores = (
        validate_ratios_and_scores(
            indicators,
        )
    )

    reproducibility = (
        validate_reproducibility(
            indicators,
        )
    )

    print(
        "PASS"
    )

    report = build_report(
        integrity,
        participants,
        structural,
        candidate,
        ratios_scores,
        reproducibility,
    )

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

    print(
        "\n" + "-" * 72
    )

    print(
        "M9.2 VALIDATION SUMMARY"
    )

    print(
        "-" * 72
    )

    print(
        f"Candidate count: "
        f"{candidate['candidate_count']:,}"
    )

    print(
        f"Candidate rate: "
        f"{candidate['candidate_rate']:.4%}"
    )

    print(
        f"Mean evidence score: "
        f"{ratios_scores['mean_evidence_score']:.4f}"
    )

    print(
        f"Maximum evidence score: "
        f"{ratios_scores['maximum_evidence_score']:.4f}"
    )

    print(
        f"Recomputed candidate count: "
        f"{reproducibility['recomputed_candidate_count']:,}"
    )

    print(
        f"Stored candidate count: "
        f"{reproducibility['stored_candidate_count']:,}"
    )

    print(
        f"All boolean checks: "
        f"{'PASS' if report['all_boolean_checks_pass'] else 'FAIL'}"
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "M9.2 INDEPENDENT VALIDATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()