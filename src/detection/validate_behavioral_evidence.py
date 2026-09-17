from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PEELING_FILE = (
    ROOT
    / "data"
    / "derived"
    / "peeling_chain_indicators.parquet"
)

MIXING_FILE = (
    ROOT
    / "data"
    / "derived"
    / "mixing_pattern_indicators.parquet"
)

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
    / "m9_3_behavioral_validation.json"
)

TXID_COLUMN = "txid"

PEELING_WEIGHT = 0.50
MIXING_WEIGHT = 0.50

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
            f"Required file not found: {path}"
        )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    require_file(PEELING_FILE)
    require_file(MIXING_FILE)
    require_file(BEHAVIORAL_FILE)

    peeling = pd.read_parquet(
        PEELING_FILE
    )

    mixing = pd.read_parquet(
        MIXING_FILE
    )

    behavioral = pd.read_parquet(
        BEHAVIORAL_FILE
    )

    return (
        peeling,
        mixing,
        behavioral,
    )


def validate_input_integrity(
    frame: pd.DataFrame,
    name: str,
) -> dict:

    if TXID_COLUMN not in frame.columns:
        raise ValueError(
            f"{name} does not contain txid."
        )

    if frame[
        TXID_COLUMN
    ].isna().any():

        raise ValueError(
            f"{name} contains null TXIDs."
        )

    if frame[
        TXID_COLUMN
    ].duplicated().any():

        raise ValueError(
            f"{name} contains duplicate TXIDs."
        )

    txids = (
        frame[
            TXID_COLUMN
        ].map(normalize_txid)
    )

    if (txids == "").any():
        raise ValueError(
            f"{name} contains invalid TXIDs."
        )

    return {
        "rows": int(
            len(frame)
        ),
        "unique_txids": int(
            txids.nunique()
        ),
    }


def validate_required_columns(
    peeling: pd.DataFrame,
    mixing: pd.DataFrame,
    behavioral: pd.DataFrame,
) -> None:

    peeling_required = {
        "txid",
        "time_step",
        "peeling_chain_candidate",
        "peeling_evidence_score",
    }

    mixing_required = {
        "txid",
        "time_step",
        "mixing_pattern_candidate",
        "mixing_evidence_score",
    }

    behavioral_required = {
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

    missing_peeling = (
        peeling_required
        - set(peeling.columns)
    )

    missing_mixing = (
        mixing_required
        - set(mixing.columns)
    )

    missing_behavioral = (
        behavioral_required
        - set(behavioral.columns)
    )

    if missing_peeling:
        raise ValueError(
            "Peeling artifact missing columns: "
            f"{sorted(missing_peeling)}"
        )

    if missing_mixing:
        raise ValueError(
            "Mixing artifact missing columns: "
            f"{sorted(missing_mixing)}"
        )

    if missing_behavioral:
        raise ValueError(
            "Behavioral artifact missing columns: "
            f"{sorted(missing_behavioral)}"
        )


def validate_txid_alignment(
    peeling: pd.DataFrame,
    mixing: pd.DataFrame,
    behavioral: pd.DataFrame,
) -> dict:

    peeling_ids = set(
        peeling[
            TXID_COLUMN
        ].map(normalize_txid)
    )

    mixing_ids = set(
        mixing[
            TXID_COLUMN
        ].map(normalize_txid)
    )

    behavioral_ids = set(
        behavioral[
            TXID_COLUMN
        ].map(normalize_txid)
    )

    peeling_mixing_match = (
        peeling_ids == mixing_ids
    )

    peeling_behavioral_match = (
        peeling_ids == behavioral_ids
    )

    mixing_behavioral_match = (
        mixing_ids == behavioral_ids
    )

    checks = {
        "peeling_mixing_txid_sets_match": bool(
            peeling_mixing_match
        ),
        "peeling_behavioral_txid_sets_match": bool(
            peeling_behavioral_match
        ),
        "mixing_behavioral_txid_sets_match": bool(
            mixing_behavioral_match
        ),
    }

    if not all(checks.values()):

        raise ValueError(
            "TXID alignment validation failed."
        )

    return {
        **checks,
        "aligned_txids": int(
            len(behavioral_ids)
        ),
    }


def validate_time_alignment(
    peeling: pd.DataFrame,
    mixing: pd.DataFrame,
    behavioral: pd.DataFrame,
) -> dict:

    peeling_times = (
        peeling[
            [
                TXID_COLUMN,
                "time_step",
            ]
        ]
        .copy()
    )

    mixing_times = (
        mixing[
            [
                TXID_COLUMN,
                "time_step",
            ]
        ]
        .copy()
    )

    behavioral_times = (
        behavioral[
            [
                TXID_COLUMN,
                "time_step",
            ]
        ]
        .copy()
    )

    for frame in (
        peeling_times,
        mixing_times,
        behavioral_times,
    ):

        frame[
            TXID_COLUMN
        ] = (
            frame[
                TXID_COLUMN
            ].map(normalize_txid)
        )

        frame[
            "time_step"
        ] = frame[
            "time_step"
        ].astype(int)

    peeling_times = (
        peeling_times
        .set_index(TXID_COLUMN)[
            "time_step"
        ]
    )

    mixing_times = (
        mixing_times
        .set_index(TXID_COLUMN)[
            "time_step"
        ]
    )

    behavioral_times = (
        behavioral_times
        .set_index(TXID_COLUMN)[
            "time_step"
        ]
    )

    common_ids = sorted(
        behavioral_times.index
    )

    peeling_mismatch = (
        peeling_times.loc[
            common_ids
        ]
        != behavioral_times.loc[
            common_ids
        ]
    )

    mixing_mismatch = (
        mixing_times.loc[
            common_ids
        ]
        != behavioral_times.loc[
            common_ids
        ]
    )

    peeling_mismatch_count = int(
        peeling_mismatch.sum()
    )

    mixing_mismatch_count = int(
        mixing_mismatch.sum()
    )

    if peeling_mismatch_count:
        raise ValueError(
            "Peeling/behavioral time-step mismatch: "
            f"{peeling_mismatch_count:,}"
        )

    if mixing_mismatch_count:
        raise ValueError(
            "Mixing/behavioral time-step mismatch: "
            f"{mixing_mismatch_count:,}"
        )

    return {
        "peeling_behavioral_mismatches": (
            peeling_mismatch_count
        ),
        "mixing_behavioral_mismatches": (
            mixing_mismatch_count
        ),
        "time_alignment_pass": True,
    }


def validate_scores(
    behavioral: pd.DataFrame,
) -> dict:

    score_columns = [
        "peeling_evidence_score",
        "mixing_evidence_score",
        "behavioral_evidence_score",
        "peeling_mixing_interaction",
    ]

    checks = {}

    for column in score_columns:

        values = behavioral[
            column
        ].to_numpy(
            dtype=float
        )

        finite = bool(
            np.isfinite(
                values
            ).all()
        )

        checks[
            f"{column}_finite"
        ] = finite

        if not finite:
            raise ValueError(
                f"Non-finite values in {column}."
            )

    bounded_columns = [
        "peeling_evidence_score",
        "mixing_evidence_score",
        "behavioral_evidence_score",
    ]

    for column in bounded_columns:

        values = behavioral[
            column
        ].to_numpy(
            dtype=float
        )

        in_range = bool(
            (
                (values >= 0)
                & (values <= 1)
            ).all()
        )

        checks[
            f"{column}_in_range"
        ] = in_range

        if not in_range:
            raise ValueError(
                f"{column} contains values outside [0,1]."
            )

    return {
        **checks,
        "mean_peeling_score": float(
            behavioral[
                "peeling_evidence_score"
            ].mean()
        ),
        "mean_mixing_score": float(
            behavioral[
                "mixing_evidence_score"
            ].mean()
        ),
        "mean_behavioral_score": float(
            behavioral[
                "behavioral_evidence_score"
            ].mean()
        ),
        "median_behavioral_score": float(
            behavioral[
                "behavioral_evidence_score"
            ].median()
        ),
        "maximum_behavioral_score": float(
            behavioral[
                "behavioral_evidence_score"
            ].max()
        ),
    }


def validate_fusion_formula(
    behavioral: pd.DataFrame,
) -> dict:

    expected_score = (
        PEELING_WEIGHT
        * behavioral[
            "peeling_evidence_score"
        ]
        + MIXING_WEIGHT
        * behavioral[
            "mixing_evidence_score"
        ]
    )

    actual_score = behavioral[
        "behavioral_evidence_score"
    ]

    score_matches = np.isclose(
        actual_score.to_numpy(
            dtype=float
        ),
        expected_score.to_numpy(
            dtype=float
        ),
        rtol=1e-10,
        atol=1e-12,
    )

    formula_correct = bool(
        score_matches.all()
    )

    if not formula_correct:
        raise ValueError(
            "Behavioral evidence fusion formula "
            "does not match configured weights."
        )

    return {
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
        "fusion_formula_correct": (
            formula_correct
        ),
    }


def validate_signal_logic(
    behavioral: pd.DataFrame,
) -> dict:

    peeling_candidate = (
        behavioral[
            "peeling_chain_candidate"
        ].astype(bool)
    )

    mixing_candidate = (
        behavioral[
            "mixing_pattern_candidate"
        ].astype(bool)
    )

    expected_signal_count = (
        peeling_candidate.astype(int)
        + mixing_candidate.astype(int)
    )

    actual_signal_count = behavioral[
        "behavioral_signal_count"
    ].astype(int)

    signal_count_matches = (
        expected_signal_count
        == actual_signal_count
    )

    if not bool(
        signal_count_matches.all()
    ):
        raise ValueError(
            "Behavioral signal counts are inconsistent."
        )

    expected_multiple = (
        expected_signal_count
        >= 2
    )

    actual_multiple = behavioral[
        "multiple_behavioral_signals"
    ].astype(bool)

    multiple_matches = (
        expected_multiple
        == actual_multiple
    )

    if not bool(
        multiple_matches.all()
    ):
        raise ValueError(
            "Multiple behavioral signal flags "
            "are inconsistent."
        )

    expected_agreement = (
        (
            behavioral[
                "peeling_evidence_score"
            ]
            > 0
        )
        &
        (
            behavioral[
                "mixing_evidence_score"
            ]
            > 0
        )
    )

    actual_agreement = behavioral[
        "behavioral_signal_agreement"
    ].astype(bool)

    agreement_matches = (
        expected_agreement
        == actual_agreement
    )

    if not bool(
        agreement_matches.all()
    ):
        raise ValueError(
            "Behavioral evidence agreement flags "
            "are inconsistent."
        )

    return {
        "signal_count_correct": True,
        "multiple_signal_flag_correct": True,
        "agreement_flag_correct": True,
        "peeling_candidates": int(
            peeling_candidate.sum()
        ),
        "mixing_candidates": int(
            mixing_candidate.sum()
        ),
        "dual_candidate_count": int(
            (
                peeling_candidate
                & mixing_candidate
            ).sum()
        ),
        "positive_evidence_agreement_count": int(
            expected_agreement.sum()
        ),
    }


def validate_interaction(
    behavioral: pd.DataFrame,
) -> dict:

    expected_interaction = (
        behavioral[
            "peeling_evidence_score"
        ]
        * behavioral[
            "mixing_evidence_score"
        ]
    )

    actual_interaction = behavioral[
        "peeling_mixing_interaction"
    ]

    matches = np.isclose(
        actual_interaction.to_numpy(
            dtype=float
        ),
        expected_interaction.to_numpy(
            dtype=float
        ),
        rtol=1e-10,
        atol=1e-12,
    )

    interaction_correct = bool(
        matches.all()
    )

    if not interaction_correct:
        raise ValueError(
            "Peeling/mixing interaction values "
            "are inconsistent."
        )

    return {
        "interaction_formula_correct": (
            interaction_correct
        ),
        "mean_interaction": float(
            actual_interaction.mean()
        ),
        "maximum_interaction": float(
            actual_interaction.max()
        ),
    }


def expected_evidence_level(
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
    behavioral: pd.DataFrame,
) -> dict:

    scores = behavioral[
        "behavioral_evidence_score"
    ].to_numpy(
        dtype=float
    )

    stored_levels = (
        behavioral[
            "behavioral_evidence_level"
        ]
        .astype(str)
        .to_numpy()
    )

    expected_levels = np.array(
        [
            expected_evidence_level(
                float(score)
            )
            for score in scores
        ],
        dtype=object,
    )

    matches = (
        stored_levels
        == expected_levels
    )

    level_logic_correct = bool(
        matches.all()
    )

    if not level_logic_correct:
        raise ValueError(
            "Behavioral evidence levels are "
            "inconsistent with score boundaries."
        )

    distribution = {
        str(key): int(value)
        for key, value in (
            behavioral[
                "behavioral_evidence_level"
            ]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    return {
        "level_logic_correct": (
            level_logic_correct
        ),
        "distribution": distribution,
    }


def validate_candidate_consistency(
    peeling: pd.DataFrame,
    mixing: pd.DataFrame,
    behavioral: pd.DataFrame,
) -> dict:

    peeling_map = (
        peeling[
            [
                TXID_COLUMN,
                "peeling_chain_candidate",
                "peeling_evidence_score",
            ]
        ]
        .copy()
    )

    mixing_map = (
        mixing[
            [
                TXID_COLUMN,
                "mixing_pattern_candidate",
                "mixing_evidence_score",
            ]
        ]
        .copy()
    )

    behavioral_map = (
        behavioral[
            [
                TXID_COLUMN,
                "peeling_chain_candidate",
                "mixing_pattern_candidate",
                "peeling_evidence_score",
                "mixing_evidence_score",
            ]
        ]
        .copy()
    )

    for frame in (
        peeling_map,
        mixing_map,
        behavioral_map,
    ):

        frame[
            TXID_COLUMN
        ] = (
            frame[
                TXID_COLUMN
            ].map(normalize_txid)
        )

    peeling_map = (
        peeling_map
        .set_index(TXID_COLUMN)
    )

    mixing_map = (
        mixing_map
        .set_index(TXID_COLUMN)
    )

    behavioral_map = (
        behavioral_map
        .set_index(TXID_COLUMN)
    )

    ids = sorted(
        behavioral_map.index
    )

    peeling_candidate_match = (
        peeling_map.loc[
            ids,
            "peeling_chain_candidate",
        ].astype(bool)
        ==
        behavioral_map.loc[
            ids,
            "peeling_chain_candidate",
        ].astype(bool)
    )

    mixing_candidate_match = (
        mixing_map.loc[
            ids,
            "mixing_pattern_candidate",
        ].astype(bool)
        ==
        behavioral_map.loc[
            ids,
            "mixing_pattern_candidate",
        ].astype(bool)
    )

    peeling_score_match = np.isclose(
        peeling_map.loc[
            ids,
            "peeling_evidence_score",
        ].to_numpy(
            dtype=float
        ),
        behavioral_map.loc[
            ids,
            "peeling_evidence_score",
        ].to_numpy(
            dtype=float
        ),
        rtol=1e-10,
        atol=1e-12,
    )

    mixing_score_match = np.isclose(
        mixing_map.loc[
            ids,
            "mixing_evidence_score",
        ].to_numpy(
            dtype=float
        ),
        behavioral_map.loc[
            ids,
            "mixing_evidence_score",
        ].to_numpy(
            dtype=float
        ),
        rtol=1e-10,
        atol=1e-12,
    )

    checks = {
        "peeling_candidate_preserved": bool(
            peeling_candidate_match.all()
        ),
        "mixing_candidate_preserved": bool(
            mixing_candidate_match.all()
        ),
        "peeling_score_preserved": bool(
            peeling_score_match.all()
        ),
        "mixing_score_preserved": bool(
            mixing_score_match.all()
        ),
    }

    if not all(checks.values()):
        raise ValueError(
            "Source evidence was not preserved "
            "in the behavioral artifact."
        )

    return checks


def validate_time_range(
    behavioral: pd.DataFrame,
) -> dict:

    values = behavioral[
        "time_step"
    ].to_numpy(
        dtype=int
    )

    finite = bool(
        np.isfinite(values).all()
    )

    valid_range = bool(
        (
            (values >= 1)
            & (values <= 49)
        ).all()
    )

    if not finite:
        raise ValueError(
            "Non-finite time-step values."
        )

    if not valid_range:
        raise ValueError(
            "Time-step values outside 1-49."
        )

    return {
        "time_step_finite": finite,
        "time_step_range_valid": valid_range,
        "minimum_time_step": int(
            values.min()
        ),
        "maximum_time_step": int(
            values.max()
        ),
        "unique_time_steps": int(
            np.unique(values).size
        ),
    }


def build_report(
    integrity: dict,
    alignment: dict,
    time_alignment: dict,
    scores: dict,
    fusion: dict,
    signals: dict,
    interaction: dict,
    levels: dict,
    consistency: dict,
    time_range: dict,
) -> dict:

    boolean_sections = (
        integrity,
        alignment,
        time_alignment,
        scores,
        fusion,
        signals,
        interaction,
        levels,
        consistency,
        time_range,
    )

    all_checks = {}

    for section in boolean_sections:

        for key, value in section.items():

            if isinstance(
                value,
                (
                    bool,
                    np.bool_,
                ),
            ):

                all_checks[
                    key
                ] = bool(value)

    overall_pass = bool(
        all(
            all_checks.values()
        )
    )

    return {
        "milestone": "M9.3",
        "validation_type": (
            "independent_behavioral_evidence_validation"
        ),
        "status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
        "checks": all_checks,
        "input_integrity": integrity,
        "txid_alignment": alignment,
        "time_alignment": time_alignment,
        "score_validation": scores,
        "fusion_validation": fusion,
        "signal_validation": signals,
        "interaction_validation": interaction,
        "evidence_level_validation": levels,
        "source_consistency": consistency,
        "time_range_validation": time_range,
        "summary": {
            "rows": int(
                len(
                    pd.read_parquet(
                        BEHAVIORAL_FILE
                    )
                )
            ),
            "peeling_candidates": int(
                signals[
                    "peeling_candidates"
                ]
            ),
            "mixing_candidates": int(
                signals[
                    "mixing_candidates"
                ]
            ),
            "dual_candidate_count": int(
                signals[
                    "dual_candidate_count"
                ]
            ),
            "positive_evidence_agreement_count": int(
                signals[
                    "positive_evidence_agreement_count"
                ]
            ),
            "mean_behavioral_score": float(
                scores[
                    "mean_behavioral_score"
                ]
            ),
            "median_behavioral_score": float(
                scores[
                    "median_behavioral_score"
                ]
            ),
            "maximum_behavioral_score": float(
                scores[
                    "maximum_behavioral_score"
                ]
            ),
        },
        "interpretation": (
            "The behavioral layer combines two independently "
            "generated structural evidence channels. "
            "The resulting score is an evidence-prioritization "
            "measure, not a probability of illicit activity "
            "and not proof of identity, ownership, intent, "
            "mixer usage, or guilt."
        ),
        "limitations": {
            "peeling": (
                "Peeling-chain value reduction uses "
                "transaction-level output BTC because the "
                "source address relationship files do not "
                "contain address-level BTC allocations."
            ),
            "mixing": (
                "Mixing detection is based on observable "
                "fan-in/fan-out participant structure and "
                "does not establish actual mixer usage."
            ),
        },
        "offline_processing": True,
        "artifacts": {
            "behavioral_evidence": str(
                BEHAVIORAL_FILE
            ),
            "validation_report": str(
                REPORT_FILE
            ),
        },
    }


def main() -> None:

    print("=" * 72)
    print(
        "M9.3 INDEPENDENT BEHAVIORAL VALIDATION"
    )
    print("=" * 72)

    (
        peeling,
        mixing,
        behavioral,
    ) = load_inputs()

    print(
        f"Peeling rows: "
        f"{len(peeling):,}"
    )

    print(
        f"Mixing rows: "
        f"{len(mixing):,}"
    )

    print(
        f"Behavioral rows: "
        f"{len(behavioral):,}"
    )

    print(
        "\n[1/10] Validating input integrity..."
    )

    peeling_integrity = (
        validate_input_integrity(
            peeling,
            "Peeling artifact",
        )
    )

    mixing_integrity = (
        validate_input_integrity(
            mixing,
            "Mixing artifact",
        )
    )

    behavioral_integrity = (
        validate_input_integrity(
            behavioral,
            "Behavioral artifact",
        )
    )

    integrity = {
        "peeling_integrity": True,
        "mixing_integrity": True,
        "behavioral_integrity": True,
        "peeling_rows": int(
            peeling_integrity["rows"]
        ),
        "mixing_rows": int(
            mixing_integrity["rows"]
        ),
        "behavioral_rows": int(
            behavioral_integrity["rows"]
        ),
    }

    print(
        "PASS"
    )

    print(
        "\n[2/10] Validating required schemas..."
    )

    validate_required_columns(
        peeling,
        mixing,
        behavioral,
    )

    print(
        "PASS"
    )

    print(
        "\n[3/10] Validating TXID alignment..."
    )

    alignment = validate_txid_alignment(
        peeling,
        mixing,
        behavioral,
    )

    print(
        f"PASS — "
        f"{alignment['aligned_txids']:,} aligned TXIDs"
    )

    print(
        "\n[4/10] Validating time-step alignment..."
    )

    time_alignment = validate_time_alignment(
        peeling,
        mixing,
        behavioral,
    )

    print(
        "PASS"
    )

    print(
        "\n[5/10] Validating evidence scores..."
    )

    scores = validate_scores(
        behavioral
    )

    print(
        "PASS"
    )

    print(
        "\n[6/10] Validating 50/50 fusion formula..."
    )

    fusion = validate_fusion_formula(
        behavioral
    )

    print(
        "PASS"
    )

    print(
        "\n[7/10] Validating behavioral signal logic..."
    )

    signals = validate_signal_logic(
        behavioral
    )

    print(
        f"PASS — "
        f"dual candidates "
        f"{signals['dual_candidate_count']:,}"
    )

    print(
        "\n[8/10] Validating interaction and "
        "evidence levels..."
    )

    interaction = validate_interaction(
        behavioral
    )

    levels = validate_evidence_levels(
        behavioral
    )

    print(
        "PASS"
    )

    print(
        "\n[9/10] Validating source evidence preservation..."
    )

    consistency = (
        validate_candidate_consistency(
            peeling,
            mixing,
            behavioral,
        )
    )

    print(
        "PASS"
    )

    print(
        "\n[10/10] Validating time range and "
        "final consistency..."
    )

    time_range = validate_time_range(
        behavioral
    )

    print(
        "PASS"
    )

    report = build_report(
        integrity,
        alignment,
        time_alignment,
        scores,
        fusion,
        signals,
        interaction,
        levels,
        consistency,
        time_range,
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
        "M9.3 VALIDATION SUMMARY"
    )

    print(
        "-" * 72
    )

    print(
        f"Transactions: "
        f"{alignment['aligned_txids']:,}"
    )

    print(
        f"Peeling candidates: "
        f"{signals['peeling_candidates']:,}"
    )

    print(
        f"Mixing candidates: "
        f"{signals['mixing_candidates']:,}"
    )

    print(
        f"Dual candidate signals: "
        f"{signals['dual_candidate_count']:,}"
    )

    print(
        f"Positive evidence agreement: "
        f"{signals['positive_evidence_agreement_count']:,}"
    )

    print(
        f"Mean behavioral evidence: "
        f"{scores['mean_behavioral_score']:.4f}"
    )

    print(
        f"Median behavioral evidence: "
        f"{scores['median_behavioral_score']:.4f}"
    )

    print(
        f"Maximum behavioral evidence: "
        f"{scores['maximum_behavioral_score']:.4f}"
    )

    print(
        f"Fusion formula: "
        f"{'PASS' if fusion['fusion_formula_correct'] else 'FAIL'}"
    )

    print(
        f"Evidence-level logic: "
        f"{'PASS' if levels['level_logic_correct'] else 'FAIL'}"
    )

    print(
        f"Source consistency: "
        f"{'PASS' if all(consistency.values()) else 'FAIL'}"
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "M9.3 INDEPENDENT VALIDATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Status: {report['status']}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()