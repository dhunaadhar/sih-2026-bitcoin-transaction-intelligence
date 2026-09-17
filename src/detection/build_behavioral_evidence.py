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

OUTPUT_DIR = (
    ROOT
    / "data"
    / "derived"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "behavioral_evidence.parquet"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "behavior"
)

REPORT_FILE = (
    REPORT_DIR
    / "m9_3_behavioral_evidence.json"
)

TXID_COLUMN = "txid"

PEELING_WEIGHT = 0.50
MIXING_WEIGHT = 0.50

EVIDENCE_LOW_MAX = 0.20
EVIDENCE_MODERATE_MAX = 0.50
EVIDENCE_HIGH_MAX = 0.75


def normalize_txid(value) -> str:
    """
    Normalize TXIDs so equivalent numeric representations
    are treated as the same identifier.
    """

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


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if not PEELING_FILE.exists():
        raise FileNotFoundError(
            "Peeling-chain artifact not found: "
            f"{PEELING_FILE}"
        )

    if not MIXING_FILE.exists():
        raise FileNotFoundError(
            "Mixing-pattern artifact not found: "
            f"{MIXING_FILE}"
        )

    peeling = pd.read_parquet(
        PEELING_FILE
    )

    mixing = pd.read_parquet(
        MIXING_FILE
    )

    return (
        peeling,
        mixing,
    )


def validate_input(
    frame: pd.DataFrame,
    name: str,
) -> None:

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


def validate_required_columns(
    peeling: pd.DataFrame,
    mixing: pd.DataFrame,
) -> None:

    peeling_required = {
        "txid",
        "time_step",
        "peeling_direct_continuation",
        "peeling_gradual_value_reduction",
        "peeling_chain_candidate",
        "peeling_evidence_score",
        "continuation_address_count",
        "previous_transaction_count",
        "best_previous_txid",
        "best_previous_time_step",
        "temporal_gap",
        "best_value_reduction_ratio",
        "current_output_btc_total",
        "current_input_btc_total",
        "current_output_address_count",
        "current_input_address_count",
    }

    missing_peeling = (
        peeling_required
        - set(peeling.columns)
    )

    if missing_peeling:
        raise ValueError(
            "Peeling artifact missing required columns: "
            f"{sorted(missing_peeling)}"
        )

    mixing_required = {
        "txid",
        "time_step",
        "input_address_count",
        "output_address_count",
        "total_participant_count",
        "participant_density",
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

    missing_mixing = (
        mixing_required
        - set(mixing.columns)
    )

    if missing_mixing:
        raise ValueError(
            "Mixing artifact missing required columns: "
            f"{sorted(missing_mixing)}"
        )


def validate_alignment(
    peeling: pd.DataFrame,
    mixing: pd.DataFrame,
) -> None:

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

    if peeling_ids != mixing_ids:

        missing_from_mixing = (
            peeling_ids
            - mixing_ids
        )

        missing_from_peeling = (
            mixing_ids
            - peeling_ids
        )

        raise ValueError(
            "Behavioral artifacts have different TXID sets. "
            f"Missing from mixing: "
            f"{len(missing_from_mixing):,}; "
            f"missing from peeling: "
            f"{len(missing_from_peeling):,}."
        )


def validate_weights() -> None:

    total_weight = (
        PEELING_WEIGHT
        + MIXING_WEIGHT
    )

    if not np.isclose(
        total_weight,
        1.0,
    ):
        raise ValueError(
            "Behavioral evidence weights must sum to 1.0. "
            f"Current sum={total_weight}"
        )


def build_behavioral_evidence(
    peeling: pd.DataFrame,
    mixing: pd.DataFrame,
) -> pd.DataFrame:

    validate_weights()

    peeling = peeling.copy()
    mixing = mixing.copy()

    peeling[
        TXID_COLUMN
    ] = (
        peeling[
            TXID_COLUMN
        ].map(normalize_txid)
    )

    mixing[
        TXID_COLUMN
    ] = (
        mixing[
            TXID_COLUMN
        ].map(normalize_txid)
    )

    peeling_columns = [
        "txid",
        "time_step",
        "peeling_direct_continuation",
        "peeling_gradual_value_reduction",
        "peeling_chain_candidate",
        "peeling_evidence_score",
        "continuation_address_count",
        "previous_transaction_count",
        "best_previous_txid",
        "best_previous_time_step",
        "temporal_gap",
        "best_value_reduction_ratio",
        "current_output_btc_total",
        "current_input_btc_total",
        "current_output_address_count",
        "current_input_address_count",
    ]

    mixing_columns = [
        "txid",
        "time_step",
        "input_address_count",
        "output_address_count",
        "total_participant_count",
        "participant_density",
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
    ]

    merged = (
        peeling[
            peeling_columns
        ]
        .merge(
            mixing[
                mixing_columns
            ],
            on=TXID_COLUMN,
            how="inner",
            validate="one_to_one",
            suffixes=(
                "_peeling",
                "_mixing",
            ),
        )
    )

    if len(merged) != len(peeling):
        raise ValueError(
            "Behavioral merge lost transactions. "
            f"Peeling rows={len(peeling):,}; "
            f"merged rows={len(merged):,}."
        )

    if len(merged) != len(mixing):
        raise ValueError(
            "Behavioral merge lost transactions. "
            f"Mixing rows={len(mixing):,}; "
            f"merged rows={len(merged):,}."
        )

    peeling_time = (
        merged[
            "time_step_peeling"
        ]
        .astype(int)
    )

    mixing_time = (
        merged[
            "time_step_mixing"
        ]
        .astype(int)
    )

    time_mismatch = int(
        (
            peeling_time
            != mixing_time
        ).sum()
    )

    if time_mismatch:
        raise ValueError(
            "Time-step mismatch detected for "
            f"{time_mismatch:,} transactions."
        )

    merged[
        "time_step"
    ] = peeling_time

    merged.drop(
        columns=[
            "time_step_peeling",
            "time_step_mixing",
        ],
        inplace=True,
    )

    score_columns = [
        "peeling_evidence_score",
        "mixing_evidence_score",
    ]

    for column in score_columns:

        merged[
            column
        ] = pd.to_numeric(
            merged[
                column
            ],
            errors="coerce",
        )

        values = merged[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():

            raise ValueError(
                f"Invalid non-finite values in {column}."
            )

        if (
            (values < 0)
            | (values > 1)
        ).any():

            raise ValueError(
                f"{column} contains values outside [0,1]."
            )

    peeling_score = (
        merged[
            "peeling_evidence_score"
        ]
    )

    mixing_score = (
        merged[
            "mixing_evidence_score"
        ]
    )

    peeling_candidate = (
        merged[
            "peeling_chain_candidate"
        ].astype(bool)
    )

    mixing_candidate = (
        merged[
            "mixing_pattern_candidate"
        ].astype(bool)
    )

    merged[
        "behavioral_evidence_score"
    ] = (
        PEELING_WEIGHT
        * peeling_score
        + MIXING_WEIGHT
        * mixing_score
    )

    merged[
        "behavioral_signal_count"
    ] = (
        peeling_candidate.astype(int)
        + mixing_candidate.astype(int)
    )

    merged[
        "multiple_behavioral_signals"
    ] = (
        merged[
            "behavioral_signal_count"
        ]
        >= 2
    )

    merged[
        "behavioral_signal_agreement"
    ] = (
        (
            peeling_score
            > 0
        )
        &
        (
            mixing_score
            > 0
        )
    )

    merged[
        "peeling_mixing_interaction"
    ] = (
        peeling_score
        * mixing_score
    )

    merged[
        "behavioral_evidence_level"
    ] = pd.cut(
        merged[
            "behavioral_evidence_score"
        ],
        bins=[
            -np.inf,
            EVIDENCE_LOW_MAX,
            EVIDENCE_MODERATE_MAX,
            EVIDENCE_HIGH_MAX,
            np.inf,
        ],
        labels=[
            "LOW",
            "MODERATE",
            "HIGH",
            "VERY_HIGH",
        ],
        right=False,
    ).astype(str)

    return merged


def validate_output(
    result: pd.DataFrame,
    expected_txids: set[str],
) -> dict:

    result_ids = set(
        result[
            TXID_COLUMN
        ].map(normalize_txid)
    )

    if len(result) != len(
        expected_txids
    ):
        raise ValueError(
            "Behavioral evidence row count does not "
            "match expected TXID count."
        )

    if result[
        TXID_COLUMN
    ].duplicated().any():

        raise ValueError(
            "Duplicate TXIDs in behavioral evidence."
        )

    if result_ids != expected_txids:
        raise ValueError(
            "Behavioral evidence TXID coverage "
            "does not match input artifacts."
        )

    score_columns = [
        "peeling_evidence_score",
        "mixing_evidence_score",
        "behavioral_evidence_score",
        "peeling_mixing_interaction",
    ]

    for column in score_columns:

        values = result[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():

            raise ValueError(
                f"Non-finite values in {column}."
            )

    bounded_score_columns = [
        "peeling_evidence_score",
        "mixing_evidence_score",
        "behavioral_evidence_score",
    ]

    for column in bounded_score_columns:

        values = result[
            column
        ].to_numpy(
            dtype=float
        )

        if (
            (values < 0)
            | (values > 1)
        ).any():

            raise ValueError(
                f"{column} contains values outside [0,1]."
            )

    expected_behavioral_score = (
        PEELING_WEIGHT
        * result[
            "peeling_evidence_score"
        ]
        + MIXING_WEIGHT
        * result[
            "mixing_evidence_score"
        ]
    )

    score_matches = np.isclose(
        result[
            "behavioral_evidence_score"
        ].to_numpy(
            dtype=float
        ),
        expected_behavioral_score.to_numpy(
            dtype=float
        ),
        rtol=1e-10,
        atol=1e-12,
    )

    if not score_matches.all():
        raise ValueError(
            "Behavioral evidence scores are inconsistent "
            "with the configured channel weights."
        )

    expected_signal_count = (
        result[
            "peeling_chain_candidate"
        ].astype(int)
        + result[
            "mixing_pattern_candidate"
        ].astype(int)
    )

    if not (
        result[
            "behavioral_signal_count"
        ].to_numpy(
            dtype=int
        )
        == expected_signal_count.to_numpy(
            dtype=int
        )
    ).all():

        raise ValueError(
            "Behavioral signal counts are inconsistent "
            "with the source candidate flags."
        )

    expected_multiple = (
        expected_signal_count
        >= 2
    )

    if not (
        result[
            "multiple_behavioral_signals"
        ].to_numpy(
            dtype=bool
        )
        == expected_multiple.to_numpy(
            dtype=bool
        )
    ).all():

        raise ValueError(
            "Multiple-signal flags are inconsistent."
        )

    expected_agreement = (
        (
            result[
                "peeling_evidence_score"
            ]
            > 0
        )
        &
        (
            result[
                "mixing_evidence_score"
            ]
            > 0
        )
    )

    if not (
        result[
            "behavioral_signal_agreement"
        ].to_numpy(
            dtype=bool
        )
        == expected_agreement.to_numpy(
            dtype=bool
        )
    ).all():

        raise ValueError(
            "Behavioral signal agreement flags "
            "are inconsistent."
        )

    expected_interaction = (
        result[
            "peeling_evidence_score"
        ]
        * result[
            "mixing_evidence_score"
        ]
    )

    if not np.isclose(
        result[
            "peeling_mixing_interaction"
        ].to_numpy(
            dtype=float
        ),
        expected_interaction.to_numpy(
            dtype=float
        ),
        rtol=1e-10,
        atol=1e-12,
    ).all():

        raise ValueError(
            "Peeling/mixing interaction values "
            "are inconsistent."
        )

    time_values = result[
        "time_step"
    ].to_numpy(
        dtype=int
    )

    if (
        (time_values < 1)
        | (time_values > 49)
    ).any():

        raise ValueError(
            "Invalid time-step values."
        )

    if result[
        "behavioral_evidence_level"
    ].isna().any():

        raise ValueError(
            "Missing behavioral evidence levels."
        )

    return {
        "rows": int(
            len(result)
        ),
        "unique_txids": int(
            result[
                TXID_COLUMN
            ].nunique()
        ),
        "time_steps": {
            "min": int(
                result[
                    "time_step"
                ].min()
            ),
            "max": int(
                result[
                    "time_step"
                ].max()
            ),
            "unique": int(
                result[
                    "time_step"
                ].nunique()
            ),
        },
        "peeling_candidates": int(
            result[
                "peeling_chain_candidate"
            ].sum()
        ),
        "mixing_candidates": int(
            result[
                "mixing_pattern_candidate"
            ].sum()
        ),
        "multiple_behavioral_signals": int(
            result[
                "multiple_behavioral_signals"
            ].sum()
        ),
        "behavioral_signal_agreement": int(
            result[
                "behavioral_signal_agreement"
            ].sum()
        ),
        "mean_peeling_score": float(
            result[
                "peeling_evidence_score"
            ].mean()
        ),
        "mean_mixing_score": float(
            result[
                "mixing_evidence_score"
            ].mean()
        ),
        "mean_behavioral_score": float(
            result[
                "behavioral_evidence_score"
            ].mean()
        ),
        "median_behavioral_score": float(
            result[
                "behavioral_evidence_score"
            ].median()
        ),
        "maximum_behavioral_score": float(
            result[
                "behavioral_evidence_score"
            ].max()
        ),
        "evidence_level_distribution": {
            str(key): int(value)
            for key, value in (
                result[
                    "behavioral_evidence_level"
                ]
                .value_counts()
                .sort_index()
                .items()
            )
        },
    }


def build_report(
    validation: dict,
) -> dict:

    return {
        "milestone": "M9.3",
        "status": "PASS",
        "detector": (
            "behavioral_evidence_fusion"
        ),
        "purpose": (
            "Combine independently generated peeling-chain "
            "and mixing-pattern structural evidence into "
            "a single auditable behavioral evidence layer."
        ),
        "source_artifacts": {
            "peeling": str(
                PEELING_FILE
            ),
            "mixing": str(
                MIXING_FILE
            ),
        },
        "methodology": {
            "peeling_weight": float(
                PEELING_WEIGHT
            ),
            "mixing_weight": float(
                MIXING_WEIGHT
            ),
            "combined_score": (
                "0.50 * peeling evidence + "
                "0.50 * mixing evidence"
            ),
            "independent_channels_preserved": True,
            "candidate_channels_preserved": True,
            "interaction_feature": (
                "peeling evidence multiplied by "
                "mixing evidence"
            ),
            "agreement_definition": (
                "Both independent evidence scores are "
                "strictly greater than zero."
            ),
            "multiple_signal_definition": (
                "Both peeling-chain and mixing-pattern "
                "candidate flags are true."
            ),
        },
        "validation": validation,
        "interpretation": (
            "Behavioral evidence is structural investigative "
            "evidence for downstream prioritization. "
            "It is not a probability of illicit activity "
            "and does not establish mixer usage, ownership, "
            "identity, intent, or guilt."
        ),
        "source_limitations": {
            "peeling": (
                "The source address relationship data do not "
                "provide individual address-level BTC amounts. "
                "Peeling value-reduction evidence therefore "
                "uses transaction-level output BTC reduction."
            ),
            "mixing": (
                "The source AddrTx and TxAddr edge lists provide "
                "address/transaction relationships but not "
                "individual address-level BTC allocations or "
                "equal-denomination output information."
            ),
        },
        "offline_processing": True,
        "artifacts": {
            "behavioral_evidence": str(
                OUTPUT_FILE
            ),
            "report": str(
                REPORT_FILE
            ),
        },
    }


def main() -> None:

    print("=" * 72)
    print(
        "M9.3 BEHAVIORAL EVIDENCE FUSION"
    )
    print("=" * 72)

    peeling, mixing = load_inputs()

    print(
        f"Peeling rows: "
        f"{len(peeling):,}"
    )

    print(
        f"Mixing rows: "
        f"{len(mixing):,}"
    )

    print(
        "\nValidating source artifacts..."
    )

    validate_input(
        peeling,
        "Peeling artifact",
    )

    validate_input(
        mixing,
        "Mixing artifact",
    )

    validate_required_columns(
        peeling,
        mixing,
    )

    print(
        "Source schemas: PASS"
    )

    print(
        "\nNormalizing and validating TXID alignment..."
    )

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

    validate_alignment(
        peeling,
        mixing,
    )

    print(
        f"Aligned TXIDs: "
        f"{len(peeling_ids):,}"
    )

    print(
        "TXID alignment: PASS"
    )

    print(
        "\nBuilding behavioral evidence fusion..."
    )

    result = build_behavioral_evidence(
        peeling,
        mixing,
    )

    print(
        "Evidence fusion: PASS"
    )

    print(
        "\nRunning internal validation..."
    )

    validation = validate_output(
        result,
        peeling_ids,
    )

    print(
        "Output validation: PASS"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    report = build_report(
        validation
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
        "BEHAVIORAL EVIDENCE SUMMARY"
    )

    print(
        "-" * 72
    )

    print(
        f"Transactions processed: "
        f"{validation['rows']:,}"
    )

    print(
        f"Time-step range: "
        f"{validation['time_steps']['min']}-"
        f"{validation['time_steps']['max']}"
    )

    print(
        f"Peeling candidates: "
        f"{validation['peeling_candidates']:,}"
    )

    print(
        f"Mixing candidates: "
        f"{validation['mixing_candidates']:,}"
    )

    print(
        f"Transactions with both candidate signals: "
        f"{validation['multiple_behavioral_signals']:,}"
    )

    print(
        f"Transactions with positive evidence "
        f"from both channels: "
        f"{validation['behavioral_signal_agreement']:,}"
    )

    print(
        f"Mean peeling evidence: "
        f"{validation['mean_peeling_score']:.4f}"
    )

    print(
        f"Mean mixing evidence: "
        f"{validation['mean_mixing_score']:.4f}"
    )

    print(
        f"Mean behavioral evidence: "
        f"{validation['mean_behavioral_score']:.4f}"
    )

    print(
        f"Median behavioral evidence: "
        f"{validation['median_behavioral_score']:.4f}"
    )

    print(
        f"Maximum behavioral evidence: "
        f"{validation['maximum_behavioral_score']:.4f}"
    )

    print(
        "\nEvidence levels:"
    )

    for (
        level,
        count,
    ) in validation[
        "evidence_level_distribution"
    ].items():

        print(
            f"  {level}: {count:,}"
        )

    print(
        "\n" + "=" * 72
    )

    print(
        "M9.3 BEHAVIORAL EVIDENCE FUSION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Artifact: {OUTPUT_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()