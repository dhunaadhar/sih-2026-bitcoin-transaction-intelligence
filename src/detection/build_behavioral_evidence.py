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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:

    if not PEELING_FILE.exists():
        raise FileNotFoundError(
            f"Peeling-chain artifact not found: {PEELING_FILE}"
        )

    if not MIXING_FILE.exists():
        raise FileNotFoundError(
            f"Mixing-pattern artifact not found: {MIXING_FILE}"
        )

    peeling = pd.read_parquet(
        PEELING_FILE
    )

    mixing = pd.read_parquet(
        MIXING_FILE
    )

    return peeling, mixing


def validate_input(
    frame: pd.DataFrame,
    name: str,
) -> None:

    if "txid" not in frame.columns:
        raise ValueError(
            f"{name} does not contain txid."
        )

    if frame["txid"].isna().any():
        raise ValueError(
            f"{name} contains null TXIDs."
        )

    if frame["txid"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate TXIDs."
        )


def validate_alignment(
    peeling: pd.DataFrame,
    mixing: pd.DataFrame,
) -> None:

    peeling_ids = set(
        peeling["txid"].astype(str)
    )

    mixing_ids = set(
        mixing["txid"].astype(str)
    )

    if peeling_ids != mixing_ids:

        missing_from_mixing = (
            peeling_ids - mixing_ids
        )

        missing_from_peeling = (
            mixing_ids - peeling_ids
        )

        raise ValueError(
            "Behavioral artifacts have different TXID sets. "
            f"Missing from mixing: "
            f"{len(missing_from_mixing)}; "
            f"missing from peeling: "
            f"{len(missing_from_peeling)}."
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


def build_behavioral_evidence(
    peeling: pd.DataFrame,
    mixing: pd.DataFrame,
) -> pd.DataFrame:

    peeling = peeling.copy()
    mixing = mixing.copy()

    peeling["txid"] = (
        peeling["txid"]
        .astype(str)
    )

    mixing["txid"] = (
        mixing["txid"]
        .astype(str)
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

    merged = peeling[
        peeling_columns
    ].merge(
        mixing[
            mixing_columns
        ],
        on="txid",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_peeling",
            "_mixing",
        ),
    )

    if len(merged) != len(peeling):
        raise ValueError(
            "Behavioral merge lost transactions. "
            f"Peeling rows={len(peeling):,}, "
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

    time_mismatch = (
        peeling_time
        != mixing_time
    ).sum()

    if time_mismatch:
        raise ValueError(
            f"Time-step mismatch detected for "
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

    merged[
        "peeling_evidence_score"
    ] = pd.to_numeric(
        merged[
            "peeling_evidence_score"
        ],
        errors="coerce",
    )

    merged[
        "mixing_evidence_score"
    ] = pd.to_numeric(
        merged[
            "mixing_evidence_score"
        ],
        errors="coerce",
    )

    if merged[
        "peeling_evidence_score"
    ].isna().any():

        raise ValueError(
            "Invalid peeling evidence score values."
        )

    if merged[
        "mixing_evidence_score"
    ].isna().any():

        raise ValueError(
            "Invalid mixing evidence score values."
        )

    merged[
        "behavioral_evidence_score"
    ] = (
        0.50
        * merged[
            "peeling_evidence_score"
        ]
        + 0.50
        * merged[
            "mixing_evidence_score"
        ]
    )

    merged[
        "behavioral_signal_count"
    ] = (
        merged[
            "peeling_chain_candidate"
        ].astype(int)
        + merged[
            "mixing_pattern_candidate"
        ].astype(int)
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
            merged[
                "peeling_evidence_score"
            ]
            > 0
        )
        &
        (
            merged[
                "mixing_evidence_score"
            ]
            > 0
        )
    )

    merged[
        "peeling_mixing_interaction"
    ] = (
        merged[
            "peeling_evidence_score"
        ]
        * merged[
            "mixing_evidence_score"
        ]
    )

    merged[
        "behavioral_evidence_level"
    ] = pd.cut(
        merged[
            "behavioral_evidence_score"
        ],
        bins=[
            -np.inf,
            0.20,
            0.50,
            0.75,
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
) -> dict:

    if result["txid"].duplicated().any():
        raise ValueError(
            "Duplicate TXIDs in behavioral evidence."
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

    for column in [
        "peeling_evidence_score",
        "mixing_evidence_score",
        "behavioral_evidence_score",
    ]:

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

    signal_count = result[
        "behavioral_signal_count"
    ].to_numpy(
        dtype=int
    )

    if (
        (signal_count < 0)
        | (signal_count > 2)
    ).any():

        raise ValueError(
            "Invalid behavioral signal count."
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

    return {
        "rows": int(
            len(result)
        ),
        "unique_txids": int(
            result["txid"].nunique()
        ),
        "time_steps": {
            "min": int(
                result["time_step"].min()
            ),
            "max": int(
                result["time_step"].max()
            ),
            "unique": int(
                result["time_step"].nunique()
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
        "max_behavioral_score": float(
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


def main() -> None:

    print("=" * 72)
    print("M9.3 BEHAVIORAL EVIDENCE LAYER")
    print("=" * 72)

    peeling, mixing = (
        load_inputs()
    )

    print(
        f"Peeling rows: "
        f"{len(peeling):,}"
    )

    print(
        f"Mixing rows: "
        f"{len(mixing):,}"
    )

    print(
        "\nValidating input schemas..."
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
        "Input schemas: PASS"
    )

    print(
        "\nValidating TXID alignment..."
    )

    validate_alignment(
        peeling,
        mixing,
    )

    print(
        "TXID alignment: PASS"
    )

    print(
        "\nBuilding combined behavioral evidence..."
    )

    result = build_behavioral_evidence(
        peeling,
        mixing,
    )

    validation = validate_output(
        result
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

    report = {
        "milestone": "M9.3",
        "status": "PASS",
        "purpose": (
            "Combine independently generated peeling-chain "
            "and mixing/fan-in-fan-out structural evidence "
            "into an auditable behavioral evidence layer."
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
            "peeling_weight": 0.50,
            "mixing_weight": 0.50,
            "combined_score": (
                "0.5 * peeling evidence + "
                "0.5 * mixing evidence"
            ),
            "independent_channels_preserved": True,
            "identity_claim": False,
            "illicit_activity_claim": False,
            "interpretation": (
                "Behavioral signals represent structural "
                "evidence for downstream risk analysis. "
                "They do not establish mixer usage, "
                "illicit activity, ownership, or identity."
            ),
        },
        "validation": validation,
        "artifacts": {
            "behavioral_evidence": str(
                OUTPUT_FILE
            ),
            "report": str(
                REPORT_FILE
            ),
        },
    }

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
    print("BEHAVIORAL EVIDENCE SUMMARY")
    print("-" * 72)

    print(
        f"Transactions processed: "
        f"{validation['rows']:,}"
    )

    print(
        f"Time-step range: "
        f"{validation['time_steps']['min']}–"
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
        f"Transactions with both behavioral signals: "
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
        f"Maximum behavioral evidence: "
        f"{validation['max_behavioral_score']:.4f}"
    )

    print(
        "\nEvidence levels:"
    )

    for level, count in (
        validation[
            "evidence_level_distribution"
        ].items()
    ):

        print(
            f"  {level}: {count:,}"
        )

    print("\n" + "=" * 72)
    print("M9.3 COMPLETE")
    print("=" * 72)

    print(
        f"Artifact: {OUTPUT_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()