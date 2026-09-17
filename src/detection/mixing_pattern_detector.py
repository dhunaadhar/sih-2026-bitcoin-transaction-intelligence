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

ADDR_TX_FILE = (
    ROOT
    / ".."
    / "data"
    / "external"
    / "elipticpp"
    / "actor"
    / "AddrTx_edgelist.csv"
)

TX_ADDR_FILE = (
    ROOT
    / ".."
    / "data"
    / "external"
    / "elipticpp"
    / "actor"
    / "TxAddr_edgelist.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "derived"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "mixing_pattern_indicators.parquet"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "behavior"
)

REPORT_FILE = (
    REPORT_DIR
    / "m9_2_mixing_patterns.json"
)

TXID_COLUMN = "txid"
TIME_COLUMN = "time_step"

FAN_IN_THRESHOLD = 5
FAN_OUT_THRESHOLD = 5
HIGH_PARTICIPANT_THRESHOLD = 10
EXTREME_PARTICIPANT_THRESHOLD = 25

BALANCED_RATIO_MIN = 0.40
BALANCED_RATIO_MAX = 2.50


def normalize_txid(value) -> str:
    """
    Normalize TXIDs so numerically equivalent representations such as:

        4304541
        4304541.0
        "4304541"
        "4304541.0"

    become the same canonical string.
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

        return format(numeric, ".15g")

    except (ValueError, TypeError):
        return text


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if not CANONICAL_FILE.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {CANONICAL_FILE}"
        )

    if not ADDR_TX_FILE.exists():
        raise FileNotFoundError(
            f"AddrTx edge list not found: {ADDR_TX_FILE}"
        )

    if not TX_ADDR_FILE.exists():
        raise FileNotFoundError(
            f"TxAddr edge list not found: {TX_ADDR_FILE}"
        )

    transactions = pd.read_parquet(
        CANONICAL_FILE
    )

    addr_tx = pd.read_csv(
        ADDR_TX_FILE
    )

    tx_addr = pd.read_csv(
        TX_ADDR_FILE
    )

    return (
        transactions,
        addr_tx,
        tx_addr,
    )


def validate_source_data(
    transactions: pd.DataFrame,
    addr_tx: pd.DataFrame,
    tx_addr: pd.DataFrame,
) -> None:

    required_transaction_columns = {
        TXID_COLUMN,
        TIME_COLUMN,
        "input_address_count",
        "output_address_count",
        "input_btc_total",
        "output_btc_total",
        "total_btc",
    }

    missing = (
        required_transaction_columns
        - set(transactions.columns)
    )

    if missing:
        raise ValueError(
            "Canonical dataset missing required columns: "
            f"{sorted(missing)}"
        )

    required_addr_tx = {
        "input_address",
        "txId",
    }

    if required_addr_tx - set(addr_tx.columns):
        raise ValueError(
            "AddrTx edge list must contain: "
            f"{sorted(required_addr_tx)}"
        )

    required_tx_addr = {
        "txId",
        "output_address",
    }

    if required_tx_addr - set(tx_addr.columns):
        raise ValueError(
            "TxAddr edge list must contain: "
            f"{sorted(required_tx_addr)}"
        )

    if transactions[TXID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate TXIDs found in canonical dataset."
        )

    if transactions[TXID_COLUMN].isna().any():
        raise ValueError(
            "Null TXIDs found in canonical dataset."
        )


def normalize_source_txids(
    transactions: pd.DataFrame,
    addr_tx: pd.DataFrame,
    tx_addr: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    transactions = transactions.copy()
    addr_tx = addr_tx.copy()
    tx_addr = tx_addr.copy()

    transactions[TXID_COLUMN] = (
        transactions[TXID_COLUMN]
        .map(normalize_txid)
    )

    addr_tx["txId"] = (
        addr_tx["txId"]
        .map(normalize_txid)
    )

    tx_addr["txId"] = (
        tx_addr["txId"]
        .map(normalize_txid)
    )

    return (
        transactions,
        addr_tx,
        tx_addr,
    )


def validate_txid_alignment(
    transactions: pd.DataFrame,
    addr_tx: pd.DataFrame,
    tx_addr: pd.DataFrame,
) -> dict:
    canonical_ids = set(
        transactions[
            TXID_COLUMN
        ]
    )

    input_ids = set(
        addr_tx[
            "txId"
        ]
    )

    output_ids = set(
        tx_addr[
            "txId"
        ]
    )

    input_overlap = (
        canonical_ids
        & input_ids
    )

    output_overlap = (
        canonical_ids
        & output_ids
    )

    both_overlap = (
        input_overlap
        & output_overlap
    )

    return {
        "canonical_unique_txids": len(
            canonical_ids
        ),
        "addr_tx_unique_txids": len(
            input_ids
        ),
        "tx_addr_unique_txids": len(
            output_ids
        ),
        "canonical_input_overlap": len(
            input_overlap
        ),
        "canonical_output_overlap": len(
            output_overlap
        ),
        "canonical_both_input_output_overlap": len(
            both_overlap
        ),
    }


def build_edge_counts(
    addr_tx: pd.DataFrame,
    tx_addr: pd.DataFrame,
) -> tuple[
    dict[str, int],
    dict[str, int],
]:

    input_edges = (
        addr_tx[
            [
                "input_address",
                "txId",
            ]
        ]
        .drop_duplicates()
    )

    output_edges = (
        tx_addr[
            [
                "txId",
                "output_address",
            ]
        ]
        .drop_duplicates()
    )

    input_counts = (
        input_edges
        .groupby("txId")[
            "input_address"
        ]
        .nunique()
        .astype(int)
        .to_dict()
    )

    output_counts = (
        output_edges
        .groupby("txId")[
            "output_address"
        ]
        .nunique()
        .astype(int)
        .to_dict()
    )

    return (
        input_counts,
        output_counts,
    )


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:

    if denominator <= 0:
        return 0.0

    return float(
        numerator / denominator
    )


def calculate_mixing_evidence(
    input_count: int,
    output_count: int,
) -> dict:

    total_participants = (
        input_count
        + output_count
    )

    fan_in = (
        input_count
        >= FAN_IN_THRESHOLD
    )

    fan_out = (
        output_count
        >= FAN_OUT_THRESHOLD
    )

    high_participant = (
        total_participants
        >= HIGH_PARTICIPANT_THRESHOLD
    )

    extreme_participant = (
        total_participants
        >= EXTREME_PARTICIPANT_THRESHOLD
    )

    if input_count > 0:
        input_output_ratio = (
            output_count
            / input_count
        )
    else:
        input_output_ratio = 0.0

    balanced = (
        input_count > 0
        and output_count > 0
        and BALANCED_RATIO_MIN
        <= input_output_ratio
        <= BALANCED_RATIO_MAX
    )

    evidence_score = 0.0

    if fan_in:
        evidence_score += 0.20

    if fan_out:
        evidence_score += 0.20

    if balanced:
        evidence_score += 0.20

    if high_participant:
        evidence_score += 0.15

    if extreme_participant:
        evidence_score += 0.15

    if fan_in and fan_out:
        evidence_score += 0.10

    evidence_score = min(
        evidence_score,
        1.0,
    )

    pure_fan_in = (
        fan_in
        and not fan_out
    )

    pure_fan_out = (
        fan_out
        and not fan_in
    )

    balanced_fan_pattern = (
        fan_in
        and fan_out
        and balanced
    )

    mixing_pattern_candidate = (
        balanced_fan_pattern
        and high_participant
    )

    return {
        "fan_in": fan_in,
        "fan_out": fan_out,
        "pure_fan_in": pure_fan_in,
        "pure_fan_out": pure_fan_out,
        "balanced_fan_pattern": (
            balanced_fan_pattern
        ),
        "high_participant_count": (
            high_participant
        ),
        "extreme_participant_count": (
            extreme_participant
        ),
        "input_output_ratio": float(
            input_output_ratio
        ),
        "mixing_pattern_candidate": (
            mixing_pattern_candidate
        ),
        "mixing_evidence_score": (
            evidence_score
        ),
    }


def build_indicators(
    transactions: pd.DataFrame,
    input_counts: dict[str, int],
    output_counts: dict[str, int],
) -> pd.DataFrame:

    rows = []

    transaction_columns = [
        TXID_COLUMN,
        TIME_COLUMN,
        "input_address_count",
        "output_address_count",
        "input_btc_total",
        "output_btc_total",
        "total_btc",
    ]

    for row in transactions[
        transaction_columns
    ].itertuples(index=False):

        txid = normalize_txid(
            row.txid
        )

        source_input_count = int(
            input_counts.get(
                txid,
                0,
            )
        )

        source_output_count = int(
            output_counts.get(
                txid,
                0,
            )
        )

        input_count = (
            source_input_count
        )

        output_count = (
            source_output_count
        )

        evidence = calculate_mixing_evidence(
            input_count,
            output_count,
        )

        total_participants = (
            input_count
            + output_count
        )

        participant_density = safe_ratio(
            total_participants,
            max(
                input_count,
                output_count,
                1,
            ),
        )

        rows.append(
            {
                "txid": txid,
                "time_step": int(
                    row.time_step
                ),
                "input_address_count": (
                    input_count
                ),
                "output_address_count": (
                    output_count
                ),
                "total_participant_count": (
                    total_participants
                ),
                "participant_density": (
                    participant_density
                ),
                "input_btc_total": (
                    float(row.input_btc_total)
                    if pd.notna(
                        row.input_btc_total
                    )
                    else np.nan
                ),
                "output_btc_total": (
                    float(row.output_btc_total)
                    if pd.notna(
                        row.output_btc_total
                    )
                    else np.nan
                ),
                "total_btc": (
                    float(row.total_btc)
                    if pd.notna(
                        row.total_btc
                    )
                    else np.nan
                ),
                **evidence,
            }
        )

    return pd.DataFrame(
        rows
    )


def validate_output(
    result: pd.DataFrame,
    transactions: pd.DataFrame,
) -> dict:

    if len(result) != len(
        transactions
    ):
        raise ValueError(
            "Output row count does not match "
            "canonical transaction count."
        )

    if result["txid"].duplicated().any():
        raise ValueError(
            "Duplicate TXIDs in mixing indicators."
        )

    canonical_ids = set(
        transactions[
            TXID_COLUMN
        ].map(normalize_txid)
    )

    result_ids = set(
        result["txid"]
        .map(normalize_txid)
    )

    if canonical_ids != result_ids:
        raise ValueError(
            "Mixing indicators do not contain "
            "exactly the canonical TXID set."
        )

    score = result[
        "mixing_evidence_score"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        score
    ).all():
        raise ValueError(
            "Non-finite mixing evidence scores."
        )

    if (
        (score < 0)
        | (score > 1)
    ).any():
        raise ValueError(
            "Mixing evidence scores outside [0,1]."
        )

    ratio = result[
        "input_output_ratio"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        ratio
    ).all():
        raise ValueError(
            "Non-finite input/output ratios."
        )

    return {
        "rows": int(
            len(result)
        ),
        "unique_txids": int(
            result["txid"].nunique()
        ),
        "fan_in_count": int(
            result["fan_in"].sum()
        ),
        "fan_out_count": int(
            result["fan_out"].sum()
        ),
        "balanced_fan_pattern_count": int(
            result[
                "balanced_fan_pattern"
            ].sum()
        ),
        "high_participant_count": int(
            result[
                "high_participant_count"
            ].sum()
        ),
        "extreme_participant_count": int(
            result[
                "extreme_participant_count"
            ].sum()
        ),
        "mixing_candidate_count": int(
            result[
                "mixing_pattern_candidate"
            ].sum()
        ),
        "mixing_candidate_rate": float(
            result[
                "mixing_pattern_candidate"
            ].mean()
        ),
        "mean_evidence_score": float(
            result[
                "mixing_evidence_score"
            ].mean()
        ),
        "max_evidence_score": float(
            result[
                "mixing_evidence_score"
            ].max()
        ),
    }


def main() -> None:

    print("=" * 72)
    print("M9.2 MIXING / FAN-IN-FAN-OUT DETECTOR")
    print("=" * 72)

    transactions, addr_tx, tx_addr = (
        load_data()
    )

    print(
        f"Canonical transactions: "
        f"{len(transactions):,}"
    )

    print(
        f"AddrTx rows: "
        f"{len(addr_tx):,}"
    )

    print(
        f"TxAddr rows: "
        f"{len(tx_addr):,}"
    )

    validate_source_data(
        transactions,
        addr_tx,
        tx_addr,
    )

    print(
        "\nNormalizing TXID representations..."
    )

    (
        transactions,
        addr_tx,
        tx_addr,
    ) = normalize_source_txids(
        transactions,
        addr_tx,
        tx_addr,
    )

    alignment = validate_txid_alignment(
        transactions,
        addr_tx,
        tx_addr,
    )

    print(
        f"Canonical unique TXIDs: "
        f"{alignment['canonical_unique_txids']:,}"
    )

    print(
        f"AddrTx unique TXIDs: "
        f"{alignment['addr_tx_unique_txids']:,}"
    )

    print(
        f"TxAddr unique TXIDs: "
        f"{alignment['tx_addr_unique_txids']:,}"
    )

    print(
        f"Canonical ↔ AddrTx overlap: "
        f"{alignment['canonical_input_overlap']:,}"
    )

    print(
        f"Canonical ↔ TxAddr overlap: "
        f"{alignment['canonical_output_overlap']:,}"
    )

    print(
        f"Canonical TXIDs with both sides: "
        f"{alignment['canonical_both_input_output_overlap']:,}"
    )

    if (
        alignment["canonical_both_input_output_overlap"]
        == 0
    ):
        raise ValueError(
            "TXID alignment failed: zero canonical "
            "transactions have both input and output "
            "source relationships."
        )

    print(
        "\nBuilding transaction participant counts..."
    )

    input_counts, output_counts = (
        build_edge_counts(
            addr_tx,
            tx_addr,
        )
    )

    print(
        f"Transactions with source input "
        f"relationships: {len(input_counts):,}"
    )

    print(
        f"Transactions with source output "
        f"relationships: {len(output_counts):,}"
    )

    print(
        "\nBuilding structural indicators..."
    )

    result = build_indicators(
        transactions,
        input_counts,
        output_counts,
    )

    validation = validate_output(
        result,
        transactions,
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
        "milestone": "M9.2",
        "status": "PASS",
        "detector": (
            "mixing_fan_in_fan_out"
        ),
        "thresholds": {
            "fan_in_threshold": (
                FAN_IN_THRESHOLD
            ),
            "fan_out_threshold": (
                FAN_OUT_THRESHOLD
            ),
            "high_participant_threshold": (
                HIGH_PARTICIPANT_THRESHOLD
            ),
            "extreme_participant_threshold": (
                EXTREME_PARTICIPANT_THRESHOLD
            ),
            "balanced_ratio_min": (
                BALANCED_RATIO_MIN
            ),
            "balanced_ratio_max": (
                BALANCED_RATIO_MAX
            ),
        },
        "txid_alignment": alignment,
        "methodology": {
            "participant_counts": (
                "Unique transaction input/output "
                "addresses from AddrTx and TxAddr "
                "edge lists."
            ),
            "txid_normalization": (
                "Numeric TXID representations are "
                "normalized so integer and floating-point "
                "representations of the same TXID align."
            ),
            "duplicate_handling": (
                "Duplicate source relationships are "
                "deduplicated before counting."
            ),
            "amount_limitation": (
                "The source edge lists do not provide "
                "individual address-level BTC amounts. "
                "Therefore the detector does not infer "
                "address-level value allocation."
            ),
        },
        "validation": validation,
        "artifacts": {
            "indicators": str(
                OUTPUT_FILE
            ),
            "report": str(
                REPORT_FILE
            ),
        },
        "interpretation_note": (
            "Fan-in/fan-out structure can be compatible "
            "with mixing-like transaction behavior, but "
            "structure alone does not establish the use "
            "of a mixer or illicit activity."
        ),
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
    print("MIXING / STRUCTURAL SUMMARY")
    print("-" * 72)

    print(
        f"Transactions processed: "
        f"{validation['rows']:,}"
    )

    print(
        f"Fan-in transactions: "
        f"{validation['fan_in_count']:,}"
    )

    print(
        f"Fan-out transactions: "
        f"{validation['fan_out_count']:,}"
    )

    print(
        f"Balanced fan-in/fan-out: "
        f"{validation['balanced_fan_pattern_count']:,}"
    )

    print(
        f"High-participant transactions: "
        f"{validation['high_participant_count']:,}"
    )

    print(
        f"Extreme-participant transactions: "
        f"{validation['extreme_participant_count']:,}"
    )

    print(
        f"Mixing-pattern candidates: "
        f"{validation['mixing_candidate_count']:,}"
    )

    print(
        f"Candidate rate: "
        f"{validation['mixing_candidate_rate']:.4%}"
    )

    print(
        f"Mean evidence score: "
        f"{validation['mean_evidence_score']:.4f}"
    )

    print(
        f"Maximum evidence score: "
        f"{validation['max_evidence_score']:.4f}"
    )

    print("\n" + "=" * 72)
    print("M9.2 COMPLETE")
    print("=" * 72)

    print(
        f"Indicators: {OUTPUT_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()