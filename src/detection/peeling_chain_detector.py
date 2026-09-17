from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set

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
    / "peeling_chain_indicators.parquet"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "behavior"
)

REPORT_FILE = (
    REPORT_DIR
    / "m9_1_peeling_chain.json"
)

TXID_COLUMN = "txid"
TIME_COLUMN = "time_step"

MIN_CHAIN_LENGTH = 3
MAX_HISTORY_STEPS = 10

# A subsequent transaction is considered a candidate continuation
# when its total output value is meaningfully lower than the prior
# transaction's total output value.
MIN_VALUE_REDUCTION_RATIO = 0.05

# Avoid treating extremely large drops as the strongest peeling
# evidence because the detector is intended to capture gradual
# continuation rather than arbitrary value changes.
MAX_VALUE_REDUCTION_RATIO = 0.95


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    return transactions, addr_tx, tx_addr


def validate_source_data(
    transactions: pd.DataFrame,
    addr_tx: pd.DataFrame,
    tx_addr: pd.DataFrame,
) -> None:
    required_transaction_columns = {
        TXID_COLUMN,
        TIME_COLUMN,
        "output_btc_total",
        "input_btc_total",
        "output_address_count",
        "input_address_count",
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

    if set(
        ["input_address", "txId"]
    ) - set(addr_tx.columns):
        raise ValueError(
            "AddrTx edge list must contain "
            "input_address and txId."
        )

    if set(
        ["txId", "output_address"]
    ) - set(tx_addr.columns):
        raise ValueError(
            "TxAddr edge list must contain "
            "txId and output_address."
        )

    if transactions[TXID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs in canonical dataset."
        )

    if transactions[TXID_COLUMN].isna().any():
        raise ValueError(
            "Null transaction IDs in canonical dataset."
        )


def build_transaction_address_maps(
    addr_tx: pd.DataFrame,
    tx_addr: pd.DataFrame,
) -> tuple[
    Dict[str, Set[str]],
    Dict[str, Set[str]],
]:
    """
    Build transaction -> input addresses and
    transaction -> output addresses.

    Only identifiers are used here. No address-level BTC amount
    is inferred because the source edge lists do not contain it.
    """

    tx_inputs: Dict[str, Set[str]] = defaultdict(set)
    tx_outputs: Dict[str, Set[str]] = defaultdict(set)

    for row in addr_tx[
        ["input_address", "txId"]
    ].itertuples(index=False):
        address = str(row.input_address)
        txid = str(row.txId)

        if address and txid:
            tx_inputs[txid].add(address)

    for row in tx_addr[
        ["txId", "output_address"]
    ].itertuples(index=False):
        txid = str(row.txId)
        address = str(row.output_address)

        if txid and address:
            tx_outputs[txid].add(address)

    return tx_inputs, tx_outputs


def build_address_history(
    transactions: pd.DataFrame,
    tx_inputs: Dict[str, Set[str]],
    tx_outputs: Dict[str, Set[str]],
) -> Dict[str, List[tuple[int, str, str]]]:
    """
    Build historical address activity.

    Each entry is:
        (time_step, txid, role)

    role is either:
        input
        output
    """

    history: Dict[
        str,
        List[tuple[int, str, str]]
    ] = defaultdict(list)

    tx_time = {
        str(row.txid): int(row.time_step)
        for row in transactions[
            [TXID_COLUMN, TIME_COLUMN]
        ].itertuples(index=False)
    }

    for txid, addresses in tx_inputs.items():
        if txid not in tx_time:
            continue

        time_step = tx_time[txid]

        for address in addresses:
            history[address].append(
                (
                    time_step,
                    txid,
                    "input",
                )
            )

    for txid, addresses in tx_outputs.items():
        if txid not in tx_time:
            continue

        time_step = tx_time[txid]

        for address in addresses:
            history[address].append(
                (
                    time_step,
                    txid,
                    "output",
                )
            )

    for address in history:
        history[address].sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

    return history


def build_transaction_lookup(
    transactions: pd.DataFrame,
) -> Dict[str, dict]:
    lookup = {}

    required = [
        TXID_COLUMN,
        TIME_COLUMN,
        "output_btc_total",
        "input_btc_total",
        "output_address_count",
        "input_address_count",
    ]

    for row in transactions[
        required
    ].itertuples(index=False):
        lookup[str(row.txid)] = {
            "time_step": int(
                row.time_step
            ),
            "output_btc_total": (
                float(row.output_btc_total)
                if pd.notna(
                    row.output_btc_total
                )
                else np.nan
            ),
            "input_btc_total": (
                float(row.input_btc_total)
                if pd.notna(
                    row.input_btc_total
                )
                else np.nan
            ),
            "output_address_count": int(
                row.output_address_count
            )
            if pd.notna(
                row.output_address_count
            )
            else 0,
            "input_address_count": int(
                row.input_address_count
            )
            if pd.notna(
                row.input_address_count
            )
            else 0,
        }

    return lookup


def find_previous_activity(
    address: str,
    current_time: int,
    history: Dict[str, List[tuple[int, str, str]]],
    max_history_steps: int,
) -> List[tuple[int, str, str]]:
    previous = []

    minimum_time = (
        current_time
        - max_history_steps
    )

    for (
        time_step,
        txid,
        role,
    ) in history.get(
        address,
        [],
    ):
        if (
            minimum_time
            <= time_step
            < current_time
        ):
            previous.append(
                (
                    time_step,
                    txid,
                    role,
                )
            )

    return previous


def calculate_value_reduction(
    previous_value: float,
    current_value: float,
) -> float | None:
    if (
        not np.isfinite(previous_value)
        or not np.isfinite(current_value)
        or previous_value <= 0
    ):
        return None

    reduction = (
        previous_value
        - current_value
    ) / previous_value

    return float(reduction)


def detect_transaction(
    txid: str,
    current_time: int,
    tx_inputs: Dict[str, Set[str]],
    tx_outputs: Dict[str, Set[str]],
    history: Dict[str, List[tuple[int, str, str]]],
    tx_lookup: Dict[str, dict],
) -> dict:
    current = tx_lookup[txid]

    current_inputs = tx_inputs.get(
        txid,
        set(),
    )

    current_outputs = tx_outputs.get(
        txid,
        set(),
    )

    continuation_addresses = set()
    previous_transactions = set()

    best_reduction = None
    best_previous_tx = None
    best_previous_time = None

    for address in current_inputs:
        previous_activity = find_previous_activity(
            address,
            current_time,
            history,
            MAX_HISTORY_STEPS,
        )

        for (
            previous_time,
            previous_txid,
            previous_role,
        ) in previous_activity:

            if previous_role != "output":
                continue

            continuation_addresses.add(
                address
            )

            previous_transactions.add(
                previous_txid
            )

            if previous_txid not in tx_lookup:
                continue

            previous = tx_lookup[
                previous_txid
            ]

            reduction = calculate_value_reduction(
                previous[
                    "output_btc_total"
                ],
                current[
                    "output_btc_total"
                ],
            )

            if reduction is None:
                continue

            if not (
                MIN_VALUE_REDUCTION_RATIO
                <= reduction
                <= MAX_VALUE_REDUCTION_RATIO
            ):
                continue

            if (
                best_reduction is None
                or reduction > best_reduction
            ):
                best_reduction = reduction
                best_previous_tx = (
                    previous_txid
                )
                best_previous_time = (
                    previous_time
                )

    temporal_gap = None

    if best_previous_time is not None:
        temporal_gap = (
            current_time
            - best_previous_time
        )

    address_continuation_count = len(
        continuation_addresses
    )

    previous_transaction_count = len(
        previous_transactions
    )

    direct_continuation = (
        address_continuation_count > 0
    )

    gradual_reduction = (
        best_reduction is not None
    )

    chain_candidate = (
        direct_continuation
        and gradual_reduction
        and temporal_gap is not None
        and temporal_gap <= MAX_HISTORY_STEPS
    )

    evidence_score = 0.0

    if direct_continuation:
        evidence_score += 0.35

    if gradual_reduction:
        evidence_score += 0.35

    if (
        temporal_gap is not None
        and temporal_gap == 1
    ):
        evidence_score += 0.20

    if (
        address_continuation_count >= 2
    ):
        evidence_score += 0.10

    evidence_score = min(
        evidence_score,
        1.0,
    )

    return {
        "txid": txid,
        "time_step": current_time,
        "peeling_direct_continuation": (
            direct_continuation
        ),
        "peeling_gradual_value_reduction": (
            gradual_reduction
        ),
        "peeling_chain_candidate": (
            chain_candidate
        ),
        "peeling_evidence_score": (
            evidence_score
        ),
        "continuation_address_count": (
            address_continuation_count
        ),
        "previous_transaction_count": (
            previous_transaction_count
        ),
        "best_previous_txid": (
            best_previous_tx
        ),
        "best_previous_time_step": (
            best_previous_time
        ),
        "temporal_gap": (
            temporal_gap
        ),
        "best_value_reduction_ratio": (
            best_reduction
        ),
        "current_output_btc_total": (
            current["output_btc_total"]
        ),
        "current_input_btc_total": (
            current["input_btc_total"]
        ),
        "current_output_address_count": (
            current[
                "output_address_count"
            ]
        ),
        "current_input_address_count": (
            current[
                "input_address_count"
            ]
        ),
    }


def build_behavioral_features(
    transactions: pd.DataFrame,
    tx_inputs: Dict[str, Set[str]],
    tx_outputs: Dict[str, Set[str]],
) -> pd.DataFrame:
    history = build_address_history(
        transactions,
        tx_inputs,
        tx_outputs,
    )

    tx_lookup = build_transaction_lookup(
        transactions
    )

    # Important causal rule:
    # history contains the complete graph for lookup, but for
    # each transaction only entries with time_step < current_time
    # are allowed by find_previous_activity().
    rows = []

    transactions_sorted = (
        transactions.sort_values(
            [TIME_COLUMN, TXID_COLUMN]
        )
    )

    for row in transactions_sorted[
        [TXID_COLUMN, TIME_COLUMN]
    ].itertuples(index=False):

        txid = str(row.txid)
        current_time = int(
            row.time_step
        )

        rows.append(
            detect_transaction(
                txid,
                current_time,
                tx_inputs,
                tx_outputs,
                history,
                tx_lookup,
            )
        )

    return pd.DataFrame(rows)


def validate_output(
    result: pd.DataFrame,
    transactions: pd.DataFrame,
) -> dict:
    if len(result) != len(
        transactions
    ):
        raise ValueError(
            "Output row count mismatch."
        )

    if result["txid"].duplicated().any():
        raise ValueError(
            "Duplicate TXIDs in behavioral output."
        )

    canonical_ids = set(
        transactions[
            TXID_COLUMN
        ].astype(str)
    )

    result_ids = set(
        result["txid"].astype(str)
    )

    if canonical_ids != result_ids:
        raise ValueError(
            "Behavioral output TXIDs do not "
            "match canonical dataset."
        )

    score_values = result[
        "peeling_evidence_score"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        score_values
    ).all():
        raise ValueError(
            "Non-finite peeling evidence scores."
        )

    if (
        (score_values < 0)
        | (score_values > 1)
    ).any():
        raise ValueError(
            "Peeling evidence scores outside [0,1]."
        )

    return {
        "rows": int(len(result)),
        "unique_txids": int(
            result["txid"].nunique()
        ),
        "candidate_count": int(
            result[
                "peeling_chain_candidate"
            ].sum()
        ),
        "candidate_rate": float(
            result[
                "peeling_chain_candidate"
            ].mean()
        ),
        "direct_continuation_count": int(
            result[
                "peeling_direct_continuation"
            ].sum()
        ),
        "gradual_reduction_count": int(
            result[
                "peeling_gradual_value_reduction"
            ].sum()
        ),
        "max_evidence_score": float(
            result[
                "peeling_evidence_score"
            ].max()
        ),
        "mean_evidence_score": float(
            result[
                "peeling_evidence_score"
            ].mean()
        ),
    }


def main() -> None:
    print("=" * 72)
    print("M9.1 PEELING-CHAIN BEHAVIORAL DETECTOR")
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

    tx_inputs, tx_outputs = (
        build_transaction_address_maps(
            addr_tx,
            tx_addr,
        )
    )

    print(
        f"Transactions with input addresses: "
        f"{len(tx_inputs):,}"
    )

    print(
        f"Transactions with output addresses: "
        f"{len(tx_outputs):,}"
    )

    print(
        "\nBuilding causal behavioral indicators..."
    )

    result = build_behavioral_features(
        transactions,
        tx_inputs,
        tx_outputs,
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
        "milestone": "M9.1",
        "status": "PASS",
        "detector": "peeling_chain",
        "methodology": {
            "min_chain_length": (
                MIN_CHAIN_LENGTH
            ),
            "max_history_steps": (
                MAX_HISTORY_STEPS
            ),
            "min_value_reduction_ratio": (
                MIN_VALUE_REDUCTION_RATIO
            ),
            "max_value_reduction_ratio": (
                MAX_VALUE_REDUCTION_RATIO
            ),
            "causal_rule": (
                "Only address activity from time_step "
                "< current transaction time_step is used."
            ),
            "address_amount_limitation": (
                "Source address edge lists contain "
                "identifiers but no individual address-level "
                "BTC amounts. Value reduction therefore uses "
                "transaction-level output BTC totals."
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
            "A peeling-chain candidate is behavioral evidence "
            "of a possible sequential value-transfer pattern. "
            "It is not proof of mixing, illicit activity, "
            "ownership, identity, or guilt."
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
    print("PEELING-CHAIN SUMMARY")
    print("-" * 72)

    print(
        f"Transactions processed: "
        f"{validation['rows']:,}"
    )

    print(
        f"Direct continuations: "
        f"{validation['direct_continuation_count']:,}"
    )

    print(
        f"Gradual value reductions: "
        f"{validation['gradual_reduction_count']:,}"
    )

    print(
        f"Chain candidates: "
        f"{validation['candidate_count']:,}"
    )

    print(
        f"Candidate rate: "
        f"{validation['candidate_rate']:.4%}"
    )

    print(
        f"Maximum evidence score: "
        f"{validation['max_evidence_score']:.4f}"
    )

    print(
        f"Mean evidence score: "
        f"{validation['mean_evidence_score']:.4f}"
    )

    print("\n" + "=" * 72)
    print("M9.1 COMPLETE")
    print("=" * 72)

    print(
        f"Indicators: {OUTPUT_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()