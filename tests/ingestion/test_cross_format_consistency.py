from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

# A valid peeling chain must contain at least this many
# sequentially connected transactions.
MIN_CHAIN_LENGTH = 3

# Only historical activity within this number of time steps
# is considered for causal reconstruction.
MAX_HISTORY_STEPS = 10

# A valid value-reduction edge must reduce the previous
# transaction's output value by at least this amount.
MIN_VALUE_REDUCTION_RATIO = 0.05

# Extremely large reductions are excluded because this detector
# is intended to capture gradual continuation rather than
# arbitrary value disappearance.
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

    if transactions[
        TXID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs in canonical dataset."
        )

    if transactions[
        TXID_COLUMN
    ].isna().any():
        raise ValueError(
            "Null transaction IDs in canonical dataset."
        )

    if transactions[
        TIME_COLUMN
    ].isna().any():
        raise ValueError(
            "Null time steps in canonical dataset."
        )

    if (
        transactions[
            TIME_COLUMN
        ]
        .astype(float)
        .lt(0)
        .any()
    ):
        raise ValueError(
            "Negative time steps in canonical dataset."
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

    Only identifiers are used here.

    No address-level BTC amount is inferred because the
    source edge lists do not contain address-level values.
    """

    tx_inputs: Dict[
        str,
        Set[str]
    ] = defaultdict(set)

    tx_outputs: Dict[
        str,
        Set[str]
    ] = defaultdict(set)

    for row in addr_tx[
        ["input_address", "txId"]
    ].itertuples(index=False):

        address = str(
            row.input_address
        )

        txid = str(
            row.txId
        )

        if (
            address
            and address.lower() != "nan"
            and txid
            and txid.lower() != "nan"
        ):
            tx_inputs[
                txid
            ].add(
                address
            )

    for row in tx_addr[
        ["txId", "output_address"]
    ].itertuples(index=False):

        txid = str(
            row.txId
        )

        address = str(
            row.output_address
        )

        if (
            txid
            and txid.lower() != "nan"
            and address
            and address.lower() != "nan"
        ):
            tx_outputs[
                txid
            ].add(
                address
            )

    return (
        tx_inputs,
        tx_outputs,
    )


def build_address_history(
    transactions: pd.DataFrame,
    tx_inputs: Dict[str, Set[str]],
    tx_outputs: Dict[str, Set[str]],
) -> Dict[
    str,
    List[Tuple[int, str, str]]
]:
    """
    Build address activity history.

    Each entry is:

        (time_step, txid, role)

    role is either:

        input
        output

    The history is used only to discover causal
    predecessor relationships. Every predecessor is
    later required to have time_step < current_time.
    """

    history: Dict[
        str,
        List[Tuple[int, str, str]]
    ] = defaultdict(list)

    tx_time = {
        str(row.txid): int(
            row.time_step
        )
        for row in transactions[
            [TXID_COLUMN, TIME_COLUMN]
        ].itertuples(index=False)
    }

    for txid, addresses in tx_inputs.items():

        if txid not in tx_time:
            continue

        time_step = tx_time[
            txid
        ]

        for address in addresses:
            history[
                address
            ].append(
                (
                    time_step,
                    txid,
                    "input",
                )
            )

    for txid, addresses in tx_outputs.items():

        if txid not in tx_time:
            continue

        time_step = tx_time[
            txid
        ]

        for address in addresses:
            history[
                address
            ].append(
                (
                    time_step,
                    txid,
                    "output",
                )
            )

    for address in history:
        history[
            address
        ].sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )

    return history


def build_transaction_lookup(
    transactions: pd.DataFrame,
) -> Dict[str, dict]:

    lookup: Dict[
        str,
        dict
    ] = {}

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

        txid = str(
            row.txid
        )

        lookup[
            txid
        ] = {
            "time_step": int(
                row.time_step
            ),
            "output_btc_total": (
                float(
                    row.output_btc_total
                )
                if pd.notna(
                    row.output_btc_total
                )
                else np.nan
            ),
            "input_btc_total": (
                float(
                    row.input_btc_total
                )
                if pd.notna(
                    row.input_btc_total
                )
                else np.nan
            ),
            "output_address_count": (
                int(
                    row.output_address_count
                )
                if pd.notna(
                    row.output_address_count
                )
                else 0
            ),
            "input_address_count": (
                int(
                    row.input_address_count
                )
                if pd.notna(
                    row.input_address_count
                )
                else 0
            ),
        }

    return lookup


def find_previous_activity(
    address: str,
    current_time: int,
    history: Dict[
        str,
        List[Tuple[int, str, str]]
    ],
    max_history_steps: int,
) -> List[
    Tuple[int, str, str]
]:

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

        # Strict causal restriction:
        # same-time-step and future activity is excluded.
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
        not np.isfinite(
            previous_value
        )
        or not np.isfinite(
            current_value
        )
        or previous_value <= 0
    ):
        return None

    reduction = (
        previous_value
        - current_value
    ) / previous_value

    return float(
        reduction
    )


def build_causal_predecessors(
    txid: str,
    current_time: int,
    tx_inputs: Dict[
        str,
        Set[str]
    ],
    history: Dict[
        str,
        List[Tuple[int, str, str]]
    ],
    tx_lookup: Dict[str, dict],
) -> Dict[
    str,
    Set[str]
]:
    """
    Find causal predecessor transactions for a transaction.

    A predecessor must satisfy all of the following:

    1. Its output address is used as an input address
       by the current transaction.
    2. Its time_step is strictly smaller than the current
       transaction's time_step.
    3. The time difference is within MAX_HISTORY_STEPS.
    4. Its output BTC total is positive and finite.
    5. The current output BTC total is finite.
    6. The value reduction is within the configured range.

    The returned mapping is:

        predecessor_txid -> reused_addresses
    """

    current = tx_lookup[
        txid
    ]

    current_outputs_value = (
        current[
            "output_btc_total"
        ]
    )

    predecessors: Dict[
        str,
        Set[str]
    ] = defaultdict(set)

    for address in tx_inputs.get(
        txid,
        set(),
    ):

        previous_activity = (
            find_previous_activity(
                address=address,
                current_time=current_time,
                history=history,
                max_history_steps=MAX_HISTORY_STEPS,
            )
        )

        for (
            previous_time,
            previous_txid,
            previous_role,
        ) in previous_activity:

            if previous_role != "output":
                continue

            if previous_txid == txid:
                continue

            if previous_txid not in tx_lookup:
                continue

            previous = tx_lookup[
                previous_txid
            ]

            # Defensive causal check.
            if previous[
                "time_step"
            ] >= current_time:
                continue

            reduction = (
                calculate_value_reduction(
                    previous[
                        "output_btc_total"
                    ],
                    current_outputs_value,
                )
            )

            if reduction is None:
                continue

            if not (
                MIN_VALUE_REDUCTION_RATIO
                <= reduction
                <= MAX_VALUE_REDUCTION_RATIO
            ):
                continue

            predecessors[
                previous_txid
            ].add(
                address
            )

    return predecessors


def build_all_causal_predecessors(
    transactions: pd.DataFrame,
    tx_inputs: Dict[
        str,
        Set[str]
    ],
    history: Dict[
        str,
        List[Tuple[int, str, str]]
    ],
    tx_lookup: Dict[str, dict],
) -> Dict[
    str,
    Dict[str, Set[str]]
]:
    """
    Build the causal predecessor graph.

    This is constructed once and reused during chain
    reconstruction to avoid repeatedly scanning the
    address history.
    """

    predecessor_map: Dict[
        str,
        Dict[str, Set[str]]
    ] = {}

    transactions_sorted = (
        transactions.sort_values(
            [
                TIME_COLUMN,
                TXID_COLUMN,
            ]
        )
    )

    for row in transactions_sorted[
        [TXID_COLUMN, TIME_COLUMN]
    ].itertuples(index=False):

        txid = str(
            row.txid
        )

        current_time = int(
            row.time_step
        )

        predecessor_map[
            txid
        ] = build_causal_predecessors(
            txid=txid,
            current_time=current_time,
            tx_inputs=tx_inputs,
            history=history,
            tx_lookup=tx_lookup,
        )

    return predecessor_map


def reconstruct_longest_chain(
    txid: str,
    predecessor_map: Dict[
        str,
        Dict[str, Set[str]]
    ],
    tx_lookup: Dict[str, dict],
) -> tuple[
    List[str],
    List[str],
]:
    """
    Reconstruct the longest causal peeling chain ending
    at txid.

    A chain is ordered oldest -> newest.

    Example:

        TX-A -> TX-B -> TX-C

    returns:

        [TX-A, TX-B, TX-C]

    The second returned list contains the addresses that
    establish the continuation edges.

    The search is bounded by MAX_HISTORY_STEPS and uses
    memoization to avoid repeated work.

    Cycles are explicitly prevented even though the causal
    time ordering should already make cycles impossible.
    """

    memo: Dict[
        str,
        tuple[List[str], List[str]]
    ] = {}

    def dfs(
        current_txid: str,
        visited: Set[str],
    ) -> tuple[
        List[str],
        List[str],
    ]:

        if current_txid in memo:
            cached_chain, cached_addresses = memo[
                current_txid
            ]

            return (
                list(cached_chain),
                list(cached_addresses),
            )

        if current_txid in visited:
            return (
                [current_txid],
                [],
            )

        visited_next = set(
            visited
        )

        visited_next.add(
            current_txid
        )

        predecessors = predecessor_map.get(
            current_txid,
            {},
        )

        if not predecessors:
            result = (
                [current_txid],
                [],
            )

            memo[
                current_txid
            ] = result

            return (
                list(result[0]),
                list(result[1]),
            )

        best_chain = [
            current_txid
        ]

        best_addresses: List[
            str
        ] = []

        current_time = tx_lookup[
            current_txid
        ][
            "time_step"
        ]

        for (
            predecessor_txid,
            addresses,
        ) in predecessors.items():

            if predecessor_txid in visited_next:
                continue

            if predecessor_txid not in tx_lookup:
                continue

            predecessor_time = (
                tx_lookup[
                    predecessor_txid
                ][
                    "time_step"
                ]
            )

            if predecessor_time >= current_time:
                continue

            previous_chain, previous_addresses = (
                dfs(
                    predecessor_txid,
                    visited_next,
                )
            )

            candidate_chain = (
                previous_chain
                + [current_txid]
            )

            candidate_addresses = (
                previous_addresses
                + [
                    sorted(
                        addresses
                    )[0]
                ]
            )

            if len(
                candidate_chain
            ) > len(
                best_chain
            ):
                best_chain = candidate_chain
                best_addresses = (
                    candidate_addresses
                )

            elif len(
                candidate_chain
            ) == len(
                best_chain
            ):

                candidate_key = tuple(
                    candidate_chain
                )

                best_key = tuple(
                    best_chain
                )

                if candidate_key < best_key:
                    best_chain = candidate_chain
                    best_addresses = (
                        candidate_addresses
                    )

        result = (
            best_chain,
            best_addresses,
        )

        memo[
            current_txid
        ] = result

        return (
            list(result[0]),
            list(result[1]),
        )

    return dfs(
        txid,
        set(),
    )


def calculate_chain_reduction(
    chain: List[str],
    tx_lookup: Dict[str, dict],
) -> float | None:
    """
    Calculate the total output-value reduction from the
    oldest transaction in the reconstructed chain to the
    current transaction.

    This is intentionally transaction-level because the
    source address edge lists do not provide address-level
    BTC amounts.
    """

    if len(
        chain
    ) < 2:
        return None

    first_tx = tx_lookup[
        chain[0]
    ]

    current_tx = tx_lookup[
        chain[-1]
    ]

    first_value = first_tx[
        "output_btc_total"
    ]

    current_value = current_tx[
        "output_btc_total"
    ]

    return calculate_value_reduction(
        first_value,
        current_value,
    )


def validate_chain(
    chain: List[str],
    tx_lookup: Dict[str, dict],
) -> tuple[
    bool,
    float | None,
    int | None,
]:
    """
    Validate a reconstructed chain.

    Requirements:

    - at least MIN_CHAIN_LENGTH transactions;
    - transaction IDs must be unique;
    - every edge must move strictly forward in time;
    - every edge must remain within MAX_HISTORY_STEPS;
    - every edge must satisfy the configured value-reduction range.
    """

    if len(
        chain
    ) < MIN_CHAIN_LENGTH:
        return (
            False,
            None,
            None,
        )

    if len(
        chain
    ) != len(
        set(chain)
    ):
        return (
            False,
            None,
            None,
        )

    reductions = []

    for index in range(
        1,
        len(chain),
    ):

        previous_txid = chain[
            index - 1
        ]

        current_txid = chain[
            index
        ]

        if (
            previous_txid not in tx_lookup
            or current_txid not in tx_lookup
        ):
            return (
                False,
                None,
                None,
            )

        previous_time = tx_lookup[
            previous_txid
        ][
            "time_step"
        ]

        current_time = tx_lookup[
            current_txid
        ][
            "time_step"
        ]

        gap = (
            current_time
            - previous_time
        )

        if gap <= 0:
            return (
                False,
                None,
                None,
            )

        if gap > MAX_HISTORY_STEPS:
            return (
                False,
                None,
                None,
            )

        reduction = (
            calculate_value_reduction(
                tx_lookup[
                    previous_txid
                ][
                    "output_btc_total"
                ],
                tx_lookup[
                    current_txid
                ][
                    "output_btc_total"
                ],
            )
        )

        if reduction is None:
            return (
                False,
                None,
                None,
            )

        if not (
            MIN_VALUE_REDUCTION_RATIO
            <= reduction
            <= MAX_VALUE_REDUCTION_RATIO
        ):
            return (
                False,
                None,
                None,
            )

        reductions.append(
            reduction
        )

    total_reduction = (
        calculate_chain_reduction(
            chain,
            tx_lookup,
        )
    )

    temporal_span = (
        tx_lookup[
            chain[-1]
        ][
            "time_step"
        ]
        - tx_lookup[
            chain[0]
        ][
            "time_step"
        ]
    )

    if temporal_span <= 0:
        return (
            False,
            None,
            None,
        )

    return (
        True,
        total_reduction,
        int(
            temporal_span
        ),
    )


def calculate_evidence_score(
    direct_continuation: bool,
    gradual_reduction: bool,
    chain_valid: bool,
    chain_length: int,
    temporal_span: int | None,
    continuation_address_count: int,
) -> float:
    """
    Calculate structural peeling evidence.

    The score is an evidence-strength score in [0,1],
    not an illicitness probability.

    A genuine multi-hop chain receives substantially more
    evidence than a single continuation.
    """

    score = 0.0

    if direct_continuation:
        score += 0.25

    if gradual_reduction:
        score += 0.25

    if chain_valid:
        score += 0.25

    if chain_length >= MIN_CHAIN_LENGTH:
        additional_length = min(
            chain_length
            - MIN_CHAIN_LENGTH,
            4,
        )

        score += (
            0.05
            * additional_length
        )

    if (
        temporal_span is not None
        and temporal_span > 0
        and temporal_span <= MAX_HISTORY_STEPS
    ):
        score += 0.10

    if continuation_address_count >= 2:
        score += 0.10

    return float(
        min(
            score,
            1.0,
        )
    )


def detect_transaction(
    txid: str,
    current_time: int,
    tx_inputs: Dict[
        str,
        Set[str]
    ],
    history: Dict[
        str,
        List[Tuple[int, str, str]]
    ],
    tx_lookup: Dict[str, dict],
    predecessor_map: Dict[
        str,
        Dict[str, Set[str]]
    ],
) -> dict:

    current = tx_lookup[
        txid
    ]

    current_inputs = tx_inputs.get(
        txid,
        set(),
    )

    continuation_addresses = set()
    previous_transactions = set()

    best_reduction = None
    best_previous_tx = None
    best_previous_time = None

    # First determine direct historical continuation.
    for address in current_inputs:

        previous_activity = (
            find_previous_activity(
                address,
                current_time,
                history,
                MAX_HISTORY_STEPS,
            )
        )

        for (
            previous_time,
            previous_txid,
            previous_role,
        ) in previous_activity:

            if previous_role != "output":
                continue

            if previous_txid == txid:
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

            reduction = (
                calculate_value_reduction(
                    previous[
                        "output_btc_total"
                    ],
                    current[
                        "output_btc_total"
                    ],
                )
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

    direct_continuation = (
        len(
            continuation_addresses
        ) > 0
    )

    gradual_reduction = (
        best_reduction is not None
    )

    # Reconstruct the longest actual causal chain.
    chain, chain_addresses = (
        reconstruct_longest_chain(
            txid,
            predecessor_map,
            tx_lookup,
        )
    )

    (
        chain_valid,
        chain_total_reduction,
        chain_temporal_span,
    ) = validate_chain(
        chain,
        tx_lookup,
    )

    chain_length = len(
        chain
    )

    # A chain candidate now explicitly requires the configured
    # minimum number of sequential transactions.
    chain_candidate = (
        chain_valid
        and chain_length >= MIN_CHAIN_LENGTH
    )

    evidence_score = (
        calculate_evidence_score(
            direct_continuation=direct_continuation,
            gradual_reduction=gradual_reduction,
            chain_valid=chain_valid,
            chain_length=chain_length,
            temporal_span=chain_temporal_span,
            continuation_address_count=(
                len(
                    continuation_addresses
                )
            ),
        )
    )

    return {
        # Existing fields preserved for M9.3 compatibility.
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
            len(
                continuation_addresses
            )
        ),
        "previous_transaction_count": (
            len(
                previous_transactions
            )
        ),
        "best_previous_txid": (
            best_previous_tx
        ),
        "best_previous_time_step": (
            best_previous_time
        ),
        "temporal_gap": (
            (
                current_time
                - best_previous_time
            )
            if best_previous_time is not None
            else None
        ),
        "best_value_reduction_ratio": (
            best_reduction
        ),
        "current_output_btc_total": (
            current[
                "output_btc_total"
            ]
        ),
        "current_input_btc_total": (
            current[
                "input_btc_total"
            ]
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

        # New hardened chain diagnostics.
        "peeling_chain_length": (
            chain_length
        ),
        "peeling_chain_txids": (
            json.dumps(
                chain,
                ensure_ascii=False,
            )
        ),
        "peeling_chain_addresses": (
            json.dumps(
                chain_addresses,
                ensure_ascii=False,
            )
        ),
        "peeling_chain_valid": (
            chain_valid
        ),
        "peeling_chain_total_value_reduction_ratio": (
            chain_total_reduction
        ),
        "peeling_chain_temporal_span": (
            chain_temporal_span
        ),
    }


def build_behavioral_features(
    transactions: pd.DataFrame,
    tx_inputs: Dict[
        str,
        Set[str]
    ],
    tx_outputs: Dict[
        str,
        Set[str]
    ],
) -> pd.DataFrame:

    history = build_address_history(
        transactions,
        tx_inputs,
        tx_outputs,
    )

    tx_lookup = build_transaction_lookup(
        transactions
    )

    print(
        "Building causal predecessor graph..."
    )

    predecessor_map = (
        build_all_causal_predecessors(
            transactions=transactions,
            tx_inputs=tx_inputs,
            history=history,
            tx_lookup=tx_lookup,
        )
    )

    predecessor_edge_count = sum(
        len(
            predecessors
        )
        for predecessors in predecessor_map.values()
    )

    print(
        "Causal predecessor edges: "
        f"{predecessor_edge_count:,}"
    )

    rows = []

    transactions_sorted = (
        transactions.sort_values(
            [
                TIME_COLUMN,
                TXID_COLUMN,
            ]
        )
    )

    for row in transactions_sorted[
        [TXID_COLUMN, TIME_COLUMN]
    ].itertuples(index=False):

        txid = str(
            row.txid
        )

        current_time = int(
            row.time_step
        )

        rows.append(
            detect_transaction(
                txid=txid,
                current_time=current_time,
                tx_inputs=tx_inputs,
                history=history,
                tx_lookup=tx_lookup,
                predecessor_map=predecessor_map,
            )
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
            "Output row count mismatch."
        )

    if result[
        "txid"
    ].duplicated().any():
        raise ValueError(
            "Duplicate TXIDs in behavioral output."
        )

    canonical_ids = set(
        transactions[
            TXID_COLUMN
        ].astype(str)
    )

    result_ids = set(
        result[
            "txid"
        ].astype(str)
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

    chain_lengths = result[
        "peeling_chain_length"
    ].to_numpy(
        dtype=int
    )

    if (
        chain_lengths < 1
    ).any():
        raise ValueError(
            "Invalid peeling chain length."
        )

    invalid_candidate_lengths = (
        result.loc[
            result[
                "peeling_chain_candidate"
            ],
            "peeling_chain_length",
        ]
        < MIN_CHAIN_LENGTH
    )

    if len(
        invalid_candidate_lengths
    ) > 0:
        raise ValueError(
            "Peeling-chain candidates exist "
            "below MIN_CHAIN_LENGTH."
        )

    candidate_without_valid_chain = (
        result[
            "peeling_chain_candidate"
        ]
        & ~result[
            "peeling_chain_valid"
        ]
    )

    if candidate_without_valid_chain.any():
        raise ValueError(
            "Peeling-chain candidate without "
            "valid reconstructed chain."
        )

    temporal_span_values = result[
        "peeling_chain_temporal_span"
    ].dropna().to_numpy(
        dtype=float
    )

    if len(
        temporal_span_values
    ) > 0:

        if (
            temporal_span_values <= 0
        ).any():
            raise ValueError(
                "Non-positive chain temporal span."
            )

        if (
            temporal_span_values
            > MAX_HISTORY_STEPS
        ).any():
            raise ValueError(
                "Chain temporal span exceeds "
                "MAX_HISTORY_STEPS."
            )

    return {
        "rows": int(
            len(result)
        ),
        "unique_txids": int(
            result[
                "txid"
            ].nunique()
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
        "valid_chain_count": int(
            result[
                "peeling_chain_valid"
            ].sum()
        ),
        "minimum_chain_length": int(
            result[
                "peeling_chain_length"
            ].min()
        ),
        "maximum_chain_length": int(
            result[
                "peeling_chain_length"
            ].max()
        ),
        "mean_chain_length": float(
            result[
                "peeling_chain_length"
            ].mean()
        ),
        "chains_at_or_above_minimum": int(
            (
                result[
                    "peeling_chain_length"
                ]
                >= MIN_CHAIN_LENGTH
            ).sum()
        ),
        "maximum_evidence_score": float(
            result[
                "peeling_evidence_score"
            ].max()
        ),
        "mean_evidence_score": float(
            result[
                "peeling_evidence_score"
            ].mean()
        ),
        "predecessor_edge_count": int(
            sum(
                len(
                    predecessors
                )
                for predecessors in
                build_all_causal_predecessors(
                    transactions=transactions,
                    tx_inputs={},
                    history={},
                    tx_lookup={},
                ).values()
            )
        )
        if False
        else None,
    }


def main() -> None:

    print("=" * 72)
    print(
        "M9.1 PEELING-CHAIN "
        "BEHAVIORAL DETECTOR"
    )
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

    chain_distribution = (
        result[
            "peeling_chain_length"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    candidate_chain_distribution = (
        result.loc[
            result[
                "peeling_chain_candidate"
            ],
            "peeling_chain_length",
        ]
        .value_counts()
        .sort_index()
        .to_dict()
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
            "chain_definition": (
                "A peeling-chain candidate requires at "
                "least MIN_CHAIN_LENGTH sequential "
                "transactions connected through reused "
                "output-to-input addresses, with strict "
                "historical ordering and configured "
                "transaction-level value reduction on "
                "every chain edge."
            ),
            "causal_rule": (
                "Only address activity from time_step "
                "< current transaction time_step is used."
            ),
            "same_time_step_excluded": True,
            "future_activity_excluded": True,
            "address_amount_limitation": (
                "Source address edge lists contain "
                "identifiers but no individual address-level "
                "BTC amounts. Value reduction therefore uses "
                "transaction-level output BTC totals."
            ),
        },
        "validation": validation,
        "chain_length_distribution": {
            str(
                key
            ): int(
                value
            )
            for key, value in chain_distribution.items()
        },
        "candidate_chain_length_distribution": {
            str(
                key
            ): int(
                value
            )
            for key, value in (
                candidate_chain_distribution.items()
            )
        },
        "artifacts": {
            "indicators": str(
                OUTPUT_FILE
            ),
            "report": str(
                REPORT_FILE
            ),
        },
        "interpretation_note": (
            "A peeling-chain candidate is behavioral "
            "evidence of a possible sequential "
            "value-transfer pattern. It is not proof "
            "of mixing, illicit activity, ownership, "
            "identity, intent, or guilt."
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

    print(
        "\n" + "-" * 72
    )
    print(
        "PEELING-CHAIN SUMMARY"
    )
    print(
        "-" * 72
    )

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
        f"Valid reconstructed chains: "
        f"{validation['valid_chain_count']:,}"
    )

    print(
        f"Chain candidates (>= "
        f"{MIN_CHAIN_LENGTH} transactions): "
        f"{validation['candidate_count']:,}"
    )

    print(
        f"Candidate rate: "
        f"{validation['candidate_rate']:.4%}"
    )

    print(
        f"Minimum reconstructed chain length: "
        f"{validation['minimum_chain_length']}"
    )

    print(
        f"Maximum reconstructed chain length: "
        f"{validation['maximum_chain_length']}"
    )

    print(
        f"Mean reconstructed chain length: "
        f"{validation['mean_chain_length']:.4f}"
    )

    print(
        f"Maximum evidence score: "
        f"{validation['maximum_evidence_score']:.4f}"
    )

    print(
        f"Mean evidence score: "
        f"{validation['mean_evidence_score']:.4f}"
    )

    print(
        "\n" + "=" * 72
    )
    print(
        "M9.1 HARDENED DETECTOR COMPLETE"
    )
    print(
        "=" * 72
    )

    print(
        f"Indicators: {OUTPUT_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()