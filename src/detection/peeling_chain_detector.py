from __future__ import annotations

import json
import time as time_module
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

OUTPUT_DIR = ROOT / "data" / "derived"
OUTPUT_FILE = OUTPUT_DIR / "peeling_chain_indicators.parquet"

REPORT_DIR = ROOT / "reports" / "behavior"
REPORT_FILE = REPORT_DIR / "m9_1_peeling_chain.json"


TXID_COLUMN = "txid"
TIME_COLUMN = "time_step"

MIN_CHAIN_LENGTH = 3
MAX_HISTORY_STEPS = 10

MIN_VALUE_REDUCTION_RATIO = 0.05
MAX_VALUE_REDUCTION_RATIO = 0.95

# Limits the depth of the causal DFS.
# This is deliberately finite so highly connected graphs cannot
# cause pathological recursive exploration.
MAX_CHAIN_LENGTH = 10


def normalize_txid(value) -> str:
    """
    Normalize transaction identifiers so that identifiers such as
    4304541 and 4304541.0 are treated consistently.
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    if text.lower() == "nan":
        return ""

    try:
        numeric = float(text)

        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except (ValueError, TypeError):
        pass

    return text


def normalize_address(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return ""

    return text


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

    transactions = pd.read_parquet(CANONICAL_FILE)
    addr_tx = pd.read_csv(ADDR_TX_FILE)
    tx_addr = pd.read_csv(TX_ADDR_FILE)

    return transactions, addr_tx, tx_addr


def prepare_transactions(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        TXID_COLUMN,
        TIME_COLUMN,
        "output_btc_total",
        "input_btc_total",
        "output_address_count",
        "input_address_count",
    }

    missing = required - set(transactions.columns)

    if missing:
        raise ValueError(
            "Canonical dataset missing required columns: "
            f"{sorted(missing)}"
        )

    result = transactions.copy()

    result[TXID_COLUMN] = (
        result[TXID_COLUMN]
        .map(normalize_txid)
    )

    result[TIME_COLUMN] = pd.to_numeric(
        result[TIME_COLUMN],
        errors="raise",
    ).astype(int)

    if result[TXID_COLUMN].eq("").any():
        raise ValueError(
            "Canonical dataset contains empty TXIDs."
        )

    if result[TXID_COLUMN].duplicated().any():
        raise ValueError(
            "Canonical dataset contains duplicate TXIDs."
        )

    if result[TIME_COLUMN].isna().any():
        raise ValueError(
            "Canonical dataset contains null time steps."
        )

    if (
        result[TIME_COLUMN] < 0
    ).any():
        raise ValueError(
            "Canonical dataset contains negative time steps."
        )

    return result


def validate_source_data(
    transactions: pd.DataFrame,
    addr_tx: pd.DataFrame,
    tx_addr: pd.DataFrame,
) -> None:

    if not {
        "input_address",
        "txId",
    }.issubset(addr_tx.columns):

        raise ValueError(
            "AddrTx edge list must contain "
            "input_address and txId."
        )

    if not {
        "txId",
        "output_address",
    }.issubset(tx_addr.columns):

        raise ValueError(
            "TxAddr edge list must contain "
            "txId and output_address."
        )

    if transactions[
        TXID_COLUMN
    ].isna().any():

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

    tx_inputs: Dict[
        str,
        Set[str],
    ] = defaultdict(set)

    tx_outputs: Dict[
        str,
        Set[str],
    ] = defaultdict(set)

    for row in addr_tx[
        ["input_address", "txId"]
    ].itertuples(index=False):

        address = normalize_address(
            row.input_address
        )

        txid = normalize_txid(
            row.txId
        )

        if address and txid:
            tx_inputs[txid].add(address)

    for row in tx_addr[
        ["txId", "output_address"]
    ].itertuples(index=False):

        txid = normalize_txid(
            row.txId
        )

        address = normalize_address(
            row.output_address
        )

        if txid and address:
            tx_outputs[txid].add(address)

    return tx_inputs, tx_outputs


def build_transaction_lookup(
    transactions: pd.DataFrame,
) -> Dict[str, dict]:

    lookup: Dict[str, dict] = {}

    columns = [
        TXID_COLUMN,
        TIME_COLUMN,
        "output_btc_total",
        "input_btc_total",
        "output_address_count",
        "input_address_count",
    ]

    for row in transactions[
        columns
    ].itertuples(index=False):

        txid = normalize_txid(
            row.txid
        )

        output_value = (
            float(row.output_btc_total)
            if pd.notna(row.output_btc_total)
            else np.nan
        )

        input_value = (
            float(row.input_btc_total)
            if pd.notna(row.input_btc_total)
            else np.nan
        )

        output_count = (
            int(row.output_address_count)
            if pd.notna(row.output_address_count)
            else 0
        )

        input_count = (
            int(row.input_address_count)
            if pd.notna(row.input_address_count)
            else 0
        )

        lookup[txid] = {
            "time_step": int(row.time_step),
            "output_btc_total": output_value,
            "input_btc_total": input_value,
            "output_address_count": output_count,
            "input_address_count": input_count,
        }

    return lookup


def build_address_history(
    transactions: pd.DataFrame,
    tx_inputs: Dict[str, Set[str]],
    tx_outputs: Dict[str, Set[str]],
) -> Dict[
    str,
    List[Tuple[int, str, str]],
]:

    tx_time = {
        normalize_txid(row.txid): int(row.time_step)
        for row in transactions[
            [TXID_COLUMN, TIME_COLUMN]
        ].itertuples(index=False)
    }

    history: Dict[
        str,
        List[Tuple[int, str, str]],
    ] = defaultdict(list)

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
                item[2],
            )
        )

    return history


def calculate_value_reduction(
    previous_value: float,
    current_value: float,
) -> float | None:

    if not np.isfinite(previous_value):
        return None

    if not np.isfinite(current_value):
        return None

    if previous_value <= 0:
        return None

    reduction = (
        previous_value - current_value
    ) / previous_value

    if not np.isfinite(reduction):
        return None

    return float(reduction)


def is_valid_reduction(
    reduction: float | None,
) -> bool:

    if reduction is None:
        return False

    return (
        MIN_VALUE_REDUCTION_RATIO
        <= reduction
        <= MAX_VALUE_REDUCTION_RATIO
    )


def build_causal_relationships(
    transactions: pd.DataFrame,
    tx_inputs: Dict[str, Set[str]],
    history: Dict[str, List[Tuple[int, str, str]]],
    tx_lookup: Dict[str, dict],
) -> tuple[
    Dict[str, Dict[str, Set[str]]],
    Dict[str, Set[str]],
    Dict[str, Set[str]],
    Dict[str, float],
    Dict[str, str],
    Dict[str, int],
]:

    """
    Build all direct historical relationships once.

    Returns:

    predecessor_map:
        current_txid -> predecessor_txid -> reused addresses

    continuation_addresses:
        current_txid -> addresses with prior output activity

    previous_transactions:
        current_txid -> prior transaction IDs

    best_reduction:
        current_txid -> strongest valid one-step reduction

    best_previous_txid:
        current_txid -> predecessor producing strongest reduction

    best_previous_time:
        current_txid -> time step of strongest predecessor
    """

    predecessor_map: Dict[
        str,
        Dict[str, Set[str]],
    ] = {}

    continuation_addresses: Dict[
        str,
        Set[str],
    ] = defaultdict(set)

    previous_transactions: Dict[
        str,
        Set[str],
    ] = defaultdict(set)

    best_reduction: Dict[
        str,
        float,
    ] = {}

    best_previous_txid: Dict[
        str,
        str,
    ] = {}

    best_previous_time: Dict[
        str,
        int,
    ] = {}

    ordered_transactions = (
        transactions.sort_values(
            [
                TIME_COLUMN,
                TXID_COLUMN,
            ]
        )
    )

    for row in ordered_transactions[
        [TXID_COLUMN, TIME_COLUMN]
    ].itertuples(index=False):

        txid = normalize_txid(
            row.txid
        )

        current_time = int(
            row.time_step
        )

        current_info = tx_lookup[
            txid
        ]

        current_output_value = (
            current_info[
                "output_btc_total"
            ]
        )

        current_predecessors: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        for address in tx_inputs.get(
            txid,
            set(),
        ):

            minimum_time = (
                current_time
                - MAX_HISTORY_STEPS
            )

            for (
                previous_time,
                previous_txid,
                previous_role,
            ) in history.get(
                address,
                [],
            ):

                # Only previous OUTPUT activity establishes
                # the output -> input continuation used here.
                if previous_role != "output":
                    continue

                # Strict causal ordering.
                if previous_time < minimum_time:
                    continue

                if previous_time >= current_time:
                    continue

                if previous_txid == txid:
                    continue

                if previous_txid not in tx_lookup:
                    continue

                continuation_addresses[
                    txid
                ].add(
                    address
                )

                previous_transactions[
                    txid
                ].add(
                    previous_txid
                )

                previous_info = tx_lookup[
                    previous_txid
                ]

                reduction = (
                    calculate_value_reduction(
                        previous_info[
                            "output_btc_total"
                        ],
                        current_output_value,
                    )
                )

                if is_valid_reduction(
                    reduction
                ):

                    current_predecessors[
                        previous_txid
                    ].add(
                        address
                    )

                    assert reduction is not None

                    existing = best_reduction.get(
                        txid
                    )

                    if (
                        existing is None
                        or reduction > existing
                        or (
                            reduction == existing
                            and previous_txid
                            < best_previous_txid.get(
                                txid,
                                previous_txid,
                            )
                        )
                    ):
                        best_reduction[
                            txid
                        ] = reduction

                        best_previous_txid[
                            txid
                        ] = previous_txid

                        best_previous_time[
                            txid
                        ] = previous_time

        predecessor_map[
            txid
        ] = dict(
            current_predecessors
        )

    return (
        predecessor_map,
        continuation_addresses,
        previous_transactions,
        best_reduction,
        best_previous_txid,
        best_previous_time,
    )


def reconstruct_longest_chain(
    txid: str,
    predecessor_map: Dict[
        str,
        Dict[str, Set[str]],
    ],
    tx_lookup: Dict[str, dict],
) -> tuple[
    List[str],
    List[str],
]:

    memo: Dict[
        str,
        tuple[List[str], List[str]],
    ] = {}

    def choose_better(
        candidate_chain: List[str],
        candidate_addresses: List[str],
        current_chain: List[str],
        current_addresses: List[str],
    ) -> tuple[
        List[str],
        List[str],
    ]:

        if len(candidate_chain) > len(
            current_chain
        ):
            return (
                candidate_chain,
                candidate_addresses,
            )

        if len(candidate_chain) < len(
            current_chain
        ):
            return (
                current_chain,
                current_addresses,
            )

        # Deterministic tie-breaking.
        if tuple(candidate_chain) < tuple(
            current_chain
        ):
            return (
                candidate_chain,
                candidate_addresses,
            )

        return (
            current_chain,
            current_addresses,
        )

    def dfs(
        current_txid: str,
        visited: Set[str],
    ) -> tuple[
        List[str],
        List[str],
    ]:

        if current_txid in memo:
            chain, addresses = memo[
                current_txid
            ]

            return (
                list(chain),
                list(addresses),
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

        current_time = tx_lookup[
            current_txid
        ][
            "time_step"
        ]

        best_chain = [
            current_txid
        ]

        best_addresses: List[str] = []

        if len(
            best_chain
        ) >= MAX_CHAIN_LENGTH:
            return (
                best_chain,
                best_addresses,
            )

        for (
            predecessor_txid,
            reused_addresses,
        ) in sorted(
            predecessors.items()
        ):

            if predecessor_txid in visited_next:
                continue

            if predecessor_txid not in tx_lookup:
                continue

            predecessor_time = tx_lookup[
                predecessor_txid
            ][
                "time_step"
            ]

            # Defensive strict causal constraint.
            if predecessor_time >= current_time:
                continue

            # Every reconstructed edge is already within the
            # historical window, but retain the check here.
            if (
                current_time
                - predecessor_time
                > MAX_HISTORY_STEPS
            ):
                continue

            previous_chain, previous_addresses = (
                dfs(
                    predecessor_txid,
                    visited_next,
                )
            )

            if (
                len(previous_chain)
                >= MAX_CHAIN_LENGTH
            ):
                previous_chain = previous_chain[
                    -MAX_CHAIN_LENGTH:
                ]

                if previous_chain[-1] != predecessor_txid:
                    continue

            candidate_chain = (
                previous_chain
                + [current_txid]
            )

            if len(
                candidate_chain
            ) > MAX_CHAIN_LENGTH:

                candidate_chain = (
                    candidate_chain[
                        -MAX_CHAIN_LENGTH:
                    ]
                )

            selected_address = sorted(
                reused_addresses
            )[0]

            candidate_addresses = (
                previous_addresses
                + [selected_address]
            )

            (
                best_chain,
                best_addresses,
            ) = choose_better(
                candidate_chain,
                candidate_addresses,
                best_chain,
                best_addresses,
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


def calculate_chain_metrics(
    chain: List[str],
    tx_lookup: Dict[str, dict],
    tx_inputs: Dict[str, Set[str]],
    tx_outputs: Dict[str, Set[str]],
) -> tuple[
    bool,
    float | None,
    float | None,
    float | None,
    int | None,
    List[int],
    List[int],
]:

    if len(
        chain
    ) < MIN_CHAIN_LENGTH:
        return (
            False,
            None,
            None,
            None,
            None,
            [],
            [],
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
            None,
            None,
            [],
            [],
        )

    reductions: List[
        float
    ] = []

    gaps: List[
        int
    ] = []

    link_address_counts: List[
        int
    ] = []

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
                None,
                None,
                [],
                [],
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
                None,
                None,
                [],
                [],
            )

        if gap > MAX_HISTORY_STEPS:
            return (
                False,
                None,
                None,
                None,
                None,
                [],
                [],
            )

        reduction = calculate_value_reduction(
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

        if not is_valid_reduction(
            reduction
        ):
            return (
                False,
                None,
                None,
                None,
                None,
                [],
                [],
            )

        shared_addresses = (
            tx_outputs.get(
                previous_txid,
                set(),
            )
            & tx_inputs.get(
                current_txid,
                set(),
            )
        )

        if not shared_addresses:
            return (
                False,
                None,
                None,
                None,
                None,
                [],
                [],
            )

        assert reduction is not None

        reductions.append(
            reduction
        )

        gaps.append(
            int(gap)
        )

        link_address_counts.append(
            len(shared_addresses)
        )

    oldest_value = tx_lookup[
        chain[0]
    ][
        "output_btc_total"
    ]

    current_value = tx_lookup[
        chain[-1]
    ][
        "output_btc_total"
    ]

    total_reduction = (
        calculate_value_reduction(
            oldest_value,
            current_value,
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

    return (
        True,
        total_reduction,
        float(
            min(reductions)
        ),
        float(
            max(reductions)
        ),
        int(
            temporal_span
        ),
        gaps,
        link_address_counts,
    )


def calculate_evidence_score(
    direct_continuation: bool,
    gradual_reduction: bool,
    chain_valid: bool,
    chain_length: int,
    temporal_span: int | None,
    continuation_address_count: int,
    chain_gaps: List[int],
) -> float:

    score = 0.0

    if direct_continuation:
        score += 0.25

    if gradual_reduction:
        score += 0.25

    if chain_valid:
        score += 0.25

    # Additional evidence for genuinely multi-hop behavior.
    if chain_length >= MIN_CHAIN_LENGTH:
        additional_length = min(
            chain_length - MIN_CHAIN_LENGTH,
            4,
        )

        score += (
            0.05
            * additional_length
        )

    if chain_gaps:
        consecutive_links = sum(
            1
            for gap in chain_gaps
            if gap == 1
        )

        if consecutive_links == len(
            chain_gaps
        ):
            score += 0.10

    elif (
        temporal_span is not None
        and temporal_span > 0
    ):
        score += 0.05

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
    tx_lookup: Dict[str, dict],
    tx_inputs: Dict[str, Set[str]],
    tx_outputs: Dict[str, Set[str]],
    predecessor_map: Dict[
        str,
        Dict[str, Set[str]],
    ],
    continuation_addresses: Dict[
        str,
        Set[str],
    ],
    previous_transactions: Dict[
        str,
        Set[str],
    ],
    best_reduction: Dict[
        str,
        float,
    ],
    best_previous_txid: Dict[
        str,
        str,
    ],
    best_previous_time: Dict[
        str,
        int,
    ],
) -> dict:

    current = tx_lookup[
        txid
    ]

    direct_continuation = bool(
        continuation_addresses.get(
            txid,
            set(),
        )
    )

    gradual_reduction = (
        txid in best_reduction
    )

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
        chain_min_reduction,
        chain_max_reduction,
        chain_temporal_span,
        chain_gaps,
        chain_link_address_counts,
    ) = calculate_chain_metrics(
        chain,
        tx_lookup,
        tx_inputs,
        tx_outputs,
    )

    chain_length = len(
        chain
    )

    chain_candidate = (
        chain_valid
        and chain_length >= MIN_CHAIN_LENGTH
    )

    evidence_score = calculate_evidence_score(
        direct_continuation=direct_continuation,
        gradual_reduction=gradual_reduction,
        chain_valid=chain_valid,
        chain_length=chain_length,
        temporal_span=chain_temporal_span,
        continuation_address_count=len(
            continuation_addresses.get(
                txid,
                set(),
            )
        ),
        chain_gaps=chain_gaps,
    )

    previous_txid = best_previous_txid.get(
        txid
    )

    previous_time = best_previous_time.get(
        txid
    )

    return {
        "txid": txid,
        "time_step": int(
            current["time_step"]
        ),

        # Existing downstream-compatible fields.
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
                continuation_addresses.get(
                    txid,
                    set(),
                )
            )
        ),
        "previous_transaction_count": (
            len(
                previous_transactions.get(
                    txid,
                    set(),
                )
            )
        ),
        "best_previous_txid": (
            previous_txid
        ),
        "best_previous_time_step": (
            previous_time
        ),
        "temporal_gap": (
            (
                int(
                    current["time_step"]
                )
                - previous_time
            )
            if previous_time is not None
            else None
        ),
        "best_value_reduction_ratio": (
            best_reduction.get(
                txid
            )
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

        # Hardened chain diagnostics.
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
        "peeling_chain_min_edge_reduction_ratio": (
            chain_min_reduction
        ),
        "peeling_chain_max_edge_reduction_ratio": (
            chain_max_reduction
        ),
        "peeling_chain_temporal_span": (
            chain_temporal_span
        ),
        "peeling_chain_temporal_gaps": (
            json.dumps(
                chain_gaps
            )
        ),
        "peeling_chain_link_address_counts": (
            json.dumps(
                chain_link_address_counts
            )
        ),
    }


def build_behavioral_features(
    transactions: pd.DataFrame,
    tx_inputs: Dict[str, Set[str]],
    tx_outputs: Dict[str, Set[str]],
) -> tuple[pd.DataFrame, dict]:

    start = time_module.perf_counter()

    history_start = (
        time_module.perf_counter()
    )

    history = build_address_history(
        transactions,
        tx_inputs,
        tx_outputs,
    )

    history_seconds = (
        time_module.perf_counter()
        - history_start
    )

    tx_lookup = build_transaction_lookup(
        transactions
    )

    relationship_start = (
        time_module.perf_counter()
    )

    (
        predecessor_map,
        continuation_addresses,
        previous_transactions,
        best_reduction,
        best_previous_txid,
        best_previous_time,
    ) = build_causal_relationships(
        transactions=transactions,
        tx_inputs=tx_inputs,
        history=history,
        tx_lookup=tx_lookup,
    )

    relationship_seconds = (
        time_module.perf_counter()
        - relationship_start
    )

    predecessor_edge_count = sum(
        len(
            predecessors
        )
        for predecessors in predecessor_map.values()
    )

    chain_start = (
        time_module.perf_counter()
    )

    rows = []

    ordered_transactions = (
        transactions.sort_values(
            [
                TIME_COLUMN,
                TXID_COLUMN,
            ]
        )
    )

    for row in ordered_transactions[
        [TXID_COLUMN, TIME_COLUMN]
    ].itertuples(index=False):

        txid = normalize_txid(
            row.txid
        )

        rows.append(
            detect_transaction(
                txid=txid,
                tx_lookup=tx_lookup,
                tx_inputs=tx_inputs,
                tx_outputs=tx_outputs,
                predecessor_map=predecessor_map,
                continuation_addresses=(
                    continuation_addresses
                ),
                previous_transactions=(
                    previous_transactions
                ),
                best_reduction=(
                    best_reduction
                ),
                best_previous_txid=(
                    best_previous_txid
                ),
                best_previous_time=(
                    best_previous_time
                ),
            )
        )

    chain_seconds = (
        time_module.perf_counter()
        - chain_start
    )

    total_seconds = (
        time_module.perf_counter()
        - start
    )

    runtime = {
        "history_build_seconds": (
            history_seconds
        ),
        "relationship_build_seconds": (
            relationship_seconds
        ),
        "chain_reconstruction_seconds": (
            chain_seconds
        ),
        "total_seconds": (
            total_seconds
        ),
        "predecessor_edge_count": (
            int(
                predecessor_edge_count
            )
        ),
    }

    return (
        pd.DataFrame(rows),
        runtime,
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
            "Duplicate TXIDs in output."
        )

    canonical_ids = set(
        transactions[
            TXID_COLUMN
        ].map(normalize_txid)
    )

    result_ids = set(
        result[
            "txid"
        ].map(normalize_txid)
    )

    if canonical_ids != result_ids:
        raise ValueError(
            "Output TXIDs do not match canonical dataset."
        )

    scores = result[
        "peeling_evidence_score"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        scores
    ).all():
        raise ValueError(
            "Non-finite evidence scores."
        )

    if (
        (scores < 0)
        | (scores > 1)
    ).any():
        raise ValueError(
            "Evidence scores outside [0,1]."
        )

    lengths = result[
        "peeling_chain_length"
    ].to_numpy(
        dtype=int
    )

    if (
        lengths < 1
    ).any():
        raise ValueError(
            "Invalid chain length."
        )

    candidate_mask = result[
        "peeling_chain_candidate"
    ]

    if (
        result.loc[
            candidate_mask,
            "peeling_chain_length",
        ]
        < MIN_CHAIN_LENGTH
    ).any():

        raise ValueError(
            "A chain candidate has fewer than "
            f"{MIN_CHAIN_LENGTH} transactions."
        )

    if (
        result.loc[
            candidate_mask,
            "peeling_chain_valid",
        ]
        == False
    ).any():

        raise ValueError(
            "Chain candidate marked invalid."
        )

    valid_chain_lengths = result.loc[
        result[
            "peeling_chain_valid"
        ],
        "peeling_chain_length",
    ]

    if (
        valid_chain_lengths
        < MIN_CHAIN_LENGTH
    ).any():

        raise ValueError(
            "Invalid chain marked as valid."
        )

    temporal_spans = (
        result[
            "peeling_chain_temporal_span"
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(
        temporal_spans
    ):

        if (
            temporal_spans <= 0
        ).any():

            raise ValueError(
                "Non-positive temporal span."
            )

    total_reductions = (
        result[
            "peeling_chain_total_value_reduction_ratio"
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(
        total_reductions
    ):

        if not np.isfinite(
            total_reductions
        ).all():

            raise ValueError(
                "Non-finite chain reductions."
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
        "candidate_count": int(
            candidate_mask.sum()
        ),
        "candidate_rate": float(
            candidate_mask.mean()
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
    }


def main() -> None:

    overall_start = (
        time_module.perf_counter()
    )

    print("=" * 72)
    print(
        "M9.1 PEELING-CHAIN "
        "BEHAVIORAL DETECTOR"
    )
    print("=" * 72)

    transactions, addr_tx, tx_addr = (
        load_data()
    )

    transactions = prepare_transactions(
        transactions
    )

    validate_source_data(
        transactions,
        addr_tx,
        tx_addr,
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

    (
        tx_inputs,
        tx_outputs,
    ) = build_transaction_address_maps(
        addr_tx,
        tx_addr,
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

    result, runtime = (
        build_behavioral_features(
            transactions,
            tx_inputs,
            tx_outputs,
        )
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

    runtime[
        "overall_seconds"
    ] = (
        time_module.perf_counter()
        - overall_start
    )

    report = {
        "milestone": "M9.1",
        "status": "PASS",
        "detector": "peeling_chain",
        "hardening": {
            "enabled": True,
            "minimum_chain_length": (
                MIN_CHAIN_LENGTH
            ),
            "maximum_chain_length": (
                MAX_CHAIN_LENGTH
            ),
            "strict_causal_ordering": True,
            "same_time_step_excluded": True,
            "future_activity_excluded": True,
            "historical_window_steps": (
                MAX_HISTORY_STEPS
            ),
            "every_chain_edge_requires_valid_value_reduction": (
                True
            ),
        },
        "methodology": {
            "min_value_reduction_ratio": (
                MIN_VALUE_REDUCTION_RATIO
            ),
            "max_value_reduction_ratio": (
                MAX_VALUE_REDUCTION_RATIO
            ),
            "chain_definition": (
                "A chain candidate requires at least "
                "MIN_CHAIN_LENGTH sequential transactions "
                "connected through reused output-to-input "
                "addresses. Every chain edge must be strictly "
                "historical and satisfy the configured "
                "transaction-level value reduction range."
            ),
            "chain_selection": (
                "Longest causal chain is selected "
                "deterministically, with TXID ordering "
                "used as the tie-breaker."
            ),
            "value_semantics": (
                "Value reduction uses transaction-level "
                "output BTC totals because the supplied "
                "AddrTx and TxAddr edge lists do not provide "
                "address-level BTC amounts."
            ),
        },
        "runtime": runtime,
        "validation": validation,
        "chain_length_distribution": {
            str(key): int(value)
            for key, value in (
                chain_distribution.items()
            )
        },
        "candidate_chain_length_distribution": {
            str(key): int(value)
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
            "evidence of a possible sequential value-transfer "
            "pattern. It is not proof of mixing, illicit "
            "activity, ownership, identity, intent, or guilt."
        ),
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
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
        "\nRuntime:"
    )

    print(
        f"  History construction: "
        f"{runtime['history_build_seconds']:.2f}s"
    )

    print(
        f"  Causal relationships: "
        f"{runtime['relationship_build_seconds']:.2f}s"
    )

    print(
        f"  Chain reconstruction: "
        f"{runtime['chain_reconstruction_seconds']:.2f}s"
    )

    print(
        f"  Total: "
        f"{runtime['overall_seconds']:.2f}s"
    )

    print(
        "\nCausal predecessor edges: "
        f"{runtime['predecessor_edge_count']:,}"
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