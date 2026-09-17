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

INDICATORS_FILE = (
    ROOT
    / "data"
    / "derived"
    / "peeling_chain_indicators.parquet"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "behavior"
)

REPORT_FILE = (
    REPORT_DIR
    / "m9_1_peeling_chain_validation.json"
)

MIN_CHAIN_LENGTH = 3
MAX_HISTORY_STEPS = 10
MIN_REDUCTION = 0.05
MAX_REDUCTION = 0.95


def normalize_txid(value) -> str:
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


def calculate_reduction(
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


def main() -> None:

    print("=" * 72)
    print("M9.1 PEELING-CHAIN FINAL VALIDATION")
    print("=" * 72)

    transactions = pd.read_parquet(
        CANONICAL_FILE
    )

    indicators = pd.read_parquet(
        INDICATORS_FILE
    )

    addr_tx = pd.read_csv(
        ADDR_TX_FILE
    )

    tx_addr = pd.read_csv(
        TX_ADDR_FILE
    )

    transactions = transactions.copy()

    transactions["txid"] = (
        transactions["txid"]
        .map(normalize_txid)
    )

    transactions["time_step"] = (
        pd.to_numeric(
            transactions["time_step"],
            errors="raise",
        )
        .astype(int)
    )

    tx_lookup = {}

    for row in transactions[
        [
            "txid",
            "time_step",
            "output_btc_total",
        ]
    ].itertuples(index=False):

        tx_lookup[
            row.txid
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
        }

    tx_inputs = {}

    for row in addr_tx[
        [
            "input_address",
            "txId",
        ]
    ].itertuples(index=False):

        txid = normalize_txid(
            row.txId
        )

        address = normalize_address(
            row.input_address
        )

        if not txid or not address:
            continue

        tx_inputs.setdefault(
            txid,
            set(),
        ).add(
            address
        )

    tx_outputs = {}

    for row in tx_addr[
        [
            "txId",
            "output_address",
        ]
    ].itertuples(index=False):

        txid = normalize_txid(
            row.txId
        )

        address = normalize_address(
            row.output_address
        )

        if not txid or not address:
            continue

        tx_outputs.setdefault(
            txid,
            set(),
        ).add(
            address
        )

    candidates = indicators[
        indicators[
            "peeling_chain_candidate"
        ]
    ].copy()

    print(
        f"Candidate chains: {len(candidates):,}"
    )

    checks = {
        "candidate_count_matches_valid": True,
        "minimum_chain_length": True,
        "candidate_length_consistency": True,
        "txid_sequence_valid": True,
        "txid_coverage_valid": True,
        "strict_temporal_order": True,
        "history_window_valid": True,
        "address_continuity": True,
        "edge_reduction_valid": True,
        "stored_temporal_span_valid": True,
        "stored_total_reduction_valid": True,
    }

    failures = []

    chain_length_counts = {}

    maximum_chain_length = 0

    for row in candidates.itertuples(
        index=False
    ):

        txid = normalize_txid(
            row.txid
        )

        try:
            chain = json.loads(
                row.peeling_chain_txids
            )
        except Exception as exc:
            checks[
                "txid_sequence_valid"
            ] = False

            failures.append(
                {
                    "txid": txid,
                    "reason": (
                        "Invalid JSON chain: "
                        f"{exc}"
                    ),
                }
            )

            continue

        chain = [
            normalize_txid(value)
            for value in chain
        ]

        chain_length = len(
            chain
        )

        chain_length_counts[
            chain_length
        ] = (
            chain_length_counts.get(
                chain_length,
                0,
            )
            + 1
        )

        maximum_chain_length = max(
            maximum_chain_length,
            chain_length,
        )

        if chain_length < MIN_CHAIN_LENGTH:
            checks[
                "minimum_chain_length"
            ] = False

            failures.append(
                {
                    "txid": txid,
                    "reason": (
                        "Chain shorter than "
                        f"{MIN_CHAIN_LENGTH}"
                    ),
                }
            )

        if chain_length != int(
            row.peeling_chain_length
        ):
            checks[
                "candidate_length_consistency"
            ] = False

            failures.append(
                {
                    "txid": txid,
                    "reason": (
                        "Stored chain length does "
                        "not match TXID sequence length."
                    ),
                }
            )

        if not chain:
            checks[
                "txid_sequence_valid"
            ] = False
            continue

        if chain[-1] != txid:
            checks[
                "txid_sequence_valid"
            ] = False

            failures.append(
                {
                    "txid": txid,
                    "reason": (
                        "Current TXID is not the "
                        "last transaction in chain."
                    ),
                }
            )

        if len(chain) != len(
            set(chain)
        ):
            checks[
                "txid_sequence_valid"
            ] = False

            failures.append(
                {
                    "txid": txid,
                    "reason": (
                        "Duplicate transaction "
                        "inside chain."
                    ),
                }
            )

        for chain_txid in chain:

            if chain_txid not in tx_lookup:
                checks[
                    "txid_coverage_valid"
                ] = False

                failures.append(
                    {
                        "txid": txid,
                        "reason": (
                            "Chain contains TXID "
                            f"absent from canonical data: "
                            f"{chain_txid}"
                        ),
                    }
                )

        reductions = []
        gaps = []

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
                continue

            previous = tx_lookup[
                previous_txid
            ]

            current = tx_lookup[
                current_txid
            ]

            previous_time = previous[
                "time_step"
            ]

            current_time = current[
                "time_step"
            ]

            gap = (
                current_time
                - previous_time
            )

            gaps.append(
                gap
            )

            if gap <= 0:
                checks[
                    "strict_temporal_order"
                ] = False

                failures.append(
                    {
                        "txid": txid,
                        "reason": (
                            "Non-increasing "
                            "chain time."
                        ),
                        "chain": chain,
                    }
                )

            if gap > MAX_HISTORY_STEPS:
                checks[
                    "history_window_valid"
                ] = False

                failures.append(
                    {
                        "txid": txid,
                        "reason": (
                            "Chain edge exceeds "
                            "historical window."
                        ),
                        "gap": gap,
                        "chain": chain,
                    }
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
                checks[
                    "address_continuity"
                ] = False

                failures.append(
                    {
                        "txid": txid,
                        "reason": (
                            "Adjacent chain transactions "
                            "do not share an output->input "
                            "address."
                        ),
                        "previous_txid": previous_txid,
                        "current_txid": current_txid,
                    }
                )

            reduction = calculate_reduction(
                previous[
                    "output_btc_total"
                ],
                current[
                    "output_btc_total"
                ],
            )

            if reduction is None:
                checks[
                    "edge_reduction_valid"
                ] = False

                failures.append(
                    {
                        "txid": txid,
                        "reason": (
                            "Edge reduction could "
                            "not be calculated."
                        ),
                        "previous_txid": previous_txid,
                        "current_txid": current_txid,
                    }
                )

            else:

                reductions.append(
                    reduction
                )

                if not (
                    MIN_REDUCTION
                    <= reduction
                    <= MAX_REDUCTION
                ):

                    checks[
                        "edge_reduction_valid"
                    ] = False

                    failures.append(
                        {
                            "txid": txid,
                            "reason": (
                                "Edge reduction outside "
                                "configured range."
                            ),
                            "reduction": reduction,
                            "previous_txid": previous_txid,
                            "current_txid": current_txid,
                        }
                    )

        if gaps:

            calculated_span = (
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

            stored_span = int(
                row.peeling_chain_temporal_span
            )

            if calculated_span != stored_span:
                checks[
                    "stored_temporal_span_valid"
                ] = False

                failures.append(
                    {
                        "txid": txid,
                        "reason": (
                            "Stored temporal span "
                            "does not match chain."
                        ),
                    }
                )

        if reductions:

            calculated_total_reduction = (
                calculate_reduction(
                    tx_lookup[
                        chain[0]
                    ][
                        "output_btc_total"
                    ],
                    tx_lookup[
                        chain[-1]
                    ][
                        "output_btc_total"
                    ],
                )
            )

            stored_total_reduction = (
                row.peeling_chain_total_value_reduction_ratio
            )

            if (
                calculated_total_reduction is None
                or pd.isna(
                    stored_total_reduction
                )
                or not np.isclose(
                    calculated_total_reduction,
                    float(
                        stored_total_reduction
                    ),
                    rtol=1e-9,
                    atol=1e-9,
                )
            ):

                checks[
                    "stored_total_reduction_valid"
                ] = False

                failures.append(
                    {
                        "txid": txid,
                        "reason": (
                            "Stored total reduction "
                            "does not match chain."
                        ),
                    }
                )

    if (
        indicators[
            "peeling_chain_candidate"
        ]
        .sum()
        != indicators[
            "peeling_chain_valid"
        ]
        .sum()
    ):
        checks[
            "candidate_count_matches_valid"
        ] = False

    all_pass = all(
        checks.values()
    )

    report = {
        "milestone": "M9.1",
        "validation": "final_causal_chain_validation",
        "status": (
            "PASS"
            if all_pass
            else "FAIL"
        ),
        "checks": checks,
        "candidate_count": int(
            len(candidates)
        ),
        "maximum_chain_length": int(
            maximum_chain_length
        ),
        "chain_length_distribution": {
            str(key): int(value)
            for key, value in sorted(
                chain_length_counts.items()
            )
        },
        "failure_count": int(
            len(failures)
        ),
        "failures": failures[:100],
        "failure_output_truncated": (
            len(failures) > 100
        ),
        "parameters": {
            "minimum_chain_length": (
                MIN_CHAIN_LENGTH
            ),
            "maximum_history_steps": (
                MAX_HISTORY_STEPS
            ),
            "minimum_edge_reduction": (
                MIN_REDUCTION
            ),
            "maximum_edge_reduction": (
                MAX_REDUCTION
            ),
        },
        "interpretation": (
            "Validation establishes structural and causal "
            "consistency of reconstructed peeling chains. "
            "The result does not establish illicit activity, "
            "identity, ownership, intent, or guilt."
        ),
    }

    REPORT_DIR.mkdir(
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
        )

    print(
        "\n" + "-" * 72
    )
    print(
        "M9.1 FINAL VALIDATION"
    )
    print(
        "-" * 72
    )

    for name, status in checks.items():

        print(
            f"{name}: "
            f"{'PASS' if status else 'FAIL'}"
        )

    print(
        f"\nCandidates validated: "
        f"{len(candidates):,}"
    )

    print(
        f"Maximum chain length: "
        f"{maximum_chain_length}"
    )

    print(
        f"Validation failures: "
        f"{len(failures):,}"
    )

    print(
        "\n" + "=" * 72
    )

    if all_pass:
        print(
            "M9.1 FINAL VALIDATION PASS"
        )
    else:
        print(
            "M9.1 FINAL VALIDATION FAIL"
        )

    print(
        "=" * 72
    )

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()