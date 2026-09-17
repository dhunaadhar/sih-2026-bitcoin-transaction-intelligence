"""
M10.3 — Network ↔ Blockchain Correlation

Purpose
-------
Correlate externally supplied network observations with the canonical
blockchain transaction dataset.

Pipeline
--------
Raw Network CSV
    ↓
M10.2 Network Ingestion
    ↓
Normalized + validated + deduplicated network observations
    ↓
M10.3 Exact TXID Correlation
    ↓
Canonical Blockchain Transaction Dataset

Design principles
-----------------
1. M10.2 remains the authoritative network normalization layer.
2. Exact TXID correlation is the primary deterministic relationship.
3. Network timestamps are preserved in UTC.
4. Elliptic++ time_step is retained as ordered temporal context.
5. No wall-clock blockchain timestamp is fabricated.
6. No IP ownership, wallet ownership, identity, or illicit activity
   is inferred.
7. Synthetic network observations remain explicitly identified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple

import pandas as pd

from network_ingestion import ingest_network_file


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

NETWORK_INPUT = (
    ROOT
    / "data"
    / "raw"
    / "network"
    / "m10_2_synthetic_fixture.csv"
)

BLOCKCHAIN_INPUT = (
    ROOT
    / "data"
    / "canonical"
    / "canonical_transactions.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "derived"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "network"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "network_blockchain_correlations.parquet"
)

REPORT_FILE = (
    REPORT_DIR
    / "m10_3_network_blockchain_correlation.json"
)


# ---------------------------------------------------------------------------
# TXID normalization
# ---------------------------------------------------------------------------

def normalize_txid(value) -> str:
    """
    Normalize numeric/string TXID representations.

    Examples
    --------
    4304541
        -> "4304541"

    "4304541"
        -> "4304541"

    "4304541.0"
        -> "4304541"
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        try:
            numeric = float(text)

            if numeric.is_integer():
                text = str(int(numeric))

        except ValueError:
            pass

    return text


# ---------------------------------------------------------------------------
# Network ingestion
# ---------------------------------------------------------------------------

def load_network_observations(
    path: Path,
) -> Tuple[pd.DataFrame, dict]:
    """
    Run the M10.2 network ingestion pipeline.

    M10.3 deliberately consumes the M10.2 normalized output rather than
    bypassing the ingestion layer.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Network input not found: {path}"
        )

    print()
    print("Running M10.2 network ingestion...")

    network_df, ingestion_report = ingest_network_file(
        path,
        "SYNTHETIC_M10_2_TEST",
    )

    network_df = network_df.copy()

    # Defensive TXID normalization.
    network_df["txid"] = (
        network_df["txid"]
        .map(normalize_txid)
    )

    if network_df["txid"].eq("").any():
        raise ValueError(
            "M10.2 output contains empty TXIDs."
        )

    if network_df["timestamp"].isna().any():
        raise ValueError(
            "M10.2 output contains invalid timestamps."
        )

    if network_df["network_event_id"].isna().any():
        raise ValueError(
            "M10.2 output contains missing network_event_id values."
        )

    if network_df["network_event_id"].duplicated().any():
        raise ValueError(
            "M10.2 output contains duplicate network_event_id values."
        )

    return network_df, ingestion_report


# ---------------------------------------------------------------------------
# Blockchain loading
# ---------------------------------------------------------------------------

def load_blockchain_transactions(
    path: Path,
) -> pd.DataFrame:
    """
    Load the canonical blockchain transaction dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Canonical blockchain dataset not found: {path}"
        )

    print()
    print(
        "Reading canonical blockchain data:",
        path,
    )

    df = pd.read_parquet(path)

    required_columns = {
        "txid",
        "time_step",
    }

    missing_columns = sorted(
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Canonical dataset is missing required "
            f"columns: {missing_columns}"
        )

    df = df.copy()

    df["txid"] = (
        df["txid"]
        .map(normalize_txid)
    )

    if df["txid"].eq("").any():
        raise ValueError(
            "Canonical dataset contains empty TXIDs."
        )

    if df["txid"].duplicated().any():
        raise ValueError(
            "Canonical dataset contains duplicate TXIDs."
        )

    return df


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def build_correlation(
    network_df: pd.DataFrame,
    blockchain_df: pd.DataFrame,
    ingestion_report: dict,
) -> Tuple[pd.DataFrame, dict]:
    """
    Perform exact TXID network ↔ blockchain correlation.

    Parameters
    ----------
    network_df:
        Normalized and deduplicated network observations produced by M10.2.

    blockchain_df:
        Canonical blockchain transaction dataset.

    ingestion_report:
        Validation/provenance report generated by M10.2.

    Multiple network observations may legitimately refer to the same
    blockchain transaction, therefore the relationship is many-to-one.
    """

    # ---------------------------------------------------------------
    # Blockchain lookup fields
    # ---------------------------------------------------------------

    blockchain_columns = [
        "txid",
        "time_step",
    ]

    optional_blockchain_columns = [
        "output_btc_total",
        "input_btc_total",
        "fees",
        "size",
        "input_count",
        "output_count",
    ]

    for column in optional_blockchain_columns:

        if column in blockchain_df.columns:
            blockchain_columns.append(column)

    blockchain_lookup = (
        blockchain_df[
            blockchain_columns
        ]
        .copy()
    )

    # Prefix blockchain-side fields.
    rename_map = {
        column: f"blockchain_{column}"
        for column in blockchain_columns
        if column != "txid"
    }

    blockchain_lookup = (
        blockchain_lookup.rename(
            columns=rename_map
        )
    )

    # ---------------------------------------------------------------
    # Exact TXID join
    # ---------------------------------------------------------------

    result = network_df.merge(
        blockchain_lookup,
        on="txid",
        how="left",
        indicator=True,
        validate="many_to_one",
    )

    result["txid_match"] = (
        result["_merge"]
        == "both"
    )

    result["txid_match_type"] = (
        result["txid_match"]
        .map(
            {
                True: "EXACT_TXID",
                False: "NO_BLOCKCHAIN_TXID_MATCH",
            }
        )
    )

    # ---------------------------------------------------------------
    # Network timestamp
    # ---------------------------------------------------------------

    result["network_timestamp_utc"] = (
        pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="coerce",
        )
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
    )

    # ---------------------------------------------------------------
    # Temporal context
    # ---------------------------------------------------------------

    result[
        "temporal_correlation_status"
    ] = (
        result["txid_match"]
        .map(
            {
                True: (
                    "BLOCKCHAIN_TIME_STEP_AVAILABLE"
                ),
                False: (
                    "NO_BLOCKCHAIN_TIME_STEP"
                ),
            }
        )
    )

    result[
        "temporal_evidence_note"
    ] = (
        result["txid_match"]
        .map(
            {
                True: (
                    "Exact TXID matched. "
                    "The blockchain time_step is "
                    "available as ordered temporal "
                    "context. It is not a wall-clock "
                    "timestamp."
                ),
                False: (
                    "No exact TXID match. "
                    "No blockchain temporal "
                    "correlation assigned."
                ),
            }
        )
    )

    result = result.drop(
        columns=["_merge"]
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    matched = result[
        "txid_match"
    ]

    report = {
        "module": "M10.3",
        "status": "PASS",
        "purpose": (
            "Network-to-blockchain correlation "
            "using exact TXID matching with "
            "conservative temporal context."
        ),
        "network_input": str(
            NETWORK_INPUT
        ),
        "blockchain_input": str(
            BLOCKCHAIN_INPUT
        ),
        "network_ingestion_module": "M10.2",
        "network_rows_after_ingestion": int(
            len(network_df)
        ),
        "blockchain_rows": int(
            len(blockchain_df)
        ),
        "correlation_rows": int(
            len(result)
        ),
        "exact_txid_matches": int(
            matched.sum()
        ),
        "unmatched_network_observations": int(
            (~matched).sum()
        ),
        "unique_network_txids": int(
            result["txid"].nunique()
        ),
        "unique_matched_blockchain_txids": int(
            result.loc[
                matched,
                "txid",
            ].nunique()
        ),
        "temporal_context_available": int(
            result[
                "blockchain_time_step"
            ].notna().sum()
        ),
        "wall_clock_temporal_delta_computed": False,
        "identity_inference_performed": False,
        "wallet_ownership_inference_performed": False,
        "illicit_activity_inference_performed": False,
        "network_observations_are_real": False,
        "notes": [
            (
                "M10.3 consumes observations through "
                "the M10.2 ingestion layer."
            ),
            (
                "Exact TXID matching is deterministic "
                "correlation, not identity attribution."
            ),
            (
                "Elliptic++ provides an ordered "
                "time_step rather than a wall-clock "
                "transaction timestamp."
            ),
            (
                "No wall-clock temporal delta is fabricated."
            ),
            (
                "The current network fixture is synthetic "
                "and is used only to validate pipeline mechanics."
            ),
            (
                "Network correlation does not establish "
                "IP ownership, wallet ownership, identity, "
                "or illicit activity."
            ),
        ],
        "m10_2_ingestion_report": ingestion_report,
        "output": str(
            OUTPUT_FILE
        ),
    }

    return result, report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "M10.3 NETWORK ↔ BLOCKCHAIN CORRELATION"
    )
    print("=" * 60)

    print(
        "Network input:",
        NETWORK_INPUT,
    )

    print(
        "Blockchain input:",
        BLOCKCHAIN_INPUT,
    )

    # ---------------------------------------------------------------
    # M10.2 network ingestion
    # ---------------------------------------------------------------

    network_df, ingestion_report = (
        load_network_observations(
            NETWORK_INPUT
        )
    )

    print(
        "Network observations after M10.2:",
        f"{len(network_df):,}",
    )

    print(
        "Duplicates removed by M10.2:",
        ingestion_report[
            "duplicates_removed"
        ],
    )

    # ---------------------------------------------------------------
    # Blockchain dataset
    # ---------------------------------------------------------------

    blockchain_df = (
        load_blockchain_transactions(
            BLOCKCHAIN_INPUT
        )
    )

    print(
        "Blockchain transactions:",
        f"{len(blockchain_df):,}",
    )

    # ---------------------------------------------------------------
    # Correlation
    # ---------------------------------------------------------------

    result, report = build_correlation(
        network_df,
        blockchain_df,
        ingestion_report,
    )

    # ---------------------------------------------------------------
    # Integrity checks
    # ---------------------------------------------------------------

    # Correlation must preserve every network observation.
    if len(result) != len(network_df):
        raise AssertionError(
            "Correlation changed the number of "
            "network observations."
        )

    # Network event IDs must remain unique.
    if result[
        "network_event_id"
    ].duplicated().any():

        raise AssertionError(
            "Correlation produced duplicate "
            "network_event_id values."
        )

    # The synthetic fixture is expected to contain
    # at least one TXID present in the canonical dataset.
    if result[
        "txid_match"
    ].sum() == 0:

        raise AssertionError(
            "Synthetic correlation test produced "
            "zero exact TXID matches."
        )

    # Every matched transaction must have blockchain time_step.
    matched_without_time = (
        result.loc[
            result["txid_match"],
            "blockchain_time_step",
        ]
        .isna()
        .sum()
    )

    if matched_without_time != 0:
        raise AssertionError(
            "Matched blockchain transactions "
            "are missing time_step values."
        )

    # Unmatched observations should not have blockchain time_step.
    unmatched_with_time = (
        result.loc[
            ~result["txid_match"],
            "blockchain_time_step",
        ]
        .notna()
        .sum()
    )

    if unmatched_with_time != 0:
        raise AssertionError(
            "Unmatched network observations "
            "unexpectedly contain blockchain "
            "time_step values."
        )

    # ---------------------------------------------------------------
    # Save parquet
    # ---------------------------------------------------------------

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    output_sha256 = hashlib.sha256(
        OUTPUT_FILE.read_bytes()
    ).hexdigest()

    report[
        "output_sha256"
    ] = output_sha256

    # ---------------------------------------------------------------
    # Save report
    # ---------------------------------------------------------------

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------

    print()
    print(
        "EXACT TXID MATCHES:",
        int(
            result[
                "txid_match"
            ].sum()
        ),
    )

    print(
        "UNMATCHED NETWORK OBSERVATIONS:",
        int(
            (
                ~result[
                    "txid_match"
                ]
            ).sum()
        ),
    )

    print(
        "BLOCKCHAIN TIME_STEP AVAILABLE:",
        int(
            result[
                "blockchain_time_step"
            ]
            .notna()
            .sum()
        ),
    )

    print(
        "WALL-CLOCK TEMPORAL DELTA COMPUTED: NO"
    )

    print()
    print(
        "Correlation preview:"
    )

    print("-" * 60)

    preview_columns = [
        "network_event_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "txid",
        "txid_match",
        "blockchain_time_step",
        "txid_match_type",
        "observation_source",
    ]

    print(
        result[
            preview_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Output:",
        OUTPUT_FILE,
    )

    print(
        "Report:",
        REPORT_FILE,
    )

    print()
    print(
        "M10.3 COMPLETE"
    )


if __name__ == "__main__":
    main()