from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from network_schema import (
    CANONICAL_COLUMNS,
    REQUIRED_COLUMNS,
    normalize_network_dataframe,
)


ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    ROOT
    / "reports"
    / "network"
)

REPORT_FILE = (
    REPORT_DIR
    / "m10_2_network_ingestion.json"
)


def calculate_file_sha256(
    file_path: Path,
) -> str:
    sha256 = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as handle:

        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


def load_network_csv(
    input_file: Path,
) -> pd.DataFrame:

    if not input_file.exists():
        raise FileNotFoundError(
            f"Network input file not found: "
            f"{input_file}"
        )

    if input_file.suffix.lower() != ".csv":
        raise ValueError(
            "M10.2 currently accepts CSV input only."
        )

    return pd.read_csv(
        input_file
    )


def validate_required_columns(
    frame: pd.DataFrame,
) -> None:

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Network input is missing required columns: "
            f"{missing}"
        )


def add_network_event_ids(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    frame = frame.copy()

    if (
        "network_event_id" in frame.columns
        and frame[
            "network_event_id"
        ].notna().all()
    ):
        return frame

    identifiers = []

    for row in frame[
        [
            "timestamp",
            "src_ip",
            "src_port",
            "dst_ip",
            "dst_port",
            "txid",
        ]
    ].itertuples(
        index=False,
        name=None,
    ):

        raw = "|".join(
            "" if value is None else str(value)
            for value in row
        )

        identifiers.append(
            hashlib.sha256(
                raw.encode(
                    "utf-8"
                )
            ).hexdigest()[:24]
        )

    frame[
        "network_event_id"
    ] = identifiers

    return frame


def add_provenance(
    frame: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:

    frame = frame.copy()

    existing = (
        frame[
            "observation_source"
        ]
        .astype("string")
    )

    missing = (
        existing.isna()
        | (
            existing.str.strip()
            == ""
        )
    )

    frame.loc[
        missing,
        "observation_source"
    ] = source_name

    return frame


def deduplicate_network_events(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:

    before = len(frame)

    dedup_columns = [
        "timestamp",
        "src_ip",
        "src_port",
        "dst_ip",
        "dst_port",
        "txid",
        "geo_country",
        "asn",
        "protocol",
        "direction",
        "observation_source",
    ]

    frame = (
        frame
        .drop_duplicates(
            subset=dedup_columns,
            keep="first",
        )
        .reset_index(drop=True)
    )

    duplicates_removed = (
        before - len(frame)
    )

    return (
        frame,
        duplicates_removed,
    )


def validate_normalized_network(
    frame: pd.DataFrame,
) -> dict:

    required_nulls = {
        column: int(
            frame[column].isna().sum()
        )
        for column in REQUIRED_COLUMNS
    }

    invalid_required = {
        column: count
        for column, count in required_nulls.items()
        if count > 0
    }

    if invalid_required:
        raise ValueError(
            "Required network fields contain invalid/null "
            f"values after normalization: {invalid_required}"
        )

    invalid_ports = (
        (
            frame["src_port"] < 0
        )
        | (
            frame["src_port"] > 65535
        )
        | (
            frame["dst_port"] < 0
        )
        | (
            frame["dst_port"] > 65535
        )
    ).sum()

    if invalid_ports:
        raise ValueError(
            f"Invalid port values remain: {invalid_ports}"
        )

    if frame[
        "network_event_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate network_event_id values remain."
        )

    return {
        "rows": int(
            len(frame)
        ),
        "unique_network_event_ids": int(
            frame[
                "network_event_id"
            ].nunique()
        ),
        "unique_txids": int(
            frame[
                "txid"
            ].nunique()
        ),
        "unique_source_ips": int(
            frame[
                "src_ip"
            ].nunique()
        ),
        "unique_destination_ips": int(
            frame[
                "dst_ip"
            ].nunique()
        ),
        "unique_asns": int(
            frame[
                "asn"
            ].nunique()
        ),
        "unique_countries": int(
            frame[
                "geo_country"
            ].nunique()
        ),
        "time_min": (
            frame[
                "timestamp"
            ]
            .min()
            .isoformat()
        ),
        "time_max": (
            frame[
                "timestamp"
            ]
            .max()
            .isoformat()
        ),
    }


def ingest_network_file(
    input_file: Path,
    observation_source: str | None = None,
) -> tuple[pd.DataFrame, dict]:

    print(
        f"\nReading network CSV: "
        f"{input_file}"
    )

    raw = load_network_csv(
        input_file
    )

    raw_rows = len(raw)

    print(
        f"Raw rows: {raw_rows:,}"
    )

    validate_required_columns(
        raw
    )

    normalized, schema_report = (
        normalize_network_dataframe(
            raw
        )
    )

    normalized = add_network_event_ids(
        normalized
    )

    source_name = (
        observation_source
        or input_file.name
    )

    normalized = add_provenance(
        normalized,
        source_name,
    )

    normalized, duplicates_removed = (
        deduplicate_network_events(
            normalized
        )
    )

    validation = (
        validate_normalized_network(
            normalized
        )
    )

    report = {
        "input_file": str(
            input_file
        ),
        "input_sha256": (
            calculate_file_sha256(
                input_file
            )
        ),
        "raw_rows": int(
            raw_rows
        ),
        "duplicates_removed": int(
            duplicates_removed
        ),
        "normalized_rows": int(
            len(normalized)
        ),
        "schema_normalization": (
            schema_report
        ),
        "validation": validation,
        "canonical_columns": (
            CANONICAL_COLUMNS
        ),
        "observation_source": source_name,
    }

    return (
        normalized,
        report,
    )


def save_ingestion_report(
    report: dict,
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
        )


def main() -> None:

    print("=" * 72)
    print("M10.2 NETWORK INGESTION + PROVENANCE")
    print("=" * 72)

    print(
        "\nThis module validates and normalizes "
        "externally supplied network observations."
    )

    print(
        "It does not generate network observations."
    )

    print(
        "\nM10.2 module status: READY"
    )

    print(
        "\nRequired input columns:"
    )

    for column in REQUIRED_COLUMNS:
        print(
            f"  - {column}"
        )

    print(
        "\nCanonical output columns:"
    )

    for column in CANONICAL_COLUMNS:
        print(
            f"  - {column}"
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "milestone": "M10.2",
        "status": "PASS",
        "purpose": (
            "Provide an offline network-observation "
            "ingestion layer with normalization, "
            "deduplication, validation, and provenance."
        ),
        "input_format": [
            "CSV"
        ],
        "output_schema": CANONICAL_COLUMNS,
        "features": [
            "required-field validation",
            "timestamp normalization",
            "IP normalization",
            "port validation",
            "ASN normalization",
            "country normalization",
            "TXID normalization",
            "network event ID generation",
            "provenance assignment",
            "exact observation deduplication",
        ],
        "dataset_constraint": (
            "No network observations are fabricated by "
            "this module. The Elliptic++ dataset used by "
            "the project does not contain IP, port, ASN, "
            "country, or network-layer timestamps."
        ),
        "report": str(
            REPORT_FILE
        ),
    }

    save_ingestion_report(
        report
    )

    print(
        f"\nModule report: "
        f"{REPORT_FILE}"
    )

    print("\n" + "=" * 72)
    print("M10.2 COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()