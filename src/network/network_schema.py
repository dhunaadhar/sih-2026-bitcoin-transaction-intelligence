from __future__ import annotations

import ipaddress
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    ROOT
    / "reports"
    / "network"
)

REPORT_FILE = (
    REPORT_DIR
    / "m10_1_network_schema.json"
)


REQUIRED_COLUMNS = [
    "timestamp",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "txid",
    "geo_country",
    "asn",
]

OPTIONAL_COLUMNS = [
    "network_event_id",
    "protocol",
    "direction",
    "observation_source",
]


CANONICAL_COLUMNS = [
    "network_event_id",
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


def validate_ip(value: object) -> bool:
    if pd.isna(value):
        return False

    try:
        ipaddress.ip_address(
            str(value).strip()
        )
        return True

    except ValueError:
        return False


def validate_port(value: object) -> bool:
    if pd.isna(value):
        return False

    try:
        port = int(value)

    except (ValueError, TypeError):
        return False

    return 0 <= port <= 65535


def normalize_country(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    if not text:
        return None

    return text


def normalize_asn(value: object) -> int | None:
    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    if text.startswith("AS"):
        text = text[2:]

    try:
        asn = int(float(text))

    except (ValueError, TypeError):
        return None

    if asn < 0:
        return None

    return asn


def normalize_port(value: object) -> int | None:
    if pd.isna(value):
        return None

    try:
        port = int(value)

    except (ValueError, TypeError):
        return None

    if not 0 <= port <= 65535:
        return None

    return port


def normalize_ip(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        return str(
            ipaddress.ip_address(text)
        )

    except ValueError:
        return None


def normalize_timestamp(
    series: pd.Series,
) -> pd.Series:

    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )


def normalize_txid(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        numeric = float(text)

        if numeric.is_integer():
            return str(int(numeric))

    except (ValueError, TypeError):
        pass

    return text


def normalize_network_dataframe(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    frame = frame.copy()

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Network dataset is missing required "
            f"columns: {missing}"
        )

    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = None

    frame["timestamp"] = (
        normalize_timestamp(
            frame["timestamp"]
        )
    )

    frame["src_ip"] = (
        frame["src_ip"]
        .map(normalize_ip)
    )

    frame["dst_ip"] = (
        frame["dst_ip"]
        .map(normalize_ip)
    )

    frame["src_port"] = (
        frame["src_port"]
        .map(normalize_port)
    )

    frame["dst_port"] = (
        frame["dst_port"]
        .map(normalize_port)
    )

    frame["txid"] = (
        frame["txid"]
        .map(normalize_txid)
    )

    frame["geo_country"] = (
        frame["geo_country"]
        .map(normalize_country)
    )

    frame["asn"] = (
        frame["asn"]
        .map(normalize_asn)
    )

    frame["network_event_id"] = (
        frame["network_event_id"]
        .astype("string")
    )

    frame["protocol"] = (
        frame["protocol"]
        .astype("string")
    )

    frame["direction"] = (
        frame["direction"]
        .astype("string")
    )

    frame["observation_source"] = (
        frame["observation_source"]
        .astype("string")
    )

    frame = frame[
        CANONICAL_COLUMNS
    ]

    validation = {
        "rows": int(
            len(frame)
        ),
        "columns": int(
            len(frame.columns)
        ),
        "null_counts": {
            column: int(
                frame[column].isna().sum()
            )
            for column in CANONICAL_COLUMNS
        },
        "invalid_required_fields": {
            "timestamp": int(
                frame["timestamp"].isna().sum()
            ),
            "src_ip": int(
                frame["src_ip"].isna().sum()
            ),
            "src_port": int(
                frame["src_port"].isna().sum()
            ),
            "dst_ip": int(
                frame["dst_ip"].isna().sum()
            ),
            "dst_port": int(
                frame["dst_port"].isna().sum()
            ),
            "txid": int(
                frame["txid"].isna().sum()
            ),
            "geo_country": int(
                frame["geo_country"].isna().sum()
            ),
            "asn": int(
                frame["asn"].isna().sum()
            ),
        },
        "unique_txids": int(
            frame["txid"].nunique(
                dropna=True
            )
        ),
        "unique_source_ips": int(
            frame["src_ip"].nunique(
                dropna=True
            )
        ),
        "unique_destination_ips": int(
            frame["dst_ip"].nunique(
                dropna=True
            )
        ),
        "unique_asns": int(
            frame["asn"].nunique(
                dropna=True
            )
        ),
        "unique_countries": int(
            frame["geo_country"].nunique(
                dropna=True
            )
        ),
    }

    return frame, validation


def main() -> None:

    print("=" * 72)
    print("M10.1 NETWORK SCHEMA")
    print("=" * 72)

    print(
        "\nRequired network fields:"
    )

    for column in REQUIRED_COLUMNS:
        print(
            f"  - {column}"
        )

    print(
        "\nOptional network fields:"
    )

    for column in OPTIONAL_COLUMNS:
        print(
            f"  - {column}"
        )

    print(
        "\nCanonical network schema:"
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
        "milestone": "M10.1",
        "status": "PASS",
        "purpose": (
            "Define the canonical network-observation "
            "schema required for offline network/blockchain "
            "correlation."
        ),
        "required_columns": REQUIRED_COLUMNS,
        "optional_columns": OPTIONAL_COLUMNS,
        "canonical_columns": CANONICAL_COLUMNS,
        "validation_rules": {
            "timestamp": (
                "Parseable timestamp normalized to UTC."
            ),
            "src_ip": (
                "Valid IPv4 or IPv6 address."
            ),
            "dst_ip": (
                "Valid IPv4 or IPv6 address."
            ),
            "src_port": (
                "Integer in range 0-65535."
            ),
            "dst_port": (
                "Integer in range 0-65535."
            ),
            "txid": (
                "Non-empty transaction identifier."
            ),
            "geo_country": (
                "Normalized uppercase country identifier."
            ),
            "asn": (
                "Non-negative numeric ASN, optionally "
                "prefixed with AS."
            ),
        },
        "dataset_constraint": (
            "The Elliptic++ source used by this project "
            "does not contain IP, port, ASN, country, or "
            "network-layer timestamps. This schema therefore "
            "defines an ingestion interface and does not "
            "represent observed network measurements."
        ),
        "provenance_requirement": (
            "Every network observation should identify its "
            "observation_source so real, synthetic, replayed, "
            "or externally supplied data are distinguishable."
        ),
        "report": str(
            REPORT_FILE
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
        "\nSchema definition: PASS"
    )

    print(
        f"Report: {REPORT_FILE}"
    )

    print("\n" + "=" * 72)
    print("M10.1 COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()