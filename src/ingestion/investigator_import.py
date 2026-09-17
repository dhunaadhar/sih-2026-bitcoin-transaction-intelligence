"""
M3.5 — Investigator Dataset Import Engine

Purpose
-------
Provide a reusable offline import engine for investigator-supplied
case data.

Supported formats
-----------------
    CSV
    JSON
    XML

Supported investigator data types
----------------------------------
    BLOCKCHAIN_TRANSACTION
    NETWORK_OBSERVATION
    IP_INTELLIGENCE

Design principles
-----------------
    - Offline-first
    - Explicit schema validation
    - Normalization before validation
    - Exact-record deduplication
    - Full source provenance
    - No silent rejection of invalid records
    - No modification of the existing M3 ingestion pipeline
    - Case-ready import metadata
"""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .deduplicator import deduplicate_records
from .provenance import file_sha256


# ============================================================================
# Constants
# ============================================================================

SUPPORTED_FORMATS = {
    ".csv",
    ".json",
    ".xml",
}


DATA_TYPES = {
    "BLOCKCHAIN_TRANSACTION",
    "NETWORK_OBSERVATION",
    "IP_INTELLIGENCE",
}


COMMON_ALIASES = {
    "time": "timestamp",
    "time_stamp": "timestamp",
    "datetime": "timestamp",
    "tx_id": "txid",
    "transaction_id": "txid",
    "source_ip": "src_ip",
    "sourceip": "src_ip",
    "destination_ip": "dst_ip",
    "destinationip": "dst_ip",
    "source_port": "src_port",
    "sourceport": "src_port",
    "destination_port": "dst_port",
    "destinationport": "dst_port",
    "fee_btc": "fee",
    "input_amount_btc": "input_amount",
    "output_amount_btc": "output_amount",
}


FIELD_ALIASES = {
    **COMMON_ALIASES,

    # Blockchain
    "input_addresses": "input_wallets",
    "input_address": "input_wallets",
    "input_wallet": "input_wallets",
    "output_addresses": "output_wallets",
    "output_address": "output_wallets",
    "output_wallet": "output_wallets",

    # IP intelligence
    "networktype": "network_type",
    "network_type_name": "network_type",
    "provider_name": "provider",
    "confidence_score": "confidence",
    "validfrom": "valid_from",
    "validto": "valid_to",

    # Network
    "country": "geo_country",
}


BLOCKCHAIN_REQUIRED_FIELDS = {
    "txid",
    "timestamp",
    "input_wallets",
    "output_wallets",
    "input_amount",
    "output_amount",
    "fee",
}


NETWORK_REQUIRED_FIELDS = {
    "timestamp",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "txid",
}


IP_INTELLIGENCE_REQUIRED_FIELDS = {
    "ip",
    "network_type",
    "confidence",
}


LIST_FIELDS = {
    "input_wallets",
    "output_wallets",
}


INTEGER_FIELDS = {
    "src_port",
    "dst_port",
}


NUMERIC_FIELDS = {
    "input_amount",
    "output_amount",
    "fee",
    "confidence",
}


IP_FIELDS = {
    "src_ip",
    "dst_ip",
    "ip",
}


TIMESTAMP_FIELDS = {
    "timestamp",
    "valid_from",
    "valid_to",
}


# ============================================================================
# Data classes
# ============================================================================

@dataclass(frozen=True)
class RecordValidation:
    valid: bool
    missing_fields: tuple[str, ...]
    invalid_fields: tuple[str, ...]
    errors: tuple[str, ...]


# ============================================================================
# General utilities
# ============================================================================

def canonical_field_name(name: Any) -> str:
    key = str(name).strip().lower()

    key = key.replace("-", "_")
    key = key.replace(" ", "_")

    return FIELD_ALIASES.get(
        key,
        key,
    )


def normalize_txid(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        try:
            numeric = float(text)

            if numeric.is_integer():
                return str(int(numeric))

        except ValueError:
            pass

    return text


def normalize_wallets(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    text = str(value).strip()

    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]

        except json.JSONDecodeError:
            pass

    for separator in ("|", ";", ","):
        if separator in text:
            return [
                item.strip()
                for item in text.split(separator)
                if item.strip()
            ]

    return [text]


def normalize_number(
    value: Any,
    field: str,
) -> int | float:
    if value is None:
        raise ValueError(
            f"{field} cannot be empty"
        )

    if isinstance(value, str):
        value = value.strip()

        if not value:
            raise ValueError(
                f"{field} cannot be empty"
            )

    if field in INTEGER_FIELDS:
        numeric = float(value)

        if not numeric.is_integer():
            raise ValueError(
                f"{field} must be an integer"
            )

        return int(numeric)

    return float(value)


def normalize_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError(
            "Record must be a dictionary."
        )

    normalized: dict[str, Any] = {}

    for key, value in record.items():
        field = canonical_field_name(key)

        if field in LIST_FIELDS:
            normalized[field] = normalize_wallets(
                value
            )

        elif field in NUMERIC_FIELDS:
            normalized[field] = normalize_number(
                value,
                field,
            )

        elif field in INTEGER_FIELDS:
            normalized[field] = normalize_number(
                value,
                field,
            )

        elif field == "txid":
            normalized[field] = normalize_txid(
                value
            )

        elif field in IP_FIELDS:
            normalized[field] = (
                str(value).strip()
                if value is not None
                else ""
            )

        elif isinstance(value, str):
            normalized[field] = value.strip()

        else:
            normalized[field] = value

    return normalized


# ============================================================================
# File readers
# ============================================================================

def read_csv(
    path: Path,
) -> Iterator[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(
                "CSV file does not contain a header row."
            )

        for row in reader:
            yield dict(row)


def read_json(
    path: Path,
) -> Iterator[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        data = json.load(handle)

    if isinstance(data, dict):

        if isinstance(
            data.get("records"),
            list,
        ):
            data = data["records"]

        elif isinstance(
            data.get("data"),
            list,
        ):
            data = data["data"]

        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError(
            "JSON input must contain an object or an array of objects."
        )

    for record in data:
        if not isinstance(record, dict):
            raise ValueError(
                "Every JSON record must be an object."
            )

        yield record


def read_xml(
    path: Path,
) -> Iterator[dict[str, Any]]:
    tree = ET.parse(path)
    root = tree.getroot()

    elements = []

    if root.tag.lower() in {
        "record",
        "transaction",
        "observation",
        "ip",
        "item",
    }:
        elements.append(root)

    elements.extend(
        root.findall(".//record")
    )

    elements.extend(
        root.findall(".//transaction")
    )

    elements.extend(
        root.findall(".//observation")
    )

    elements.extend(
        root.findall(".//item")
    )

    # Remove duplicate element references while preserving order.
    unique_elements = []
    seen_ids = set()

    for element in elements:
        identifier = id(element)

        if identifier not in seen_ids:
            seen_ids.add(identifier)
            unique_elements.append(element)

    if not unique_elements:
        raise ValueError(
            "XML input does not contain recognizable record elements."
        )

    for element in unique_elements:
        record: dict[str, Any] = {}

        for child in element:
            tag = child.tag

            if "}" in tag:
                tag = tag.split(
                    "}",
                    1,
                )[1]

            record[tag] = child.text or ""

        yield record


def iter_input_records(
    path: str | Path,
) -> Iterator[dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Input path is not a file: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        yield from read_csv(path)

    elif suffix == ".json":
        yield from read_json(path)

    elif suffix == ".xml":
        yield from read_xml(path)

    else:
        raise ValueError(
            f"Unsupported input format: {suffix}. "
            f"Supported formats: CSV, JSON, XML."
        )


# ============================================================================
# Validation helpers
# ============================================================================

def valid_timestamp(
    value: Any,
) -> bool:
    if not isinstance(
        value,
        str,
    ):
        return False

    if not value.strip():
        return False

    try:
        datetime.fromisoformat(
            value.replace(
                "Z",
                "+00:00",
            )
        )

        return True

    except ValueError:
        return False


def valid_ip(
    value: Any,
) -> bool:
    if not isinstance(
        value,
        str,
    ):
        return False

    if not value.strip():
        return False

    try:
        ipaddress.ip_address(
            value
        )

        return True

    except ValueError:
        return False


def valid_port(
    value: Any,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return False

    if not isinstance(
        value,
        int,
    ):
        return False

    return 0 <= value <= 65535


def valid_number(
    value: Any,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return False

    return isinstance(
        value,
        (int, float),
    )


def valid_wallet_list(
    value: Any,
) -> bool:
    return (
        isinstance(
            value,
            list,
        )
        and all(
            isinstance(
                item,
                str,
            )
            and item.strip()
            for item in value
        )
    )


def validate_common_fields(
    record: dict[str, Any],
) -> tuple[set[str], set[str], list[str]]:
    invalid: set[str] = set()
    errors: list[str] = []

    for field in TIMESTAMP_FIELDS:

        if field not in record:
            continue

        value = record[field]

        if value in (
            None,
            "",
        ):
            continue

        if not valid_timestamp(value):
            invalid.add(field)

    for field in IP_FIELDS:

        if field not in record:
            continue

        value = record[field]

        if value in (
            None,
            "",
        ):
            continue

        if not valid_ip(value):
            invalid.add(field)

    for field in INTEGER_FIELDS:

        if field not in record:
            continue

        value = record[field]

        if not valid_port(value):
            invalid.add(field)

    for field in NUMERIC_FIELDS:

        if field not in record:
            continue

        value = record[field]

        if not valid_number(value):
            invalid.add(field)

    return (
        invalid,
        set(),
        errors,
    )


# ============================================================================
# Type-specific validation
# ============================================================================

def validate_blockchain_record(
    record: dict[str, Any],
) -> RecordValidation:

    required = BLOCKCHAIN_REQUIRED_FIELDS

    missing = tuple(
        sorted(
            required - set(record)
        )
    )

    invalid: set[str] = set()
    errors: list[str] = []

    common_invalid, _, common_errors = (
        validate_common_fields(record)
    )

    invalid.update(
        common_invalid
    )

    errors.extend(
        common_errors
    )

    if "txid" in record:
        if not normalize_txid(
            record["txid"]
        ):
            invalid.add("txid")

    if "timestamp" in record:
        if not valid_timestamp(
            record["timestamp"]
        ):
            invalid.add("timestamp")

    for field in (
        "input_wallets",
        "output_wallets",
    ):
        if field in record:
            if not valid_wallet_list(
                record[field]
            ):
                invalid.add(field)

    for field in (
        "input_amount",
        "output_amount",
        "fee",
    ):
        if field in record:
            if not valid_number(
                record[field]
            ):
                invalid.add(field)
            elif float(
                record[field]
            ) < 0:
                invalid.add(field)

    if missing:
        errors.append(
            "Missing required fields: "
            + ", ".join(missing)
        )

    if invalid:
        errors.append(
            "Invalid field values/types: "
            + ", ".join(
                sorted(invalid)
            )
        )

    return RecordValidation(
        valid=(
            not missing
            and not invalid
        ),
        missing_fields=missing,
        invalid_fields=tuple(
            sorted(invalid)
        ),
        errors=tuple(errors),
    )


def validate_network_record(
    record: dict[str, Any],
) -> RecordValidation:

    required = NETWORK_REQUIRED_FIELDS

    missing = tuple(
        sorted(
            required - set(record)
        )
    )

    invalid: set[str] = set()
    errors: list[str] = []

    if "timestamp" in record:
        if not valid_timestamp(
            record["timestamp"]
        ):
            invalid.add("timestamp")

    for field in (
        "src_ip",
        "dst_ip",
    ):
        if field in record:
            if not valid_ip(
                record[field]
            ):
                invalid.add(field)

    for field in (
        "src_port",
        "dst_port",
    ):
        if field in record:
            if not valid_port(
                record[field]
            ):
                invalid.add(field)

    if "txid" in record:
        if not normalize_txid(
            record["txid"]
        ):
            invalid.add("txid")

    if missing:
        errors.append(
            "Missing required fields: "
            + ", ".join(missing)
        )

    if invalid:
        errors.append(
            "Invalid field values/types: "
            + ", ".join(
                sorted(invalid)
            )
        )

    return RecordValidation(
        valid=(
            not missing
            and not invalid
        ),
        missing_fields=missing,
        invalid_fields=tuple(
            sorted(invalid)
        ),
        errors=tuple(errors),
    )


def validate_ip_intelligence_record(
    record: dict[str, Any],
) -> RecordValidation:

    required = IP_INTELLIGENCE_REQUIRED_FIELDS

    missing = tuple(
        sorted(
            required - set(record)
        )
    )

    invalid: set[str] = set()
    errors: list[str] = []

    if "ip" in record:
        if not valid_ip(
            record["ip"]
        ):
            invalid.add("ip")

    if "network_type" in record:
        value = record["network_type"]

        if not isinstance(
            value,
            str,
        ) or not value.strip():
            invalid.add(
                "network_type"
            )

    if "confidence" in record:
        if not valid_number(
            record["confidence"]
        ):
            invalid.add("confidence")
        elif not (
            0.0
            <= float(
                record["confidence"]
            )
            <= 1.0
        ):
            invalid.add("confidence")

    for field in (
        "valid_from",
        "valid_to",
    ):
        if field in record:
            value = record[field]

            if value not in (
                None,
                "",
            ):
                if not valid_timestamp(
                    value
                ):
                    invalid.add(field)

    if missing:
        errors.append(
            "Missing required fields: "
            + ", ".join(missing)
        )

    if invalid:
        errors.append(
            "Invalid field values/types: "
            + ", ".join(
                sorted(invalid)
            )
        )

    return RecordValidation(
        valid=(
            not missing
            and not invalid
        ),
        missing_fields=missing,
        invalid_fields=tuple(
            sorted(invalid)
        ),
        errors=tuple(errors),
    )


def validate_record(
    record: dict[str, Any],
    data_type: str,
) -> RecordValidation:

    if data_type == "BLOCKCHAIN_TRANSACTION":
        return validate_blockchain_record(
            record
        )

    if data_type == "NETWORK_OBSERVATION":
        return validate_network_record(
            record
        )

    if data_type == "IP_INTELLIGENCE":
        return validate_ip_intelligence_record(
            record
        )

    raise ValueError(
        f"Unsupported investigator data type: {data_type}"
    )


# ============================================================================
# Import manifest
# ============================================================================

def build_import_id(
    source_sha256: str,
    data_type: str,
) -> str:

    payload = (
        source_sha256
        + "|"
        + data_type
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]


def build_import_manifest(
    path: Path,
    data_type: str,
    records_read: int,
    valid_records: int,
    invalid_records: int,
    duplicates_removed: int,
    source_sha256: str,
    import_id: str,
) -> dict[str, Any]:

    return {
        "import_id": import_id,
        "data_type": data_type,
        "source_file": str(
            path.resolve()
        ),
        "source_filename": path.name,
        "source_format": (
            path.suffix
            .lower()
            .lstrip(".")
        ),
        "source_sha256": source_sha256,
        "imported_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "records_read": records_read,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "duplicate_records_removed": (
            duplicates_removed
        ),
        "unique_valid_records": (
            valid_records
        ),
        "offline_processing": True,
        "provenance_available": True,
    }


# ============================================================================
# Main import engine
# ============================================================================

def import_investigator_data(
    path: str | Path,
    data_type: str,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:

    path = Path(path)

    data_type = str(
        data_type
    ).strip().upper()

    if data_type not in DATA_TYPES:
        raise ValueError(
            f"Unsupported data type: {data_type}. "
            f"Supported types: "
            f"{sorted(DATA_TYPES)}"
        )

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {path.suffix}. "
            f"Supported formats: CSV, JSON, XML."
        )

    source_sha256 = file_sha256(
        path
    )

    import_id = build_import_id(
        source_sha256,
        data_type,
    )

    valid_records: list[
        dict[str, Any]
    ] = []

    invalid_records: list[
        dict[str, Any]
    ] = []

    records_read = 0

    try:

        for row_number, raw_record in enumerate(
            iter_input_records(path),
            start=1,
        ):

            records_read += 1

            try:

                normalized = normalize_record(
                    raw_record
                )

                validation = validate_record(
                    normalized,
                    data_type,
                )

                if validation.valid:

                    valid_records.append(
                        normalized
                    )

                else:

                    invalid_records.append(
                        {
                            "row": row_number,
                            "missing_fields": (
                                list(
                                    validation.missing_fields
                                )
                            ),
                            "invalid_fields": (
                                list(
                                    validation.invalid_fields
                                )
                            ),
                            "errors": (
                                list(
                                    validation.errors
                                )
                            ),
                            "record": normalized,
                        }
                    )

            except (
                TypeError,
                ValueError,
            ) as exc:

                invalid_records.append(
                    {
                        "row": row_number,
                        "missing_fields": [],
                        "invalid_fields": [],
                        "errors": [
                            str(exc)
                        ],
                        "record": raw_record,
                    }
                )

    except (
        OSError,
        ValueError,
        ET.ParseError,
        json.JSONDecodeError,
    ) as exc:

        invalid_records.append(
            {
                "row": None,
                "missing_fields": [],
                "invalid_fields": [],
                "errors": [
                    str(exc)
                ],
                "record": None,
            }
        )

    unique_records, duplicates_removed = (
        deduplicate_records(
            valid_records
        )
    )

    manifest = build_import_manifest(
        path=path,
        data_type=data_type,
        records_read=records_read,
        valid_records=len(
            unique_records
        ),
        invalid_records=len(
            invalid_records
        ),
        duplicates_removed=(
            duplicates_removed
        ),
        source_sha256=source_sha256,
        import_id=import_id,
    )

    result = {
        "status": (
            "PASS"
            if not invalid_records
            else "WARNING"
        ),
        "import_id": import_id,
        "data_type": data_type,
        "records": unique_records,
        "invalid_records": invalid_records,
        "provenance": manifest,
        "statistics": {
            "records_read": records_read,
            "valid_records_before_deduplication": (
                len(valid_records)
            ),
            "valid_records": len(
                unique_records
            ),
            "invalid_records": len(
                invalid_records
            ),
            "duplicates_removed": (
                duplicates_removed
            ),
        },
    }

    if output_directory is not None:

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_import_artifacts(
            result,
            output_directory,
        )

    return result


# ============================================================================
# Artifact writing
# ============================================================================

def write_json(
    path: Path,
    data: Any,
) -> None:

    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_import_artifacts(
    result: dict[str, Any],
    output_directory: Path,
) -> None:

    import_id = result[
        "import_id"
    ]

    records_path = (
        output_directory
        / f"{import_id}_normalized_records.json"
    )

    errors_path = (
        output_directory
        / f"{import_id}_validation_errors.json"
    )

    manifest_path = (
        output_directory
        / f"{import_id}_manifest.json"
    )

    write_json(
        records_path,
        result["records"],
    )

    write_json(
        errors_path,
        result["invalid_records"],
    )

    write_json(
        manifest_path,
        result["provenance"],
    )


# ============================================================================
# CLI
# ============================================================================

def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "M3.5 Investigator Dataset Import Engine"
        )
    )

    parser.add_argument(
        "file",
        help=(
            "Path to investigator-supplied "
            "CSV, JSON or XML file."
        ),
    )

    parser.add_argument(
        "--type",
        required=True,
        choices=sorted(DATA_TYPES),
        help=(
            "Investigator data type."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output directory "
            "for normalized records, "
            "validation errors and manifest."
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 72)
    print(
        "M3.5 INVESTIGATOR DATASET IMPORT ENGINE"
    )
    print("=" * 72)

    print(
        f"Input: {args.file}"
    )

    print(
        f"Data type: {args.type}"
    )

    result = import_investigator_data(
        path=args.file,
        data_type=args.type,
        output_directory=args.output,
    )

    print()
    print(
        f"IMPORT ID: "
        f"{result['import_id']}"
    )

    print(
        f"STATUS: "
        f"{result['status']}"
    )

    print(
        f"RECORDS READ: "
        f"{result['statistics']['records_read']:,}"
    )

    print(
        f"VALID RECORDS: "
        f"{result['statistics']['valid_records']:,}"
    )

    print(
        f"INVALID RECORDS: "
        f"{result['statistics']['invalid_records']:,}"
    )

    print(
        f"DUPLICATES REMOVED: "
        f"{result['statistics']['duplicates_removed']:,}"
    )

    print(
        f"SOURCE SHA256: "
        f"{result['provenance']['source_sha256']}"
    )

    if args.output:
        print()
        print(
            f"Artifacts: "
            f"{Path(args.output).resolve()}"
        )

    print()
    print(
        "M3.5 INVESTIGATOR DATASET IMPORT COMPLETE"
    )


if __name__ == "__main__":
    main()