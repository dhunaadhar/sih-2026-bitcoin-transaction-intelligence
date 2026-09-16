from __future__ import annotations

import csv
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator

from .deduplicator import deduplicate_records
from .provenance import build_provenance
from .schema_validator import validate_record


LOGGER = logging.getLogger(__name__)


FIELD_ALIASES = {
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
}


def _canonical_field_name(name: Any) -> str:
    key = str(name).strip().lower()
    return FIELD_ALIASES.get(key, key)


def _normalize_wallets(value: Any) -> list[str]:
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

    for separator in ("|", ";"):
        if separator in text:
            return [
                item.strip()
                for item in text.split(separator)
                if item.strip()
            ]

    return [text]


def _normalize_number(value: Any, field: str) -> int | float:
    if value is None or value == "":
        raise ValueError(f"{field} cannot be empty")

    if field in INTEGER_FIELDS:
        return int(float(value))

    return float(value)


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError("Record must be a dictionary")

    normalized: dict[str, Any] = {}

    for key, value in record.items():
        canonical_key = _canonical_field_name(key)

        if canonical_key in LIST_FIELDS:
            normalized[canonical_key] = _normalize_wallets(value)

        elif canonical_key in NUMERIC_FIELDS:
            normalized[canonical_key] = _normalize_number(
                value,
                canonical_key,
            )

        elif canonical_key in INTEGER_FIELDS:
            normalized[canonical_key] = _normalize_number(
                value,
                canonical_key,
            )

        elif isinstance(value, str):
            normalized[canonical_key] = value.strip()

        else:
            normalized[canonical_key] = value

    return normalized


def _read_csv(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for record in reader:
            yield dict(record)


def _read_json(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        data = data.get("records", [data])

    if not isinstance(data, list):
        raise ValueError(
            "JSON input must contain an object or array of records"
        )

    for record in data:
        if not isinstance(record, dict):
            raise ValueError(
                "Each JSON record must be an object"
            )

        yield record


def _read_xml(path: Path) -> Iterator[dict[str, Any]]:
    root = ET.parse(path).getroot()

    if root.tag == "record":
        records = [root]
    else:
        records = root.findall(".//record")

    for element in records:
        record: dict[str, Any] = {}

        for child in element:
            record[child.tag] = child.text or ""

        yield record


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)

    readers = {
        ".csv": _read_csv,
        ".json": _read_json,
        ".xml": _read_xml,
    }

    suffix = path.suffix.lower()

    if suffix not in readers:
        raise ValueError(
            f"Unsupported input format: {suffix}"
        )

    yield from readers[suffix](path)


def ingest(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    normalized_valid_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []

    records_read = 0

    try:
        for number, raw in enumerate(
            iter_records(path),
            start=1,
        ):
            records_read += 1

            try:
                record = normalize_record(raw)
                result = validate_record(record)

                if not result.valid:
                    invalid_records.append(
                        {
                            "row": number,
                            "errors": result.errors,
                        }
                    )
                    continue

                normalized_valid_records.append(record)

            except (TypeError, ValueError) as exc:
                invalid_records.append(
                    {
                        "row": number,
                        "errors": (str(exc),),
                    }
                )

    except (OSError, ValueError, ET.ParseError, json.JSONDecodeError) as exc:
        LOGGER.error(
            "Unable to read input file %s: %s",
            path,
            exc,
        )

        provenance = build_provenance(
            path=path,
            records_read=records_read,
            valid_records=0,
            invalid_records=len(invalid_records) + 1,
            duplicates_removed=0,
        )

        return {
            "records": [],
            "invalid_records": invalid_records
            + [
                {
                    "row": None,
                    "errors": (str(exc),),
                }
            ],
            "provenance": provenance,
        }

    unique_records, duplicates_removed = deduplicate_records(
        normalized_valid_records
    )

    provenance = build_provenance(
        path=path,
        records_read=records_read,
        valid_records=len(unique_records),
        invalid_records=len(invalid_records),
        duplicates_removed=duplicates_removed,
    )

    LOGGER.info(
        "Ingestion complete: %s",
        provenance,
    )

    return {
        "records": unique_records,
        "invalid_records": invalid_records,
        "provenance": provenance,
    }