from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import ipaddress


REQUIRED_FIELDS = {
    "timestamp",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "txid",
    "input_wallets",
    "output_wallets",
    "input_amount",
    "output_amount",
    "fee",
    "script_type",
    "geo_country",
    "asn",
}

INTEGER_FIELDS = {"src_port", "dst_port"}

NUMERIC_FIELDS = {
    "input_amount",
    "output_amount",
    "fee",
}

STRING_FIELDS = {
    "timestamp",
    "src_ip",
    "dst_ip",
    "txid",
    "script_type",
    "geo_country",
    "asn",
}

LIST_FIELDS = {
    "input_wallets",
    "output_wallets",
}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    missing_fields: tuple[str, ...]
    invalid_fields: tuple[str, ...]
    errors: tuple[str, ...]


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False

    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _valid_ip(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False

    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _valid_port(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 65535
    )


def _valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_wallet_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(
            isinstance(item, str) and item.strip()
            for item in value
        )
    )


def validate_record(record: dict[str, Any]) -> ValidationResult:
    if not isinstance(record, dict):
        return ValidationResult(
            valid=False,
            missing_fields=tuple(sorted(REQUIRED_FIELDS)),
            invalid_fields=(),
            errors=("Record must be a dictionary.",),
        )

    missing = tuple(sorted(REQUIRED_FIELDS - record.keys()))
    invalid: set[str] = set()
    errors: list[str] = []

    if missing:
        errors.append(
            "Missing required fields: " + ", ".join(missing)
        )

    if "timestamp" in record:
        if not _valid_timestamp(record["timestamp"]):
            invalid.add("timestamp")

    for field in ("src_ip", "dst_ip"):
        if field in record:
            if not _valid_ip(record[field]):
                invalid.add(field)

    for field in INTEGER_FIELDS:
        if field in record:
            if not _valid_port(record[field]):
                invalid.add(field)

    if "txid" in record:
        value = record["txid"]

        if not (
            isinstance(value, (int, str))
            and not isinstance(value, bool)
            and str(value).strip()
        ):
            invalid.add("txid")

    for field in NUMERIC_FIELDS:
        if field in record:
            if not _valid_number(record[field]):
                invalid.add(field)

    for field in LIST_FIELDS:
        if field in record:
            if not _valid_wallet_list(record[field]):
                invalid.add(field)

    for field in STRING_FIELDS - {
        "timestamp",
        "src_ip",
        "dst_ip",
        "txid",
    }:
        if field in record:
            value = record[field]

            if not isinstance(value, str) or not value.strip():
                invalid.add(field)

    if invalid:
        errors.append(
            "Invalid field values/types: "
            + ", ".join(sorted(invalid))
        )

    return ValidationResult(
        valid=not missing and not invalid,
        missing_fields=missing,
        invalid_fields=tuple(sorted(invalid)),
        errors=tuple(errors),
    )