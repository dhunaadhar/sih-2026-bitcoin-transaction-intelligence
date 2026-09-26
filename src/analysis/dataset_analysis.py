from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean, median
from typing import Any


def _number(record: dict[str, Any], key: str) -> float | None:
    if key not in record or record.get(key) is None:
        return None

    value = record.get(key)

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _wallets(record: dict[str, Any], key: str) -> list[str]:
    value = record.get(key, [])

    if isinstance(value, list):
        return [
            str(item)
            for item in value
            if item is not None and str(item)
        ]

    if value is None:
        return []

    return [str(value)]


def _percent(value: int | float, total: int) -> float:
    if not total:
        return 0.0

    return round((value / total) * 100, 2)


def _btc_stats(
    values: list[float],
    *,
    available: bool,
) -> dict[str, Any]:
    if not available or not values:
        return {
            "available": False,
            "total": None,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "available": True,
        "total": round(sum(values), 8),
        "mean": round(mean(values), 8),
        "median": round(median(values), 8),
        "min": round(min(values), 8),
        "max": round(max(values), 8),
    }


def _distribution(
    values: list[str],
) -> list[dict[str, Any]]:
    counts = Counter(values)
    total = len(values)

    return [
        {
            "value": value,
            "count": count,
            "percentage": _percent(count, total),
        }
        for value, count in counts.most_common()
    ]


def _field_available(
    records: list[dict[str, Any]],
    key: str,
) -> bool:
    return any(
        key in record
        and record.get(key) is not None
        for record in records
    )


def _wallet_field_available(
    records: list[dict[str, Any]],
    key: str,
) -> bool:
    return any(
        key in record
        and record.get(key) not in (None, [], "")
        for record in records
    )


def analyze_dataset(
    records: list[dict[str, Any]],
    *,
    records_read: int | None = None,
    invalid_records: int = 0,
    duplicates_removed: int = 0,
) -> dict[str, Any]:

    records = [
        record
        for record in records
        if isinstance(record, dict)
    ]

    record_count = len(records)

    txids = [
        str(record["txid"])
        for record in records
        if record.get("txid") is not None
    ]

    input_wallets = [
        wallet
        for record in records
        for wallet in _wallets(record, "input_wallets")
    ]

    output_wallets = [
        wallet
        for record in records
        for wallet in _wallets(record, "output_wallets")
    ]

    all_wallets = set(input_wallets) | set(output_wallets)

    input_field_available = _field_available(
        records,
        "input_amount",
    )

    output_field_available = _field_available(
        records,
        "output_amount",
    )

    fee_field_available = _field_available(
        records,
        "fee",
    )

    input_wallet_field_available = _wallet_field_available(
        records,
        "input_wallets",
    )

    output_wallet_field_available = _wallet_field_available(
        records,
        "output_wallets",
    )

    input_amounts = [
        value
        for record in records
        for value in [_number(record, "input_amount")]
        if value is not None
    ]

    output_amounts = [
        value
        for record in records
        for value in [_number(record, "output_amount")]
        if value is not None
    ]

    fees = [
        value
        for record in records
        for value in [_number(record, "fee")]
        if value is not None
    ]

    timestamps: list[datetime] = []

    for record in records:
        value = record.get("timestamp")

        if not value:
            continue

        try:
            timestamps.append(
                datetime.fromisoformat(
                    str(value).replace("Z", "+00:00")
                )
            )
        except ValueError:
            continue

    input_stats = _btc_stats(
        input_amounts,
        available=input_field_available,
    )

    output_stats = _btc_stats(
        output_amounts,
        available=output_field_available,
    )

    fee_stats = _btc_stats(
        fees,
        available=fee_field_available,
    )

    countries = [
        str(record["geo_country"])
        for record in records
        if record.get("geo_country")
    ]

    asns = [
        str(record["asn"])
        for record in records
        if record.get("asn")
    ]

    script_types = [
        str(record["script_type"])
        for record in records
        if record.get("script_type")
    ]

    src_ips = {
        str(record["src_ip"])
        for record in records
        if record.get("src_ip")
    }

    dst_ips = {
        str(record["dst_ip"])
        for record in records
        if record.get("dst_ip")
    }

    src_ports = set()

    for record in records:
        value = record.get("src_port")

        if value is None:
            continue

        try:
            src_ports.add(int(value))
        except (TypeError, ValueError):
            continue

    dst_ports = set()

    for record in records:
        value = record.get("dst_port")

        if value is None:
            continue

        try:
            dst_ports.add(int(value))
        except (TypeError, ValueError):
            continue

    wallet_activity: Counter[str] = Counter()

    for record in records:
        for wallet in (
            _wallets(record, "input_wallets")
            + _wallets(record, "output_wallets")
        ):
            wallet_activity[wallet] += 1

    wallet_values: dict[str, float] = {}

    if input_field_available or output_field_available:
        for record in records:
            input_value = _number(
                record,
                "input_amount",
            )

            output_value = _number(
                record,
                "output_amount",
            )

            if input_value is not None:
                for wallet in _wallets(
                    record,
                    "input_wallets",
                ):
                    wallet_values[wallet] = (
                        wallet_values.get(wallet, 0.0)
                        + input_value
                    )

            if output_value is not None:
                for wallet in _wallets(
                    record,
                    "output_wallets",
                ):
                    wallet_values[wallet] = (
                        wallet_values.get(wallet, 0.0)
                        + output_value
                    )

    top_wallets = [
        {
            "wallet": wallet,
            "activity_count": count,
            "associated_btc": (
                round(
                    wallet_values.get(wallet, 0.0),
                    8,
                )
                if input_field_available
                or output_field_available
                else None
            ),
        }
        for wallet, count in wallet_activity.most_common(20)
    ]

    top_transactions = []

    for record in records:
        input_value = _number(
            record,
            "input_amount",
        )

        output_value = _number(
            record,
            "output_amount",
        )

        fee_value = _number(
            record,
            "fee",
        )

        top_transactions.append(
            {
                "txid": str(
                    record.get(
                        "txid",
                        "",
                    )
                ),
                "input_amount": (
                    round(input_value, 8)
                    if input_value is not None
                    else None
                ),
                "output_amount": (
                    round(output_value, 8)
                    if output_value is not None
                    else None
                ),
                "fee": (
                    round(fee_value, 8)
                    if fee_value is not None
                    else None
                ),
                "timestamp": record.get(
                    "timestamp"
                ),
            }
        )

    top_transactions.sort(
        key=lambda item: (
            item["input_amount"]
            if item["input_amount"] is not None
            else float("-inf")
        ),
        reverse=True,
    )

    top_transactions = top_transactions[:20]

    temporal_distribution = Counter()

    for timestamp in timestamps:
        temporal_distribution[
            timestamp.strftime("%Y-%m-%d")
        ] += 1

    financial_fields_available = (
        input_field_available
        or output_field_available
        or fee_field_available
    )

    if (
        input_field_available
        and output_field_available
        and fee_field_available
        and input_stats["total"] is not None
        and output_stats["total"] is not None
        and fee_stats["total"] is not None
    ):
        conservation_delta = round(
            input_stats["total"]
            - output_stats["total"]
            - fee_stats["total"],
            8,
        )
    else:
        conservation_delta = None

    return {
        "scope": "imported_dataset",

        "records": {
            "analyzed": record_count,
            "records_read": (
                record_count
                if records_read is None
                else records_read
            ),
            "invalid": invalid_records,
            "duplicates_removed": duplicates_removed,
        },

        "transactions": {
            "unique_txids": len(set(txids)),
        },

        "wallets": {
            "available": (
                input_wallet_field_available
                or output_wallet_field_available
            ),
            "unique_input_wallets": (
                len(set(input_wallets))
                if input_wallet_field_available
                else None
            ),
            "unique_output_wallets": (
                len(set(output_wallets))
                if output_wallet_field_available
                else None
            ),
            "unique_wallets": (
                len(all_wallets)
                if (
                    input_wallet_field_available
                    or output_wallet_field_available
                )
                else None
            ),
            "top_wallets": (
                top_wallets
                if (
                    input_wallet_field_available
                    or output_wallet_field_available
                )
                else []
            ),
        },

        "financial": {
            "available": financial_fields_available,
            "input": input_stats,
            "output": output_stats,
            "fees": fee_stats,
            "conservation_delta": conservation_delta,
        },

        "time": {
            "start": (
                min(timestamps).isoformat()
                if timestamps
                else None
            ),
            "end": (
                max(timestamps).isoformat()
                if timestamps
                else None
            ),
            "records_with_valid_timestamp": len(
                timestamps
            ),
            "daily_distribution": [
                {
                    "date": date,
                    "count": count,
                }
                for date, count in sorted(
                    temporal_distribution.items()
                )
            ],
        },

        "network": {
            "unique_source_ips": len(src_ips),
            "unique_destination_ips": len(dst_ips),
            "unique_source_ports": len(src_ports),
            "unique_destination_ports": len(dst_ports),
        },

        "distributions": {
            "countries": _distribution(countries),
            "asns": _distribution(asns),
            "script_types": _distribution(script_types),
        },

        "top_transactions": top_transactions,

        "availability": {
            "risk_analysis": False,
            "behavioral_analysis": False,
            "graph_analysis": False,
            "financial_analysis": financial_fields_available,
            "wallet_analysis": (
                input_wallet_field_available
                or output_wallet_field_available
            ),
            "reason": (
                "The imported dataset analysis only reports "
                "signals directly supported by the imported "
                "schema. Existing benchmark risk and graph "
                "results are not mixed into case-specific "
                "analysis."
            ),
            "financial_reason": (
                "Financial analysis is available only when "
                "the imported records contain input_amount, "
                "output_amount, or fee fields."
            ),
            "wallet_reason": (
                "Wallet analysis is available only when "
                "the imported records contain input_wallets "
                "or output_wallets fields."
            ),
        },
    }