from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    ROOT / "data" / "raw" / "network" / "network_intelligence_feed.csv"
)

DEFAULT_OUTPUT = (
    ROOT / "data" / "derived" / "network_intelligence_feed.parquet"
)

DEFAULT_REPORT = (
    ROOT / "reports" / "network" / "m13_2_network_intelligence_feed.json"
)


REQUIRED_COLUMNS = {
    "ip",
    "network_type",
    "provider",
    "confidence",
    "source",
}

OPTIONAL_COLUMNS = {
    "valid_from",
    "valid_to",
    "asn",
    "country",
}


ALLOWED_TYPES = {
    "VPN",
    "PROXY",
    "TOR_EXIT",
    "TOR_RELAY",
    "HOSTING",
    "DATACENTER",
    "RESIDENTIAL",
    "MOBILE",
    "EDUCATIONAL",
    "GOVERNMENT",
    "UNKNOWN",
}


def normalize_ip(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return None


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def normalize_network_type(value: Any) -> str:
    text = normalize_text(value)

    if text is None:
        return "UNKNOWN"

    text = text.upper()

    aliases = {
        "TOR": "TOR_EXIT",
        "TOR EXIT": "TOR_EXIT",
        "TOR RELAY": "TOR_RELAY",
        "EXIT NODE": "TOR_EXIT",
        "TOR EXIT NODE": "TOR_EXIT",
        "DATA CENTER": "DATACENTER",
        "DATA-CENTER": "DATACENTER",
    }

    text = aliases.get(text, text)

    if text not in ALLOWED_TYPES:
        raise ValueError(
            f"Unsupported network_type '{value}'. "
            f"Allowed values: {sorted(ALLOWED_TYPES)}"
        )

    return text


def normalize_confidence(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return 0.0

    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid confidence value: {value}"
        ) from exc

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Confidence must be between 0 and 1: {value}"
        )

    return value


def normalize_asn(value: Any) -> str | None:
    text = normalize_text(value)

    if text is None:
        return None

    text_upper = text.upper()

    if text_upper.startswith("AS"):
        numeric = text_upper[2:]
    else:
        numeric = text_upper

    if not numeric.isdigit():
        raise ValueError(
            f"Invalid ASN value: {value}"
        )

    return f"AS{numeric}"


def normalize_country(value: Any) -> str | None:
    text = normalize_text(value)

    if text is None:
        return None

    text = text.upper()

    if len(text) != 2 or not text.isalpha():
        raise ValueError(
            f"Country must be an ISO-3166 alpha-2 code: {value}"
        )

    return text


def build_record_id(row: pd.Series) -> str:
    fields = [
        str(row["ip"]),
        str(row["network_type"]),
        str(row["provider"] or ""),
        f"{float(row['confidence']):.8f}",
        str(row["source"] or ""),
        str(row["valid_from"] or ""),
        str(row["valid_to"] or ""),
        str(row["asn"] or ""),
        str(row["country"] or ""),
    ]

    payload = "|".join(fields)

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def ingest_feed(
    input_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Offline intelligence feed not found:\n{input_path}"
        )

    raw = pd.read_csv(input_path)

    missing = REQUIRED_COLUMNS - set(raw.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Add optional columns if absent.
    for column in OPTIONAL_COLUMNS:
        if column not in raw.columns:
            raw[column] = None

    normalized = pd.DataFrame()

    normalized["ip"] = raw["ip"].map(normalize_ip)

    invalid_ip_count = int(normalized["ip"].isna().sum())

    if invalid_ip_count:
        raise ValueError(
            f"Feed contains {invalid_ip_count} invalid IP records."
        )

    normalized["network_type"] = raw[
        "network_type"
    ].map(normalize_network_type)

    normalized["provider"] = raw[
        "provider"
    ].map(normalize_text)

    normalized["confidence"] = raw[
        "confidence"
    ].map(normalize_confidence)

    normalized["source"] = raw[
        "source"
    ].map(normalize_text)

    normalized["valid_from"] = raw[
        "valid_from"
    ].map(normalize_text)

    normalized["valid_to"] = raw[
        "valid_to"
    ].map(normalize_text)

    normalized["asn"] = raw[
        "asn"
    ].map(normalize_asn)

    normalized["country"] = raw[
        "country"
    ].map(normalize_country)

    normalized["record_id"] = normalized.apply(
        build_record_id,
        axis=1,
    )

    duplicate_mask = normalized.duplicated(
        subset=["record_id"],
        keep="first",
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    normalized = normalized.loc[
        ~duplicate_mask
    ].reset_index(drop=True)

    # Conflicting records for the same IP are allowed because
    # providers may supply multiple intelligence observations.
    conflicting_ip_count = 0

    for ip, group in normalized.groupby("ip"):
        if len(group) <= 1:
            continue

        classifications = set(
            group["network_type"].tolist()
        )

        if len(classifications) > 1:
            conflicting_ip_count += 1

    report = {
        "input_path": str(input_path),
        "raw_rows": int(len(raw)),
        "normalized_rows": int(len(normalized)),
        "duplicate_records_removed": duplicate_count,
        "unique_ips": int(normalized["ip"].nunique()),
        "conflicting_ip_classification_groups": conflicting_ip_count,
        "network_type_distribution": {
            str(k): int(v)
            for k, v in normalized[
                "network_type"
            ].value_counts().to_dict().items()
        },
        "status": "PASS",
    }

    return normalized, report


def validate_feed(
    df: pd.DataFrame,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    checks["rows_present"] = len(df) > 0

    checks["required_columns"] = REQUIRED_COLUMNS.issubset(
        set(df.columns)
    )

    checks["unique_record_ids"] = (
        df["record_id"].nunique() == len(df)
    )

    checks["valid_ips"] = bool(
        df["ip"].map(normalize_ip).notna().all()
    )

    checks["valid_network_types"] = bool(
        df["network_type"].isin(
            ALLOWED_TYPES
        ).all()
    )

    checks["confidence_range"] = bool(
        (
            (df["confidence"] >= 0.0)
            & (df["confidence"] <= 1.0)
        ).all()
    )

    checks["sources_present"] = bool(
        df["source"].notna().all()
        & (df["source"].astype(str).str.len() > 0).all()
    )

    checks["providers_optional"] = True

    checks["asn_values_valid"] = bool(
        df["asn"].dropna().map(normalize_asn).notna().all()
    )

    checks["country_values_valid"] = bool(
        df["country"].dropna().map(normalize_country).notna().all()
    )

    passed = all(checks.values())

    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
    }


def run(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
) -> None:
    print("=" * 72)
    print("M13.2 OFFLINE NETWORK INTELLIGENCE FEED")
    print("=" * 72)

    print(f"Input: {input_path}")

    df, ingestion_report = ingest_feed(
        input_path
    )

    print(
        f"Raw records: "
        f"{ingestion_report['raw_rows']:,}"
    )

    print(
        f"Normalized records: "
        f"{ingestion_report['normalized_rows']:,}"
    )

    print(
        f"Unique IPs: "
        f"{ingestion_report['unique_ips']:,}"
    )

    validation = validate_feed(df)

    print("\nValidation:")

    for name, passed in validation["checks"].items():
        print(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if validation["status"] != "PASS":
        raise RuntimeError(
            "M13.2 feed validation failed."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    report = {
        "milestone": "M13.2",
        "component": "offline_network_intelligence_feed",
        "status": "PASS",
        "ingestion": ingestion_report,
        "validation": validation,
        "schema": {
            "required_columns": sorted(
                REQUIRED_COLUMNS
            ),
            "optional_columns": sorted(
                OPTIONAL_COLUMNS
            ),
            "allowed_network_types": sorted(
                ALLOWED_TYPES
            ),
        },
        "methodology": {
            "offline_only": True,
            "source_attribution_required": True,
            "confidence_range": [0.0, 1.0],
            "identity_inference": False,
            "illicitness_inference": False,
        },
        "limitations": [
            (
                "The feed is an intelligence source and does not "
                "prove that a transaction is illicit."
            ),
            (
                "VPN, proxy, and Tor classifications depend on the "
                "quality, provenance, temporal validity, and coverage "
                "of the supplied offline intelligence source."
            ),
            (
                "Multiple intelligence records for the same IP are "
                "retained rather than silently selecting a classification."
            ),
        ],
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            sort_keys=True,
        )

    print("\n" + "=" * 72)
    print("M13.2 RESULT")
    print("=" * 72)
    print(
        f"OUTPUT RECORDS: {len(df):,}"
    )
    print(
        f"UNIQUE IPS: {df['ip'].nunique():,}"
    )
    print(
        f"NETWORK TYPES: "
        f"{df['network_type'].nunique():,}"
    )
    print(
        f"OUTPUT: {output_path}"
    )
    print(
        f"REPORT: {report_path}"
    )
    print("\nM13.2 OFFLINE NETWORK INTELLIGENCE FEED COMPLETE")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M13.2 offline VPN/proxy/Tor "
            "intelligence feed ingestion"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()