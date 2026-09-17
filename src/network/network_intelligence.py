from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

NETWORK_FEATURES_PATH = (
    ROOT / "data" / "derived" / "network_transaction_features.parquet"
)

OUTPUT_PATH = (
    ROOT / "data" / "derived" / "network_intelligence.parquet"
)

REPORT_PATH = (
    ROOT / "reports" / "network" / "m13_1_network_intelligence.json"
)

DEFAULT_INTELLIGENCE_PATH = (
    ROOT / "data" / "raw" / "network" / "network_intelligence_feed.csv"
)

REQUIRED_NETWORK_COLUMNS = {
    "txid",
    "network_observation_count",
    "network_unique_source_ip_count",
    "network_unique_destination_ip_count",
    "network_unique_endpoint_count",
    "network_unique_asn_count",
    "network_unique_country_count",
    "network_exact_txid_match",
    "network_observation_available",
}


FEED_COLUMNS = {
    "ip",
    "network_type",
    "provider",
    "confidence",
    "source",
    "valid_from",
    "valid_to",
}


ALLOWED_NETWORK_TYPES = {
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


def normalize_network_type(value: Any) -> str:
    if value is None:
        return "UNKNOWN"

    text = str(value).strip().upper()

    if not text:
        return "UNKNOWN"

    aliases = {
        "TOR": "TOR_EXIT",
        "TOR EXIT": "TOR_EXIT",
        "TOR RELAY": "TOR_RELAY",
        "EXIT": "TOR_EXIT",
        "VPN": "VPN",
        "PROXY": "PROXY",
        "DATACENTER": "DATACENTER",
        "DATA CENTER": "DATACENTER",
        "HOSTING": "HOSTING",
        "RESIDENTIAL": "RESIDENTIAL",
        "MOBILE": "MOBILE",
        "EDUCATIONAL": "EDUCATIONAL",
        "GOVERNMENT": "GOVERNMENT",
    }

    normalized = aliases.get(text, text)

    if normalized not in ALLOWED_NETWORK_TYPES:
        raise ValueError(
            f"Unsupported network_type '{value}'. "
            f"Allowed values: {sorted(ALLOWED_NETWORK_TYPES)}"
        )

    return normalized


def normalize_confidence(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return 0.0

    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid confidence value: {value}") from exc

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"Confidence must be between 0 and 1, got {confidence}"
        )

    return confidence


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def load_network_features() -> pd.DataFrame:
    if not NETWORK_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Network feature artifact not found: {NETWORK_FEATURES_PATH}"
        )

    df = pd.read_parquet(NETWORK_FEATURES_PATH)

    missing = REQUIRED_NETWORK_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Network feature artifact is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df["txid"] = df["txid"].astype(str)

    if df["txid"].duplicated().any():
        raise ValueError("Network feature artifact contains duplicate TXIDs.")

    return df


def load_intelligence_feed(
    feed_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if feed_path is None:
        feed_path = DEFAULT_INTELLIGENCE_PATH

    if not feed_path.exists():
        empty = pd.DataFrame(
            columns=[
                "ip",
                "network_type",
                "provider",
                "confidence",
                "source",
                "valid_from",
                "valid_to",
            ]
        )

        return empty, {
            "available": False,
            "path": str(feed_path),
            "rows": 0,
            "reason": "No offline intelligence feed supplied.",
        }

    feed = pd.read_csv(feed_path)

    missing = FEED_COLUMNS - set(feed.columns)

    if missing:
        raise ValueError(
            f"Network intelligence feed is missing columns: {sorted(missing)}"
        )

    feed = feed.copy()

    feed["ip"] = feed["ip"].map(normalize_ip)

    if feed["ip"].isna().any():
        invalid_count = int(feed["ip"].isna().sum())
        raise ValueError(
            f"Network intelligence feed contains {invalid_count} invalid IP rows."
        )

    feed["network_type"] = feed["network_type"].map(normalize_network_type)

    feed["provider"] = feed["provider"].map(normalize_text)
    feed["source"] = feed["source"].map(normalize_text)

    feed["confidence"] = feed["confidence"].map(normalize_confidence)

    feed["valid_from"] = feed["valid_from"].map(normalize_text)
    feed["valid_to"] = feed["valid_to"].map(normalize_text)

    if feed.empty:
        return feed, {
            "available": True,
            "path": str(feed_path),
            "rows": 0,
            "reason": "Feed exists but contains no records.",
        }

    duplicate_mask = feed.duplicated(
        subset=[
            "ip",
            "network_type",
            "provider",
            "confidence",
            "source",
            "valid_from",
            "valid_to",
        ],
        keep="first",
    )

    duplicate_count = int(duplicate_mask.sum())

    feed = feed.loc[~duplicate_mask].reset_index(drop=True)

    return feed, {
        "available": True,
        "path": str(feed_path),
        "rows": int(len(feed)),
        "duplicates_removed": duplicate_count,
        "reason": "Offline intelligence feed loaded.",
    }


def build_ip_index(
    feed: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}

    if feed.empty:
        return index

    for row in feed.to_dict(orient="records"):
        ip = str(row["ip"])

        index.setdefault(ip, []).append(row)

    return index


def classify_transaction_network(
    row: pd.Series,
    ip_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    Produce network-intelligence evidence for one transaction.

    Important:
    The transaction-level network feature artifact currently contains
    aggregate network counts but does not preserve individual source and
    destination IP addresses. Therefore a transaction can only receive
    provider/type intelligence when the upstream artifact contains an
    IP-level intelligence mapping.

    With the current M10 artifact and no supplied feed, the correct state
    is UNKNOWN rather than an inferred VPN/proxy/Tor classification.
    """

    observation_available = bool(
        row.get("network_observation_available", False)
    )

    if not observation_available:
        return {
            "network_intelligence_available": False,
            "network_classification": "UNKNOWN",
            "network_classification_confidence": 0.0,
            "network_provider": None,
            "network_intelligence_source": None,
            "vpn_evidence": 0.0,
            "proxy_evidence": 0.0,
            "tor_evidence": 0.0,
            "hosting_evidence": 0.0,
            "network_intelligence_reason": "No network observation available.",
        }

    # M10 aggregates do not expose source/destination IP addresses.
    # We therefore do not reverse-engineer or guess the IP classification.
    return {
        "network_intelligence_available": False,
        "network_classification": "UNKNOWN",
        "network_classification_confidence": 0.0,
        "network_provider": None,
        "network_intelligence_source": None,
        "vpn_evidence": 0.0,
        "proxy_evidence": 0.0,
        "tor_evidence": 0.0,
        "hosting_evidence": 0.0,
        "network_intelligence_reason": (
            "Network observation exists, but the current aggregated "
            "transaction artifact does not retain IP-level addresses. "
            "No VPN/proxy/Tor classification was inferred."
        ),
    }


def build_network_intelligence(
    network_df: pd.DataFrame,
    feed: pd.DataFrame,
) -> pd.DataFrame:
    ip_index = build_ip_index(feed)

    records: list[dict[str, Any]] = []

    for _, row in network_df.iterrows():
        txid = str(row["txid"])

        result = classify_transaction_network(
            row=row,
            ip_index=ip_index,
        )

        record = {
            "txid": txid,
            "time_step": row.get("time_step"),
            "network_observation_count": row.get(
                "network_observation_count", 0
            ),
            **result,
        }

        records.append(record)

    result_df = pd.DataFrame(records)

    if len(result_df) != len(network_df):
        raise ValueError(
            "Network intelligence row count does not match "
            "network transaction feature row count."
        )

    return result_df


def calculate_report(
    network_df: pd.DataFrame,
    intelligence_df: pd.DataFrame,
    feed_report: dict[str, Any],
) -> dict[str, Any]:
    classification_counts = (
        intelligence_df["network_classification"]
        .value_counts(dropna=False)
        .to_dict()
    )

    observation_count = int(
        intelligence_df["network_intelligence_available"].sum()
    )

    vpn_count = int(
        (intelligence_df["network_classification"] == "VPN").sum()
    )

    proxy_count = int(
        (intelligence_df["network_classification"] == "PROXY").sum()
    )

    tor_count = int(
        intelligence_df["network_classification"]
        .isin(["TOR_EXIT", "TOR_RELAY"])
        .sum()
    )

    return {
        "milestone": "M13.1",
        "component": "offline_network_intelligence",
        "status": "PASS",
        "methodology": {
            "purpose": (
                "Interpret supplied IP-level network intelligence "
                "without fabricating VPN, proxy, or Tor classifications."
            ),
            "classification_is_identity": False,
            "classification_is_illicitness_probability": False,
            "unknown_when_insufficient_evidence": True,
            "offline_only": True,
        },
        "input": {
            "network_feature_rows": int(len(network_df)),
            "network_feature_unique_txids": int(
                network_df["txid"].nunique()
            ),
        },
        "offline_feed": feed_report,
        "output": {
            "rows": int(len(intelligence_df)),
            "unique_txids": int(intelligence_df["txid"].nunique()),
            "intelligence_available_transactions": observation_count,
            "vpn_classifications": vpn_count,
            "proxy_classifications": proxy_count,
            "tor_related_classifications": tor_count,
            "classification_distribution": classification_counts,
        },
        "limitations": [
            (
                "The current M10 transaction-level network artifact "
                "contains aggregate network features and does not retain "
                "individual source/destination IP addresses."
            ),
            (
                "The current synthetic network fixture provides only "
                "one transaction with network observations."
            ),
            (
                "No VPN, proxy, or Tor status is inferred without an "
                "explicit offline intelligence source."
            ),
            (
                "Network classification does not establish wallet "
                "ownership, real-world identity, intent, or illicit activity."
            ),
        ],
    }


def validate_output(
    network_df: pd.DataFrame,
    intelligence_df: pd.DataFrame,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    checks["row_count_match"] = (
        len(network_df) == len(intelligence_df)
    )

    checks["unique_txids"] = (
        intelligence_df["txid"].nunique() == len(intelligence_df)
    )

    checks["txid_coverage"] = set(network_df["txid"]) == set(
        intelligence_df["txid"]
    )

    checks["confidence_finite"] = bool(
        pd.to_numeric(
            intelligence_df["network_classification_confidence"],
            errors="coerce",
        )
        .notna()
        .all()
    )

    checks["confidence_range"] = bool(
        (
            (
                intelligence_df["network_classification_confidence"]
                >= 0.0
            )
            & (
                intelligence_df["network_classification_confidence"]
                <= 1.0
            )
        ).all()
    )

    checks["finite_evidence"] = bool(
        pd.to_numeric(
            intelligence_df["vpn_evidence"],
            errors="coerce",
        ).notna().all()
        and pd.to_numeric(
            intelligence_df["proxy_evidence"],
            errors="coerce",
        ).notna().all()
        and pd.to_numeric(
            intelligence_df["tor_evidence"],
            errors="coerce",
        ).notna().all()
        and pd.to_numeric(
            intelligence_df["hosting_evidence"],
            errors="coerce",
        ).notna().all()
    )

    checks["valid_classifications"] = bool(
        intelligence_df["network_classification"]
        .isin(ALLOWED_NETWORK_TYPES)
        .all()
    )

    passed = all(checks.values())

    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
    }


def run_m13_1(
    feed_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print("=" * 72)
    print("M13.1 OFFLINE NETWORK INTELLIGENCE")
    print("=" * 72)

    print("Loading M10 network transaction features...")

    network_df = load_network_features()

    print(f"Network feature rows: {len(network_df):,}")
    print(f"Unique TXIDs: {network_df['txid'].nunique():,}")

    print("\nLoading offline intelligence feed...")

    feed, feed_report = load_intelligence_feed(feed_path)

    print(f"Feed available: {feed_report['available']}")
    print(f"Feed rows: {len(feed):,}")

    print("\nBuilding network intelligence...")

    intelligence_df = build_network_intelligence(
        network_df=network_df,
        feed=feed,
    )

    validation = validate_output(
        network_df=network_df,
        intelligence_df=intelligence_df,
    )

    print("\nValidation:")

    for name, passed in validation["checks"].items():
        print(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    report = calculate_report(
        network_df=network_df,
        intelligence_df=intelligence_df,
        feed_report=feed_report,
    )

    report["validation"] = validation

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    intelligence_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    with REPORT_PATH.open(
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
    print("M13.1 RESULT")
    print("=" * 72)
    print(f"OUTPUT ROWS: {len(intelligence_df):,}")
    print(
        "INTELLIGENCE-CLASSIFIED TRANSACTIONS: "
        f"{intelligence_df['network_intelligence_available'].sum():,}"
    )
    print(
        "VPN CLASSIFICATIONS: "
        f"{(intelligence_df['network_classification'] == 'VPN').sum():,}"
    )
    print(
        "PROXY CLASSIFICATIONS: "
        f"{(intelligence_df['network_classification'] == 'PROXY').sum():,}"
    )
    print(
        "TOR-RELATED CLASSIFICATIONS: "
        f"{intelligence_df['network_classification'].isin(['TOR_EXIT', 'TOR_RELAY']).sum():,}"
    )
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")

    if validation["status"] != "PASS":
        raise RuntimeError(
            "M13.1 validation failed."
        )

    print("\nM13.1 OFFLINE NETWORK INTELLIGENCE COMPLETE")

    return intelligence_df, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M13.1 offline VPN/proxy/Tor intelligence"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Build network intelligence artifact",
    )

    run_parser.add_argument(
        "--feed",
        type=Path,
        default=None,
        help=(
            "Optional offline CSV intelligence feed. "
            "Defaults to data/raw/network/"
            "network_intelligence_feed.csv"
        ),
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Build and validate the M13.1 artifact",
    )

    validate_parser.add_argument(
        "--feed",
        type=Path,
        default=None,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"run", "validate"}:
        run_m13_1(
            feed_path=args.feed,
        )


if __name__ == "__main__":
    main()