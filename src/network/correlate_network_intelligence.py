from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

NETWORK_CORRELATION_PATH = (
    ROOT
    / "data"
    / "derived"
    / "network_blockchain_correlations.parquet"
)

INTELLIGENCE_FEED_PATH = (
    ROOT
    / "data"
    / "derived"
    / "network_intelligence_feed.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "derived"
    / "transaction_network_intelligence.parquet"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "network"
    / "m13_3_transaction_network_intelligence.json"
)


REQUIRED_CORRELATION_COLUMNS = {
    "network_event_id",
    "timestamp",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "txid",
    "geo_country",
    "asn",
    "txid_match",
}

REQUIRED_FEED_COLUMNS = {
    "ip",
    "network_type",
    "provider",
    "confidence",
    "source",
    "record_id",
}


CLASSIFICATION_TO_EVIDENCE = {
    "VPN": "vpn",
    "PROXY": "proxy",
    "TOR_EXIT": "tor",
    "TOR_RELAY": "tor",
    "HOSTING": "hosting",
    "DATACENTER": "hosting",
}


def normalize_ip(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None

    return text


def normalize_txid(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        numeric_part = text[:-2]

        if numeric_part.isdigit():
            text = numeric_part

    return text


def load_correlation_data() -> pd.DataFrame:
    if not NETWORK_CORRELATION_PATH.exists():
        raise FileNotFoundError(
            "Network correlation artifact not found:\n"
            f"{NETWORK_CORRELATION_PATH}"
        )

    df = pd.read_parquet(
        NETWORK_CORRELATION_PATH
    )

    missing = (
        REQUIRED_CORRELATION_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Network correlation artifact is missing "
            f"columns: {sorted(missing)}"
        )

    df = df.copy()

    df["txid"] = df["txid"].map(
        normalize_txid
    )

    df["src_ip"] = df["src_ip"].map(
        normalize_ip
    )

    df["dst_ip"] = df["dst_ip"].map(
        normalize_ip
    )

    return df


def load_intelligence_feed() -> pd.DataFrame:
    if not INTELLIGENCE_FEED_PATH.exists():
        raise FileNotFoundError(
            "Network intelligence feed not found:\n"
            f"{INTELLIGENCE_FEED_PATH}"
        )

    df = pd.read_parquet(
        INTELLIGENCE_FEED_PATH
    )

    missing = (
        REQUIRED_FEED_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Network intelligence feed is missing "
            f"columns: {sorted(missing)}"
        )

    df = df.copy()

    df["ip"] = df["ip"].map(
        normalize_ip
    )

    df["network_type"] = (
        df["network_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["provider"] = (
        df["provider"]
        .where(df["provider"].notna(), None)
    )

    df["source"] = (
        df["source"]
        .where(df["source"].notna(), None)
    )

    df["confidence"] = pd.to_numeric(
        df["confidence"],
        errors="coerce",
    ).fillna(0.0)

    return df


def build_ip_index(
    feed: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    index: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for record in feed.to_dict(
        orient="records"
    ):
        ip = record["ip"]

        if ip is None:
            continue

        index.setdefault(
            ip,
            [],
        ).append(record)

    return index


def choose_best_record(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not records:
        return None

    return max(
        records,
        key=lambda record: float(
            record.get(
                "confidence",
                0.0,
            )
        ),
    )


def classify_ip(
    ip: str | None,
    ip_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if ip is None:
        return {
            "classification": "UNKNOWN",
            "provider": None,
            "confidence": 0.0,
            "source": None,
        }

    records = ip_index.get(
        ip,
        [],
    )

    best = choose_best_record(
        records
    )

    if best is None:
        return {
            "classification": "UNKNOWN",
            "provider": None,
            "confidence": 0.0,
            "source": None,
        }

    return {
        "classification": str(
            best["network_type"]
        ),
        "provider": best.get(
            "provider"
        ),
        "confidence": float(
            best.get(
                "confidence",
                0.0,
            )
        ),
        "source": best.get(
            "source"
        ),
    }


def build_observation_records(
    correlation_df: pd.DataFrame,
    ip_index: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for row in correlation_df.itertuples(
        index=False
    ):
        txid = normalize_txid(
            getattr(row, "txid")
        )

        src_ip = normalize_ip(
            getattr(row, "src_ip")
        )

        dst_ip = normalize_ip(
            getattr(row, "dst_ip")
        )

        src_info = classify_ip(
            src_ip,
            ip_index,
        )

        dst_info = classify_ip(
            dst_ip,
            ip_index,
        )

        src_classification = (
            src_info["classification"]
        )

        dst_classification = (
            dst_info["classification"]
        )

        records.append(
            {
                "network_event_id": getattr(
                    row,
                    "network_event_id",
                ),
                "txid": txid,
                "timestamp": getattr(
                    row,
                    "timestamp",
                ),
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_network_type": src_classification,
                "src_provider": src_info[
                    "provider"
                ],
                "src_confidence": src_info[
                    "confidence"
                ],
                "src_source": src_info[
                    "source"
                ],
                "dst_network_type": dst_classification,
                "dst_provider": dst_info[
                    "provider"
                ],
                "dst_confidence": dst_info[
                    "confidence"
                ],
                "dst_source": dst_info[
                    "source"
                ],
                "src_vpn_evidence": float(
                    src_classification == "VPN"
                )
                * src_info["confidence"],
                "dst_vpn_evidence": float(
                    dst_classification == "VPN"
                )
                * dst_info["confidence"],
                "src_proxy_evidence": float(
                    src_classification == "PROXY"
                )
                * src_info["confidence"],
                "dst_proxy_evidence": float(
                    dst_classification == "PROXY"
                )
                * dst_info["confidence"],
                "src_tor_evidence": float(
                    src_classification
                    in {
                        "TOR_EXIT",
                        "TOR_RELAY",
                    }
                )
                * src_info["confidence"],
                "dst_tor_evidence": float(
                    dst_classification
                    in {
                        "TOR_EXIT",
                        "TOR_RELAY",
                    }
                )
                * dst_info["confidence"],
                "src_hosting_evidence": float(
                    src_classification
                    in {
                        "HOSTING",
                        "DATACENTER",
                    }
                )
                * src_info["confidence"],
                "dst_hosting_evidence": float(
                    dst_classification
                    in {
                        "HOSTING",
                        "DATACENTER",
                    }
                )
                * dst_info["confidence"],
            }
        )

    return pd.DataFrame(records)


def aggregate_transaction_intelligence(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(
            columns=[
                "txid",
                "network_intelligence_observation_count",
                "network_intelligence_available",
                "network_intelligence_classification",
                "network_intelligence_confidence",
                "network_intelligence_provider_count",
                "network_intelligence_source_count",
                "vpn_evidence",
                "proxy_evidence",
                "tor_evidence",
                "hosting_evidence",
                "vpn_observation_count",
                "proxy_observation_count",
                "tor_observation_count",
                "hosting_observation_count",
                "classified_source_ip_count",
                "classified_destination_ip_count",
                "network_intelligence_evidence_score",
                "network_intelligence_evidence_level",
            ]
        )

    grouped: list[dict[str, Any]] = []

    for txid, group in observations.groupby(
        "txid",
        sort=False,
    ):
        classifications: list[str] = []

        providers: set[str] = set()
        sources: set[str] = set()

        classified_source_ips: set[str] = set()
        classified_destination_ips: set[str] = set()

        for _, row in group.iterrows():
            src_class = str(
                row["src_network_type"]
            )

            dst_class = str(
                row["dst_network_type"]
            )

            if src_class != "UNKNOWN":
                classifications.append(
                    src_class
                )

                if row["src_provider"]:
                    providers.add(
                        str(row["src_provider"])
                    )

                if row["src_source"]:
                    sources.add(
                        str(row["src_source"])
                    )

                if row["src_ip"]:
                    classified_source_ips.add(
                        str(row["src_ip"])
                    )

            if dst_class != "UNKNOWN":
                classifications.append(
                    dst_class
                )

                if row["dst_provider"]:
                    providers.add(
                        str(row["dst_provider"])
                    )

                if row["dst_source"]:
                    sources.add(
                        str(row["dst_source"])
                    )

                if row["dst_ip"]:
                    classified_destination_ips.add(
                        str(row["dst_ip"])
                    )

        vpn_evidence = max(
            float(
                group["src_vpn_evidence"].max()
            ),
            float(
                group["dst_vpn_evidence"].max()
            ),
        )

        proxy_evidence = max(
            float(
                group["src_proxy_evidence"].max()
            ),
            float(
                group["dst_proxy_evidence"].max()
            ),
        )

        tor_evidence = max(
            float(
                group["src_tor_evidence"].max()
            ),
            float(
                group["dst_tor_evidence"].max()
            ),
        )

        hosting_evidence = max(
            float(
                group["src_hosting_evidence"].max()
            ),
            float(
                group["dst_hosting_evidence"].max()
            ),
        )

        classification_counts = Counter(
            classifications
        )

        priority_order = [
            "TOR_EXIT",
            "TOR_RELAY",
            "VPN",
            "PROXY",
            "DATACENTER",
            "HOSTING",
            "RESIDENTIAL",
            "MOBILE",
            "EDUCATIONAL",
            "GOVERNMENT",
        ]

        selected_classification = "UNKNOWN"

        for classification in priority_order:
            if classification in classification_counts:
                selected_classification = (
                    classification
                )
                break

        confidence_values: list[float] = []

        for column in [
            "src_confidence",
            "dst_confidence",
        ]:
            confidence_values.extend(
                pd.to_numeric(
                    group[column],
                    errors="coerce",
                )
                .dropna()
                .tolist()
            )

        classification_confidence = (
            max(confidence_values)
            if confidence_values
            else 0.0
        )

        evidence_values = [
            vpn_evidence,
            proxy_evidence,
            tor_evidence,
            hosting_evidence,
        ]

        positive_evidence = [
            value
            for value in evidence_values
            if value > 0
        ]

        if positive_evidence:
            network_evidence_score = max(
                positive_evidence
            )
        else:
            network_evidence_score = 0.0

        if tor_evidence >= 0.90:
            evidence_level = "VERY_HIGH"
        elif tor_evidence >= 0.75:
            evidence_level = "HIGH"
        elif (
            vpn_evidence >= 0.90
            or proxy_evidence >= 0.90
        ):
            evidence_level = "HIGH"
        elif (
            vpn_evidence >= 0.75
            or proxy_evidence >= 0.75
            or hosting_evidence >= 0.90
        ):
            evidence_level = "MODERATE"
        elif network_evidence_score > 0:
            evidence_level = "LOW"
        else:
            evidence_level = "NONE"

        grouped.append(
            {
                "txid": txid,
                "network_intelligence_observation_count": int(
                    len(group)
                ),
                "network_intelligence_available": bool(
                    positive_evidence
                ),
                "network_intelligence_classification": (
                    selected_classification
                ),
                "network_intelligence_confidence": (
                    classification_confidence
                ),
                "network_intelligence_provider_count": (
                    len(providers)
                ),
                "network_intelligence_source_count": (
                    len(sources)
                ),
                "vpn_evidence": vpn_evidence,
                "proxy_evidence": proxy_evidence,
                "tor_evidence": tor_evidence,
                "hosting_evidence": hosting_evidence,
                "vpn_observation_count": int(
                    (
                        group[
                            [
                                "src_network_type",
                                "dst_network_type",
                            ]
                        ]
                        == "VPN"
                    )
                    .any(axis=1)
                    .sum()
                ),
                "proxy_observation_count": int(
                    (
                        group[
                            [
                                "src_network_type",
                                "dst_network_type",
                            ]
                        ]
                        == "PROXY"
                    )
                    .any(axis=1)
                    .sum()
                ),
                "tor_observation_count": int(
                    (
                        group[
                            [
                                "src_network_type",
                                "dst_network_type",
                            ]
                        ]
                        .isin(
                            [
                                "TOR_EXIT",
                                "TOR_RELAY",
                            ]
                        )
                    )
                    .any(axis=1)
                    .sum()
                ),
                "hosting_observation_count": int(
                    (
                        group[
                            [
                                "src_network_type",
                                "dst_network_type",
                            ]
                        ]
                        .isin(
                            [
                                "HOSTING",
                                "DATACENTER",
                            ]
                        )
                    )
                    .any(axis=1)
                    .sum()
                ),
                "classified_source_ip_count": len(
                    classified_source_ips
                ),
                "classified_destination_ip_count": len(
                    classified_destination_ips
                ),
                "network_intelligence_evidence_score": (
                    network_evidence_score
                ),
                "network_intelligence_evidence_level": (
                    evidence_level
                ),
            }
        )

    return pd.DataFrame(grouped)


def validate_output(
    correlation_df: pd.DataFrame,
    observation_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    checks["observation_rows_match"] = (
        len(correlation_df)
        == len(observation_df)
    )

    checks["transaction_rows_positive"] = (
        len(transaction_df) > 0
    )

    checks["unique_transaction_txids"] = (
        transaction_df["txid"].nunique()
        == len(transaction_df)
    )

    correlation_txids = set(
        correlation_df["txid"]
        .dropna()
        .astype(str)
    )

    observation_txids = set(
        observation_df["txid"]
        .dropna()
        .astype(str)
    )

    transaction_txids = set(
        transaction_df["txid"]
        .dropna()
        .astype(str)
    )

    checks["observation_txid_subset"] = (
        observation_txids
        .issubset(correlation_txids)
    )

    checks["transaction_txid_subset"] = (
        transaction_txids
        .issubset(observation_txids)
    )

    checks["evidence_finite"] = bool(
        pd.to_numeric(
            transaction_df[
                "network_intelligence_evidence_score"
            ],
            errors="coerce",
        )
        .notna()
        .all()
    )

    checks["evidence_range"] = bool(
        (
            (
                transaction_df[
                    "network_intelligence_evidence_score"
                ]
                >= 0.0
            )
            & (
                transaction_df[
                    "network_intelligence_evidence_score"
                ]
                <= 1.0
            )
        ).all()
    )

    checks["confidence_range"] = bool(
        (
            (
                transaction_df[
                    "network_intelligence_confidence"
                ]
                >= 0.0
            )
            & (
                transaction_df[
                    "network_intelligence_confidence"
                ]
                <= 1.0
            )
        ).all()
    )

    valid_levels = {
        "NONE",
        "LOW",
        "MODERATE",
        "HIGH",
        "VERY_HIGH",
    }

    checks["valid_evidence_levels"] = bool(
        transaction_df[
            "network_intelligence_evidence_level"
        ]
        .isin(valid_levels)
        .all()
    )

    passed = all(
        checks.values()
    )

    return {
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "checks": checks,
    }


def run() -> None:
    print("=" * 72)
    print("M13.3 NETWORK INTELLIGENCE CORRELATION")
    print("=" * 72)

    print(
        "Loading M10 network-blockchain correlations..."
    )

    correlation_df = load_correlation_data()

    print(
        f"Correlation observations: "
        f"{len(correlation_df):,}"
    )

    print(
        f"Unique correlation TXIDs: "
        f"{correlation_df['txid'].nunique():,}"
    )

    print(
        "\nLoading M13.2 offline intelligence feed..."
    )

    feed_df = load_intelligence_feed()

    print(
        f"Intelligence records: "
        f"{len(feed_df):,}"
    )

    print(
        f"Intelligence IPs: "
        f"{feed_df['ip'].nunique():,}"
    )

    ip_index = build_ip_index(
        feed_df
    )

    print(
        "\nCorrelating IP observations "
        "with intelligence..."
    )

    observation_df = build_observation_records(
        correlation_df,
        ip_index,
    )

    transaction_df = (
        aggregate_transaction_intelligence(
            observation_df
        )
    )

    validation = validate_output(
        correlation_df,
        observation_df,
        transaction_df,
    )

    print("\nValidation:")

    for name, passed in validation[
        "checks"
    ].items():
        print(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if validation["status"] != "PASS":
        raise RuntimeError(
            "M13.3 validation failed."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    transaction_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    classification_counts = (
        transaction_df[
            "network_intelligence_classification"
        ]
        .value_counts()
        .to_dict()
    )

    report = {
        "milestone": "M13.3",
        "component": (
            "ip_level_network_intelligence_correlation"
        ),
        "status": "PASS",
        "inputs": {
            "network_correlation_rows": int(
                len(correlation_df)
            ),
            "intelligence_feed_rows": int(
                len(feed_df)
            ),
            "intelligence_feed_unique_ips": int(
                feed_df["ip"].nunique()
            ),
        },
        "outputs": {
            "observation_rows": int(
                len(observation_df)
            ),
            "transaction_rows": int(
                len(transaction_df)
            ),
            "classified_transactions": int(
                transaction_df[
                    "network_intelligence_available"
                ].sum()
            ),
            "vpn_transactions": int(
                (
                    transaction_df[
                        "vpn_observation_count"
                    ]
                    > 0
                ).sum()
            ),
            "proxy_transactions": int(
                (
                    transaction_df[
                        "proxy_observation_count"
                    ]
                    > 0
                ).sum()
            ),
            "tor_transactions": int(
                (
                    transaction_df[
                        "tor_observation_count"
                    ]
                    > 0
                ).sum()
            ),
            "hosting_transactions": int(
                (
                    transaction_df[
                        "hosting_observation_count"
                    ]
                    > 0
                ).sum()
            ),
            "classification_distribution": {
                str(k): int(v)
                for k, v in classification_counts.items()
            },
        },
        "validation": validation,
        "methodology": {
            "ip_level_matching": True,
            "source_attribution_preserved": True,
            "confidence_preserved": True,
            "best_record_selection": (
                "Highest-confidence intelligence record "
                "for each observed IP."
            ),
            "identity_inference": False,
            "illicitness_inference": False,
            "offline_only": True,
        },
        "limitations": [
            (
                "The current M10 network fixture is synthetic "
                "and contains only three network observations."
            ),
            (
                "A classification describes the supplied "
                "network intelligence for an IP; it does not "
                "establish ownership or identity."
            ),
            (
                "VPN/proxy/Tor evidence is dependent on the "
                "quality and temporal validity of the offline feed."
            ),
        ],
    }

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
    print("M13.3 RESULT")
    print("=" * 72)

    print(
        f"NETWORK OBSERVATIONS: "
        f"{len(observation_df):,}"
    )

    print(
        f"TRANSACTIONS WITH INTELLIGENCE: "
        f"{transaction_df['network_intelligence_available'].sum():,}"
    )

    print(
        f"VPN TRANSACTIONS: "
        f"{(transaction_df['vpn_observation_count'] > 0).sum():,}"
    )

    print(
        f"PROXY TRANSACTIONS: "
        f"{(transaction_df['proxy_observation_count'] > 0).sum():,}"
    )

    print(
        f"TOR TRANSACTIONS: "
        f"{(transaction_df['tor_observation_count'] > 0).sum():,}"
    )

    print(
        f"HOSTING/DATACENTER TRANSACTIONS: "
        f"{(transaction_df['hosting_observation_count'] > 0).sum():,}"
    )

    print(
        f"\nOutput: {OUTPUT_PATH}"
    )

    print(
        f"Report: {REPORT_PATH}"
    )

    print(
        "\nM13.3 NETWORK INTELLIGENCE CORRELATION COMPLETE"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M13.3 IP-level network intelligence "
            "correlation engine"
        )
    )

    parser.add_argument(
        "command",
        choices=["run"],
        help="Run M13.3 correlation",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run()


if __name__ == "__main__":
    main()