from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


# ======================================================================
# PATHS
# ======================================================================

ROOT = Path(__file__).resolve().parents[2]

# Elliptic++ source dataset is outside the final repository.
EXTERNAL_DATASET_ROOT = Path(
    r"D:\Hackathons\SIH 2026\data\external\elipticpp"
)

CANONICAL_PATH = (
    ROOT
    / "data"
    / "canonical"
    / "canonical_transactions.parquet"
)

ADDR_TX_PATH = (
    EXTERNAL_DATASET_ROOT
    / "actor"
    / "AddrTx_edgelist.csv"
)

# IMPORTANT: keep the separator between actor and filename.
TX_ADDR_PATH = (
    EXTERNAL_DATASET_ROOT
    / "actor"
    / "TxAddr_edgelist.csv"
)

NETWORK_CORRELATION_PATH = (
    ROOT
    / "data"
    / "derived"
    / "network_blockchain_correlations.parquet"
)

RISK_PATH = (
    ROOT
    / "data"
    / "derived"
    / "unified_risk_scores.parquet"
)

ALERT_PATH = (
    ROOT
    / "data"
    / "derived"
    / "ranked_alerts.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "derived"
    / "investigation_graph"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "graph"
)

NODES_PATH = (
    OUTPUT_DIR
    / "investigation_graph_nodes.parquet"
)

EDGES_PATH = (
    OUTPUT_DIR
    / "investigation_graph_edges.parquet"
)

GRAPHML_PATH = (
    OUTPUT_DIR
    / "investigation_graph_top_alerts.graphml"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "unified_investigation_graph_summary.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "m12_1_unified_investigation_graph.json"
)

GRAPHML_REPORT_PATH = (
    REPORT_DIR
    / "m12_1_graphml_investigation_view.json"
)

# GraphML is an analyst-facing subset.
GRAPHML_SEED_LIMIT = 100
GRAPHML_NODE_LIMIT = 5000


# ======================================================================
# NORMALIZATION
# ======================================================================

def normalize_txid(value: Any) -> str | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return None

    if text.endswith(".0"):
        try:
            numeric = float(text)

            if numeric.is_integer():
                return str(int(numeric))
        except ValueError:
            pass

    return text


def normalize_identifier(value: Any) -> str | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return None

    return text


def normalize_ip(value: Any) -> str | None:
    return normalize_identifier(value)


def safe_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (int, float, str, bool)):
        return value

    return str(value)


def stable_edge_id(
    source: str,
    target: str,
    relation: str,
) -> str:
    raw = (
        f"{source}|{target}|{relation}"
        .encode("utf-8")
    )

    return hashlib.sha256(
        raw
    ).hexdigest()[:24]


# ======================================================================
# LOADING
# ======================================================================

def load_canonical() -> pd.DataFrame:
    print(
        "Loading canonical transactions..."
    )

    if not CANONICAL_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: "
            f"{CANONICAL_PATH}"
        )

    df = pd.read_parquet(
        CANONICAL_PATH
    )

    required = {"txid"}

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Canonical dataset missing "
            f"columns: {sorted(missing)}"
        )

    df["txid"] = (
        df["txid"]
        .map(normalize_txid)
    )

    df = df[
        df["txid"].notna()
    ].copy()

    if df["txid"].duplicated().any():
        raise ValueError(
            "Canonical dataset contains "
            "duplicate TXIDs."
        )

    print(
        f"Canonical transactions: "
        f"{len(df):,}"
    )

    return df


def load_address_edges() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    print(
        "Loading Elliptic++ address "
        "transaction edges..."
    )

    if not ADDR_TX_PATH.exists():
        raise FileNotFoundError(
            f"AddrTx source not found: "
            f"{ADDR_TX_PATH}"
        )

    if not TX_ADDR_PATH.exists():
        raise FileNotFoundError(
            f"TxAddr source not found: "
            f"{TX_ADDR_PATH}"
        )

    print(
        f"AddrTx: {ADDR_TX_PATH}"
    )

    print(
        f"TxAddr: {TX_ADDR_PATH}"
    )

    addr_tx = pd.read_csv(
        ADDR_TX_PATH,
        usecols=[
            "input_address",
            "txId",
        ],
        dtype=str,
    )

    tx_addr = pd.read_csv(
        TX_ADDR_PATH,
        usecols=[
            "txId",
            "output_address",
        ],
        dtype=str,
    )

    addr_tx["input_address"] = (
        addr_tx["input_address"]
        .map(normalize_identifier)
    )

    addr_tx["txId"] = (
        addr_tx["txId"]
        .map(normalize_txid)
    )

    tx_addr["txId"] = (
        tx_addr["txId"]
        .map(normalize_txid)
    )

    tx_addr["output_address"] = (
        tx_addr["output_address"]
        .map(normalize_identifier)
    )

    addr_tx = (
        addr_tx
        .dropna()
        .drop_duplicates()
    )

    tx_addr = (
        tx_addr
        .dropna()
        .drop_duplicates()
    )

    print(
        f"Input-address edges: "
        f"{len(addr_tx):,}"
    )

    print(
        f"Output-address edges: "
        f"{len(tx_addr):,}"
    )

    return addr_tx, tx_addr


def load_optional_table(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        print(
            f"{label}: not found; "
            "continuing without it."
        )

        return pd.DataFrame()

    print(
        f"Loading {label}..."
    )

    df = pd.read_parquet(
        path
    )

    print(
        f"{label} rows: "
        f"{len(df):,}"
    )

    return df


# ======================================================================
# LOOKUPS
# ======================================================================

def build_risk_lookup(
    risk: pd.DataFrame,
) -> dict[str, dict[str, Any]]:

    lookup = {}

    if (
        risk.empty
        or "txid" not in risk.columns
    ):
        return lookup

    risk = risk.copy()

    risk["txid"] = (
        risk["txid"]
        .map(normalize_txid)
    )

    for _, row in (
        risk
        .dropna(subset=["txid"])
        .iterrows()
    ):

        lookup[
            row["txid"]
        ] = {
            key: safe_value(value)
            for key, value in row.items()
            if key != "txid"
        }

    return lookup


def build_alert_lookup(
    alerts: pd.DataFrame,
) -> dict[str, dict[str, Any]]:

    lookup = {}

    if (
        alerts.empty
        or "txid" not in alerts.columns
    ):
        return lookup

    alerts = alerts.copy()

    alerts["txid"] = (
        alerts["txid"]
        .map(normalize_txid)
    )

    for _, row in (
        alerts
        .dropna(subset=["txid"])
        .iterrows()
    ):

        lookup[
            row["txid"]
        ] = {
            key: safe_value(value)
            for key, value in row.items()
            if key != "txid"
        }

    return lookup


# ======================================================================
# GRAPH BUILDING
# ======================================================================

def build_graph(
    canonical: pd.DataFrame,
    addr_tx: pd.DataFrame,
    tx_addr: pd.DataFrame,
    network: pd.DataFrame,
    risk: pd.DataFrame,
    alerts: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:

    nodes: dict[
        str,
        dict[str, Any],
    ] = {}

    edges: dict[
        str,
        dict[str, Any],
    ] = {}

    def add_node(
        node_id: str,
        node_type: str,
        **attributes: Any,
    ) -> None:

        if node_id not in nodes:

            nodes[node_id] = {
                "node_id": node_id,
                "node_type": node_type,
            }

        for key, value in (
            attributes.items()
        ):

            if value is None:
                continue

            if (
                key not in nodes[node_id]
                or nodes[node_id][key] is None
            ):
                nodes[node_id][key] = value

    def add_edge(
        source: str,
        target: str,
        relation: str,
        evidence_source: str,
        **attributes: Any,
    ) -> None:

        edge_id = stable_edge_id(
            source,
            target,
            relation,
        )

        if edge_id in edges:
            return

        record = {
            "edge_id": edge_id,
            "source": source,
            "target": target,
            "relation": relation,
            "evidence_source": evidence_source,
        }

        record.update(
            {
                key: value
                for key, value in attributes.items()
                if value is not None
            }
        )

        edges[edge_id] = record

    risk_lookup = build_risk_lookup(
        risk
    )

    alert_lookup = build_alert_lookup(
        alerts
    )

    # ================================================================
    # TRANSACTION NODES
    # ================================================================

    print(
        "Building transaction nodes..."
    )

    for row in (
        canonical.itertuples(
            index=False
        )
    ):

        txid = normalize_txid(
            row.txid
        )

        if txid is None:
            continue

        node_id = f"tx:{txid}"

        attributes = {
            "node_role": "transaction",
            "time_step": safe_value(
                getattr(
                    row,
                    "time_step",
                    None,
                )
            ),
        }

        for column in (
            "output_btc_total",
            "input_btc_total",
            "fee_btc",
            "size_bytes",
            "input_address_count",
            "output_address_count",
        ):

            if hasattr(
                row,
                column,
            ):

                attributes[
                    column
                ] = safe_value(
                    getattr(
                        row,
                        column,
                    )
                )

        risk_data = (
            risk_lookup.get(
                txid,
                {},
            )
        )

        alert_data = (
            alert_lookup.get(
                txid,
                {},
            )
        )

        risk_mapping = {
            "risk_score": "risk_score",
            "predicted_class": "predicted_class",
            "confidence": "ml_confidence",
            "anomaly_signal": "anomaly_signal",
            "behavioral_signal": "behavioral_signal",
            "entity_signal": "entity_signal",
            "network_signal": "network_signal",
        }

        for source_key, target_key in (
            risk_mapping.items()
        ):

            if source_key in risk_data:

                attributes[
                    target_key
                ] = risk_data[
                    source_key
                ]

        for key in (
            "investigation_rank",
            "alert_status",
            "priority_band",
        ):

            if key in alert_data:

                attributes[
                    key
                ] = alert_data[key]

        add_node(
            node_id,
            "transaction",
            **attributes,
        )

    transaction_ids = set(
        canonical["txid"]
    )

    # ================================================================
    # WALLET NODES + INPUT EDGES
    # ================================================================

    print(
        "Building wallet-to-transaction "
        "relationships..."
    )

    input_count = 0

    for row in (
        addr_tx.itertuples(
            index=False
        )
    ):

        address = normalize_identifier(
            row.input_address
        )

        txid = normalize_txid(
            row.txId
        )

        if (
            address is None
            or txid is None
            or txid not in transaction_ids
        ):
            continue

        wallet_node = (
            f"wallet:{address}"
        )

        tx_node = (
            f"tx:{txid}"
        )

        add_node(
            wallet_node,
            "wallet",
            address=address,
            node_role="wallet",
        )

        add_edge(
            wallet_node,
            tx_node,
            "INPUT_TO",
            "Elliptic++ AddrTx_edgelist",
        )

        input_count += 1

    # ================================================================
    # OUTPUT EDGES
    # ================================================================

    output_count = 0

    for row in (
        tx_addr.itertuples(
            index=False
        )
    ):

        txid = normalize_txid(
            row.txId
        )

        address = normalize_identifier(
            row.output_address
        )

        if (
            address is None
            or txid is None
            or txid not in transaction_ids
        ):
            continue

        wallet_node = (
            f"wallet:{address}"
        )

        tx_node = (
            f"tx:{txid}"
        )

        add_node(
            wallet_node,
            "wallet",
            address=address,
            node_role="wallet",
        )

        add_edge(
            tx_node,
            wallet_node,
            "OUTPUT_TO",
            "Elliptic++ TxAddr_edgelist",
        )

        output_count += 1

    # ================================================================
    # NETWORK EDGES
    # ================================================================

    print(
        "Building network-to-transaction "
        "relationships..."
    )

    network_rows = 0
    network_ip_observations = 0

    source_ips = set()
    destination_ips = set()

    if (
        not network.empty
        and "txid" in network.columns
    ):

        network = network.copy()

        network["txid"] = (
            network["txid"]
            .map(normalize_txid)
        )

        if "src_ip" in network.columns:

            network["src_ip"] = (
                network["src_ip"]
                .map(normalize_ip)
            )

        if "dst_ip" in network.columns:

            network["dst_ip"] = (
                network["dst_ip"]
                .map(normalize_ip)
            )

        for row in (
            network.itertuples(
                index=False
            )
        ):

            txid = normalize_txid(
                getattr(
                    row,
                    "txid",
                    None,
                )
            )

            if (
                txid is None
                or txid not in transaction_ids
            ):
                continue

            network_rows += 1

            tx_node = (
                f"tx:{txid}"
            )

            src_ip = normalize_ip(
                getattr(
                    row,
                    "src_ip",
                    None,
                )
            )

            dst_ip = normalize_ip(
                getattr(
                    row,
                    "dst_ip",
                    None,
                )
            )

            if src_ip:

                ip_node = (
                    f"ip:{src_ip}"
                )

                add_node(
                    ip_node,
                    "ip",
                    ip_address=src_ip,
                    node_role="network_source",
                )

                add_edge(
                    ip_node,
                    tx_node,
                    "OBSERVED_SOURCE",
                    "M10 network-blockchain correlation",
                    observation_timestamp=safe_value(
                        getattr(
                            row,
                            "timestamp",
                            None,
                        )
                    ),
                    src_port=safe_value(
                        getattr(
                            row,
                            "src_port",
                            None,
                        )
                    ),
                    dst_port=safe_value(
                        getattr(
                            row,
                            "dst_port",
                            None,
                        )
                    ),
                    geo_country=safe_value(
                        getattr(
                            row,
                            "geo_country",
                            None,
                        )
                    ),
                    asn=safe_value(
                        getattr(
                            row,
                            "asn",
                            None,
                        )
                    ),
                )

                source_ips.add(
                    src_ip
                )

                network_ip_observations += 1

            if dst_ip:

                ip_node = (
                    f"ip:{dst_ip}"
                )

                add_node(
                    ip_node,
                    "ip",
                    ip_address=dst_ip,
                    node_role="network_destination",
                )

                add_edge(
                    tx_node,
                    ip_node,
                    "OBSERVED_DESTINATION",
                    "M10 network-blockchain correlation",
                    observation_timestamp=safe_value(
                        getattr(
                            row,
                            "timestamp",
                            None,
                        )
                    ),
                    src_port=safe_value(
                        getattr(
                            row,
                            "src_port",
                            None,
                        )
                    ),
                    dst_port=safe_value(
                        getattr(
                            row,
                            "dst_port",
                            None,
                        )
                    ),
                    geo_country=safe_value(
                        getattr(
                            row,
                            "geo_country",
                            None,
                        )
                    ),
                    asn=safe_value(
                        getattr(
                            row,
                            "asn",
                            None,
                        )
                    ),
                )

                destination_ips.add(
                    dst_ip
                )

                network_ip_observations += 1

    # ================================================================
    # ADDRESS REUSE CONTINUITY
    # ================================================================

    print(
        "Building transaction continuity "
        "relationships..."
    )

    tx_inputs: defaultdict[
        str,
        set[str],
    ] = defaultdict(set)

    for row in (
        addr_tx.itertuples(
            index=False
        )
    ):

        address = normalize_identifier(
            row.input_address
        )

        txid = normalize_txid(
            row.txId
        )

        if (
            address
            and txid
            and txid in transaction_ids
        ):

            tx_inputs[
                txid
            ].add(address)

    tx_time = {}

    for row in (
        canonical.itertuples(
            index=False
        )
    ):

        txid = normalize_txid(
            row.txid
        )

        if txid is None:
            continue

        tx_time[
            txid
        ] = safe_value(
            getattr(
                row,
                "time_step",
                None,
            )
        )

    address_spenders: defaultdict[
        str,
        list[str],
    ] = defaultdict(list)

    for txid, addresses in (
        tx_inputs.items()
    ):

        for address in addresses:

            address_spenders[
                address
            ].append(
                txid
            )

    continuity_count = 0

    for address, txids in (
        address_spenders.items()
    ):

        if len(txids) < 2:
            continue

        txids = sorted(
            txids,
            key=lambda tx: (
                tx_time.get(tx) is None,
                tx_time.get(tx),
                tx,
            ),
        )

        for previous_tx, current_tx in zip(
            txids,
            txids[1:],
        ):

            if previous_tx == current_tx:
                continue

            previous_node = (
                f"tx:{previous_tx}"
            )

            current_node = (
                f"tx:{current_tx}"
            )

            add_edge(
                previous_node,
                current_node,
                "ADDRESS_REUSE_CONTINUITY",
                "Derived from temporal address-spend history",
                shared_address=address,
            )

            continuity_count += 1

    # ================================================================
    # DATAFRAME CREATION
    # ================================================================

    nodes_df = pd.DataFrame(
        list(
            nodes.values()
        )
    )

    edges_df = pd.DataFrame(
        list(
            edges.values()
        )
    )

    if nodes_df.empty:

        nodes_df = pd.DataFrame(
            columns=[
                "node_id",
                "node_type",
            ]
        )

    if edges_df.empty:

        edges_df = pd.DataFrame(
            columns=[
                "edge_id",
                "source",
                "target",
                "relation",
                "evidence_source",
            ]
        )

    node_type_counts = Counter(
        nodes_df["node_type"]
    )

    edge_type_counts = Counter(
        edges_df["relation"]
    )

    summary = {
        "milestone": "M12.1",
        "status": "PASS",
        "graph_type": (
            "Unified Bitcoin investigation graph"
        ),
        "scope": {
            "total_nodes": int(
                len(nodes_df)
            ),
            "total_edges": int(
                len(edges_df)
            ),
            "transaction_nodes": int(
                node_type_counts.get(
                    "transaction",
                    0,
                )
            ),
            "wallet_nodes": int(
                node_type_counts.get(
                    "wallet",
                    0,
                )
            ),
            "ip_nodes": int(
                node_type_counts.get(
                    "ip",
                    0,
                )
            ),
        },
        "node_types": {
            str(key): int(value)
            for key, value in (
                node_type_counts.items()
            )
        },
        "edge_types": {
            str(key): int(value)
            for key, value in (
                edge_type_counts.items()
            )
        },
        "source_statistics": {
            "canonical_transactions": int(
                len(canonical)
            ),
            "input_address_edges_processed": int(
                input_count
            ),
            "output_address_edges_processed": int(
                output_count
            ),
            "network_rows_processed": int(
                network_rows
            ),
            "network_ip_transaction_observations": int(
                network_ip_observations
            ),
            "address_reuse_continuity_edges": int(
                continuity_count
            ),
        },
        "network_coverage": {
            "unique_source_ips": int(
                len(source_ips)
            ),
            "unique_destination_ips": int(
                len(destination_ips)
            ),
        },
        "limitations": [
            (
                "Elliptic++ address edge lists do not "
                "contain individual address-level BTC amounts."
            ),
            (
                "Wallet nodes represent blockchain addresses "
                "and are not asserted to be real-world identities."
            ),
            (
                "IP-to-transaction relationships require "
                "supplied network observations with exact TXID correlation."
            ),
            (
                "Current network observations are synthetic "
                "and have very limited coverage."
            ),
            (
                "Address reuse continuity is structural evidence "
                "and does not prove common ownership."
            ),
            (
                "Graph relationships do not establish illicit "
                "activity, ownership, identity, intent, or guilt."
            ),
        ],
    }

    return (
        nodes_df,
        edges_df,
        summary,
    )


# ======================================================================
# VALIDATION
# ======================================================================

def validate_graph_tables(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> None:

    print(
        "Validating graph outputs..."
    )

    if nodes_df.empty:
        raise ValueError(
            "Graph contains zero nodes."
        )

    required_nodes = {
        "node_id",
        "node_type",
    }

    missing_nodes = (
        required_nodes
        - set(nodes_df.columns)
    )

    if missing_nodes:
        raise ValueError(
            f"Node table missing: "
            f"{sorted(missing_nodes)}"
        )

    if nodes_df[
        "node_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate node IDs detected."
        )

    required_edges = {
        "edge_id",
        "source",
        "target",
        "relation",
        "evidence_source",
    }

    missing_edges = (
        required_edges
        - set(edges_df.columns)
    )

    if missing_edges:
        raise ValueError(
            f"Edge table missing: "
            f"{sorted(missing_edges)}"
        )

    if edges_df[
        "edge_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate edge IDs detected."
        )

    node_ids = set(
        nodes_df[
            "node_id"
        ]
    )

    missing_sources = (
        set(
            edges_df[
                "source"
            ]
        )
        - node_ids
    )

    missing_targets = (
        set(
            edges_df[
                "target"
            ]
        )
        - node_ids
    )

    if missing_sources:
        raise ValueError(
            "Edges contain source "
            "nodes that do not exist."
        )

    if missing_targets:
        raise ValueError(
            "Edges contain target "
            "nodes that do not exist."
        )

    print(
        "Graph validation: PASS"
    )


# ======================================================================
# GRAPHML SUBSET
# ======================================================================

def select_graphml_view(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:

    transactions = nodes_df[
        nodes_df["node_type"]
        == "transaction"
    ].copy()

    if "investigation_rank" in (
        transactions.columns
    ):

        transactions[
            "_rank"
        ] = pd.to_numeric(
            transactions[
                "investigation_rank"
            ],
            errors="coerce",
        )

    else:

        transactions[
            "_rank"
        ] = float("inf")

    if "risk_score" in (
        transactions.columns
    ):

        transactions[
            "_risk"
        ] = pd.to_numeric(
            transactions[
                "risk_score"
            ],
            errors="coerce",
        )

    else:

        transactions[
            "_risk"
        ] = 0.0

    transactions = (
        transactions
        .sort_values(
            [
                "_rank",
                "_risk",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )
        .head(
            GRAPHML_SEED_LIMIT
        )
    )

    selected_ids = set(
        transactions[
            "node_id"
        ]
    )

    # One-hop expansion around selected
    # transaction nodes.
    candidate_edges = edges_df[
        edges_df["source"].isin(
            selected_ids
        )
        | edges_df["target"].isin(
            selected_ids
        )
    ].copy()

    candidate_edges = (
        candidate_edges
        .head(
            GRAPHML_NODE_LIMIT
        )
    )

    for _, row in (
        candidate_edges.iterrows()
    ):

        selected_ids.add(
            row["source"]
        )

        selected_ids.add(
            row["target"]
        )

        if (
            len(selected_ids)
            >= GRAPHML_NODE_LIMIT
        ):
            break

    view_nodes = nodes_df[
        nodes_df["node_id"].isin(
            selected_ids
        )
    ].copy()

    view_edges = candidate_edges[
        candidate_edges["source"].isin(
            set(
                view_nodes[
                    "node_id"
                ]
            )
        )
        & candidate_edges["target"].isin(
            set(
                view_nodes[
                    "node_id"
                ]
            )
        )
    ].copy()

    metadata = {
        "seed_transaction_count": int(
            len(transactions)
        ),
        "view_node_count": int(
            len(view_nodes)
        ),
        "view_edge_count": int(
            len(view_edges)
        ),
        "seed_limit": int(
            GRAPHML_SEED_LIMIT
        ),
        "node_limit": int(
            GRAPHML_NODE_LIMIT
        ),
        "selection_method": (
            "Top investigation-ranked "
            "transactions with bounded "
            "one-hop graph expansion."
        ),
    }

    return (
        view_nodes,
        view_edges,
        metadata,
    )


def graphml_text(
    value: Any,
) -> str:

    value = safe_value(
        value
    )

    if value is None:
        return ""

    if isinstance(
        value,
        (
            dict,
            list,
            tuple,
            set,
        ),
    ):

        return json.dumps(
            value,
            default=str,
        )

    return str(value)


def write_graphml_view(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> dict:

    print(
        "Preparing bounded GraphML "
        "investigation view..."
    )

    (
        view_nodes,
        view_edges,
        metadata,
    ) = select_graphml_view(
        nodes_df,
        edges_df,
    )

    namespace = (
        "http://graphml.graphdrawing.org/xmlns"
    )

    xsi_namespace = (
        "http://www.w3.org/2001/XMLSchema-instance"
    )

    ET.register_namespace(
        "",
        namespace,
    )

    ET.register_namespace(
        "xsi",
        xsi_namespace,
    )

    root = ET.Element(
        f"{{{namespace}}}graphml"
    )

    root.set(
        f"{{{xsi_namespace}}}schemaLocation",
        (
            f"{namespace} "
            "http://graphml.graphdrawing.org/"
            "1.0/graphml.xsd"
        ),
    )

    # --------------------------------------------------------------
    # Determine attributes
    # --------------------------------------------------------------

    node_columns = [
        column
        for column in view_nodes.columns
        if column != "node_id"
        and not column.startswith("_")
    ]

    edge_columns = [
        column
        for column in view_edges.columns
        if column not in {
            "edge_id",
            "source",
            "target",
        }
        and not column.startswith("_")
    ]

    for column in node_columns:

        key = ET.SubElement(
            root,
            f"{{{namespace}}}key",
        )

        key.set(
            "id",
            f"node_{column}",
        )

        key.set(
            "for",
            "node",
        )

        key.set(
            "attr.name",
            column,
        )

        key.set(
            "attr.type",
            "string",
        )

    for column in edge_columns:

        key = ET.SubElement(
            root,
            f"{{{namespace}}}key",
        )

        key.set(
            "id",
            f"edge_{column}",
        )

        key.set(
            "for",
            "edge",
        )

        key.set(
            "attr.name",
            column,
        )

        key.set(
            "attr.type",
            "string",
        )

    graph_element = ET.SubElement(
        root,
        f"{{{namespace}}}graph",
    )

    graph_element.set(
        "id",
        "bitcoin_investigation_graph",
    )

    graph_element.set(
        "edgedefault",
        "directed",
    )

    # --------------------------------------------------------------
    # Nodes
    # --------------------------------------------------------------

    for _, row in (
        view_nodes.iterrows()
    ):

        node = ET.SubElement(
            graph_element,
            f"{{{namespace}}}node",
        )

        node.set(
            "id",
            str(
                row["node_id"]
            ),
        )

        for column in node_columns:

            data = ET.SubElement(
                node,
                f"{{{namespace}}}data",
            )

            data.set(
                "key",
                f"node_{column}",
            )

            data.text = graphml_text(
                row[column]
            )

    # --------------------------------------------------------------
    # Edges
    # --------------------------------------------------------------

    for _, row in (
        view_edges.iterrows()
    ):

        edge = ET.SubElement(
            graph_element,
            f"{{{namespace}}}edge",
        )

        edge.set(
            "id",
            str(
                row["edge_id"]
            ),
        )

        edge.set(
            "source",
            str(
                row["source"]
            ),
        )

        edge.set(
            "target",
            str(
                row["target"]
            ),
        )

        for column in edge_columns:

            data = ET.SubElement(
                edge,
                f"{{{namespace}}}data",
            )

            data.set(
                "key",
                f"edge_{column}",
            )

            data.text = graphml_text(
                row[column]
            )

    tree = ET.ElementTree(
        root
    )

    tree.write(
        GRAPHML_PATH,
        encoding="utf-8",
        xml_declaration=True,
    )

    metadata[
        "output"
    ] = str(
        GRAPHML_PATH
    )

    return metadata


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    print("=" * 72)
    print(
        "M12.1 UNIFIED INVESTIGATION GRAPH"
    )
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("DATA SOURCES")

    print(
        f"Canonical: "
        f"{CANONICAL_PATH}"
    )

    print(
        f"Elliptic++: "
        f"{EXTERNAL_DATASET_ROOT}"
    )

    print(
        f"AddrTx: "
        f"{ADDR_TX_PATH}"
    )

    print(
        f"TxAddr: "
        f"{TX_ADDR_PATH}"
    )

    print()

    canonical = load_canonical()

    (
        addr_tx,
        tx_addr,
    ) = load_address_edges()

    network = load_optional_table(
        NETWORK_CORRELATION_PATH,
        "Network-blockchain correlations",
    )

    risk = load_optional_table(
        RISK_PATH,
        "Unified risk scores",
    )

    alerts = load_optional_table(
        ALERT_PATH,
        "Ranked alerts",
    )

    (
        nodes_df,
        edges_df,
        summary,
    ) = build_graph(
        canonical=canonical,
        addr_tx=addr_tx,
        tx_addr=tx_addr,
        network=network,
        risk=risk,
        alerts=alerts,
    )

    validate_graph_tables(
        nodes_df,
        edges_df,
    )

    print()
    print(
        "M12.1 GRAPH RESULT"
    )

    print(
        f"NODES: "
        f"{len(nodes_df):,}"
    )

    print(
        f"EDGES: "
        f"{len(edges_df):,}"
    )

    print()
    print("NODE TYPES:")

    for node_type, count in (
        Counter(
            nodes_df["node_type"]
        ).items()
    ):

        print(
            f"  {node_type}: "
            f"{count:,}"
        )

    print()
    print("EDGE TYPES:")

    for edge_type, count in (
        Counter(
            edges_df["relation"]
        ).items()
    ):

        print(
            f"  {edge_type}: "
            f"{count:,}"
        )

    # ================================================================
    # FULL GRAPH PARQUET
    # ================================================================

    print()
    print(
        "Writing full graph Parquet artifacts..."
    )

    nodes_df.to_parquet(
        NODES_PATH,
        index=False,
    )

    edges_df.to_parquet(
        EDGES_PATH,
        index=False,
    )

    # ================================================================
    # BOUNDED GRAPHML
    # ================================================================

    print()
    print(
        "Writing bounded GraphML "
        "investigation view..."
    )

    graphml_metadata = (
        write_graphml_view(
            nodes_df,
            edges_df,
        )
    )

    summary[
        "graphml_investigation_view"
    ] = graphml_metadata

    # ================================================================
    # SUMMARY
    # ================================================================

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    with open(
        GRAPHML_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            {
                "milestone": "M12.1",
                "status": "PASS",
                "full_graph": {
                    "nodes": int(
                        len(nodes_df)
                    ),
                    "edges": int(
                        len(edges_df)
                    ),
                    "nodes_artifact": str(
                        NODES_PATH
                    ),
                    "edges_artifact": str(
                        EDGES_PATH
                    ),
                },
                "graphml_view": (
                    graphml_metadata
                ),
                "methodology": (
                    "The complete investigation graph "
                    "is preserved in Parquet. GraphML is "
                    "a bounded analyst-facing view generated "
                    "from top investigation-ranked transactions "
                    "and their one-hop relationships."
                ),
            },
            handle,
            indent=2,
        )

    print()
    print("OUTPUTS")

    print(
        f"Full nodes: "
        f"{NODES_PATH}"
    )

    print(
        f"Full edges: "
        f"{EDGES_PATH}"
    )

    print(
        f"GraphML view: "
        f"{GRAPHML_PATH}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Report: "
        f"{REPORT_PATH}"
    )

    print(
        f"GraphML report: "
        f"{GRAPHML_REPORT_PATH}"
    )

    print()
    print(
        f"FULL GRAPH NODES: "
        f"{len(nodes_df):,}"
    )

    print(
        f"FULL GRAPH EDGES: "
        f"{len(edges_df):,}"
    )

    print(
        f"GRAPHML VIEW NODES: "
        f"{graphml_metadata['view_node_count']:,}"
    )

    print(
        f"GRAPHML VIEW EDGES: "
        f"{graphml_metadata['view_edge_count']:,}"
    )

    print()
    print(
        "M12.1 UNIFIED INVESTIGATION GRAPH COMPLETE"
    )


if __name__ == "__main__":
    main()