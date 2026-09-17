from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

GRAPH_DIR = (
    ROOT
    / "data"
    / "derived"
    / "investigation_graph"
)

NODES_PATH = (
    GRAPH_DIR
    / "investigation_graph_nodes.parquet"
)

EDGES_PATH = (
    GRAPH_DIR
    / "investigation_graph_edges.parquet"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "graph"
)

REPORT_PATH = (
    REPORT_DIR
    / "m12_2_graph_investigation_queries.json"
)

DEFAULT_TOP_N = 20


def normalize_txid(
    value: Any,
) -> str | None:

    if value is None:
        return None

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


def normalize_node_id(
    value: Any,
) -> str | None:

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


class InvestigationGraph:

    def __init__(
        self,
        nodes_path: Path = NODES_PATH,
        edges_path: Path = EDGES_PATH,
    ) -> None:

        if not nodes_path.exists():
            raise FileNotFoundError(
                f"Nodes artifact not found: "
                f"{nodes_path}"
            )

        if not edges_path.exists():
            raise FileNotFoundError(
                f"Edges artifact not found: "
                f"{edges_path}"
            )

        print(
            "Loading investigation graph tables..."
        )

        self.nodes = pd.read_parquet(
            nodes_path
        )

        self.edges = pd.read_parquet(
            edges_path
        )

        self.nodes["node_id"] = (
            self.nodes["node_id"]
            .map(normalize_node_id)
        )

        self.edges["source"] = (
            self.edges["source"]
            .map(normalize_node_id)
        )

        self.edges["target"] = (
            self.edges["target"]
            .map(normalize_node_id)
        )

        print(
            f"Nodes loaded: "
            f"{len(self.nodes):,}"
        )

        print(
            f"Edges loaded: "
            f"{len(self.edges):,}"
        )

        self.node_lookup = (
            self.nodes
            .set_index("node_id")
            .to_dict(
                orient="index"
            )
        )

    def node_exists(
        self,
        node_id: str,
    ) -> bool:

        return node_id in self.node_lookup

    def get_node(
        self,
        node_id: str,
    ) -> dict[str, Any]:

        if not self.node_exists(
            node_id
        ):
            raise KeyError(
                f"Node not found: {node_id}"
            )

        return self.node_lookup[
            node_id
        ]

    def transaction_node(
        self,
        txid: str,
    ) -> str:

        normalized = normalize_txid(
            txid
        )

        if normalized is None:
            raise ValueError(
                "Invalid TXID."
            )

        return f"tx:{normalized}"

    # ================================================================
    # 1. TXID -> CONNECTED ENTITIES
    # ================================================================

    def transaction_context(
        self,
        txid: str,
    ) -> dict[str, Any]:

        node_id = self.transaction_node(
            txid
        )

        if not self.node_exists(
            node_id
        ):
            raise KeyError(
                f"Transaction not found: {txid}"
            )

        outgoing = self.edges[
            self.edges["source"]
            == node_id
        ]

        incoming = self.edges[
            self.edges["target"]
            == node_id
        ]

        connected_edges = pd.concat(
            [
                outgoing,
                incoming,
            ],
            ignore_index=True,
        ).drop_duplicates(
            subset=["edge_id"]
        )

        connected_nodes = set()

        for _, row in (
            connected_edges.iterrows()
        ):

            connected_nodes.add(
                row["source"]
            )

            connected_nodes.add(
                row["target"]
            )

        connected_nodes.discard(
            node_id
        )

        node_records = []

        for connected_id in (
            connected_nodes
        ):

            if connected_id not in (
                self.node_lookup
            ):
                continue

            record = {
                "node_id": connected_id
            }

            record.update(
                self.node_lookup[
                    connected_id
                ]
            )

            node_records.append(
                record
            )

        connected_df = pd.DataFrame(
            node_records
        )

        return {
            "query": "transaction_context",
            "txid": normalize_txid(txid),
            "transaction": self.get_node(
                node_id
            ),
            "connected_node_count": int(
                len(connected_df)
            ),
            "connected_nodes": (
                connected_df
                .to_dict(
                    orient="records"
                )
                if not connected_df.empty
                else []
            ),
            "relationships": (
                connected_edges
                .to_dict(
                    orient="records"
                )
            ),
        }

    # ================================================================
    # 2. WALLET -> CONNECTED TRANSACTIONS
    # ================================================================

    def wallet_context(
        self,
        address: str,
    ) -> dict[str, Any]:

        node_id = (
            f"wallet:{address}"
        )

        if not self.node_exists(
            node_id
        ):
            raise KeyError(
                f"Wallet not found: {address}"
            )

        outgoing = self.edges[
            self.edges["source"]
            == node_id
        ]

        incoming = self.edges[
            self.edges["target"]
            == node_id
        ]

        connected_edges = pd.concat(
            [
                outgoing,
                incoming,
            ],
            ignore_index=True,
        ).drop_duplicates(
            subset=["edge_id"]
        )

        txids = set()

        for _, row in (
            connected_edges.iterrows()
        ):

            for endpoint in (
                row["source"],
                row["target"],
            ):

                if str(endpoint).startswith(
                    "tx:"
                ):

                    txids.add(
                        str(endpoint)[3:]
                    )

        transactions = []

        for txid in txids:

            tx_node = (
                f"tx:{txid}"
            )

            if tx_node not in (
                self.node_lookup
            ):
                continue

            record = {
                "txid": txid
            }

            record.update(
                self.node_lookup[
                    tx_node
                ]
            )

            transactions.append(
                record
            )

        transactions_df = (
            pd.DataFrame(
                transactions
            )
        )

        if not transactions_df.empty:

            if "time_step" in (
                transactions_df.columns
            ):

                transactions_df = (
                    transactions_df
                    .sort_values(
                        "time_step"
                    )
                )

        return {
            "query": "wallet_context",
            "wallet": address,
            "transaction_count": int(
                len(transactions_df)
            ),
            "transactions": (
                transactions_df
                .to_dict(
                    orient="records"
                )
                if not transactions_df.empty
                else []
            ),
            "relationships": (
                connected_edges
                .to_dict(
                    orient="records"
                )
            ),
        }

    # ================================================================
    # 3. IP -> OBSERVED TRANSACTIONS
    # ================================================================

    def ip_context(
        self,
        ip_address: str,
    ) -> dict[str, Any]:

        node_id = (
            f"ip:{ip_address}"
        )

        if not self.node_exists(
            node_id
        ):
            raise KeyError(
                f"IP not found: {ip_address}"
            )

        connected_edges = self.edges[
            (
                self.edges["source"]
                == node_id
            )
            |
            (
                self.edges["target"]
                == node_id
            )
        ].copy()

        txids = set()

        for _, row in (
            connected_edges.iterrows()
        ):

            for endpoint in (
                row["source"],
                row["target"],
            ):

                if str(endpoint).startswith(
                    "tx:"
                ):

                    txids.add(
                        str(endpoint)[3:]
                    )

        transactions = []

        for txid in txids:

            tx_node = (
                f"tx:{txid}"
            )

            if tx_node not in (
                self.node_lookup
            ):
                continue

            record = {
                "txid": txid
            }

            record.update(
                self.node_lookup[
                    tx_node
                ]
            )

            transactions.append(
                record
            )

        return {
            "query": "ip_context",
            "ip_address": ip_address,
            "transaction_count": int(
                len(transactions)
            ),
            "transactions": transactions,
            "relationships": (
                connected_edges
                .to_dict(
                    orient="records"
                )
            ),
        }

    # ================================================================
    # 4. ONE-HOP INVESTIGATION NEIGHBORHOOD
    # ================================================================

    def neighborhood(
        self,
        node_id: str,
    ) -> dict[str, Any]:

        node_id = normalize_node_id(
            node_id
        )

        if node_id is None:
            raise ValueError(
                "Invalid node ID."
            )

        if not self.node_exists(
            node_id
        ):
            raise KeyError(
                f"Node not found: {node_id}"
            )

        edges = self.edges[
            (
                self.edges["source"]
                == node_id
            )
            |
            (
                self.edges["target"]
                == node_id
            )
        ].copy()

        neighbors = set()

        for _, row in (
            edges.iterrows()
        ):

            neighbors.add(
                row["source"]
            )

            neighbors.add(
                row["target"]
            )

        neighbors.discard(
            node_id
        )

        neighbor_records = []

        for neighbor in (
            neighbors
        ):

            if neighbor not in (
                self.node_lookup
            ):
                continue

            record = {
                "node_id": neighbor
            }

            record.update(
                self.node_lookup[
                    neighbor
                ]
            )

            neighbor_records.append(
                record
            )

        return {
            "query": "neighborhood",
            "node_id": node_id,
            "neighbor_count": int(
                len(neighbor_records)
            ),
            "neighbors": neighbor_records,
            "relationships": (
                edges
                .to_dict(
                    orient="records"
                )
            ),
        }

    # ================================================================
    # 5. TOP-RISK TRANSACTIONS
    # ================================================================

    def top_risk_transactions(
        self,
        top_n: int = DEFAULT_TOP_N,
    ) -> dict[str, Any]:

        transactions = self.nodes[
            self.nodes["node_type"]
            == "transaction"
        ].copy()

        if "risk_score" not in (
            transactions.columns
        ):

            raise ValueError(
                "Graph does not contain "
                "risk_score attributes."
            )

        transactions[
            "risk_score_numeric"
        ] = pd.to_numeric(
            transactions[
                "risk_score"
            ],
            errors="coerce",
        )

        transactions = (
            transactions
            .sort_values(
                "risk_score_numeric",
                ascending=False,
                na_position="last",
            )
            .head(
                top_n
            )
        )

        transactions[
            "risk_score_numeric"
        ] = transactions[
            "risk_score_numeric"
        ].round(6)

        return {
            "query": "top_risk_transactions",
            "top_n": int(top_n),
            "transactions": (
                transactions
                .drop(
                    columns=[
                        "risk_score_numeric"
                    ],
                    errors="ignore",
                )
                .to_dict(
                    orient="records"
                )
            ),
        }

    # ================================================================
    # 6. SHORTEST PATH
    # ================================================================

    def shortest_path(
        self,
        source: str,
        target: str,
        max_depth: int = 6,
    ) -> dict[str, Any]:

        source = normalize_node_id(
            source
        )

        target = normalize_node_id(
            target
        )

        if (
            source is None
            or target is None
        ):

            raise ValueError(
                "Invalid source or target."
            )

        if not self.node_exists(
            source
        ):

            raise KeyError(
                f"Source node not found: "
                f"{source}"
            )

        if not self.node_exists(
            target
        ):

            raise KeyError(
                f"Target node not found: "
                f"{target}"
            )

        if source == target:

            return {
                "query": "shortest_path",
                "source": source,
                "target": target,
                "found": True,
                "path_length": 0,
                "path": [source],
                "relationships": [],
            }

        # Build a local adjacency structure
        # without constructing a NetworkX graph.
        adjacency = defaultdict(set)

        for row in self.edges.itertuples(
            index=False
        ):

            adjacency[
                row.source
            ].add(
                row.target
            )

            adjacency[
                row.target
            ].add(
                row.source
            )

        queue = [
            (
                source,
                [source],
            )
        ]

        visited = {
            source
        }

        found_path = None

        while queue:

            current, path = (
                queue.pop(0)
            )

            if len(path) - 1 >= (
                max_depth
            ):
                continue

            for neighbor in (
                adjacency[current]
            ):

                if neighbor in visited:
                    continue

                new_path = (
                    path
                    + [neighbor]
                )

                if neighbor == target:

                    found_path = (
                        new_path
                    )

                    queue = []
                    break

                visited.add(
                    neighbor
                )

                queue.append(
                    (
                        neighbor,
                        new_path,
                    )
                )

        if found_path is None:

            return {
                "query": "shortest_path",
                "source": source,
                "target": target,
                "found": False,
                "path_length": None,
                "path": [],
                "relationships": [],
            }

        relationships = []

        for left, right in zip(
            found_path,
            found_path[1:],
        ):

            matches = self.edges[
                (
                    (
                        self.edges["source"]
                        == left
                    )
                    &
                    (
                        self.edges["target"]
                        == right
                    )
                )
                |
                (
                    (
                        self.edges["source"]
                        == right
                    )
                    &
                    (
                        self.edges["target"]
                        == left
                    )
                )
            ]

            relationships.extend(
                matches.to_dict(
                    orient="records"
                )
            )

        return {
            "query": "shortest_path",
            "source": source,
            "target": target,
            "found": True,
            "path_length": int(
                len(found_path) - 1
            ),
            "path": found_path,
            "relationships": relationships,
        }


def json_safe(
    value: Any,
) -> Any:

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key): json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):

        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):

        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        pd.Timestamp,
    ):

        return value.isoformat()

    try:

        if pd.isna(value):
            return None

    except (
        TypeError,
        ValueError,
    ):
        pass

    if hasattr(
        value,
        "item",
    ):

        try:
            return value.item()
        except (
            ValueError,
            TypeError,
        ):
            pass

    return value


def run_validation(
    graph: InvestigationGraph,
) -> dict[str, Any]:

    print()
    print(
        "=" * 72
    )

    print(
        "M12.2 GRAPH INVESTIGATION QUERY VALIDATION"
    )

    print(
        "=" * 72
    )

    results = {}

    # --------------------------------------------------------------
    # Find a transaction with graph context.
    # --------------------------------------------------------------

    transaction_nodes = graph.nodes[
        graph.nodes["node_type"]
        == "transaction"
    ].copy()

    if transaction_nodes.empty:
        raise ValueError(
            "No transaction nodes available."
        )

    if "investigation_rank" in (
        transaction_nodes.columns
    ):

        transaction_nodes[
            "_rank"
        ] = pd.to_numeric(
            transaction_nodes[
                "investigation_rank"
            ],
            errors="coerce",
        )

        transaction_nodes = (
            transaction_nodes
            .sort_values(
                "_rank",
                na_position="last",
            )
        )

    selected_txid = normalize_txid(
        transaction_nodes.iloc[0][
            "node_id"
        ].replace(
            "tx:",
            "",
            1,
        )
    )

    print()
    print(
        f"Validation TXID: "
        f"{selected_txid}"
    )

    # --------------------------------------------------------------
    # Query 1
    # --------------------------------------------------------------

    context = (
        graph.transaction_context(
            selected_txid
        )
    )

    results[
        "transaction_context"
    ] = {
        "status": "PASS",
        "txid": selected_txid,
        "connected_node_count": context[
            "connected_node_count"
        ],
        "relationship_count": len(
            context[
                "relationships"
            ]
        ),
    }

    print(
        "Transaction context: PASS"
    )

    # --------------------------------------------------------------
    # Query 2
    # --------------------------------------------------------------

    wallet_candidates = graph.nodes[
        graph.nodes["node_type"]
        == "wallet"
    ]

    if wallet_candidates.empty:
        raise ValueError(
            "No wallet nodes available."
        )

    wallet_address = (
        wallet_candidates.iloc[0][
            "address"
        ]
    )

    wallet_result = (
        graph.wallet_context(
            wallet_address
        )
    )

    results[
        "wallet_context"
    ] = {
        "status": "PASS",
        "wallet": wallet_address,
        "transaction_count": wallet_result[
            "transaction_count"
        ],
    }

    print(
        "Wallet context: PASS"
    )

    # --------------------------------------------------------------
    # Query 3
    # --------------------------------------------------------------

    ip_candidates = graph.nodes[
        graph.nodes["node_type"]
        == "ip"
    ]

    if ip_candidates.empty:
        raise ValueError(
            "No IP nodes available."
        )

    ip_address = (
        ip_candidates.iloc[0][
            "ip_address"
        ]
    )

    ip_result = (
        graph.ip_context(
            ip_address
        )
    )

    results[
        "ip_context"
    ] = {
        "status": "PASS",
        "ip_address": ip_address,
        "transaction_count": ip_result[
            "transaction_count"
        ],
    }

    print(
        "IP context: PASS"
    )

    # --------------------------------------------------------------
    # Query 4
    # --------------------------------------------------------------

    neighborhood_result = (
        graph.neighborhood(
            f"tx:{selected_txid}"
        )
    )

    results[
        "neighborhood"
    ] = {
        "status": "PASS",
        "node_id": f"tx:{selected_txid}",
        "neighbor_count": neighborhood_result[
            "neighbor_count"
        ],
    }

    print(
        "Neighborhood query: PASS"
    )

    # --------------------------------------------------------------
    # Query 5
    # --------------------------------------------------------------

    top_risk = (
        graph.top_risk_transactions(
            10
        )
    )

    results[
        "top_risk_transactions"
    ] = {
        "status": "PASS",
        "returned": len(
            top_risk[
                "transactions"
            ]
        ),
    }

    print(
        "Top-risk query: PASS"
    )

    # --------------------------------------------------------------
    # Query 6
    # --------------------------------------------------------------

    target_transaction = (
        transaction_nodes.iloc[
            min(
                1,
                len(transaction_nodes) - 1,
            )
        ][
            "node_id"
        ]
    )

    path_result = (
        graph.shortest_path(
            f"tx:{selected_txid}",
            target_transaction,
            max_depth=4,
        )
    )

    results[
        "shortest_path"
    ] = {
        "status": "PASS",
        "source": f"tx:{selected_txid}",
        "target": target_transaction,
        "found": path_result[
            "found"
        ],
        "path_length": path_result[
            "path_length"
        ],
    }

    print(
        "Shortest-path query: PASS"
    )

    # --------------------------------------------------------------
    # Overall
    # --------------------------------------------------------------

    results[
        "status"
    ] = "PASS"

    results[
        "graph_nodes"
    ] = int(
        len(graph.nodes)
    )

    results[
        "graph_edges"
    ] = int(
        len(graph.edges)
    )

    results[
        "supported_queries"
    ] = [
        "transaction_context",
        "wallet_context",
        "ip_context",
        "neighborhood",
        "top_risk_transactions",
        "shortest_path",
    ]

    return results


def print_result(
    result: dict[str, Any],
) -> None:

    print(
        json.dumps(
            json_safe(result),
            indent=2,
        )
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "M12.2 Bitcoin investigation "
            "graph query engine"
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command"
        )
    )

    tx_parser = (
        subparsers.add_parser(
            "tx"
        )
    )

    tx_parser.add_argument(
        "txid"
    )

    wallet_parser = (
        subparsers.add_parser(
            "wallet"
        )
    )

    wallet_parser.add_argument(
        "address"
    )

    ip_parser = (
        subparsers.add_parser(
            "ip"
        )
    )

    ip_parser.add_argument(
        "ip_address"
    )

    neighborhood_parser = (
        subparsers.add_parser(
            "neighborhood"
        )
    )

    neighborhood_parser.add_argument(
        "node_id"
    )

    top_parser = (
        subparsers.add_parser(
            "top-risk"
        )
    )

    top_parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
    )

    path_parser = (
        subparsers.add_parser(
            "path"
        )
    )

    path_parser.add_argument(
        "source"
    )

    path_parser.add_argument(
        "target"
    )

    path_parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
    )

    subparsers.add_parser(
        "validate"
    )

    args = parser.parse_args()

    graph = (
        InvestigationGraph()
    )

    if args.command == "tx":

        result = (
            graph.transaction_context(
                args.txid
            )
        )

        print_result(
            result
        )

    elif args.command == "wallet":

        result = (
            graph.wallet_context(
                args.address
            )
        )

        print_result(
            result
        )

    elif args.command == "ip":

        result = (
            graph.ip_context(
                args.ip_address
            )
        )

        print_result(
            result
        )

    elif args.command == "neighborhood":

        result = (
            graph.neighborhood(
                args.node_id
            )
        )

        print_result(
            result
        )

    elif args.command == "top-risk":

        result = (
            graph.top_risk_transactions(
                args.top
            )
        )

        print_result(
            result
        )

    elif args.command == "path":

        result = (
            graph.shortest_path(
                args.source,
                args.target,
                args.max_depth,
            )
        )

        print_result(
            result
        )

    elif args.command == "validate":

        results = run_validation(
            graph
        )

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            REPORT_PATH,
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                json_safe(results),
                handle,
                indent=2,
            )

        print()
        print(
            f"Report: {REPORT_PATH}"
        )

        print()
        print(
            "M12.2 GRAPH INVESTIGATION "
            "QUERY ENGINE COMPLETE"
        )

    else:

        parser.print_help()


if __name__ == "__main__":
    main()