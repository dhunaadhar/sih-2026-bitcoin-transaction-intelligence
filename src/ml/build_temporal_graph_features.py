"""
M6.13 — TEMPORAL-SAFE GRAPH FEATURE REBUILD

Builds causal graph/entity features from the original Elliptic++
address/transaction edge files.

STRICT TEMPORAL POLICY
----------------------
For every transaction at time t:

    features(transaction_t)
        use ONLY graph state from time steps < t

All transactions belonging to the same time step are therefore
evaluated against the SAME historical graph state.

Only AFTER every transaction in time t has been evaluated are all
transactions from time t inserted into the historical graph.

Source:
    AddrTx_edgelist.csv
        input_address, txId

    TxAddr_edgelist.csv
        txId, output_address

    canonical_transactions.parquet
        txid, time_step

Outputs:
    data/derived/temporal_entity_evidence.parquet
    data/derived/temporal_transaction_entity_evidence.parquet
    reports/ml/temporal_graph_features.json
"""

from pathlib import Path
from collections import defaultdict, Counter
import json

import pandas as pd
import numpy as np


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

EXTERNAL_ROOT = (
    ROOT.parent
    / "data"
    / "external"
    / "elipticpp"
)

ADDR_TX_PATH = (
    EXTERNAL_ROOT
    / "actor"
    / "AddrTx_edgelist.csv"
)

TX_ADDR_PATH = (
    EXTERNAL_ROOT
    / "actor"
    / "TxAddr_edgelist.csv"
)

CANONICAL_PATH = (
    ROOT
    / "data"
    / "canonical"
    / "canonical_transactions.parquet"
)

OUTPUT_ENTITY = (
    ROOT
    / "data"
    / "derived"
    / "temporal_entity_evidence.parquet"
)

OUTPUT_TX = (
    ROOT
    / "data"
    / "derived"
    / "temporal_transaction_entity_evidence.parquet"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "ml"
    / "temporal_graph_features.json"
)


# =====================================================================
# CONFIGURATION
# =====================================================================

CHUNK_SIZE = 500_000


# =====================================================================
# UNION-FIND
# =====================================================================

class UnionFind:
    """
    Incremental disjoint-set structure.

    Used independently for thresholds:

        t1 = pair observed >= 1 historical time
        t2 = pair observed >= 2 historical times
        t3 = pair observed >= 3 historical times
    """

    def __init__(self):
        self.parent = {}
        self.size = {}

    def add(self, node):

        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1

    def find(self, node):

        if node not in self.parent:
            self.add(node)
            return node

        root = node

        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[node] != node:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent

        return root

    def union(self, a, b):

        self.add(a)
        self.add(b)

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return root_a

        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]

        return root_a

    def component_size(self, node):

        if node not in self.parent:
            return 1

        return self.size[
            self.find(node)
        ]

    def component_id(self, node):

        if node not in self.parent:
            return None

        return self.find(node)


# =====================================================================
# HELPERS
# =====================================================================

def normalize_id(value):
    """
    Normalize transaction IDs so numeric CSV values and canonical
    transaction IDs can be matched safely.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass

    return text


def normalize_address(value):

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def build_membership_maps():

    print()
    print("Loading AddrTx transaction-input relationships...")

    if not ADDR_TX_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {ADDR_TX_PATH}"
        )

    if not TX_ADDR_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {TX_ADDR_PATH}"
        )

    input_map = defaultdict(list)
    output_map = defaultdict(list)

    # -------------------------------------------------------------
    # INPUT RELATIONSHIPS
    # -------------------------------------------------------------

    for chunk in pd.read_csv(
        ADDR_TX_PATH,
        usecols=[
            "input_address",
            "txId",
        ],
        chunksize=CHUNK_SIZE,
    ):

        chunk["txId"] = chunk["txId"].map(
            normalize_id
        )

        chunk["input_address"] = chunk[
            "input_address"
        ].map(
            normalize_address
        )

        chunk = chunk.dropna(
            subset=[
                "txId",
                "input_address",
            ]
        )

        for txid, address in zip(
            chunk["txId"],
            chunk["input_address"],
        ):

            input_map[txid].append(address)

    print(
        f"  Transactions with inputs: "
        f"{len(input_map):,}"
    )

    # -------------------------------------------------------------
    # OUTPUT RELATIONSHIPS
    # -------------------------------------------------------------

    print(
        "Loading TxAddr transaction-output relationships..."
    )

    for chunk in pd.read_csv(
        TX_ADDR_PATH,
        usecols=[
            "txId",
            "output_address",
        ],
        chunksize=CHUNK_SIZE,
    ):

        chunk["txId"] = chunk["txId"].map(
            normalize_id
        )

        chunk["output_address"] = chunk[
            "output_address"
        ].map(
            normalize_address
        )

        chunk = chunk.dropna(
            subset=[
                "txId",
                "output_address",
            ]
        )

        for txid, address in zip(
            chunk["txId"],
            chunk["output_address"],
        ):

            output_map[txid].append(address)

    print(
        f"  Transactions with outputs: "
        f"{len(output_map):,}"
    )

    # -------------------------------------------------------------
    # DEDUPLICATE ADDRESS MEMBERSHIP
    # -------------------------------------------------------------

    for txid in list(input_map):

        input_map[txid] = sorted(
            set(input_map[txid])
        )

    for txid in list(output_map):

        output_map[txid] = sorted(
            set(output_map[txid])
        )

    return input_map, output_map


# =====================================================================
# GRAPH STATE
# =====================================================================

class TemporalGraph:

    def __init__(self):

        # Historical pair occurrence count.
        self.pair_counts = Counter()

        # Historical address neighbors at each threshold.
        self.neighbors_t1 = defaultdict(set)
        self.neighbors_t2 = defaultdict(set)
        self.neighbors_t3 = defaultdict(set)

        # Dynamic connected components.
        self.uf_t1 = UnionFind()
        self.uf_t2 = UnionFind()
        self.uf_t3 = UnionFind()

        # Historical transaction counts.
        self.address_transaction_count = Counter()

        # Historical input/output counts.
        self.address_input_count = Counter()
        self.address_output_count = Counter()

    def add_address(self, address):

        self.uf_t1.add(address)
        self.uf_t2.add(address)
        self.uf_t3.add(address)

    def pair_threshold_state(
        self,
        address,
        threshold,
    ):

        if threshold == 1:

            neighbors = self.neighbors_t1
            uf = self.uf_t1

        elif threshold == 2:

            neighbors = self.neighbors_t2
            uf = self.uf_t2

        else:

            neighbors = self.neighbors_t3
            uf = self.uf_t3

        degree = len(
            neighbors.get(
                address,
                (),
            )
        )

        if degree == 0:

            cluster_size = 1
            cluster_id = None

        else:

            cluster_id = uf.component_id(
                address
            )

            cluster_size = uf.component_size(
                address
            )

        return (
            degree,
            cluster_size,
            cluster_id,
        )

    def address_stats(
        self,
        address,
    ):

        (
            degree_t1,
            cluster_t1,
            component_t1,
        ) = self.pair_threshold_state(
            address,
            1,
        )

        (
            degree_t2,
            cluster_t2,
            component_t2,
        ) = self.pair_threshold_state(
            address,
            2,
        )

        (
            degree_t3,
            cluster_t3,
            component_t3,
        ) = self.pair_threshold_state(
            address,
            3,
        )

        return {
            "address": address,

            "historical_transaction_count": int(
                self.address_transaction_count[
                    address
                ]
            ),

            "historical_input_count": int(
                self.address_input_count[
                    address
                ]
            ),

            "historical_output_count": int(
                self.address_output_count[
                    address
                ]
            ),

            "repeated_cosponsor_degree": int(
                degree_t2
            ),

            "max_shared_transaction_count": int(
                self.max_shared_count(
                    address
                )
            ),

            "cluster_size_t1": int(
                cluster_t1
            ),

            "cluster_size_t2": int(
                cluster_t2
            ),

            "cluster_size_t3": int(
                cluster_t3
            ),

            "cluster_id_t1": component_t1,

            "cluster_id_t2": component_t2,

            "cluster_id_t3": component_t3,
        }

    def max_shared_count(
        self,
        address,
    ):

        neighbors = set(
            self.neighbors_t1.get(
                address,
                (),
            )
        )

        if not neighbors:
            return 0

        maximum = 0

        for neighbor in neighbors:

            pair = (
                address,
                neighbor,
            )

            if pair[0] > pair[1]:

                pair = (
                    pair[1],
                    pair[0],
                )

            count = self.pair_counts.get(
                pair,
                0,
            )

            if count > maximum:
                maximum = count

        return maximum

    def add_transaction(
        self,
        inputs,
        outputs,
    ):

        # ---------------------------------------------------------
        # Ensure current addresses exist.
        # ---------------------------------------------------------

        for address in inputs:
            self.add_address(address)

        for address in outputs:
            self.add_address(address)

        # ---------------------------------------------------------
        # Input co-sponsorship.
        # ---------------------------------------------------------

        unique_inputs = sorted(
            set(inputs)
        )

        for i in range(
            len(unique_inputs)
        ):

            for j in range(
                i + 1,
                len(unique_inputs),
            ):

                a = unique_inputs[i]
                b = unique_inputs[j]

                if a == b:
                    continue

                pair = (
                    a,
                    b,
                )

                previous_count = (
                    self.pair_counts[pair]
                )

                new_count = (
                    previous_count + 1
                )

                self.pair_counts[pair] = (
                    new_count
                )

                # Threshold 1.
                if previous_count < 1:

                    self.neighbors_t1[
                        a
                    ].add(b)

                    self.neighbors_t1[
                        b
                    ].add(a)

                    self.uf_t1.union(
                        a,
                        b,
                    )

                # Threshold 2.
                if (
                    previous_count < 2
                    and new_count >= 2
                ):

                    self.neighbors_t2[
                        a
                    ].add(b)

                    self.neighbors_t2[
                        b
                    ].add(a)

                    self.uf_t2.union(
                        a,
                        b,
                    )

                # Threshold 3.
                if (
                    previous_count < 3
                    and new_count >= 3
                ):

                    self.neighbors_t3[
                        a
                    ].add(b)

                    self.neighbors_t3[
                        b
                    ].add(a)

                    self.uf_t3.union(
                        a,
                        b,
                    )

        # ---------------------------------------------------------
        # Update historical address counts.
        # ---------------------------------------------------------

        for address in set(inputs):

            self.address_transaction_count[
                address
            ] += 1

            self.address_input_count[
                address
            ] += 1

        for address in set(outputs):

            self.address_transaction_count[
                address
            ] += 1

            self.address_output_count[
                address
            ] += 1


# =====================================================================
# TRANSACTION FEATURE GENERATION
# =====================================================================

def aggregate_address_features(
    graph,
    addresses,
):

    addresses = sorted(
        set(addresses)
    )

    if not addresses:

        return {
            "address_count": 0,
            "repeated_count": 0,
            "repeated_ratio": 0.0,
            "max_repeated_degree": 0,
            "max_shared_count": 0,
            "distinct_clusters_t1": 0,
            "distinct_clusters_t2": 0,
            "distinct_clusters_t3": 0,
            "max_cluster_size_t1": 1,
            "max_cluster_size_t2": 1,
            "max_cluster_size_t3": 1,
        }

    stats = [
        graph.address_stats(address)
        for address in addresses
    ]

    repeated_count = sum(
        value["repeated_cosponsor_degree"] > 0
        for value in stats
    )

    clusters_t1 = {
        value["cluster_id_t1"]
        for value in stats
        if value["cluster_id_t1"] is not None
    }

    clusters_t2 = {
        value["cluster_id_t2"]
        for value in stats
        if value["cluster_id_t2"] is not None
    }

    clusters_t3 = {
        value["cluster_id_t3"]
        for value in stats
        if value["cluster_id_t3"] is not None
    }

    return {
        "address_count": len(addresses),

        "repeated_count": int(
            repeated_count
        ),

        "repeated_ratio": float(
            repeated_count / len(addresses)
        ),

        "max_repeated_degree": int(
            max(
                value[
                    "repeated_cosponsor_degree"
                ]
                for value in stats
            )
        ),

        "max_shared_count": int(
            max(
                value[
                    "max_shared_transaction_count"
                ]
                for value in stats
            )
        ),

        "distinct_clusters_t1": int(
            len(clusters_t1)
        ),

        "distinct_clusters_t2": int(
            len(clusters_t2)
        ),

        "distinct_clusters_t3": int(
            len(clusters_t3)
        ),

        "max_cluster_size_t1": int(
            max(
                value["cluster_size_t1"]
                for value in stats
            )
        ),

        "max_cluster_size_t2": int(
            max(
                value["cluster_size_t2"]
                for value in stats
            )
        ),

        "max_cluster_size_t3": int(
            max(
                value["cluster_size_t3"]
                for value in stats
            )
        ),
    }


# =====================================================================
# BUILD TRANSACTION ROW
# =====================================================================

def build_transaction_row(
    graph,
    txid,
    time_step,
    inputs,
    outputs,
):

    input_features = (
        aggregate_address_features(
            graph,
            inputs,
        )
    )

    output_features = (
        aggregate_address_features(
            graph,
            outputs,
        )
    )

    row = {
        "txid": txid,
        "time_step": int(time_step),

        "input_repeated_cosponsor_count": (
            input_features[
                "repeated_count"
            ]
        ),

        "input_repeated_cosponsor_ratio": (
            input_features[
                "repeated_ratio"
            ]
        ),

        "output_repeated_cosponsor_count": (
            output_features[
                "repeated_count"
            ]
        ),

        "output_repeated_cosponsor_ratio": (
            output_features[
                "repeated_ratio"
            ]
        ),

        "input_max_repeated_cosponsor_degree": (
            input_features[
                "max_repeated_degree"
            ]
        ),

        "output_max_repeated_cosponsor_degree": (
            output_features[
                "max_repeated_degree"
            ]
        ),

        "input_max_shared_transaction_count": (
            input_features[
                "max_shared_count"
            ]
        ),

        "output_max_shared_transaction_count": (
            output_features[
                "max_shared_count"
            ]
        ),

        "input_distinct_clusters_t1": (
            input_features[
                "distinct_clusters_t1"
            ]
        ),

        "input_distinct_clusters_t2": (
            input_features[
                "distinct_clusters_t2"
            ]
        ),

        "input_distinct_clusters_t3": (
            input_features[
                "distinct_clusters_t3"
            ]
        ),

        "output_distinct_clusters_t1": (
            output_features[
                "distinct_clusters_t1"
            ]
        ),

        "output_distinct_clusters_t2": (
            output_features[
                "distinct_clusters_t2"
            ]
        ),

        "output_distinct_clusters_t3": (
            output_features[
                "distinct_clusters_t3"
            ]
        ),

        "input_max_cluster_size_t1": (
            input_features[
                "max_cluster_size_t1"
            ]
        ),

        "input_max_cluster_size_t2": (
            input_features[
                "max_cluster_size_t2"
            ]
        ),

        "input_max_cluster_size_t3": (
            input_features[
                "max_cluster_size_t3"
            ]
        ),

        "output_max_cluster_size_t1": (
            output_features[
                "max_cluster_size_t1"
            ]
        ),

        "output_max_cluster_size_t2": (
            output_features[
                "max_cluster_size_t2"
            ]
        ),

        "output_max_cluster_size_t3": (
            output_features[
                "max_cluster_size_t3"
            ]
        ),
    }

    # -------------------------------------------------------------
    # Historical address counts.
    # -------------------------------------------------------------

    input_entity_counts = [
        graph.address_transaction_count[
            address
        ]
        for address in inputs
    ]

    output_entity_counts = [
        graph.address_transaction_count[
            address
        ]
        for address in outputs
    ]

    row[
        "entity_input_address_count"
    ] = int(
        sum(
            count > 0
            for count in input_entity_counts
        )
    )

    row[
        "entity_output_address_count"
    ] = int(
        sum(
            count > 0
            for count in output_entity_counts
        )
    )

    return row


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print(
        "M6.13 — TEMPORAL-SAFE GRAPH FEATURE REBUILD"
    )
    print("=" * 70)

    # -------------------------------------------------------------
    # Verify sources.
    # -------------------------------------------------------------

    print()
    print("Checking source files...")

    for path in [
        ADDR_TX_PATH,
        TX_ADDR_PATH,
        CANONICAL_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required source not found: {path}"
            )

        print(
            f"  OK: {path}"
        )

    # -------------------------------------------------------------
    # Load canonical time mapping.
    # -------------------------------------------------------------

    print()
    print(
        "Loading canonical transaction time steps..."
    )

    canonical = pd.read_parquet(
        CANONICAL_PATH,
        columns=[
            "txid",
            "time_step",
        ],
    )

    canonical["txid"] = canonical[
        "txid"
    ].map(
        normalize_id
    )

    canonical = canonical.dropna(
        subset=[
            "txid",
            "time_step",
        ]
    )

    if canonical["txid"].duplicated().any():

        raise ValueError(
            "Canonical dataset contains duplicate txids."
        )

    tx_time = dict(
        zip(
            canonical["txid"],
            canonical["time_step"].astype(int),
        )
    )

    print(
        f"Canonical transactions: "
        f"{len(tx_time):,}"
    )

    print(
        f"Time range: "
        f"{min(tx_time.values())}"
        f"–"
        f"{max(tx_time.values())}"
    )

    # -------------------------------------------------------------
    # Load address memberships.
    # -------------------------------------------------------------

    input_map, output_map = (
        build_membership_maps()
    )

    # -------------------------------------------------------------
    # Build chronological transaction order.
    # -------------------------------------------------------------

    print()
    print(
        "Building chronological transaction order..."
    )

    all_txids = set(tx_time)

    transactions = []

    for txid in all_txids:

        transactions.append(
            (
                int(tx_time[txid]),
                txid,
            )
        )

    transactions.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    print(
        f"Transactions to process: "
        f"{len(transactions):,}"
    )

    # -------------------------------------------------------------
    # Group transactions by time step.
    #
    # IMPORTANT:
    #
    # Every transaction in the same time step must see the same
    # historical graph. Therefore graph updates happen only after
    # the entire time step has been evaluated.
    # -------------------------------------------------------------

    transactions_by_time = defaultdict(list)

    for time_step, txid in transactions:

        transactions_by_time[
            time_step
        ].append(txid)

    # -------------------------------------------------------------
    # Graph state.
    # -------------------------------------------------------------

    graph = TemporalGraph()

    entity_rows = {}
    transaction_rows = []

    sorted_time_steps = sorted(
        transactions_by_time
    )

    total_processed = 0

    # -------------------------------------------------------------
    # Chronological time-step processing.
    # -------------------------------------------------------------

    print()
    print(
        "Building strictly causal graph features..."
    )

    for time_index, time_step in enumerate(
        sorted_time_steps,
        start=1,
    ):

        current_txids = transactions_by_time[
            time_step
        ]

        print()
        print(
            f"Processing time step "
            f"{time_step} "
            f"({len(current_txids):,} transactions)"
        )

        # =========================================================
        # PHASE 1
        #
        # Calculate ALL features against graph state containing
        # ONLY transactions from time steps < current time_step.
        #
        # No graph mutation occurs in this phase.
        # =========================================================

        for txid in current_txids:

            inputs = input_map.get(
                txid,
                [],
            )

            outputs = output_map.get(
                txid,
                [],
            )

            row = build_transaction_row(
                graph,
                txid,
                time_step,
                inputs,
                outputs,
            )

            transaction_rows.append(
                row
            )

            # -----------------------------------------------------
            # Capture first-observation entity evidence.
            #
            # This is deliberately captured BEFORE the current
            # transaction enters historical state.
            # -----------------------------------------------------

            for address in set(
                inputs + outputs
            ):

                if address not in entity_rows:

                    stats = graph.address_stats(
                        address
                    )

                    entity_rows[
                        address
                    ] = {
                        "address": address,
                        **stats,
                    }

        # =========================================================
        # PHASE 2
        #
        # ONLY NOW insert all transactions from this time step into
        # the historical graph.
        #
        # Therefore the next time step sees the complete history of
        # this time step, while transactions within this time step
        # never see one another.
        # =========================================================

        print(
            f"  Updating historical graph with "
            f"{len(current_txids):,} transactions..."
        )

        for txid in current_txids:

            inputs = input_map.get(
                txid,
                [],
            )

            outputs = output_map.get(
                txid,
                [],
            )

            graph.add_transaction(
                inputs,
                outputs,
            )

        total_processed += len(
            current_txids
        )

        print(
            f"  Cumulative transactions processed: "
            f"{total_processed:,} / "
            f"{len(transactions):,}"
        )

    # -------------------------------------------------------------
    # DataFrames.
    # -------------------------------------------------------------

    print()
    print(
        "Creating output DataFrames..."
    )

    entity_df = pd.DataFrame(
        list(
            entity_rows.values()
        )
    )

    tx_df = pd.DataFrame(
        transaction_rows
    )

    # -------------------------------------------------------------
    # Validation.
    # -------------------------------------------------------------

    print()
    print(
        "Validating temporal graph artifacts..."
    )

    if len(tx_df) != len(
        transactions
    ):

        raise ValueError(
            "Transaction evidence row count mismatch."
        )

    if tx_df["txid"].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    if tx_df["time_step"].min() != min(
        tx_time.values()
    ):

        raise ValueError(
            "Minimum time_step mismatch."
        )

    if tx_df["time_step"].max() != max(
        tx_time.values()
    ):

        raise ValueError(
            "Maximum time_step mismatch."
        )

    numeric_columns = tx_df.select_dtypes(
        include=[np.number]
    ).columns

    infinite_count = int(
        np.isinf(
            tx_df[numeric_columns].to_numpy(
                dtype=float
            )
        ).sum()
    )

    negative_count = int(
        (
            tx_df[numeric_columns] < 0
        ).sum().sum()
    )

    if infinite_count != 0:

        raise ValueError(
            f"Found {infinite_count} infinite values."
        )

    if negative_count != 0:

        raise ValueError(
            f"Found {negative_count} negative values."
        )

    print(
        f"Transaction rows: "
        f"{len(tx_df):,}"
    )

    print(
        f"Unique transaction IDs: "
        f"{tx_df['txid'].nunique():,}"
    )

    print(
        f"Entity rows: "
        f"{len(entity_df):,}"
    )

    print(
        f"Infinite values: "
        f"{infinite_count}"
    )

    print(
        f"Negative values: "
        f"{negative_count}"
    )

    # -------------------------------------------------------------
    # Explicit time-step-1 causal sanity check.
    #
    # Because no earlier time step exists, every historical-state
    # feature must be zero and every cluster-size feature must be 1
    # for observed addresses.
    # -------------------------------------------------------------

    print()
    print(
        "Checking time-step-1 causal state..."
    )

    time1 = tx_df[
        tx_df["time_step"] == 1
    ]

    zero_state_features = [
        "input_repeated_cosponsor_count",
        "input_repeated_cosponsor_ratio",
        "output_repeated_cosponsor_count",
        "output_repeated_cosponsor_ratio",
        "input_max_repeated_cosponsor_degree",
        "output_max_repeated_cosponsor_degree",
        "input_max_shared_transaction_count",
        "output_max_shared_transaction_count",
        "input_distinct_clusters_t1",
        "input_distinct_clusters_t2",
        "input_distinct_clusters_t3",
        "output_distinct_clusters_t1",
        "output_distinct_clusters_t2",
        "output_distinct_clusters_t3",
        "entity_input_address_count",
        "entity_output_address_count",
    ]

    size_features = [
        "input_max_cluster_size_t1",
        "input_max_cluster_size_t2",
        "input_max_cluster_size_t3",
        "output_max_cluster_size_t1",
        "output_max_cluster_size_t2",
        "output_max_cluster_size_t3",
    ]

    zero_violations = {}

    for feature in zero_state_features:

        count = int(
            (
                time1[feature] != 0
            ).sum()
        )

        zero_violations[
            feature
        ] = count

    singleton_violations = {}

    for feature in size_features:

        count = int(
            (
                time1[feature] != 1
            ).sum()
        )

        singleton_violations[
            feature
        ] = count

    total_zero_violations = sum(
        zero_violations.values()
    )

    total_singleton_violations = sum(
        singleton_violations.values()
    )

    print(
        f"  Time-step-1 rows: "
        f"{len(time1):,}"
    )

    print(
        f"  Zero-state violations: "
        f"{total_zero_violations:,}"
    )

    print(
        f"  Singleton-size violations: "
        f"{total_singleton_violations:,}"
    )

    if total_zero_violations != 0:

        raise ValueError(
            "Temporal causality violation at time-step 1."
        )

    if total_singleton_violations != 0:

        raise ValueError(
            "Cluster-size initialization violation at time-step 1."
        )

    print(
        "  Time-step-1 causal state: PASS"
    )

    # -------------------------------------------------------------
    # Save artifacts.
    # -------------------------------------------------------------

    OUTPUT_ENTITY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    entity_df.to_parquet(
        OUTPUT_ENTITY,
        index=False,
    )

    tx_df.to_parquet(
        OUTPUT_TX,
        index=False,
    )

    # -------------------------------------------------------------
    # Nonzero statistics.
    # -------------------------------------------------------------

    graph_columns = [
        column
        for column in tx_df.columns
        if column not in {
            "txid",
            "time_step",
        }
    ]

    nonzero_features = {}

    for column in graph_columns:

        nonzero_features[
            column
        ] = int(
            (
                tx_df[column] != 0
            ).sum()
        )

    # -------------------------------------------------------------
    # Report.
    # -------------------------------------------------------------

    report = {
        "stage": "M6.13",

        "task": (
            "temporal_safe_graph_feature_rebuild"
        ),

        "status": "PASS",

        "sources": {
            "addr_tx": str(
                ADDR_TX_PATH
            ),
            "tx_addr": str(
                TX_ADDR_PATH
            ),
            "canonical_transactions": str(
                CANONICAL_PATH
            ),
        },

        "input_statistics": {
            "canonical_transactions": int(
                len(canonical)
            ),
            "transactions_processed": int(
                len(transactions)
            ),
            "transactions_with_inputs": int(
                len(input_map)
            ),
            "transactions_with_outputs": int(
                len(output_map)
            ),
            "time_steps": int(
                len(sorted_time_steps)
            ),
        },

        "output_statistics": {
            "entity_rows": int(
                len(entity_df)
            ),
            "transaction_rows": int(
                len(tx_df)
            ),
            "unique_txids": int(
                tx_df["txid"].nunique()
            ),
            "graph_feature_columns": int(
                len(graph_columns)
            ),
            "nonzero_counts": nonzero_features,
        },

        "temporal_policy": {
            "future_transactions_used": False,
            "same_time_step_transactions_used": False,
            "current_transaction_used_before_features": False,
            "historical_condition": (
                "only transactions from strictly earlier "
                "time steps are present in graph state"
            ),
            "graph_update_order": (
                "calculate ALL features for time t -> "
                "update graph with ALL transactions at time t"
            ),
        },

        "time1_validation": {
            "rows": int(
                len(time1)
            ),
            "zero_state_violations": int(
                total_zero_violations
            ),
            "singleton_size_violations": int(
                total_singleton_violations
            ),
            "status": "PASS",
        },

        "graph_definition": {
            "relationship": (
                "input addresses appearing together "
                "in a transaction"
            ),
            "graph_type": (
                "undirected co-sponsorship graph"
            ),
            "thresholds": {
                "t1": "pair observed at least once",
                "t2": "pair observed at least twice",
                "t3": "pair observed at least three times",
            },
        },

        "limitations": [
            (
                "Temporal-safe graph structure does not "
                "establish real-world identity."
            ),
            (
                "A cluster represents structural association, "
                "not proof of common ownership."
            ),
            (
                "The global M5 graph remains available for "
                "retrospective investigation and visualization."
            ),
        ],

        "next_step": (
            "Validate the regenerated temporal artifact and "
            "then integrate causal graph evidence into a new "
            "chronological ML benchmark."
        ),
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Final output.
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "M6.13 COMPLETE"
    )
    print("=" * 70)

    print(
        f"Entity rows:       {len(entity_df):,}"
    )

    print(
        f"Transaction rows:  {len(tx_df):,}"
    )

    print(
        "Future data used:  False"
    )

    print(
        "Same-time-step data used: False"
    )

    print(
        "Current tx before feature calculation: False"
    )

    print(
        "Time-step-1 causal state: PASS"
    )

    print()
    print(
        f"Entity artifact:"
        f"\n  {OUTPUT_ENTITY}"
    )

    print(
        f"Transaction artifact:"
        f"\n  {OUTPUT_TX}"
    )

    print(
        f"Report:"
        f"\n  {REPORT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()