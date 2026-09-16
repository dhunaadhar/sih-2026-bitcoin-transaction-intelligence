from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def build_entity_clusters(
    edge_path: str | Path,
    node_path: str | Path,
    cluster_output_path: str | Path,
    summary_output_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    edge_path = Path(edge_path)
    node_path = Path(node_path)
    cluster_output_path = Path(cluster_output_path)
    summary_output_path = Path(summary_output_path)

    edges = pd.read_parquet(edge_path)
    nodes = pd.read_parquet(node_path)

    required_edge_columns = [
        "input_address",
        "output_address",
        "is_self_loop",
    ]

    required_node_columns = ["address"]

    missing_edge_columns = [
        column
        for column in required_edge_columns
        if column not in edges.columns
    ]

    missing_node_columns = [
        column
        for column in required_node_columns
        if column not in nodes.columns
    ]

    if missing_edge_columns:
        raise ValueError(
            f"Missing edge columns: {missing_edge_columns}"
        )

    if missing_node_columns:
        raise ValueError(
            f"Missing node columns: {missing_node_columns}"
        )

    addresses = nodes["address"].astype("string").reset_index(drop=True)

    if addresses.duplicated().any():
        raise ValueError("Node artifact contains duplicate addresses.")

    address_to_index = pd.Series(
        np.arange(len(addresses), dtype=np.int64),
        index=addresses,
    )

    topology_edges = edges[
        edges["is_self_loop"] == 0
    ][
        ["input_address", "output_address"]
    ].copy()

    topology_edges["source_index"] = (
        topology_edges["input_address"]
        .map(address_to_index)
    )

    topology_edges["target_index"] = (
        topology_edges["output_address"]
        .map(address_to_index)
    )

    if topology_edges[
        ["source_index", "target_index"]
    ].isna().any().any():
        raise ValueError(
            "Graph edge contains an address missing from node artifact."
        )

    source = topology_edges["source_index"].to_numpy(
        dtype=np.int64
    )

    target = topology_edges["target_index"].to_numpy(
        dtype=np.int64
    )

    # The source graph is directed.
    # For entity connectivity only, each relationship is treated
    # as an undirected connection.
    row = np.concatenate([source, target])
    col = np.concatenate([target, source])

    data = np.ones(
        len(row),
        dtype=np.uint8,
    )

    graph = coo_matrix(
        (data, (row, col)),
        shape=(len(addresses), len(addresses)),
    ).tocsr()

    graph.data[:] = 1

    component_count, labels = connected_components(
        csgraph=graph,
        directed=False,
        return_labels=True,
    )

    # Deterministic component IDs:
    # sort components by their smallest address.
    component_first_address = {}

    for index, label in enumerate(labels):
        address = str(addresses.iloc[index])

        if (
            label not in component_first_address
            or address < component_first_address[label]
        ):
            component_first_address[label] = address

    ordered_components = sorted(
        component_first_address,
        key=lambda label: component_first_address[label],
    )

    label_to_cluster_id = {
        label: cluster_number
        for cluster_number, label in enumerate(
            ordered_components,
            start=1,
        )
    }

    cluster_ids = np.array(
        [
            label_to_cluster_id[label]
            for label in labels
        ],
        dtype=np.int64,
    )

    clusters = pd.DataFrame(
        {
            "address": addresses,
            "cluster_id": cluster_ids,
        }
    )

    cluster_sizes = (
        clusters.groupby("cluster_id")
        .size()
        .rename("cluster_address_count")
    )

    clusters = clusters.merge(
        cluster_sizes,
        on="cluster_id",
        how="left",
    )

    # Degree information from the node artifact.
    degree_columns = [
        "out_degree",
        "in_degree",
        "total_degree",
        "out_edge_observations",
        "in_edge_observations",
        "self_loop_edge_count",
    ]

    available_degree_columns = [
        column
        for column in degree_columns
        if column in nodes.columns
    ]

    clusters = clusters.merge(
        nodes[
            ["address"] + available_degree_columns
        ],
        on="address",
        how="left",
        validate="one_to_one",
    )

    if clusters["cluster_id"].isna().any():
        raise ValueError("Cluster assignment contains null cluster IDs.")

    if clusters["address"].duplicated().any():
        raise ValueError(
            "Cluster artifact contains duplicate addresses."
        )

    # Cluster-level summary.
    summary = (
        clusters.groupby("cluster_id", sort=True)
        .agg(
            cluster_address_count=(
                "address",
                "count",
            ),
            cluster_total_out_degree=(
                "out_degree",
                "sum",
            ),
            cluster_total_in_degree=(
                "in_degree",
                "sum",
            ),
            cluster_max_out_degree=(
                "out_degree",
                "max",
            ),
            cluster_max_in_degree=(
                "in_degree",
                "max",
            ),
            cluster_total_edge_observations=(
                "out_edge_observations",
                "sum",
            ),
            cluster_self_loop_observations=(
                "self_loop_edge_count",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["cluster_is_singleton"] = (
        summary["cluster_address_count"] == 1
    ).astype("int8")

    summary["cluster_address_count_log1p"] = np.log1p(
        summary["cluster_address_count"]
    )

    summary["cluster_degree_sum"] = (
        summary["cluster_total_out_degree"]
        + summary["cluster_total_in_degree"]
    )

    summary = summary.sort_values(
        "cluster_id",
        kind="stable",
    ).reset_index(drop=True)

    cluster_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clusters.to_parquet(
        cluster_output_path,
        index=False,
    )

    summary.to_parquet(
        summary_output_path,
        index=False,
    )

    return clusters, summary


def main() -> None:

    project_root = Path(__file__).resolve().parents[2]

    edge_path = (
        project_root
        / "data"
        / "graph"
        / "address_edges.parquet"
    )

    node_path = (
        project_root
        / "data"
        / "graph"
        / "address_nodes.parquet"
    )

    cluster_output_path = (
        project_root
        / "data"
        / "graph"
        / "address_clusters.parquet"
    )

    summary_output_path = (
        project_root
        / "data"
        / "graph"
        / "cluster_summary.parquet"
    )

    clusters, summary = build_entity_clusters(
        edge_path=edge_path,
        node_path=node_path,
        cluster_output_path=cluster_output_path,
        summary_output_path=summary_output_path,
    )

    print("=== M5.4 ENTITY CLUSTERING ===")
    print(
        f"CLUSTER ARTIFACT: {cluster_output_path}"
    )
    print(
        f"SUMMARY ARTIFACT: {summary_output_path}"
    )
    print(
        f"ADDRESSES ASSIGNED: {len(clusters)}"
    )
    print(
        f"CLUSTERS: {len(summary)}"
    )
    print(
        "SINGLETON CLUSTERS:",
        int(summary["cluster_is_singleton"].sum()),
    )
    print(
        "LARGEST CLUSTER:",
        int(summary["cluster_address_count"].max()),
    )
    print(
        "CLUSTER NULL COUNT:",
        int(clusters.isna().sum().sum()),
    )
    print(
        "SUMMARY NULL COUNT:",
        int(summary.isna().sum().sum()),
    )
    print()
    print("TOP 10 CLUSTERS BY SIZE:")
    print(
        summary.nlargest(
            10,
            "cluster_address_count",
        )[
            [
                "cluster_id",
                "cluster_address_count",
                "cluster_total_edge_observations",
                "cluster_max_out_degree",
                "cluster_max_in_degree",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()