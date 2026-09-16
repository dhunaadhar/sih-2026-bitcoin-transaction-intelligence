from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["input_address", "output_address"]


def build_address_graph(
    edge_path: str | Path,
    edge_output_path: str | Path,
    node_output_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    edge_path = Path(edge_path)
    edge_output_path = Path(edge_output_path)
    node_output_path = Path(node_output_path)

    df = pd.read_csv(
        edge_path,
        usecols=REQUIRED_COLUMNS,
        dtype={
            "input_address": "string",
            "output_address": "string",
        },
    )

    if df.isna().any().any():
        raise ValueError("Address graph contains null endpoint values.")

    if (df["input_address"].str.len() == 0).any():
        raise ValueError("Address graph contains empty input addresses.")

    if (df["output_address"].str.len() == 0).any():
        raise ValueError("Address graph contains empty output addresses.")

    df["is_self_loop"] = (
        df["input_address"] == df["output_address"]
    ).astype("int8")

    edge_frequency = (
        df.groupby(
            ["input_address", "output_address"],
            sort=False,
            dropna=False,
        )
        .size()
        .rename("edge_observation_count")
        .reset_index()
    )

    unique_edges = edge_frequency.copy()

    unique_edges["is_self_loop"] = (
        unique_edges["input_address"]
        == unique_edges["output_address"]
    ).astype("int8")

    # Remove self-loops from topology calculations.
    topology_edges = unique_edges[
        unique_edges["is_self_loop"] == 0
    ].copy()

    # Build reciprocal relationship lookup.
    reverse_pairs = set(
        zip(
            topology_edges["output_address"],
            topology_edges["input_address"],
        )
    )

    topology_edges["has_reciprocal_edge"] = [
        int((src, dst) in reverse_pairs)
        for src, dst in zip(
            topology_edges["input_address"],
            topology_edges["output_address"],
        )
    ]

    # Add reciprocal information back to the saved directed-edge artifact.
    reciprocal_lookup = topology_edges[
        [
            "input_address",
            "output_address",
            "has_reciprocal_edge",
        ]
    ]

    unique_edges = unique_edges.merge(
        reciprocal_lookup,
        on=["input_address", "output_address"],
        how="left",
    )

    unique_edges["has_reciprocal_edge"] = (
        unique_edges["has_reciprocal_edge"]
        .fillna(0)
        .astype("int8")
    )

    # Outgoing topology statistics.
    out_stats = (
        topology_edges.groupby("input_address", sort=False)
        .agg(
            out_degree=("output_address", "nunique"),
            out_edge_observations=("edge_observation_count", "sum"),
            out_counterparty_count=("output_address", "nunique"),
        )
        .reset_index()
        .rename(columns={"input_address": "address"})
    )

    # Incoming topology statistics.
    in_stats = (
        topology_edges.groupby("output_address", sort=False)
        .agg(
            in_degree=("input_address", "nunique"),
            in_edge_observations=("edge_observation_count", "sum"),
            in_counterparty_count=("input_address", "nunique"),
        )
        .reset_index()
        .rename(columns={"output_address": "address"})
    )

    # Self-loop statistics.
    self_loop_stats = (
        unique_edges[unique_edges["is_self_loop"] == 1]
        .groupby("input_address", sort=False)
        .agg(
            self_loop_edge_count=("edge_observation_count", "sum"),
            self_loop_present=("is_self_loop", "max"),
        )
        .reset_index()
        .rename(columns={"input_address": "address"})
    )

    # Complete address universe.
    addresses = pd.DataFrame(
        {
            "address": pd.unique(
                pd.concat(
                    [
                        unique_edges["input_address"],
                        unique_edges["output_address"],
                    ],
                    ignore_index=True,
                )
            )
        }
    )

    nodes = addresses.merge(
        out_stats,
        on="address",
        how="left",
    )

    nodes = nodes.merge(
        in_stats,
        on="address",
        how="left",
    )

    nodes = nodes.merge(
        self_loop_stats,
        on="address",
        how="left",
    )

    fill_zero_columns = [
        "out_degree",
        "out_edge_observations",
        "out_counterparty_count",
        "in_degree",
        "in_edge_observations",
        "in_counterparty_count",
        "self_loop_edge_count",
        "self_loop_present",
    ]

    nodes[fill_zero_columns] = (
        nodes[fill_zero_columns]
        .fillna(0)
        .astype("int64")
    )

    nodes["total_degree"] = (
        nodes["out_degree"] + nodes["in_degree"]
    )

    nodes["total_edge_observations"] = (
        nodes["out_edge_observations"]
        + nodes["in_edge_observations"]
    )

    nodes["directional_degree_imbalance"] = (
        nodes["out_degree"] - nodes["in_degree"]
    ).abs()

    nodes["has_non_self_loop_relationship"] = (
        nodes["total_degree"] > 0
    ).astype("int8")

    nodes = nodes.sort_values(
        "address",
        kind="stable",
    ).reset_index(drop=True)

    unique_edges = unique_edges.sort_values(
        ["input_address", "output_address"],
        kind="stable",
    ).reset_index(drop=True)

    edge_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    node_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    unique_edges.to_parquet(
        edge_output_path,
        index=False,
    )

    nodes.to_parquet(
        node_output_path,
        index=False,
    )

    return unique_edges, nodes


def main() -> None:

    project_root = Path(__file__).resolve().parents[2]

    source_path = (
        project_root
        / ".."
        / "data"
        / "external"
        / "elipticpp"
        / "actor"
        / "AddrAddr_edgelist.csv"
    )

    edge_output_path = (
        project_root
        / "data"
        / "graph"
        / "address_edges.parquet"
    )

    node_output_path = (
        project_root
        / "data"
        / "graph"
        / "address_nodes.parquet"
    )

    edges, nodes = build_address_graph(
        source_path,
        edge_output_path,
        node_output_path,
    )

    print("=== M5.3 ADDRESS GRAPH ===")
    print(f"EDGE ARTIFACT: {edge_output_path}")
    print(f"NODE ARTIFACT: {node_output_path}")
    print(f"UNIQUE DIRECTED EDGES: {len(edges)}")
    print(f"UNIQUE ADDRESSES: {len(nodes)}")
    print(
        "SELF-LOOP EDGES:",
        int(edges["is_self_loop"].sum()),
    )
    print(
        "RECIPROCAL EDGES:",
        int(edges["has_reciprocal_edge"].sum()),
    )
    print(
        "MAX OUT-DEGREE:",
        int(nodes["out_degree"].max()),
    )
    print(
        "MAX IN-DEGREE:",
        int(nodes["in_degree"].max()),
    )
    print(
        "NODE NULL COUNT:",
        int(nodes.isna().sum().sum()),
    )
    print(
        "EDGE NULL COUNT:",
        int(edges.isna().sum().sum()),
    )


if __name__ == "__main__":
    main()