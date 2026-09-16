from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def compute_components(
    pairs: pd.DataFrame,
    addresses: pd.Series,
    threshold: int,
) -> np.ndarray:

    selected = pairs[
        pairs["co_spend_transaction_count"] >= threshold
    ]

    address_to_index = pd.Series(
        np.arange(len(addresses), dtype=np.int64),
        index=addresses,
    )

    if len(selected) == 0:
        return np.arange(
            1,
            len(addresses) + 1,
            dtype=np.int64,
        )

    source = selected["input_address_a"].map(
        address_to_index
    ).to_numpy(dtype=np.int64)

    target = selected["input_address_b"].map(
        address_to_index
    ).to_numpy(dtype=np.int64)

    row = np.concatenate([source, target])
    col = np.concatenate([target, source])

    data = np.ones(
        len(row),
        dtype=np.uint8,
    )

    graph = coo_matrix(
        (
            data,
            (row, col),
        ),
        shape=(len(addresses), len(addresses)),
    ).tocsr()

    graph.data[:] = 1

    _, labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )

    first_address = {}

    for index, label in enumerate(labels):
        address = str(addresses.iloc[index])

        if (
            label not in first_address
            or address < first_address[label]
        ):
            first_address[label] = address

    ordered_labels = sorted(
        first_address,
        key=lambda label: first_address[label],
    )

    label_to_cluster = {
        label: cluster_number
        for cluster_number, label in enumerate(
            ordered_labels,
            start=1,
        )
    }

    return np.array(
        [
            label_to_cluster[label]
            for label in labels
        ],
        dtype=np.int64,
    )


def build_entity_evidence(
    pair_path: str | Path,
    node_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:

    pair_path = Path(pair_path)
    node_path = Path(node_path)
    output_path = Path(output_path)

    pairs = pd.read_parquet(pair_path)

    nodes = pd.read_parquet(
        node_path,
        columns=["address"],
    )

    addresses = nodes["address"].astype("string")

    if addresses.duplicated().any():
        raise ValueError(
            "Address node artifact contains duplicate addresses."
        )

    evidence = pd.DataFrame(
        {
            "address": addresses,
        }
    )

    for threshold in [1, 2, 3]:

        print(
            f"Building candidate clusters at threshold >= {threshold}..."
        )

        cluster_ids = compute_components(
            pairs=pairs,
            addresses=addresses,
            threshold=threshold,
        )

        evidence[
            f"cosponsor_cluster_id_t{threshold}"
        ] = cluster_ids

    # Address-level repeated co-spend degree.
    repeated_pairs = pairs[
        pairs["co_spend_transaction_count"] >= 2
    ].copy()

    if len(repeated_pairs) > 0:

        degree_a = (
            repeated_pairs.groupby("input_address_a")
            .size()
            .rename("repeated_cosponsor_degree")
        )

        degree_b = (
            repeated_pairs.groupby("input_address_b")
            .size()
            .rename("repeated_cosponsor_degree")
        )

        degree = pd.concat(
            [degree_a, degree_b]
        ).groupby(level=0).sum()

        evidence = evidence.merge(
            degree.rename_axis("address")
            .reset_index(),
            on="address",
            how="left",
        )

        max_shared_a = (
            repeated_pairs.groupby("input_address_a")
            ["co_spend_transaction_count"]
            .max()
            .rename("max_shared_transaction_count")
        )

        max_shared_b = (
            repeated_pairs.groupby("input_address_b")
            ["co_spend_transaction_count"]
            .max()
            .rename("max_shared_transaction_count")
        )

        max_shared = pd.concat(
            [max_shared_a, max_shared_b]
        ).groupby(level=0).max()

        evidence = evidence.merge(
            max_shared.rename_axis("address")
            .reset_index(),
            on="address",
            how="left",
        )

    else:

        evidence["repeated_cosponsor_degree"] = 0
        evidence["max_shared_transaction_count"] = 0

    evidence[
        "repeated_cosponsor_degree"
    ] = evidence[
        "repeated_cosponsor_degree"
    ].fillna(0).astype("int64")

    evidence[
        "max_shared_transaction_count"
    ] = evidence[
        "max_shared_transaction_count"
    ].fillna(0).astype("int64")

    evidence[
        "has_repeated_cosponsor"
    ] = (
        evidence["repeated_cosponsor_degree"] > 0
    ).astype("int8")

    evidence[
        "repeated_cosponsor_degree_log1p"
    ] = np.log1p(
        evidence["repeated_cosponsor_degree"]
    )

    # Add cluster sizes for each threshold.
    for threshold in [1, 2, 3]:

        cluster_column = (
            f"cosponsor_cluster_id_t{threshold}"
        )

        size_column = (
            f"cosponsor_cluster_size_t{threshold}"
        )

        sizes = (
            evidence.groupby(cluster_column)
            .size()
            .rename(size_column)
        )

        evidence = evidence.merge(
            sizes,
            left_on=cluster_column,
            right_index=True,
            how="left",
        )

    evidence["cosponsor_cluster_size_t1_log1p"] = np.log1p(
        evidence["cosponsor_cluster_size_t1"]
    )

    evidence["cosponsor_cluster_size_t2_log1p"] = np.log1p(
        evidence["cosponsor_cluster_size_t2"]
    )

    evidence["cosponsor_cluster_size_t3_log1p"] = np.log1p(
        evidence["cosponsor_cluster_size_t3"]
    )

    evidence = evidence.sort_values(
        "address",
        kind="stable",
    ).reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence.to_parquet(
        output_path,
        index=False,
    )

    return evidence


def main() -> None:

    project_root = Path(__file__).resolve().parents[2]

    pair_path = (
        project_root
        / "data"
        / "graph"
        / "input_cosponsor_edges.parquet"
    )

    node_path = (
        project_root
        / "data"
        / "graph"
        / "address_nodes.parquet"
    )

    output_path = (
        project_root
        / "data"
        / "graph"
        / "entity_evidence.parquet"
    )

    evidence = build_entity_evidence(
        pair_path=pair_path,
        node_path=node_path,
        output_path=output_path,
    )

    print()
    print("=== M5.8 ENTITY EVIDENCE ===")
    print(f"ARTIFACT: {output_path}")
    print(f"ADDRESSES: {len(evidence)}")
    print(f"COLUMNS: {len(evidence.columns)}")
    print(
        "NULL COUNT:",
        int(evidence.isna().sum().sum()),
    )
    print(
        "ADDRESSES WITH REPEATED CO-SPONSOR:",
        int(
            evidence["has_repeated_cosponsor"].sum()
        ),
    )
    print(
        "MAX REPEATED CO-SPONSOR DEGREE:",
        int(
            evidence[
                "repeated_cosponsor_degree"
            ].max()
        ),
    )
    print(
        "MAX SHARED TRANSACTIONS:",
        int(
            evidence[
                "max_shared_transaction_count"
            ].max()
        ),
    )


if __name__ == "__main__":
    main()