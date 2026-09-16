from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def analyze_threshold(
    pairs: pd.DataFrame,
    all_addresses: pd.Series,
    threshold: int,
) -> dict:

    selected = pairs[
        pairs["co_spend_transaction_count"] >= threshold
    ]

    addresses = all_addresses.reset_index(drop=True)

    address_to_index = pd.Series(
        np.arange(len(addresses), dtype=np.int64),
        index=addresses,
    )

    if len(selected) == 0:
        return {
            "threshold": threshold,
            "selected_pairs": 0,
            "components": len(addresses),
            "singleton_components": len(addresses),
            "largest_component": 1 if len(addresses) else 0,
            "non_singleton_addresses": 0,
        }

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

    component_count, labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )

    component_sizes = np.bincount(labels)

    singleton_components = int(
        (component_sizes == 1).sum()
    )

    largest_component = int(
        component_sizes.max()
    )

    non_singleton_addresses = int(
        component_sizes[component_sizes > 1].sum()
    )

    return {
        "threshold": threshold,
        "selected_pairs": len(selected),
        "components": int(component_count),
        "singleton_components": singleton_components,
        "largest_component": largest_component,
        "non_singleton_addresses": non_singleton_addresses,
    }


def main() -> None:

    project_root = Path(__file__).resolve().parents[2]

    pair_path = (
        project_root
        / "data"
        / "graph"
        / "input_cosponsor_edges.parquet"
    )

    address_node_path = (
        project_root
        / "data"
        / "graph"
        / "address_nodes.parquet"
    )

    output_path = (
        project_root
        / "data"
        / "graph"
        / "cosponsor_cluster_thresholds.parquet"
    )

    pairs = pd.read_parquet(pair_path)

    nodes = pd.read_parquet(
        address_node_path,
        columns=["address"],
    )

    all_addresses = nodes["address"].astype("string")

    if all_addresses.duplicated().any():
        raise ValueError(
            "Address node artifact contains duplicate addresses."
        )

    results = []

    for threshold in [1, 2, 3]:
        print(
            f"Analyzing co-spend threshold >= {threshold}..."
        )

        result = analyze_threshold(
            pairs=pairs,
            all_addresses=all_addresses,
            threshold=threshold,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_parquet(
        output_path,
        index=False,
    )

    print()
    print("=== M5.7 CO-SPEND CLUSTER THRESHOLD ANALYSIS ===")
    print(results_df.to_string(index=False))
    print()
    print(f"ARTIFACT: {output_path}")
    print(
        "NULL COUNT:",
        int(results_df.isna().sum().sum()),
    )


if __name__ == "__main__":
    main()