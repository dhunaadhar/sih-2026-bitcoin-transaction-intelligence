from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_input_cosponsor_edges(
    input_edge_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:

    input_edge_path = Path(input_edge_path)
    output_path = Path(output_path)

    df = pd.read_csv(
        input_edge_path,
        dtype={
            "input_address": "string",
            "txId": "int64",
        },
    )

    if df.isna().any().any():
        raise ValueError("Input address-TX data contains null values.")

    # Remove accidental duplicate address-TX observations.
    df = df.drop_duplicates(
        ["input_address", "txId"]
    )

    # Transactions with one input cannot create a co-spend pair.
    counts = df.groupby("txId")["input_address"].nunique()

    multi_input_txids = counts[counts > 1].index

    multi = df[
        df["txId"].isin(multi_input_txids)
    ][
        ["txId", "input_address"]
    ].sort_values(
        ["txId", "input_address"],
        kind="stable",
    )

    # Self-join within each transaction to generate address pairs.
    pairs = multi.merge(
        multi,
        on="txId",
        suffixes=("_a", "_b"),
        how="inner",
    )

    pairs = pairs[
        pairs["input_address_a"]
        < pairs["input_address_b"]
    ].copy()

    # Count how many transactions jointly spend each address pair.
    pair_counts = (
        pairs.groupby(
            ["input_address_a", "input_address_b"],
            sort=False,
        )
        .size()
        .rename("co_spend_transaction_count")
        .reset_index()
    )

    pair_counts = pair_counts.sort_values(
        [
            "input_address_a",
            "input_address_b",
        ],
        kind="stable",
    ).reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_counts.to_parquet(
        output_path,
        index=False,
    )

    return pair_counts


def main() -> None:

    project_root = Path(__file__).resolve().parents[2]

    input_edge_path = (
        project_root
        / ".."
        / "data"
        / "external"
        / "elipticpp"
        / "actor"
        / "AddrTx_edgelist.csv"
    )

    output_path = (
        project_root
        / "data"
        / "graph"
        / "input_cosponsor_edges.parquet"
    )

    pairs = build_input_cosponsor_edges(
        input_edge_path=input_edge_path,
        output_path=output_path,
    )

    print("=== M5.6 INPUT CO-SPEND GRAPH ===")
    print(f"ARTIFACT: {output_path}")
    print(f"UNIQUE CO-SPEND PAIRS: {len(pairs)}")

    if len(pairs) > 0:
        print(
            "MAX CO-SPEND COUNT:",
            int(pairs["co_spend_transaction_count"].max()),
        )
        print(
            "MEDIAN CO-SPEND COUNT:",
            float(pairs["co_spend_transaction_count"].median()),
        )
        print(
            "PAIRS WITH >=2 CO-SPENDS:",
            int(
                (
                    pairs["co_spend_transaction_count"] >= 2
                ).sum()
            ),
        )
        print(
            "PAIRS WITH >=3 CO-SPENDS:",
            int(
                (
                    pairs["co_spend_transaction_count"] >= 3
                ).sum()
            ),
        )

    print(
        "NULL COUNT:",
        int(pairs.isna().sum().sum()),
    )


if __name__ == "__main__":
    main()