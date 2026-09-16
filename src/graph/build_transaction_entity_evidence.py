from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ENTITY_COLUMNS = [
    "address",
    "cosponsor_cluster_id_t1",
    "cosponsor_cluster_id_t2",
    "cosponsor_cluster_id_t3",
    "repeated_cosponsor_degree",
    "max_shared_transaction_count",
    "has_repeated_cosponsor",
    "cosponsor_cluster_size_t1",
    "cosponsor_cluster_size_t2",
    "cosponsor_cluster_size_t3",
]


def aggregate_address_evidence(
    wallets: list,
    evidence_lookup: dict[str, dict],
    prefix: str,
) -> dict:

    wallets = list(wallets)

    if not wallets:
        result = {
            f"{prefix}_address_count": 0,
            f"{prefix}_repeated_cosponsor_count": 0,
            f"{prefix}_repeated_cosponsor_ratio": 0.0,
            f"{prefix}_max_repeated_cosponsor_degree": 0,
            f"{prefix}_max_shared_transaction_count": 0,
        }

        for threshold in [1, 2, 3]:
            result[
                f"{prefix}_distinct_clusters_t{threshold}"
            ] = 0

            result[
                f"{prefix}_max_cluster_size_t{threshold}"
            ] = 0

        return result

    records = []

    for wallet in wallets:
        record = evidence_lookup.get(str(wallet))

        if record is not None:
            records.append(record)

    address_count = len(wallets)

    repeated_count = sum(
        int(record["has_repeated_cosponsor"])
        for record in records
    )

    result = {
        f"{prefix}_address_count": address_count,
        f"{prefix}_repeated_cosponsor_count": repeated_count,
        f"{prefix}_repeated_cosponsor_ratio": (
            repeated_count / address_count
            if address_count
            else 0.0
        ),
        f"{prefix}_max_repeated_cosponsor_degree": (
            max(
                (
                    int(
                        record[
                            "repeated_cosponsor_degree"
                        ]
                    )
                    for record in records
                ),
                default=0,
            )
        ),
        f"{prefix}_max_shared_transaction_count": (
            max(
                (
                    int(
                        record[
                            "max_shared_transaction_count"
                        ]
                    )
                    for record in records
                ),
                default=0,
            )
        ),
    }

    for threshold in [1, 2, 3]:

        cluster_column = (
            f"cosponsor_cluster_id_t{threshold}"
        )

        size_column = (
            f"cosponsor_cluster_size_t{threshold}"
        )

        cluster_ids = {
            record[cluster_column]
            for record in records
            if record.get(cluster_column) is not None
        }

        cluster_sizes = [
            int(record[size_column])
            for record in records
            if record.get(size_column) is not None
        ]

        result[
            f"{prefix}_distinct_clusters_t{threshold}"
        ] = len(cluster_ids)

        result[
            f"{prefix}_max_cluster_size_t{threshold}"
        ] = max(cluster_sizes, default=0)

    return result


def build_transaction_entity_evidence(
    canonical_path: str | Path,
    entity_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:

    canonical_path = Path(canonical_path)
    entity_path = Path(entity_path)
    output_path = Path(output_path)

    canonical = pd.read_parquet(
        canonical_path,
        columns=[
            "txid",
            "input_wallets",
            "output_wallets",
        ],
    )

    entity = pd.read_parquet(
        entity_path,
        columns=ENTITY_COLUMNS,
    )

    if entity["address"].duplicated().any():
        raise ValueError(
            "Entity evidence contains duplicate addresses."
        )

    entity_lookup = (
        entity.set_index("address")
        .to_dict(orient="index")
    )

    records = []

    for _, row in canonical.iterrows():

        input_wallets = list(row["input_wallets"])
        output_wallets = list(row["output_wallets"])

        input_evidence = aggregate_address_evidence(
            wallets=input_wallets,
            evidence_lookup=entity_lookup,
            prefix="input",
        )

        output_evidence = aggregate_address_evidence(
            wallets=output_wallets,
            evidence_lookup=entity_lookup,
            prefix="output",
        )

        record = {
            "txid": row["txid"],
        }

        record.update(input_evidence)
        record.update(output_evidence)

        records.append(record)

    result = pd.DataFrame(records)

    if result["txid"].duplicated().any():
        raise ValueError(
            "Transaction entity evidence contains duplicate TXIDs."
        )

    if len(result) != len(canonical):
        raise ValueError(
            "Transaction entity evidence row count mismatch."
        )

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    result[numeric_columns] = (
        result[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        output_path,
        index=False,
    )

    return result


def main() -> None:

    project_root = Path(__file__).resolve().parents[2]

    canonical_path = (
        project_root
        / "data"
        / "canonical"
        / "canonical_transactions.parquet"
    )

    entity_path = (
        project_root
        / "data"
        / "graph"
        / "entity_evidence.parquet"
    )

    output_path = (
        project_root
        / "data"
        / "graph"
        / "transaction_entity_evidence.parquet"
    )

    result = build_transaction_entity_evidence(
        canonical_path=canonical_path,
        entity_path=entity_path,
        output_path=output_path,
    )

    print("=== M5.10 TRANSACTION ENTITY EVIDENCE ===")
    print(f"ARTIFACT: {output_path}")
    print(f"ROWS: {len(result)}")
    print(f"COLUMNS: {len(result.columns)}")
    print(
        "UNIQUE TXIDS:",
        result["txid"].nunique(),
    )
    print(
        "NULL COUNT:",
        int(result.isna().sum().sum()),
    )
    print(
        "INF COUNT:",
        int(
            result.isin(
                [float("inf"), float("-inf")]
            ).sum().sum()
        ),
    )

    print(
        "TRANSACTIONS WITH INPUT REPEATED CO-SPONSOR:",
        int(
            (
                result[
                    "input_repeated_cosponsor_count"
                ]
                > 0
            ).sum()
        ),
    )

    print(
        "TRANSACTIONS WITH OUTPUT REPEATED CO-SPONSOR:",
        int(
            (
                result[
                    "output_repeated_cosponsor_count"
                ]
                > 0
            ).sum()
        ),
    )


if __name__ == "__main__":
    main()