from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_unified_features(
    transaction_path: str | Path,
    temporal_path: str | Path,
    wallet_path: str | Path,
    wallet_history_path: str | Path,
    entity_evidence_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:

    transaction_path = Path(transaction_path)
    temporal_path = Path(temporal_path)
    wallet_path = Path(wallet_path)
    wallet_history_path = Path(wallet_history_path)
    entity_evidence_path = Path(entity_evidence_path)
    output_path = Path(output_path)

    transaction = pd.read_parquet(transaction_path)
    temporal = pd.read_parquet(temporal_path)
    wallet = pd.read_parquet(wallet_path)
    wallet_history = pd.read_parquet(wallet_history_path)
    entity_evidence = pd.read_parquet(entity_evidence_path)

    # M5.10 graph evidence contains address-count fields that
    # overlap with existing transaction feature names.
    # Rename only those graph-specific fields to preserve
    # the original transaction feature names.
    entity_evidence = entity_evidence.rename(
        columns={
            "input_address_count": "entity_input_address_count",
            "output_address_count": "entity_output_address_count",
        }
    )

    row_count = len(transaction)

    if not (
        len(temporal)
        == len(wallet)
        == len(wallet_history)
        == len(entity_evidence)
        == row_count
    ):
        raise ValueError(
            "Feature artifacts have different row counts."
        )

    if transaction["txid"].duplicated().any():
        raise ValueError(
            "Transaction feature artifact contains duplicate TXIDs."
        )

    if entity_evidence["txid"].duplicated().any():
        raise ValueError(
            "Entity evidence artifact contains duplicate TXIDs."
        )

    if not transaction["txid"].equals(
        entity_evidence["txid"]
    ):
        raise ValueError(
            "Transaction and entity evidence TXID ordering does not match."
        )

    unified = pd.concat(
        [
            transaction.reset_index(drop=True),
            temporal.reset_index(drop=True),
            wallet.reset_index(drop=True),
            wallet_history.reset_index(drop=True),
            entity_evidence.drop(
                columns=["txid"]
            ).reset_index(drop=True),
        ],
        axis=1,
    )

    if unified["txid"].duplicated().any():
        raise ValueError(
            "Unified feature matrix contains duplicate TXIDs."
        )

    if unified.columns.duplicated().any():
        duplicate_columns = unified.columns[
            unified.columns.duplicated()
        ].tolist()

        raise ValueError(
            "Unified feature matrix contains duplicate "
            f"column names: {duplicate_columns}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    unified.to_parquet(
        output_path,
        index=False,
    )

    return unified


def main() -> None:

    project_root = Path(__file__).resolve().parents[2]

    derived = (
        project_root
        / "data"
        / "derived"
    )

    graph = (
        project_root
        / "data"
        / "graph"
    )

    output_path = (
        derived
        / "unified_features.parquet"
    )

    features = build_unified_features(
        transaction_path=(
            derived
            / "transaction_features.parquet"
        ),
        temporal_path=(
            derived
            / "temporal_features.parquet"
        ),
        wallet_path=(
            derived
            / "wallet_features.parquet"
        ),
        wallet_history_path=(
            derived
            / "wallet_history_features.parquet"
        ),
        entity_evidence_path=(
            graph
            / "transaction_entity_evidence.parquet"
        ),
        output_path=output_path,
    )

    print(
        f"UNIFIED FEATURE DATASET: {output_path}"
    )
    print(
        f"SHAPE: {features.shape}"
    )
    print(
        f"COLUMNS: {len(features.columns)}"
    )
    print(
        "UNIQUE TXIDS:",
        features["txid"].nunique(),
    )
    print(
        "NULL COUNT:",
        int(
            features.isna().sum().sum()
        ),
    )
    print(
        "INF COUNT:",
        int(
            features.isin(
                [
                    float("inf"),
                    float("-inf"),
                ]
            ).sum().sum()
        ),
    )


if __name__ == "__main__":
    main()