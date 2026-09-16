from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_unified_features(
    transaction_path: str | Path,
    temporal_path: str | Path,
    wallet_path: str | Path,
    wallet_history_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:

    transaction_path = Path(transaction_path)
    temporal_path = Path(temporal_path)
    wallet_path = Path(wallet_path)
    wallet_history_path = Path(wallet_history_path)
    output_path = Path(output_path)

    transaction = pd.read_parquet(transaction_path)
    temporal = pd.read_parquet(temporal_path)
    wallet = pd.read_parquet(wallet_path)
    wallet_history = pd.read_parquet(wallet_history_path)

    row_count = len(transaction)

    if not (
        len(temporal)
        == len(wallet)
        == len(wallet_history)
        == row_count
    ):
        raise ValueError(
            "Feature artifacts have different row counts."
        )

    unified = pd.concat(
        [
            transaction.reset_index(drop=True),
            temporal.reset_index(drop=True),
            wallet.reset_index(drop=True),
            wallet_history.reset_index(drop=True),
        ],
        axis=1,
    )

    if unified["txid"].duplicated().any():
        raise ValueError(
            "Unified feature matrix contains duplicate TXIDs."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(output_path, index=False)

    return unified


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    derived = project_root / "data" / "derived"

    output_path = (
        derived / "unified_features.parquet"
    )

    features = build_unified_features(
        transaction_path=(
            derived / "transaction_features.parquet"
        ),
        temporal_path=(
            derived / "temporal_features.parquet"
        ),
        wallet_path=(
            derived / "wallet_features.parquet"
        ),
        wallet_history_path=(
            derived / "wallet_history_features.parquet"
        ),
        output_path=output_path,
    )

    print(f"UNIFIED FEATURE DATASET: {output_path}")
    print(f"SHAPE: {features.shape}")
    print(f"COLUMNS: {len(features.columns)}")
    print(
        "UNIQUE TXIDS:",
        features["txid"].nunique(),
    )
    print(
        "NULL COUNT:",
        int(features.isna().sum().sum()),
    )
    print(
        "INF COUNT:",
        int(
            features.isin(
                [float("inf"), float("-inf")]
            ).sum().sum()
        ),
    )


if __name__ == "__main__":
    main()