from __future__ import annotations

from pathlib import Path

import pandas as pd

from .wallet_history_features import build_wallet_history_features


def build_wallet_history_feature_dataset(
    canonical_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    canonical_path = Path(canonical_path)
    output_path = Path(output_path)

    df = pd.read_parquet(canonical_path)

    features = build_wallet_history_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    return features


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    canonical_path = (
        project_root
        / "data"
        / "canonical"
        / "canonical_transactions.parquet"
    )

    output_path = (
        project_root
        / "data"
        / "derived"
        / "wallet_history_features.parquet"
    )

    features = build_wallet_history_feature_dataset(
        canonical_path,
        output_path,
    )

    print(f"WALLET HISTORY FEATURE DATASET: {output_path}")
    print(f"SHAPE: {features.shape}")
    print(f"COLUMNS: {len(features.columns)}")
    print(f"NULL COUNT: {int(features.isna().sum().sum())}")
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