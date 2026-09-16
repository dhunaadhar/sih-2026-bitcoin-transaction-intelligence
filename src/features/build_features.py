from __future__ import annotations

from pathlib import Path

import pandas as pd

from .transaction_features import build_transaction_features


def build_feature_dataset(
    canonical_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Build the canonical transaction feature dataset.

    The canonical source is read-only and is never modified.
    """

    canonical_path = Path(canonical_path)
    output_path = Path(output_path)

    df = pd.read_parquet(canonical_path)

    features = build_transaction_features(df)

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
        / "transaction_features.parquet"
    )

    features = build_feature_dataset(
        canonical_path=canonical_path,
        output_path=output_path,
    )

    print(f"FEATURE DATASET: {output_path}")
    print(f"SHAPE: {features.shape}")
    print(f"COLUMNS: {len(features.columns)}")
    print(
        "TRANSACTION DATA MISSING:",
        int(features["transaction_data_missing"].sum()),
    )


if __name__ == "__main__":
    main()