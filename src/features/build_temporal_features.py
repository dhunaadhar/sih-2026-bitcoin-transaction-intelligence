from __future__ import annotations

from pathlib import Path

import pandas as pd

from .temporal_features import build_temporal_features


def build_temporal_feature_dataset(
    feature_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Build the temporal feature dataset.

    The existing transaction feature artifact is read-only.
    """

    feature_path = Path(feature_path)
    output_path = Path(output_path)

    df = pd.read_parquet(feature_path)

    temporal_features = build_temporal_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporal_features.to_parquet(output_path, index=False)

    return temporal_features


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    feature_path = (
        project_root
        / "data"
        / "derived"
        / "transaction_features.parquet"
    )

    output_path = (
        project_root
        / "data"
        / "derived"
        / "temporal_features.parquet"
    )

    features = build_temporal_feature_dataset(
        feature_path=feature_path,
        output_path=output_path,
    )

    print(f"TEMPORAL FEATURE DATASET: {output_path}")
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