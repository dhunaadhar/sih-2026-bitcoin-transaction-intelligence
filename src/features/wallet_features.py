from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def build_wallet_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build transaction-local wallet relationship features.

    No global wallet statistics are calculated here, so this module
    does not introduce future-information leakage.
    """

    features = pd.DataFrame(index=df.index)

    input_counts = df["input_wallets"].map(len).astype("int64")
    output_counts = df["output_wallets"].map(len).astype("int64")

    overlap_counts = df.apply(
        lambda row: len(
            set(row["input_wallets"])
            & set(row["output_wallets"])
        ),
        axis=1,
    ).astype("int64")

    union_counts = (
        input_counts
        + output_counts
        - overlap_counts
    )

    # ---------------------------------------------------------
    # Wallet relationship structure
    # ---------------------------------------------------------

    features["wallet_overlap_count"] = overlap_counts

    features["wallet_overlap_input_ratio"] = _safe_divide(
        overlap_counts,
        input_counts,
    )

    features["wallet_overlap_output_ratio"] = _safe_divide(
        overlap_counts,
        output_counts,
    )

    features["wallet_jaccard_similarity"] = _safe_divide(
        overlap_counts,
        union_counts,
    )

    features["unique_wallet_count"] = union_counts

    features["unique_wallet_count_log1p"] = np.log1p(
        union_counts
    )

    features["has_wallet_overlap"] = (
        overlap_counts > 0
    ).astype("int8")

    features["wallet_relationship_missing"] = (
        (input_counts == 0)
        & (output_counts == 0)
    ).astype("int8")

    # ---------------------------------------------------------
    # Numerical cleanup
    # ---------------------------------------------------------

    numeric_columns = features.select_dtypes(
        include=[np.number]
    ).columns

    features[numeric_columns] = features[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return features