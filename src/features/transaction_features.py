from __future__ import annotations

import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "txid",
    "time_step",
    "total_btc",
    "fee_btc",
    "input_address_count",
    "output_address_count",
    "input_btc_min",
    "input_btc_max",
    "input_btc_mean",
    "input_btc_median",
    "input_btc_total",
    "output_btc_min",
    "output_btc_max",
    "output_btc_mean",
    "output_btc_median",
    "output_btc_total",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build transaction-level and value-distribution features.

    The input dataframe must contain the canonical transaction columns.
    The source dataframe is not modified.
    """

    features = df[BASE_COLUMNS].copy()

    # ---------------------------------------------------------
    # Missing-data indicators
    # ---------------------------------------------------------

    features["transaction_data_missing"] = (
        features["total_btc"].isna()
        | features["fee_btc"].isna()
        | features["input_address_count"].isna()
        | features["output_address_count"].isna()
    ).astype("int8")

    # ---------------------------------------------------------
    # Address-count features
    # ---------------------------------------------------------

    features["input_count_log1p"] = np.log1p(
        features["input_address_count"].fillna(0)
    )

    features["output_count_log1p"] = np.log1p(
        features["output_address_count"].fillna(0)
    )

    features["total_address_count"] = (
        features["input_address_count"]
        + features["output_address_count"]
    )

    features["address_count_imbalance"] = (
        features["input_address_count"]
        - features["output_address_count"]
    ).abs()

    features["address_count_ratio"] = _safe_divide(
        features["input_address_count"],
        features["output_address_count"],
    )

    # ---------------------------------------------------------
    # Transaction value features
    # ---------------------------------------------------------

    features["total_btc_log1p"] = np.log1p(
        features["total_btc"].clip(lower=0)
    )

    features["fee_btc_log1p"] = np.log1p(
        features["fee_btc"].clip(lower=0)
    )

    features["fee_rate_relative"] = _safe_divide(
        features["fee_btc"],
        features["total_btc"],
    )

    features["input_output_value_ratio"] = _safe_divide(
        features["input_btc_total"],
        features["output_btc_total"],
    )

    features["input_output_value_difference"] = (
        features["input_btc_total"]
        - features["output_btc_total"]
    )

    # ---------------------------------------------------------
    # Input value distribution
    # ---------------------------------------------------------

    features["input_value_range"] = (
        features["input_btc_max"]
        - features["input_btc_min"]
    )

    features["input_value_range_log1p"] = np.log1p(
        features["input_value_range"].clip(lower=0)
    )

    features["input_mean_median_ratio"] = _safe_divide(
        features["input_btc_mean"],
        features["input_btc_median"],
    )

    features["input_max_mean_ratio"] = _safe_divide(
        features["input_btc_max"],
        features["input_btc_mean"],
    )

    # ---------------------------------------------------------
    # Output value distribution
    # ---------------------------------------------------------

    features["output_value_range"] = (
        features["output_btc_max"]
        - features["output_btc_min"]
    )

    features["output_value_range_log1p"] = np.log1p(
        features["output_value_range"].clip(lower=0)
    )

    features["output_mean_median_ratio"] = _safe_divide(
        features["output_btc_mean"],
        features["output_btc_median"],
    )

    features["output_max_mean_ratio"] = _safe_divide(
        features["output_btc_max"],
        features["output_btc_mean"],
    )

    # ---------------------------------------------------------
    # Concentration proxies
    # ---------------------------------------------------------

    features["input_max_share"] = _safe_divide(
        features["input_btc_max"],
        features["input_btc_total"],
    )

    features["output_max_share"] = _safe_divide(
        features["output_btc_max"],
        features["output_btc_total"],
    )

    features["input_median_share"] = _safe_divide(
        features["input_btc_median"],
        features["input_btc_total"],
    )

    features["output_median_share"] = _safe_divide(
        features["output_btc_median"],
        features["output_btc_total"],
    )

    # ---------------------------------------------------------
    # Clean numeric representation
    # ---------------------------------------------------------

    numeric_columns = features.select_dtypes(
        include=[np.number]
    ).columns

    features[numeric_columns] = features[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return features