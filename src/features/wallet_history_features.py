from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def _history_stats(
    wallets,
    tx_counts,
    last_seen,
    current_step,
):
    """
    Read wallet history strictly before the current time step.
    """

    if not wallets:
        return {
            "seen_count": 0,
            "frequency_sum": 0.0,
            "frequency_mean": 0.0,
            "frequency_max": 0.0,
            "recency_min": 0.0,
            "recency_mean": 0.0,
            "recency_max": 0.0,
        }

    frequencies = []
    recencies = []

    for wallet in wallets:
        frequencies.append(tx_counts.get(wallet, 0))

        if wallet in last_seen:
            recencies.append(
                current_step - last_seen[wallet]
            )

    seen_count = sum(
        frequency > 0
        for frequency in frequencies
    )

    return {
        "seen_count": seen_count,
        "frequency_sum": float(sum(frequencies)),
        "frequency_mean": float(np.mean(frequencies)),
        "frequency_max": float(max(frequencies)),
        "recency_min": (
            float(min(recencies))
            if recencies
            else 0.0
        ),
        "recency_mean": (
            float(np.mean(recencies))
            if recencies
            else 0.0
        ),
        "recency_max": (
            float(max(recencies))
            if recencies
            else 0.0
        ),
    }


def build_wallet_history_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build point-in-time wallet behavioral features.

    For a transaction at time step t, only information from
    strictly earlier time steps is used.

    Transactions within the same time step do not see one
    another. Their complete time step is incorporated into
    wallet history only after all transactions in that step
    have been evaluated.
    """

    features = pd.DataFrame(index=df.index)

    columns = [
        "input_wallets_seen_before",
        "output_wallets_seen_before",
        "input_wallet_prior_frequency_sum",
        "output_wallet_prior_frequency_sum",
        "input_wallet_prior_frequency_mean",
        "output_wallet_prior_frequency_mean",
        "input_wallet_prior_frequency_max",
        "output_wallet_prior_frequency_max",
        "input_wallet_reuse_ratio",
        "output_wallet_reuse_ratio",
        "input_wallet_recency_min",
        "output_wallet_recency_min",
        "input_wallet_recency_mean",
        "output_wallet_recency_mean",
        "input_wallet_recency_max",
        "output_wallet_recency_max",
        "wallet_history_missing",
    ]

    for column in columns:
        features[column] = 0.0

    tx_counts = defaultdict(int)
    last_seen = {}

    # Process time steps chronologically.
    for time_step in sorted(df["time_step"].unique()):

        step_mask = df["time_step"] == time_step
        step_df = df.loc[step_mask]

        # Counter preserves how many transactions in the
        # current time step contain each wallet.
        pending_updates = Counter()

        for idx, row in step_df.iterrows():

            input_wallets = list(row["input_wallets"])
            output_wallets = list(row["output_wallets"])

            # IMPORTANT:
            # These statistics use ONLY history from
            # strictly earlier time steps.
            input_stats = _history_stats(
                input_wallets,
                tx_counts,
                last_seen,
                int(time_step),
            )

            output_stats = _history_stats(
                output_wallets,
                tx_counts,
                last_seen,
                int(time_step),
            )

            features.at[
                idx,
                "input_wallets_seen_before",
            ] = input_stats["seen_count"]

            features.at[
                idx,
                "output_wallets_seen_before",
            ] = output_stats["seen_count"]

            features.at[
                idx,
                "input_wallet_prior_frequency_sum",
            ] = input_stats["frequency_sum"]

            features.at[
                idx,
                "output_wallet_prior_frequency_sum",
            ] = output_stats["frequency_sum"]

            features.at[
                idx,
                "input_wallet_prior_frequency_mean",
            ] = input_stats["frequency_mean"]

            features.at[
                idx,
                "output_wallet_prior_frequency_mean",
            ] = output_stats["frequency_mean"]

            features.at[
                idx,
                "input_wallet_prior_frequency_max",
            ] = input_stats["frequency_max"]

            features.at[
                idx,
                "output_wallet_prior_frequency_max",
            ] = output_stats["frequency_max"]

            input_count = len(input_wallets)
            output_count = len(output_wallets)

            features.at[
                idx,
                "input_wallet_reuse_ratio",
            ] = (
                input_stats["seen_count"] / input_count
                if input_count
                else 0.0
            )

            features.at[
                idx,
                "output_wallet_reuse_ratio",
            ] = (
                output_stats["seen_count"] / output_count
                if output_count
                else 0.0
            )

            features.at[
                idx,
                "input_wallet_recency_min",
            ] = input_stats["recency_min"]

            features.at[
                idx,
                "output_wallet_recency_min",
            ] = output_stats["recency_min"]

            features.at[
                idx,
                "input_wallet_recency_mean",
            ] = input_stats["recency_mean"]

            features.at[
                idx,
                "output_wallet_recency_mean",
            ] = output_stats["recency_mean"]

            features.at[
                idx,
                "input_wallet_recency_max",
            ] = input_stats["recency_max"]

            features.at[
                idx,
                "output_wallet_recency_max",
            ] = output_stats["recency_max"]

            features.at[
                idx,
                "wallet_history_missing",
            ] = int(
                len(input_wallets) == 0
                and len(output_wallets) == 0
            )

            # Do NOT update tx_counts yet.
            # The current time step must remain invisible
            # to every transaction in this same time step.
            pending_updates.update(input_wallets)
            pending_updates.update(output_wallets)

        # After ALL transactions in this time step have
        # been evaluated, update historical wallet counts.
        for wallet, occurrence_count in pending_updates.items():
            tx_counts[wallet] += occurrence_count
            last_seen[wallet] = int(time_step)

    numeric_columns = features.select_dtypes(
        include=[np.number]
    ).columns

    features[numeric_columns] = features[
        numeric_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return features