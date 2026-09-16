from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.wallet_history_features import (
    build_wallet_history_features,
)


def test_wallet_history_excludes_current_time_step():
    df = pd.DataFrame(
        {
            "time_step": [1, 1, 2],
            "input_wallets": [
                ["A"],
                ["A"],
                ["A"],
            ],
            "output_wallets": [
                ["B"],
                ["C"],
                ["D"],
            ],
        }
    )

    features = build_wallet_history_features(df)

    # Both transactions at time step 1 must see no history.
    assert features.loc[0, "input_wallets_seen_before"] == 0
    assert features.loc[1, "input_wallets_seen_before"] == 0

    # The transaction at time step 2 sees wallet A from time step 1.
    assert features.loc[2, "input_wallets_seen_before"] == 1
    assert features.loc[2, "input_wallet_prior_frequency_max"] == 2


def test_wallet_history_recency_is_point_in_time():
    df = pd.DataFrame(
        {
            "time_step": [1, 3],
            "input_wallets": [
                ["A"],
                ["A"],
            ],
            "output_wallets": [
                ["B"],
                ["C"],
            ],
        }
    )

    features = build_wallet_history_features(df)

    assert features.loc[0, "input_wallets_seen_before"] == 0
    assert features.loc[1, "input_wallets_seen_before"] == 1
    assert features.loc[1, "input_wallet_recency_min"] == 2