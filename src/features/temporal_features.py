from __future__ import annotations

import numpy as np
import pandas as pd


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build relative temporal activity features from Elliptic++ time steps.

    The time_step field is treated as an ordered observation index.
    It is not interpreted as a wall-clock timestamp.
    """

    features = pd.DataFrame(index=df.index)

    time_step = df["time_step"].astype("int64")

    # Transaction volume at each observation step.
    step_counts = time_step.value_counts().sort_index()

    features["transactions_in_time_step"] = (
        time_step.map(step_counts).astype("int64")
    )

    # Relative activity compared with the global mean.
    mean_activity = float(step_counts.mean())

    features["time_step_activity_ratio"] = (
        features["transactions_in_time_step"] / mean_activity
    )

    features["time_step_activity_deviation"] = (
        features["transactions_in_time_step"] - mean_activity
    )

    features["time_step_activity_log1p"] = np.log1p(
        features["transactions_in_time_step"]
    )

    # Relative position in the observation window.
    min_step = int(time_step.min())
    max_step = int(time_step.max())

    if max_step == min_step:
        features["temporal_position"] = 0.0
    else:
        features["temporal_position"] = (
            time_step - min_step
        ) / (max_step - min_step)

    # Start/end indicators.
    features["early_period"] = (
        features["temporal_position"] <= 0.25
    ).astype("int8")

    features["late_period"] = (
        features["temporal_position"] >= 0.75
    ).astype("int8")

    return features