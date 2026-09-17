"""
M11.4 — Unified Risk Scoring Engine

Purpose
-------
Fuse independently generated investigative evidence into a single,
bounded 0–100 transaction priority score.

Evidence channels
-----------------
1. M7 Production XGBoost classification confidence
2. M8 Isolation Forest anomaly evidence
3. M9 behavioral evidence
4. M6 temporal-safe entity/graph evidence
5. M10 network evidence

Important semantics
-------------------
The resulting score is an INVESTIGATIVE PRIORITY SCORE.

It is NOT:
    - probability of illicit activity
    - proof of criminal activity
    - proof of wallet ownership
    - proof of IP ownership
    - identity attribution

Elliptic++ numeric class labels are retained as numeric classes.
No class is assumed to represent illicit or legitimate activity.

Design principles
-----------------
- Deterministic
- Bounded to [0, 100]
- Explicit evidence channels
- No future-data graph features
- No double-counting of peeling/mixing inside behavioral evidence
- Missing network observations remain neutral
- ML confidence is treated as model certainty, not illicitness
- Explanations are generated from actual contributing signals
- M7/M8 preprocessing artifacts are reused exactly as trained
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_EVIDENCE_FILE = (
    ROOT
    / "data"
    / "derived"
    / "normalized_evidence.parquet"
)

TEMPORAL_SAFE_FEATURES_FILE = (
    ROOT
    / "data"
    / "derived"
    / "unified_features_temporal_safe.parquet"
)

BEHAVIORAL_FILE = (
    ROOT
    / "data"
    / "derived"
    / "behavioral_evidence.parquet"
)

ENTITY_FILE = (
    ROOT
    / "data"
    / "derived"
    / "temporal_transaction_entity_evidence.parquet"
)

NETWORK_FILE = (
    ROOT
    / "data"
    / "derived"
    / "network_transaction_features.parquet"
)

XGB_MODEL_FILE = (
    ROOT
    / "models"
    / "production_xgboost"
    / "model.joblib"
)

XGB_SCHEMA_FILE = (
    ROOT
    / "models"
    / "production_xgboost"
    / "feature_schema.json"
)

IF_MODEL_FILE = (
    ROOT
    / "models"
    / "production_isolation_forest"
    / "model.joblib"
)

IF_SCHEMA_FILE = (
    ROOT
    / "models"
    / "production_isolation_forest"
    / "feature_schema.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "derived"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "risk"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "unified_risk_scores.parquet"
)

REPORT_FILE = (
    REPORT_DIR
    / "m11_4_unified_risk_scoring.json"
)


# ============================================================================
# Configuration
# ============================================================================

WEIGHTS = {
    "ml": 0.15,
    "anomaly": 0.20,
    "behavioral": 0.30,
    "entity": 0.20,
    "network": 0.15,
}

WEIGHT_SUM = sum(
    WEIGHTS.values()
)

if not np.isclose(
    WEIGHT_SUM,
    1.0,
):
    raise RuntimeError(
        "Risk channel weights must sum to 1.0."
    )


AGREEMENT_BONUS = {
    0: 0.00,
    1: 0.00,
    2: 0.02,
    3: 0.04,
    4: 0.06,
    5: 0.08,
}


RISK_BANDS = [
    (80.0, "VERY_HIGH"),
    (60.0, "HIGH"),
    (40.0, "MODERATE"),
    (20.0, "GUARDED"),
    (0.0, "LOW"),
]


# ============================================================================
# Utility functions
# ============================================================================

def normalize_txid(value) -> str:
    """
    Normalize TXID representations consistently.
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):

        try:
            numeric = float(text)

            if numeric.is_integer():
                return str(int(numeric))

        except ValueError:
            pass

    return text


def require_columns(
    df: pd.DataFrame,
    required: List[str],
    source_name: str,
) -> None:
    """
    Validate required columns.
    """

    missing = sorted(
        set(required)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            f"{missing}"
        )


def clamp01(
    series: pd.Series,
) -> pd.Series:
    """
    Clamp numeric values into [0,1].
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    return values.clip(
        lower=0.0,
        upper=1.0,
    )


def risk_band(
    score: float,
) -> str:
    """
    Convert a 0–100 score into an operational priority band.
    """

    for threshold, label in RISK_BANDS:

        if score >= threshold:
            return label

    return "LOW"


def safe_float(
    value,
    default: float = 0.0,
) -> float:
    """
    Convert a value safely to finite float.
    """

    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except (
        TypeError,
        ValueError,
    ):
        pass

    return default


# ============================================================================
# Artifact loading
# ============================================================================

def load_parquet(
    path: Path,
    required_columns: List[str],
    source_name: str,
) -> pd.DataFrame:
    """
    Load and validate a parquet artifact.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{source_name} not found: {path}"
        )

    df = pd.read_parquet(
        path
    )

    require_columns(
        df,
        required_columns,
        source_name,
    )

    df = df.copy()

    df["txid"] = (
        df["txid"]
        .map(normalize_txid)
    )

    if df["txid"].eq("").any():
        raise ValueError(
            f"{source_name} contains empty TXIDs."
        )

    if df["txid"].duplicated().any():
        raise ValueError(
            f"{source_name} contains duplicate TXIDs."
        )

    return df


def load_json(
    path: Path,
) -> Dict:
    """
    Load a JSON artifact.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON artifact not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================================
# Model feature preparation
# ============================================================================

def prepare_model_matrix(
    df: pd.DataFrame,
    model_features: List[str],
    source_name: str,
) -> pd.DataFrame:
    """
    Extract the exact 87 model features used during M7/M8 training.
    """

    missing = sorted(
        set(model_features)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{source_name} is missing model features: "
            f"{missing}"
        )

    X = df[
        model_features
    ].copy()

    for column in model_features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    numeric_values = X.to_numpy(
        dtype=float
    )

    if np.isinf(
        numeric_values
    ).any():

        raise ValueError(
            f"{source_name} contains infinite model features."
        )

    return X


# ============================================================================
# M7 Production XGBoost
# ============================================================================

def load_xgb_package():
    """
    Load the exact M7 production package.
    """

    if not XGB_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"M7 model not found: {XGB_MODEL_FILE}"
        )

    if not XGB_SCHEMA_FILE.exists():
        raise FileNotFoundError(
            f"M7 schema not found: {XGB_SCHEMA_FILE}"
        )

    package = joblib.load(
        XGB_MODEL_FILE
    )

    schema = load_json(
        XGB_SCHEMA_FILE
    )

    if not isinstance(
        package,
        dict,
    ):

        raise ValueError(
            "M7 model artifact is not a dictionary package."
        )

    required_keys = {
        "imputer",
        "classifier",
        "model_features",
        "label_mapping",
        "inverse_label_mapping",
    }

    missing = sorted(
        required_keys
        - set(package.keys())
    )

    if missing:
        raise ValueError(
            f"M7 model package missing keys: {missing}"
        )

    return package, schema


def xgb_predictions(
    feature_df: pd.DataFrame,
):
    """
    Run M7 production XGBoost inference using the SAME imputer
    saved during M7.1.

    87 raw features
        ->
    saved SimpleImputer(add_indicator=True)
        ->
    124 transformed features
        ->
    saved XGBClassifier
    """

    package, schema = (
        load_xgb_package()
    )

    model_features = package[
        "model_features"
    ]

    X = prepare_model_matrix(
        feature_df,
        model_features,
        "M7 temporal-safe features",
    )

    imputer = package[
        "imputer"
    ]

    classifier = package[
        "classifier"
    ]

    X_transformed = imputer.transform(
        X
    )

    expected_transformed_features = (
        classifier.n_features_in_
    )

    actual_transformed_features = (
        X_transformed.shape[1]
    )

    if (
        actual_transformed_features
        != expected_transformed_features
    ):

        raise ValueError(
            "M7 transformed feature count mismatch: "
            f"expected {expected_transformed_features}, "
            f"got {actual_transformed_features}."
        )

    probabilities = classifier.predict_proba(
        X_transformed
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "M7 predict_proba output is not two-dimensional."
        )

    if probabilities.shape[1] != 3:
        raise ValueError(
            "Expected three M7 class probabilities; "
            f"received {probabilities.shape[1]}."
        )

    row_sums = probabilities.sum(
        axis=1
    )

    if not np.allclose(
        row_sums,
        1.0,
        atol=1e-5,
    ):

        raise ValueError(
            "M7 class probabilities do not sum to 1."
        )

    predicted_indices = np.argmax(
        probabilities,
        axis=1,
    )

    inverse_mapping = package[
        "inverse_label_mapping"
    ]

    predicted_class = np.array(
        [
            int(
                inverse_mapping[
                    int(index)
                ]
            )
            for index in predicted_indices
        ],
        dtype=int,
    )

    confidence = probabilities.max(
        axis=1
    )

    return (
        probabilities,
        predicted_class,
        confidence,
        actual_transformed_features,
    )


# ============================================================================
# M8 Isolation Forest
# ============================================================================

def load_if_package():
    """
    Load the exact M8 production package.
    """

    if not IF_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"M8 model not found: {IF_MODEL_FILE}"
        )

    if not IF_SCHEMA_FILE.exists():
        raise FileNotFoundError(
            f"M8 schema not found: {IF_SCHEMA_FILE}"
        )

    package = joblib.load(
        IF_MODEL_FILE
    )

    schema = load_json(
        IF_SCHEMA_FILE
    )

    if not isinstance(
        package,
        dict,
    ):

        raise ValueError(
            "M8 model artifact is not a dictionary package."
        )

    required_keys = {
        "imputer",
        "detector",
        "model_features",
        "transformed_feature_names",
    }

    missing = sorted(
        required_keys
        - set(package.keys())
    )

    if missing:
        raise ValueError(
            f"M8 model package missing keys: {missing}"
        )

    return package, schema


def anomaly_predictions(
    feature_df: pd.DataFrame,
):
    """
    Run M8 Isolation Forest inference using the SAME imputer saved
    during M8.1.

    87 raw features
        ->
    saved SimpleImputer(add_indicator=True)
        ->
    124 transformed features
        ->
    saved IsolationForest
    """

    package, schema = (
        load_if_package()
    )

    model_features = package[
        "model_features"
    ]

    X = prepare_model_matrix(
        feature_df,
        model_features,
        "M8 temporal-safe features",
    )

    imputer = package[
        "imputer"
    ]

    detector = package[
        "detector"
    ]

    X_transformed = imputer.transform(
        X
    )

    expected_transformed_features = len(
        package[
            "transformed_feature_names"
        ]
    )

    actual_transformed_features = (
        X_transformed.shape[1]
    )

    if (
        actual_transformed_features
        != expected_transformed_features
    ):

        raise ValueError(
            "M8 transformed feature count mismatch: "
            f"expected {expected_transformed_features}, "
            f"got {actual_transformed_features}."
        )

    decision = np.asarray(
        detector.decision_function(
            X_transformed
        ),
        dtype=float,
    )

    if decision.ndim != 1:

        decision = decision.reshape(
            -1
        )

    anomaly_score = -decision

    anomaly_prediction = detector.predict(
        X_transformed
    )

    anomaly_flag = (
        anomaly_prediction
        == -1
    )

    return (
        decision,
        anomaly_score,
        anomaly_flag,
        actual_transformed_features,
    )


# ============================================================================
# Evidence loading
# ============================================================================

def load_evidence_sources():
    """
    Load all M11 evidence sources.
    """

    normalized = load_parquet(
        NORMALIZED_EVIDENCE_FILE,
        [
            "txid",
            "time_step",
            "peeling_signal",
            "mixing_signal",
            "behavioral_signal",
            "entity_signal",
            "network_signal",
            "available_evidence_channel_count",
        ],
        "M11.3 normalized evidence",
    )

    behavioral = load_parquet(
        BEHAVIORAL_FILE,
        [
            "txid",
            "peeling_evidence_score",
            "mixing_evidence_score",
            "behavioral_evidence_score",
            "behavioral_signal_count",
            "multiple_behavioral_signals",
            "behavioral_signal_agreement",
            "peeling_mixing_interaction",
            "behavioral_evidence_level",
        ],
        "M9 behavioral evidence",
    )

    entity = load_parquet(
        ENTITY_FILE,
        [
            "txid",
            "time_step",
            "input_repeated_cosponsor_count",
            "output_repeated_cosponsor_count",
            "input_max_repeated_cosponsor_degree",
            "output_max_repeated_cosponsor_degree",
            "input_max_shared_transaction_count",
            "output_max_shared_transaction_count",
            "input_distinct_clusters_t1",
            "output_distinct_clusters_t1",
            "input_max_cluster_size_t1",
            "output_max_cluster_size_t1",
            "entity_input_address_count",
            "entity_output_address_count",
        ],
        "M6 temporal entity evidence",
    )

    network = load_parquet(
        NETWORK_FILE,
        [
            "txid",
            "time_step",
            "network_observation_count",
            "network_unique_source_ip_count",
            "network_unique_destination_ip_count",
            "network_unique_endpoint_count",
            "network_observation_span_seconds",
            "network_unique_asn_count",
            "network_unique_country_count",
            "network_exact_txid_match",
            "network_temporal_context_available",
            "network_multi_observation_context",
            "network_temporal_evidence_score",
            "network_temporal_evidence_level",
            "network_observation_available",
        ],
        "M10 network transaction features",
    )

    features = load_parquet(
        TEMPORAL_SAFE_FEATURES_FILE,
        [
            "txid",
            "time_step",
        ],
        "Temporal-safe unified features",
    )

    return (
        normalized,
        behavioral,
        entity,
        network,
        features,
    )


# ============================================================================
# Base fusion table
# ============================================================================

def build_fused_dataset():
    """
    Build a one-row-per-TXID evidence table.
    """

    (
        normalized,
        behavioral,
        entity,
        network,
        feature_df,
    ) = load_evidence_sources()

    result = normalized[
        [
            "txid",
            "time_step",
            "peeling_signal",
            "mixing_signal",
            "behavioral_signal",
            "entity_signal",
            "network_signal",
            "available_evidence_channel_count",
        ]
    ].copy()

    # ------------------------------------------------------------------------
    # Behavioral explanation metadata
    # ------------------------------------------------------------------------

    behavioral_subset = behavioral[
        [
            "txid",
            "peeling_evidence_score",
            "mixing_evidence_score",
            "behavioral_evidence_score",
            "behavioral_signal_count",
            "multiple_behavioral_signals",
            "behavioral_signal_agreement",
            "peeling_mixing_interaction",
            "behavioral_evidence_level",
        ]
    ].copy()

    result = result.merge(
        behavioral_subset,
        on="txid",
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------------------
    # Entity explanation metadata
    # ------------------------------------------------------------------------

    entity_subset = entity[
        [
            "txid",
            "input_repeated_cosponsor_count",
            "output_repeated_cosponsor_count",
            "input_max_repeated_cosponsor_degree",
            "output_max_repeated_cosponsor_degree",
            "input_max_shared_transaction_count",
            "output_max_shared_transaction_count",
            "input_distinct_clusters_t1",
            "output_distinct_clusters_t1",
            "input_max_cluster_size_t1",
            "output_max_cluster_size_t1",
            "entity_input_address_count",
            "entity_output_address_count",
        ]
    ].copy()

    result = result.merge(
        entity_subset,
        on="txid",
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------------------
    # Network explanation metadata
    # ------------------------------------------------------------------------

    network_subset = network[
        [
            "txid",
            "network_observation_count",
            "network_unique_source_ip_count",
            "network_unique_destination_ip_count",
            "network_unique_endpoint_count",
            "network_observation_span_seconds",
            "network_unique_asn_count",
            "network_unique_country_count",
            "network_exact_txid_match",
            "network_temporal_context_available",
            "network_multi_observation_context",
            "network_temporal_evidence_score",
            "network_temporal_evidence_level",
            "network_observation_available",
        ]
    ].copy()

    result = result.merge(
        network_subset,
        on="txid",
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------------------
    # Feature coverage validation
    # ------------------------------------------------------------------------

    if set(
        result["txid"]
    ) != set(
        feature_df["txid"]
    ):

        raise ValueError(
            "M11 evidence TXID coverage differs from temporal-safe "
            "model feature coverage."
        )

    return (
        result,
        feature_df,
    )


# ============================================================================
# M7/M8 signal integration
# ============================================================================

def add_ml_signals(
    result: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add M7 class probabilities and model confidence.
    """

    (
        probabilities,
        predicted_class,
        confidence,
        transformed_count,
    ) = xgb_predictions(
        feature_df
    )

    if len(probabilities) != len(result):

        raise ValueError(
            "M7 inference row count does not match "
            "fusion dataset."
        )

    result = result.copy()

    result[
        "ml_class_1_probability"
    ] = probabilities[
        :,
        0
    ]

    result[
        "ml_class_2_probability"
    ] = probabilities[
        :,
        1
    ]

    result[
        "ml_class_3_probability"
    ] = probabilities[
        :,
        2
    ]

    result[
        "ml_predicted_class"
    ] = predicted_class

    result[
        "ml_confidence"
    ] = np.clip(
        confidence,
        0.0,
        1.0,
    )

    result[
        "ml_signal"
    ] = result[
        "ml_confidence"
    ]

    result[
        "ml_transformed_feature_count"
    ] = transformed_count

    return result


def add_anomaly_signals(
    result: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add M8 Isolation Forest anomaly evidence.

    The anomaly evidence is converted to an empirical percentile
    over the scoring population. Higher values indicate stronger
    relative anomaly evidence.

    This is NOT a probability.
    """

    (
        decision,
        anomaly_score,
        anomaly_flag,
        transformed_count,
    ) = anomaly_predictions(
        feature_df
    )

    if len(anomaly_score) != len(result):

        raise ValueError(
            "M8 inference row count does not match "
            "fusion dataset."
        )

    result = result.copy()

    result[
        "anomaly_decision_function"
    ] = decision

    result[
        "anomaly_raw_score"
    ] = anomaly_score

    result[
        "anomaly_flag"
    ] = anomaly_flag.astype(
        bool
    )

    result[
        "anomaly_signal"
    ] = (
        result[
            "anomaly_raw_score"
        ]
        .rank(
            method="average",
            pct=True,
        )
        .astype(float)
        .clip(
            0.0,
            1.0,
        )
    )

    result[
        "anomaly_transformed_feature_count"
    ] = transformed_count

    return result


# ============================================================================
# Score calculation
# ============================================================================

def calculate_scores(
    result: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate weighted evidence score and agreement adjustment.
    """

    result = result.copy()

    signal_columns = [
        "ml_signal",
        "anomaly_signal",
        "behavioral_signal",
        "entity_signal",
        "network_signal",
    ]

    for column in signal_columns:

        result[column] = clamp01(
            result[column]
        )

    # ------------------------------------------------------------------------
    # Independent evidence availability
    # ------------------------------------------------------------------------

    result[
        "ml_evidence_available"
    ] = (
        result[
            "ml_signal"
        ]
        > 0
    )

    result[
        "anomaly_evidence_available"
    ] = (
        result[
            "anomaly_signal"
        ]
        > 0
    )

    result[
        "behavioral_evidence_available"
    ] = (
        result[
            "behavioral_signal"
        ]
        > 0
    )

    result[
        "entity_evidence_available"
    ] = (
        result[
            "entity_signal"
        ]
        > 0
    )

    result[
        "network_evidence_available"
    ] = (
        result[
            "network_signal"
        ]
        > 0
    )

    availability_columns = [
        "ml_evidence_available",
        "anomaly_evidence_available",
        "behavioral_evidence_available",
        "entity_evidence_available",
        "network_evidence_available",
    ]

    result[
        "effective_evidence_channel_count"
    ] = (
        result[
            availability_columns
        ]
        .sum(
            axis=1
        )
        .astype(int)
    )

    # ------------------------------------------------------------------------
    # Weighted score
    # ------------------------------------------------------------------------

    result[
        "weighted_evidence_score"
    ] = (
        result[
            "ml_signal"
        ]
        * WEIGHTS["ml"]
        +
        result[
            "anomaly_signal"
        ]
        * WEIGHTS["anomaly"]
        +
        result[
            "behavioral_signal"
        ]
        * WEIGHTS["behavioral"]
        +
        result[
            "entity_signal"
        ]
        * WEIGHTS["entity"]
        +
        result[
            "network_signal"
        ]
        * WEIGHTS["network"]
    )

    # ------------------------------------------------------------------------
    # Evidence agreement bonus
    # ------------------------------------------------------------------------

    result[
        "evidence_agreement_bonus"
    ] = (
        result[
            "effective_evidence_channel_count"
        ]
        .map(
            AGREEMENT_BONUS
        )
        .fillna(
            AGREEMENT_BONUS[5]
        )
    )

    result[
        "agreement_adjusted_score"
    ] = (
        result[
            "weighted_evidence_score"
        ]
        *
        (
            1.0
            +
            result[
                "evidence_agreement_bonus"
            ]
        )
    )

    # ------------------------------------------------------------------------
    # Final 0–100 score
    # ------------------------------------------------------------------------

    result[
        "risk_score"
    ] = (
        result[
            "agreement_adjusted_score"
        ]
        .clip(
            0.0,
            1.0,
        )
        * 100.0
    )

    result[
        "risk_level"
    ] = result[
        "risk_score"
    ].map(
        risk_band
    )

    return result


# ============================================================================
# Explainability
# ============================================================================

def generate_explanation(
    row: pd.Series,
) -> str:
    """
    Generate a human-readable explanation using actual evidence.
    """

    contributors = []

    signal_descriptions = [
        (
            "behavioral",
            safe_float(
                row.get(
                    "behavioral_signal"
                )
            ),
            "behavioral structural evidence",
        ),
        (
            "anomaly",
            safe_float(
                row.get(
                    "anomaly_signal"
                )
            ),
            "statistical anomaly evidence",
        ),
        (
            "entity",
            safe_float(
                row.get(
                    "entity_signal"
                )
            ),
            "historical entity/graph evidence",
        ),
        (
            "network",
            safe_float(
                row.get(
                    "network_signal"
                )
            ),
            "network correlation evidence",
        ),
        (
            "ml",
            safe_float(
                row.get(
                    "ml_signal"
                )
            ),
            "supervised-model confidence",
        ),
    ]

    for (
        name,
        value,
        description,
    ) in signal_descriptions:

        if value >= 0.70:

            contributors.append(
                f"strong {description}"
            )

        elif value >= 0.40:

            contributors.append(
                f"moderate {description}"
            )

        elif value > 0.0:

            contributors.append(
                f"weak {description}"
            )

    # ------------------------------------------------------------------------
    # Behavioral specifics
    # ------------------------------------------------------------------------

    if bool(
        row.get(
            "multiple_behavioral_signals",
            False,
        )
    ):

        contributors.append(
            "multiple behavioral signals"
        )

    if bool(
        row.get(
            "behavioral_signal_agreement",
            False,
        )
    ):

        contributors.append(
            "agreement between behavioral channels"
        )

    if (
        safe_float(
            row.get(
                "peeling_evidence_score"
            )
        )
        >= 0.50
    ):

        contributors.append(
            "peeling-chain structural signal"
        )

    if (
        safe_float(
            row.get(
                "mixing_evidence_score"
            )
        )
        >= 0.50
    ):

        contributors.append(
            "mixing-pattern structural signal"
        )

    # ------------------------------------------------------------------------
    # Network specifics
    # ------------------------------------------------------------------------

    if bool(
        row.get(
            "network_multi_observation_context",
            False,
        )
    ):

        contributors.append(
            "multiple correlated network observations"
        )

    if (
        safe_float(
            row.get(
                "network_unique_endpoint_count"
            )
        )
        >= 2
    ):

        contributors.append(
            "network endpoint diversity"
        )

    # ------------------------------------------------------------------------
    # No positive evidence
    # ------------------------------------------------------------------------

    if not contributors:

        return (
            "No material positive evidence channel was identified "
            "by the configured scoring signals."
        )

    channel_count = int(
        safe_float(
            row.get(
                "effective_evidence_channel_count"
            )
        )
    )

    if channel_count >= 4:

        agreement_text = (
            f"{channel_count} independent evidence channels "
            "contribute to the priority score."
        )

    elif channel_count >= 2:

        agreement_text = (
            f"{channel_count} evidence channels contribute "
            "to the priority score."
        )

    else:

        agreement_text = (
            "The priority score is driven primarily by "
            "individual evidence channels."
        )

    return (
        "Contributing evidence: "
        + "; ".join(
            contributors
        )
        + ". "
        + agreement_text
        + " "
        + "This score represents investigative priority, "
        + "not proof of illicit activity or identity."
    )


def add_explanations(
    result: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add human-readable and categorical explanation fields.
    """

    result = result.copy()

    result[
        "risk_explanation"
    ] = result.apply(
        generate_explanation,
        axis=1,
    )

    result[
        "evidence_agreement"
    ] = np.select(
        [
            result[
                "effective_evidence_channel_count"
            ]
            >= 4,

            result[
                "effective_evidence_channel_count"
            ]
            >= 3,

            result[
                "effective_evidence_channel_count"
            ]
            >= 2,
        ],
        [
            "VERY_HIGH",
            "HIGH",
            "MODERATE",
        ],
        default="LOW",
    )

    return result


# ============================================================================
# Integrity validation
# ============================================================================

def validate_output(
    result: pd.DataFrame,
) -> Dict:
    """
    Validate all M11.4 output invariants.
    """

    errors = []

    if result[
        "txid"
    ].duplicated().any():

        errors.append(
            "Duplicate TXIDs."
        )

    if result[
        "txid"
    ].eq("").any():

        errors.append(
            "Empty TXIDs."
        )

    # ------------------------------------------------------------------------
    # Signal bounds
    # ------------------------------------------------------------------------

    for column in (
        "ml_signal",
        "anomaly_signal",
        "behavioral_signal",
        "entity_signal",
        "network_signal",
        "weighted_evidence_score",
        "agreement_adjusted_score",
    ):

        values = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        if values.isna().any():

            errors.append(
                f"{column} contains missing/non-numeric values."
            )

        if (
            values < 0
        ).any() or (
            values > 1
        ).any():

            errors.append(
                f"{column} is outside [0,1]."
            )

    # ------------------------------------------------------------------------
    # Final score
    # ------------------------------------------------------------------------

    risk_scores = pd.to_numeric(
        result[
            "risk_score"
        ],
        errors="coerce",
    )

    if risk_scores.isna().any():

        errors.append(
            "risk_score contains missing values."
        )

    if (
        risk_scores < 0
    ).any() or (
        risk_scores > 100
    ).any():

        errors.append(
            "risk_score is outside [0,100]."
        )

    # ------------------------------------------------------------------------
    # M7 probabilities
    # ------------------------------------------------------------------------

    probability_matrix = result[
        [
            "ml_class_1_probability",
            "ml_class_2_probability",
            "ml_class_3_probability",
        ]
    ].to_numpy(
        dtype=float
    )

    if not np.allclose(
        probability_matrix.sum(
            axis=1
        ),
        1.0,
        atol=1e-5,
    ):

        errors.append(
            "M7 class probabilities do not sum to 1."
        )

    if (
        probability_matrix < 0
    ).any() or (
        probability_matrix > 1
    ).any():

        errors.append(
            "M7 class probabilities are outside [0,1]."
        )

    # ------------------------------------------------------------------------
    # Network semantics
    # ------------------------------------------------------------------------

    no_network = ~result[
        "network_observation_available"
    ]

    if not (
        result.loc[
            no_network,
            "network_signal",
        ]
        == 0
    ).all():

        errors.append(
            "Transactions without network observations "
            "have non-zero network signal."
        )

    # ------------------------------------------------------------------------
    # Evidence count
    # ------------------------------------------------------------------------

    channel_count = result[
        "effective_evidence_channel_count"
    ]

    if (
        channel_count < 0
    ).any() or (
        channel_count > 5
    ).any():

        errors.append(
            "Evidence channel count outside [0,5]."
        )

    # ------------------------------------------------------------------------
    # M7/M8 transformed feature dimensions
    # ------------------------------------------------------------------------

    if not (
        result[
            "ml_transformed_feature_count"
        ]
        == 124
    ).all():

        errors.append(
            "M7 transformed feature count is not 124."
        )

    if not (
        result[
            "anomaly_transformed_feature_count"
        ]
        == 124
    ).all():

        errors.append(
            "M8 transformed feature count is not 124."
        )

    if errors:

        raise AssertionError(
            "M11.4 validation failed: "
            + " | ".join(
                errors
            )
        )

    return {
        "status": "PASS",
        "rows": int(
            len(result)
        ),
        "unique_txids": int(
            result[
                "txid"
            ].nunique()
        ),
        "min_risk_score": float(
            risk_scores.min()
        ),
        "max_risk_score": float(
            risk_scores.max()
        ),
        "mean_risk_score": float(
            risk_scores.mean()
        ),
        "median_risk_score": float(
            risk_scores.median()
        ),
        "network_positive_transactions": int(
            result[
                "network_observation_available"
            ].sum()
        ),
        "very_high_count": int(
            (
                result[
                    "risk_level"
                ]
                == "VERY_HIGH"
            ).sum()
        ),
        "high_count": int(
            (
                result[
                    "risk_level"
                ]
                == "HIGH"
            ).sum()
        ),
        "moderate_count": int(
            (
                result[
                    "risk_level"
                ]
                == "MODERATE"
            ).sum()
        ),
        "guarded_count": int(
            (
                result[
                    "risk_level"
                ]
                == "GUARDED"
            ).sum()
        ),
        "low_count": int(
            (
                result[
                    "risk_level"
                ]
                == "LOW"
            ).sum()
        ),
        "mean_ml_signal": float(
            result[
                "ml_signal"
            ].mean()
        ),
        "mean_anomaly_signal": float(
            result[
                "anomaly_signal"
            ].mean()
        ),
        "mean_behavioral_signal": float(
            result[
                "behavioral_signal"
            ].mean()
        ),
        "mean_entity_signal": float(
            result[
                "entity_signal"
            ].mean()
        ),
        "mean_network_signal": float(
            result[
                "network_signal"
            ].mean()
        ),
        "mean_evidence_channels": float(
            result[
                "effective_evidence_channel_count"
            ].mean()
        ),
        "m7_transformed_features": 124,
        "m8_transformed_features": 124,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "M11.4 UNIFIED RISK SCORING ENGINE"
    )
    print("=" * 65)

    print(
        "Normalized evidence:",
        NORMALIZED_EVIDENCE_FILE,
    )

    print(
        "Temporal-safe features:",
        TEMPORAL_SAFE_FEATURES_FILE,
    )

    print(
        "M7 model:",
        XGB_MODEL_FILE,
    )

    print(
        "M8 model:",
        IF_MODEL_FILE,
    )

    # ------------------------------------------------------------------------
    # Build base evidence
    # ------------------------------------------------------------------------

    print()
    print(
        "Building transaction-level evidence table..."
    )

    (
        result,
        feature_df,
    ) = build_fused_dataset()

    print(
        f"Base fusion rows: {len(result):,}"
    )

    # ------------------------------------------------------------------------
    # Deterministic ordering
    # ------------------------------------------------------------------------

    feature_df = (
        feature_df
        .sort_values(
            [
                "time_step",
                "txid",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    result = (
        result
        .sort_values(
            [
                "time_step",
                "txid",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if not (
        result[
            "txid"
        ].to_numpy()
        ==
        feature_df[
            "txid"
        ].to_numpy()
    ).all():

        raise AssertionError(
            "Fusion rows and model feature rows are not "
            "in identical TXID order."
        )

    # ------------------------------------------------------------------------
    # M7
    # ------------------------------------------------------------------------

    print()
    print(
        "Running M7 production XGBoost inference..."
    )

    result = add_ml_signals(
        result,
        feature_df,
    )

    print(
        "M7 inference complete."
    )

    print(
        "M7 transformed feature count:",
        result[
            "ml_transformed_feature_count"
        ].iloc[0],
    )

    # ------------------------------------------------------------------------
    # M8
    # ------------------------------------------------------------------------

    print()
    print(
        "Running M8 Isolation Forest inference..."
    )

    result = add_anomaly_signals(
        result,
        feature_df,
    )

    print(
        "M8 inference complete."
    )

    print(
        "M8 transformed feature count:",
        result[
            "anomaly_transformed_feature_count"
        ].iloc[0],
    )

    # ------------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------------

    print()
    print(
        "Calculating unified evidence score..."
    )

    result = calculate_scores(
        result
    )

    result = add_explanations(
        result
    )

    # ------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------

    print()
    print(
        "Validating M11.4 output..."
    )

    validation = validate_output(
        result
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    output_sha256 = hashlib.sha256(
        OUTPUT_FILE.read_bytes()
    ).hexdigest()

    # ------------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------------

    report = {
        "module": "M11.4",
        "status": validation[
            "status"
        ],
        "purpose": (
            "Fuse supervised-model confidence, anomaly evidence, "
            "behavioral evidence, temporal entity evidence and "
            "network evidence into a bounded investigative "
            "priority score."
        ),
        "score_semantics": (
            "Investigative priority score in [0,100]. "
            "It is not a probability of illicit activity and "
            "does not establish identity, ownership, authorship "
            "or criminal activity."
        ),
        "class_semantics": (
            "Elliptic++ class IDs remain numeric. No class is "
            "assumed to represent illicit or legitimate activity."
        ),
        "weights": WEIGHTS,
        "agreement_bonus": AGREEMENT_BONUS,
        "risk_bands": {
            "VERY_HIGH": "80-100",
            "HIGH": "60-79.999",
            "MODERATE": "40-59.999",
            "GUARDED": "20-39.999",
            "LOW": "0-19.999",
        },
        "validation": validation,
        "preprocessing": {
            "m7_raw_features": 87,
            "m7_transformed_features": 124,
            "m7_preprocessing": (
                "Saved SimpleImputer(add_indicator=True, strategy='median')"
            ),
            "m8_raw_features": 87,
            "m8_transformed_features": 124,
            "m8_preprocessing": (
                "Saved SimpleImputer(add_indicator=True, strategy='median')"
            ),
        },
        "network_coverage_warning": (
            "Current network evidence is based on the supplied "
            "synthetic validation fixture and therefore has "
            "extremely low coverage."
        ),
        "double_counting_controls": [
            (
                "Behavioral score is consumed as one composite "
                "channel; peeling and mixing are retained only "
                "for explanation."
            ),
            (
                "Entity score is consumed as one normalized "
                "channel; raw entity features are explanatory."
            ),
            (
                "Network score is consumed as one normalized "
                "channel; raw network features are explanatory."
            ),
            (
                "ML confidence is not interpreted as a class-risk "
                "probability."
            ),
        ],
        "model_artifacts": {
            "xgboost_model": str(
                XGB_MODEL_FILE
            ),
            "isolation_forest_model": str(
                IF_MODEL_FILE
            ),
        },
        "input_artifacts": {
            "normalized_evidence": str(
                NORMALIZED_EVIDENCE_FILE
            ),
            "temporal_safe_features": str(
                TEMPORAL_SAFE_FEATURES_FILE
            ),
            "behavioral_evidence": str(
                BEHAVIORAL_FILE
            ),
            "temporal_entity_evidence": str(
                ENTITY_FILE
            ),
            "network_transaction_features": str(
                NETWORK_FILE
            ),
        },
        "output": str(
            OUTPUT_FILE
        ),
        "output_sha256": output_sha256,
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------------

    print()
    print(
        "M11.4 RESULT"
    )
    print("-" * 65)

    print(
        f"OUTPUT ROWS: "
        f"{len(result):,}"
    )

    print(
        f"UNIQUE TXIDS: "
        f"{result['txid'].nunique():,}"
    )

    print(
        f"MEAN RISK SCORE: "
        f"{result['risk_score'].mean():.4f}"
    )

    print(
        f"MEDIAN RISK SCORE: "
        f"{result['risk_score'].median():.4f}"
    )

    print(
        f"MIN RISK SCORE: "
        f"{result['risk_score'].min():.4f}"
    )

    print(
        f"MAX RISK SCORE: "
        f"{result['risk_score'].max():.4f}"
    )

    print()
    print(
        "RISK LEVEL DISTRIBUTION:"
    )

    print(
        result[
            "risk_level"
        ]
        .value_counts()
        .reindex(
            [
                "VERY_HIGH",
                "HIGH",
                "MODERATE",
                "GUARDED",
                "LOW",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print()
    print(
        "MEAN EVIDENCE SIGNALS:"
    )

    print(
        f"ML:         "
        f"{result['ml_signal'].mean():.6f}"
    )

    print(
        f"Anomaly:    "
        f"{result['anomaly_signal'].mean():.6f}"
    )

    print(
        f"Behavioral: "
        f"{result['behavioral_signal'].mean():.6f}"
    )

    print(
        f"Entity:     "
        f"{result['entity_signal'].mean():.6f}"
    )

    print(
        f"Network:    "
        f"{result['network_signal'].mean():.6f}"
    )

    print()
    print(
        "NETWORK POSITIVE TRANSACTIONS:",
        int(
            result[
                "network_observation_available"
            ].sum()
        ),
    )

    print()
    print(
        "TOP 10 INVESTIGATIVE PRIORITY TRANSACTIONS:"
    )

    preview_columns = [
        "txid",
        "time_step",
        "risk_score",
        "risk_level",
        "ml_predicted_class",
        "ml_confidence",
        "anomaly_signal",
        "behavioral_signal",
        "entity_signal",
        "network_signal",
        "effective_evidence_channel_count",
    ]

    print(
        result[
            preview_columns
        ]
        .sort_values(
            "risk_score",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "OUTPUT:",
        OUTPUT_FILE,
    )

    print(
        "REPORT:",
        REPORT_FILE,
    )

    print()
    print(
        "M11.4 COMPLETE"
    )


if __name__ == "__main__":
    main()