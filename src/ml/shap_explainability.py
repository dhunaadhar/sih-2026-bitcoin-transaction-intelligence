from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "production_xgboost" / "model.joblib"
FEATURE_PATH = ROOT / "data" / "derived" / "unified_features_temporal_safe.parquet"
ALERT_PATH = ROOT / "data" / "derived" / "ranked_alerts.parquet"

OUTPUT_PATH = ROOT / "data" / "derived" / "shap_explanations.parquet"
REPORT_PATH = ROOT / "reports" / "ml" / "m10_shap_explainability.json"

TOP_K_ALERTS = 1000
TOP_FEATURES_PER_CLASS = 10


def ensure_paths() -> None:
    for path, label in [
        (MODEL_PATH, "Production model"),
        (FEATURE_PATH, "Temporal-safe feature dataset"),
        (ALERT_PATH, "Ranked alerts"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def load_model_package() -> dict[str, Any]:
    package = joblib.load(MODEL_PATH)

    if not isinstance(package, dict):
        raise TypeError("Production XGBoost artifact must be a dictionary.")

    required = {
        "imputer",
        "classifier",
        "model_features",
        "label_mapping",
        "inverse_label_mapping",
    }

    missing = required.difference(package.keys())

    if missing:
        raise ValueError(
            f"Production model package is missing required keys: "
            f"{sorted(missing)}"
        )

    model_features = list(package["model_features"])

    if len(model_features) != 87:
        raise ValueError(
            f"Expected 87 production model features, got {len(model_features)}"
        )

    return package


def normalize_txid_series(series: pd.Series) -> pd.Series:
    """
    Normalize TXIDs to a canonical string representation.

    The source artifacts may store TXIDs as object/string in one parquet
    and numeric/int64 in another. Numeric values such as 4304541.0 are
    normalized to '4304541' so exact joins remain deterministic.

    No source artifact is modified.
    """

    def normalize(value: Any) -> str | None:
        if pd.isna(value):
            return None

        text = str(value).strip()

        if not text:
            return None

        try:
            numeric = float(text)

            if math.isfinite(numeric) and numeric.is_integer():
                return str(int(numeric))
        except (TypeError, ValueError):
            pass

        return text

    return series.map(normalize).astype("string")


def get_transformed_feature_names(
    imputer: Any,
    model_features: list[str],
    transformed_count: int,
) -> list[str]:
    try:
        names = list(imputer.get_feature_names_out(model_features))

        if len(names) == transformed_count:
            return names
    except Exception:
        pass

    if transformed_count == len(model_features):
        return model_features

    indicator_names = [
        f"missing_indicator__{name}"
        for name in model_features
    ]

    if transformed_count == len(model_features) + len(indicator_names):
        return model_features + indicator_names

    return [
        f"transformed_feature_{i}"
        for i in range(transformed_count)
    ]


def normalize_shap_values(
    shap_values: Any,
    n_samples: int,
    n_features: int,
    predicted_indices: np.ndarray,
) -> np.ndarray:
    """
    Normalize SHAP output across supported SHAP output layouts.

    Final shape:
        (n_samples, n_features)

    Values correspond to the predicted class of each transaction.
    """

    if isinstance(shap_values, list):
        if not shap_values:
            raise ValueError("SHAP returned an empty list.")

        arrays = [np.asarray(value) for value in shap_values]

        for array in arrays:
            if array.shape != (n_samples, n_features):
                raise ValueError(
                    "Unexpected SHAP array shape: "
                    f"{array.shape}; expected "
                    f"{(n_samples, n_features)}"
                )

        stacked = np.stack(arrays, axis=2)
        rows = np.arange(n_samples)

        return stacked[rows, :, predicted_indices]

    values = np.asarray(shap_values)

    if values.ndim == 3:
        # (samples, features, classes)
        if (
            values.shape[0] == n_samples
            and values.shape[1] == n_features
        ):
            rows = np.arange(n_samples)
            return values[rows, :, predicted_indices]

        # (classes, samples, features)
        if (
            values.shape[1] == n_samples
            and values.shape[2] == n_features
        ):
            output = np.empty(
                (n_samples, n_features),
                dtype=float,
            )

            for index in range(n_samples):
                output[index] = values[
                    predicted_indices[index],
                    index,
                    :,
                ]

            return output

        raise ValueError(
            f"Unexpected 3D SHAP output shape: {values.shape}"
        )

    if values.ndim == 2:
        if values.shape != (n_samples, n_features):
            raise ValueError(
                f"Unexpected 2D SHAP output shape: {values.shape}"
            )

        return values

    raise ValueError(
        f"Unsupported SHAP output dimensionality: {values.ndim}"
    )


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return 0.0

    if not math.isfinite(result):
        return 0.0

    return result


def humanize_feature_name(name: str) -> str:
    if name.startswith("missing_indicator__"):
        base = name.replace(
            "missing_indicator__",
            "",
            1,
        )
        return f"Missing value indicator: {base}"

    if name.startswith("transformed_feature_"):
        return name.replace("_", " ").title()

    return name.replace("_", " ").strip().title()


def make_feature_description(
    feature_name: str,
    shap_value: float,
    feature_value: float,
) -> str:
    direction = (
        "increased"
        if shap_value > 0
        else "decreased"
    )

    return (
        f"{humanize_feature_name(feature_name)} {direction} "
        f"the predicted-class model output "
        f"(SHAP {shap_value:+.4f}; "
        f"feature value {feature_value:.4f})."
    )


def build_explanation(
    ranked_features: list[dict[str, Any]],
    predicted_class: int,
    confidence: float,
) -> str:
    if not ranked_features:
        return (
            f"Predicted class={predicted_class} with confidence "
            f"{confidence:.3f}. "
            "No SHAP feature contribution was available."
        )

    positive = sorted(
        [
            item
            for item in ranked_features
            if item["shap_value"] > 0
        ],
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    negative = sorted(
        [
            item
            for item in ranked_features
            if item["shap_value"] < 0
        ],
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    parts = [
        f"Predicted class={predicted_class} "
        f"with confidence {confidence:.3f}."
    ]

    if positive:
        names = ", ".join(
            humanize_feature_name(item["feature_name"])
            for item in positive[:3]
        )

        parts.append(
            f"Strongest model contributions toward the predicted "
            f"class: {names}."
        )

    if negative:
        names = ", ".join(
            humanize_feature_name(item["feature_name"])
            for item in negative[:3]
        )

        parts.append(
            f"Strongest opposing contributions: {names}."
        )

    parts.append(
        "SHAP values describe model contribution and do not "
        "establish illicit activity, ownership, identity, "
        "intent, or guilt."
    )

    return " ".join(parts)


def main() -> None:
    ensure_paths()

    print("M10 SHAP EXPLAINABILITY")
    print("=" * 72)

    print("Loading production model...")
    package = load_model_package()

    imputer = package["imputer"]
    classifier = package["classifier"]
    model_features = list(package["model_features"])
    inverse_label_mapping = package["inverse_label_mapping"]

    print(
        f"Production model features: {len(model_features)}"
    )

    print("Loading ranked alerts...")
    alerts = pd.read_parquet(ALERT_PATH)

    required_alert_columns = {
        "txid",
        "investigation_rank",
        "ml_predicted_class",
        "ml_confidence",
    }

    missing_alert_columns = (
        required_alert_columns.difference(alerts.columns)
    )

    if missing_alert_columns:
        raise ValueError(
            "ranked_alerts.parquet is missing columns: "
            f"{sorted(missing_alert_columns)}"
        )

    alerts = (
        alerts.sort_values(
            ["investigation_rank", "txid"],
            kind="mergesort",
        )
        .head(TOP_K_ALERTS)
        .copy()
    )

    if alerts["txid"].duplicated().any():
        raise ValueError(
            "Duplicate TXIDs found in selected alert queue."
        )

    print(
        f"Selected alerts for SHAP: {len(alerts):,}"
    )

    print("Loading temporal-safe feature matrix...")
    features = pd.read_parquet(FEATURE_PATH)

    if "txid" not in features.columns:
        raise ValueError(
            "unified_features_temporal_safe.parquet "
            "is missing txid."
        )

    missing_features = [
        feature
        for feature in model_features
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Temporal-safe feature matrix is missing "
            f"model features: {missing_features}"
        )

    # Preserve the original alert TXID representation for output.
    alerts["_txid_join_key"] = normalize_txid_series(
        alerts["txid"]
    )

    features["_txid_join_key"] = normalize_txid_series(
        features["txid"]
    )

    if alerts["_txid_join_key"].isna().any():
        raise ValueError(
            "Selected alerts contain null TXIDs."
        )

    if features["_txid_join_key"].isna().any():
        raise ValueError(
            "Temporal-safe features contain null TXIDs."
        )

    if alerts["_txid_join_key"].duplicated().any():
        raise ValueError(
            "Duplicate normalized TXIDs found in alerts."
        )

    if features["_txid_join_key"].duplicated().any():
        raise ValueError(
            "Duplicate normalized TXIDs found in features."
        )

    selected = alerts[
        [
            "txid",
            "investigation_rank",
            "ml_predicted_class",
            "ml_confidence",
            "_txid_join_key",
        ]
    ].merge(
        features[
            ["_txid_join_key"] + model_features
        ],
        on="_txid_join_key",
        how="left",
        validate="one_to_one",
    )

    if len(selected) != len(alerts):
        raise ValueError(
            "SHAP alert-feature join changed row count."
        )

    if selected["_txid_join_key"].isna().any():
        raise ValueError(
            "Missing normalized TXID after join."
        )

    if selected[model_features].isna().all(axis=1).any():
        raise ValueError(
            "At least one selected alert has no matching "
            "temporal-safe feature row."
        )

    selected = selected.drop(
        columns=["_txid_join_key"]
    )

    if selected["txid"].duplicated().any():
        raise ValueError(
            "Duplicate TXIDs appeared after SHAP feature join."
        )

    X = selected[model_features].copy()

    print("Applying production imputer...")
    X_transformed = imputer.transform(X)
    X_transformed = np.asarray(
        X_transformed,
        dtype=float,
    )

    expected_transformed_features = int(
        getattr(
            classifier,
            "n_features_in_",
            X_transformed.shape[1],
        )
    )

    if (
        X_transformed.shape[1]
        != expected_transformed_features
    ):
        raise ValueError(
            "Transformed feature count mismatch: "
            f"{X_transformed.shape[1]} != "
            f"{expected_transformed_features}"
        )

    transformed_feature_names = (
        get_transformed_feature_names(
            imputer,
            model_features,
            X_transformed.shape[1],
        )
    )

    if len(transformed_feature_names) != X_transformed.shape[1]:
        raise ValueError(
            "Unable to establish transformed feature names."
        )

    print(
        f"Transformed feature count: "
        f"{X_transformed.shape[1]}"
    )

    predicted_classes = (
        selected["ml_predicted_class"]
        .astype(int)
        .to_numpy()
    )

    predicted_indices = np.empty(
        len(predicted_classes),
        dtype=int,
    )

    for index, original_class in enumerate(
        predicted_classes
    ):
        matches = [
            int(internal_index)
            for internal_index, original_value
            in inverse_label_mapping.items()
            if int(original_value)
            == int(original_class)
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Unable to map original class "
                f"{original_class} to exactly one "
                "internal class index."
            )

        predicted_indices[index] = matches[0]

    print("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(classifier)

    print("Calculating SHAP values...")
    raw_shap_values = explainer.shap_values(
        X_transformed
    )

    shap_matrix = normalize_shap_values(
        raw_shap_values,
        n_samples=len(selected),
        n_features=X_transformed.shape[1],
        predicted_indices=predicted_indices,
    )

    shap_matrix = np.asarray(
        shap_matrix,
        dtype=float,
    )

    if shap_matrix.shape != X_transformed.shape:
        raise ValueError(
            "Final SHAP matrix shape mismatch: "
            f"{shap_matrix.shape} != "
            f"{X_transformed.shape}"
        )

    print("Building explanations...")

    rows: list[dict[str, Any]] = []

    for row_index in range(len(selected)):
        txid = selected.iloc[row_index]["txid"]

        predicted_class = int(
            predicted_classes[row_index]
        )

        confidence = safe_float(
            selected.iloc[row_index]["ml_confidence"]
        )

        contributions = shap_matrix[row_index]

        ranked_indices = np.argsort(
            np.abs(contributions)
        )[::-1]

        top_indices = ranked_indices[
            :TOP_FEATURES_PER_CLASS
        ]

        feature_records: list[
            dict[str, Any]
        ] = []

        for feature_index in top_indices:
            feature_name = (
                transformed_feature_names[
                    feature_index
                ]
            )

            shap_value = safe_float(
                contributions[feature_index]
            )

            feature_value = safe_float(
                X_transformed[
                    row_index,
                    feature_index,
                ]
            )

            feature_records.append(
                {
                    "feature_name": feature_name,
                    "human_readable_feature":
                        humanize_feature_name(
                            feature_name
                        ),
                    "shap_value": shap_value,
                    "absolute_shap_value":
                        abs(shap_value),
                    "transformed_feature_value":
                        feature_value,
                    "direction": (
                        "toward_predicted_class"
                        if shap_value > 0
                        else "against_predicted_class"
                    ),
                    "description":
                        make_feature_description(
                            feature_name,
                            shap_value,
                            feature_value,
                        ),
                }
            )

        positive_sum = float(
            np.sum(
                np.maximum(
                    contributions,
                    0.0,
                )
            )
        )

        negative_sum = float(
            np.sum(
                np.minimum(
                    contributions,
                    0.0,
                )
            )
        )

        absolute_sum = float(
            np.sum(
                np.abs(contributions)
            )
        )

        if absolute_sum > 0:
            positive_fraction = (
                positive_sum / absolute_sum
            )
            negative_fraction = (
                abs(negative_sum)
                / absolute_sum
            )
        else:
            positive_fraction = 0.0
            negative_fraction = 0.0

        explanation = build_explanation(
            feature_records,
            predicted_class,
            confidence,
        )

        rows.append(
            {
                "txid": txid,
                "investigation_rank": int(
                    selected.iloc[row_index][
                        "investigation_rank"
                    ]
                ),
                "ml_predicted_class":
                    predicted_class,
                "ml_confidence":
                    confidence,
                "shap_feature_count":
                    int(shap_matrix.shape[1]),
                "shap_positive_contribution":
                    positive_sum,
                "shap_negative_contribution":
                    negative_sum,
                "shap_absolute_contribution":
                    absolute_sum,
                "shap_positive_fraction":
                    positive_fraction,
                "shap_negative_fraction":
                    negative_fraction,
                "top_feature_1": (
                    feature_records[0][
                        "feature_name"
                    ]
                    if len(feature_records) > 0
                    else None
                ),
                "top_feature_1_shap": (
                    feature_records[0][
                        "shap_value"
                    ]
                    if len(feature_records) > 0
                    else 0.0
                ),
                "top_feature_2": (
                    feature_records[1][
                        "feature_name"
                    ]
                    if len(feature_records) > 1
                    else None
                ),
                "top_feature_2_shap": (
                    feature_records[1][
                        "shap_value"
                    ]
                    if len(feature_records) > 1
                    else 0.0
                ),
                "top_feature_3": (
                    feature_records[2][
                        "feature_name"
                    ]
                    if len(feature_records) > 2
                    else None
                ),
                "top_feature_3_shap": (
                    feature_records[2][
                        "shap_value"
                    ]
                    if len(feature_records) > 2
                    else 0.0
                ),
                "top_features_json": json.dumps(
                    feature_records,
                    ensure_ascii=False,
                ),
                "human_readable_explanation":
                    explanation,
                "explainability_method":
                    "SHAP TreeExplainer",
                "explainability_scope":
                    "top_1000_ranked_alerts",
                "interpretation_note":
                    (
                        "SHAP contribution explains "
                        "the supervised model's "
                        "prediction. It is not evidence "
                        "of illicit activity, identity, "
                        "ownership, or intent."
                    ),
            }
        )

    output = pd.DataFrame(rows)

    print("Validating SHAP artifact...")

    if len(output) != len(alerts):
        raise ValueError(
            "SHAP output row count does not match "
            "selected alert count."
        )

    if output["txid"].nunique() != len(output):
        raise ValueError(
            "SHAP output contains duplicate TXIDs."
        )

    if output["shap_feature_count"].nunique() != 1:
        raise ValueError(
            "SHAP feature counts are inconsistent."
        )

    if (
        output["shap_feature_count"].iloc[0]
        != X_transformed.shape[1]
    ):
        raise ValueError(
            "Stored SHAP feature count does not "
            "match transformed matrix."
        )

    numeric_columns = [
        "ml_confidence",
        "shap_positive_contribution",
        "shap_negative_contribution",
        "shap_absolute_contribution",
        "shap_positive_fraction",
        "shap_negative_fraction",
        "top_feature_1_shap",
        "top_feature_2_shap",
        "top_feature_3_shap",
    ]

    if np.isinf(
        output[numeric_columns]
        .to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "Infinite values found in SHAP output."
        )

    if output[
        "human_readable_explanation"
    ].isna().any():
        raise ValueError(
            "Missing human-readable explanations."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    report = {
        "milestone": "M10",
        "component": "SHAP Explainability",
        "status": "PASS",
        "method": "SHAP TreeExplainer",
        "model_path":
            str(MODEL_PATH.relative_to(ROOT)),
        "feature_path":
            str(FEATURE_PATH.relative_to(ROOT)),
        "alert_path":
            str(ALERT_PATH.relative_to(ROOT)),
        "output_path":
            str(OUTPUT_PATH.relative_to(ROOT)),
        "rows_explained":
            int(len(output)),
        "unique_txids":
            int(output["txid"].nunique()),
        "source_alert_queue_size":
            int(len(alerts)),
        "base_model_features":
            int(len(model_features)),
        "transformed_model_features":
            int(X_transformed.shape[1]),
        "top_features_per_transaction":
            int(TOP_FEATURES_PER_CLASS),
        "predicted_class_distribution": {
            str(int(key)): int(value)
            for key, value
            in output[
                "ml_predicted_class"
            ]
            .value_counts()
            .sort_index()
            .items()
        },
        "mean_absolute_shap_contribution":
            float(
                output[
                    "shap_absolute_contribution"
                ].mean()
            ),
        "mean_positive_shap_contribution":
            float(
                output[
                    "shap_positive_contribution"
                ].mean()
            ),
        "mean_negative_shap_contribution":
            float(
                output[
                    "shap_negative_contribution"
                ].mean()
            ),
        "validation": {
            "row_count_match": True,
            "unique_txids": True,
            "txid_normalization_join": True,
            "transformed_feature_count_match": True,
            "finite_values": True,
            "human_readable_explanations_present":
                True,
        },
        "interpretation": (
            "SHAP explanations identify feature "
            "contributions to the production "
            "supervised model prediction. They do "
            "not establish illicit activity, "
            "real-world identity, ownership, "
            "intent, or guilt."
        ),
        "network_coverage_note": (
            "SHAP explanations are independent "
            "of the current limited synthetic "
            "network-observation coverage."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("M10 SHAP RESULT")
    print("=" * 72)
    print(
        f"ROWS EXPLAINED: {len(output):,}"
    )
    print(
        f"UNIQUE TXIDS: "
        f"{output.txid.nunique():,}"
    )
    print(
        f"BASE MODEL FEATURES: "
        f"{len(model_features)}"
    )
    print(
        f"TRANSFORMED FEATURES: "
        f"{X_transformed.shape[1]}"
    )
    print(
        "MEAN ABSOLUTE SHAP: "
        f"{output.shap_absolute_contribution.mean():.6f}"
    )
    print()
    print("PREDICTED CLASS DISTRIBUTION:")
    print(
        output.ml_predicted_class
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()
    print("TOP EXPLANATION:")
    print(
        output.iloc[0][
            "human_readable_explanation"
        ]
    )
    print()
    print(
        f"OUTPUT: "
        f"{OUTPUT_PATH.relative_to(ROOT)}"
    )
    print(
        f"REPORT: "
        f"{REPORT_PATH.relative_to(ROOT)}"
    )
    print()
    print(
        "M10 SHAP EXPLAINABILITY COMPLETE"
    )


if __name__ == "__main__":
    main()