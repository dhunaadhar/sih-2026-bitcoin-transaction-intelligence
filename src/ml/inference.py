from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "production_xgboost"

MODEL_FILE = MODEL_DIR / "model.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"


class ProductionInferenceEngine:
    """
    Production inference engine for the temporal-safe XGBoost model.

    Responsibilities:
        1. Load the persisted preprocessing + classifier package.
        2. Validate incoming feature data.
        3. Select the exact 87 model features.
        4. Apply the persisted imputer.
        5. Generate class probabilities.
        6. Return the predicted original class ID and confidence.

    Important:
        Class IDs 1, 2 and 3 are preserved exactly as supplied by
        the benchmark dataset. No semantic interpretation is assigned
        here.
    """

    def __init__(
        self,
        model_file: Path = MODEL_FILE,
        schema_file: Path = SCHEMA_FILE,
    ) -> None:
        self.model_file = Path(model_file)
        self.schema_file = Path(schema_file)

        self._validate_artifacts()

        self.package = joblib.load(self.model_file)

        with self.schema_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.schema = json.load(f)

        self.imputer = self.package["imputer"]
        self.classifier = self.package["classifier"]

        self.model_features = list(
            self.package["model_features"]
        )

        self.label_mapping = {
            int(k): int(v)
            for k, v in self.package[
                "label_mapping"
            ].items()
        }

        self.inverse_label_mapping = {
            int(k): int(v)
            for k, v in self.package[
                "inverse_label_mapping"
            ].items()
        }

        self.transformed_feature_names = (
            self.imputer
            .get_feature_names_out(
                self.model_features
            )
            .tolist()
        )

        self._validate_loaded_artifacts()

    def _validate_artifacts(self) -> None:
        if not self.model_file.exists():
            raise FileNotFoundError(
                f"Production model not found: "
                f"{self.model_file}"
            )

        if not self.schema_file.exists():
            raise FileNotFoundError(
                f"Feature schema not found: "
                f"{self.schema_file}"
            )

    def _validate_loaded_artifacts(self) -> None:
        if not self.model_features:
            raise ValueError(
                "Production model contains no model features."
            )

        schema_features = list(
            self.schema.get("model_features", [])
        )

        if schema_features != self.model_features:
            raise ValueError(
                "Model feature list does not match "
                "feature_schema.json."
            )

        expected_count = int(
            self.schema.get(
                "original_feature_count",
                len(self.model_features),
            )
        )

        if expected_count != len(self.model_features):
            raise ValueError(
                "Feature count mismatch: "
                f"schema={expected_count}, "
                f"model={len(self.model_features)}."
            )

        transformed_count = len(
            self.transformed_feature_names
        )

        schema_transformed_count = int(
            self.schema.get(
                "transformed_feature_count",
                transformed_count,
            )
        )

        if transformed_count != schema_transformed_count:
            raise ValueError(
                "Transformed feature count mismatch: "
                f"schema={schema_transformed_count}, "
                f"actual={transformed_count}."
            )

        classifier_features = getattr(
            self.classifier,
            "n_features_in_",
            None,
        )

        if (
            classifier_features is not None
            and classifier_features != transformed_count
        ):
            raise ValueError(
                "Classifier feature count mismatch: "
                f"classifier={classifier_features}, "
                f"transformed={transformed_count}."
            )

    def _prepare_dataframe(
        self,
        features: Mapping[str, Any] | pd.DataFrame,
    ) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            df = features.copy()

        elif isinstance(features, Mapping):
            df = pd.DataFrame([dict(features)])

        else:
            raise TypeError(
                "features must be either a mapping "
                "or a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "No feature rows were supplied."
            )

        missing = [
            feature
            for feature in self.model_features
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing required model features: "
                f"{missing}"
            )

        # Select only the frozen model features.
        X = df[self.model_features].copy()

        # Convert all model columns to numeric.
        for column in X.columns:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            )

        numeric_values = X.to_numpy(
            dtype=float
        )

        # NaN is allowed because the persisted SimpleImputer
        # handles missing values.
        if np.isinf(numeric_values).any():
            raise ValueError(
                "Input contains infinite feature values."
            )

        return X

    def predict(
        self,
        features: Mapping[str, Any] | pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Predict one or more transactions.

        Returns:
            {
                "predictions": [
                    {
                        "predicted_class": 3,
                        "confidence": 0.91,
                        "probabilities": {
                            "class_1": 0.01,
                            "class_2": 0.08,
                            "class_3": 0.91
                        }
                    }
                ]
            }
        """

        X = self._prepare_dataframe(features)

        X_transformed = self.imputer.transform(X)

        probabilities = self.classifier.predict_proba(
            X_transformed
        )

        encoded_predictions = np.argmax(
            probabilities,
            axis=1,
        )

        predictions = []

        for encoded_class, probability_vector in zip(
            encoded_predictions,
            probabilities,
        ):
            encoded_class = int(encoded_class)

            if encoded_class not in self.inverse_label_mapping:
                raise ValueError(
                    "Classifier returned an unknown encoded "
                    f"class: {encoded_class}"
                )

            original_class = self.inverse_label_mapping[
                encoded_class
            ]

            probability_vector = np.asarray(
                probability_vector,
                dtype=float,
            )

            probability_sum = float(
                probability_vector.sum()
            )

            if probability_sum <= 0:
                raise ValueError(
                    "Invalid probability vector returned "
                    "by classifier."
                )

            # Normalize defensively against tiny floating-point
            # deviations from a sum of exactly 1.
            probability_vector = (
                probability_vector / probability_sum
            )

            class_probabilities = {}

            for encoded_index, probability in enumerate(
                probability_vector
            ):
                if encoded_index not in (
                    self.inverse_label_mapping
                ):
                    raise ValueError(
                        "Classifier probability vector "
                        "contains an unknown class index: "
                        f"{encoded_index}"
                    )

                original_label = (
                    self.inverse_label_mapping[
                        encoded_index
                    ]
                )

                class_probabilities[
                    f"class_{original_label}"
                ] = float(probability)

            confidence = float(
                probability_vector[encoded_class]
            )

            predictions.append(
                {
                    "predicted_class": original_class,
                    "confidence": confidence,
                    "probabilities": class_probabilities,
                }
            )

        return {
            "predictions": predictions,
            "model": "production_xgboost",
            "feature_count": len(self.model_features),
            "transformed_feature_count": len(
                self.transformed_feature_names
            ),
        }

    def predict_one(
        self,
        features: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Convenience method for a single transaction.
        """

        result = self.predict(features)

        if len(result["predictions"]) != 1:
            raise RuntimeError(
                "predict_one expected exactly one prediction."
            )

        return result["predictions"][0]


_ENGINE: ProductionInferenceEngine | None = None


def get_engine() -> ProductionInferenceEngine:
    """
    Return a process-level singleton inference engine.
    """

    global _ENGINE

    if _ENGINE is None:
        _ENGINE = ProductionInferenceEngine()

    return _ENGINE


def predict_transaction(
    features: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Module-level convenience function.
    """

    return get_engine().predict_one(features)


def main() -> None:
    """
    Basic artifact/inference smoke test.

    Uses the validation dataset only to obtain one real feature row.
    This does not retrain the model and does not modify the dataset.
    """

    validation_file = (
        ROOT
        / "data"
        / "evaluation"
        / "validation_temporal_safe.parquet"
    )

    if not validation_file.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: "
            f"{validation_file}"
        )

    print("=" * 72)
    print("M7.2 PRODUCTION INFERENCE ENGINE")
    print("=" * 72)

    engine = ProductionInferenceEngine()

    print(
        f"Model features: "
        f"{len(engine.model_features)}"
    )

    print(
        f"Transformed features: "
        f"{len(engine.transformed_feature_names)}"
    )

    validation = pd.read_parquet(
        validation_file
    )

    missing = [
        feature
        for feature in engine.model_features
        if feature not in validation.columns
    ]

    if missing:
        raise ValueError(
            "Validation dataset is missing model features: "
            f"{missing}"
        )

    sample = validation.iloc[
        0
    ][engine.model_features].to_dict()

    expected_label = int(
        validation.iloc[0]["label"]
    )

    result = engine.predict_one(sample)

    print("\nSmoke-test transaction:")
    print(
        f"  TXID: "
        f"{validation.iloc[0]['txid']}"
    )

    print(
        f"  Actual benchmark label: "
        f"{expected_label}"
    )

    print(
        f"  Predicted class: "
        f"{result['predicted_class']}"
    )

    print(
        f"  Confidence: "
        f"{result['confidence']:.6f}"
    )

    print("\nProbabilities:")

    for class_name, probability in (
        result["probabilities"].items()
    ):
        print(
            f"  {class_name}: "
            f"{probability:.6f}"
        )

    probability_sum = sum(
        result["probabilities"].values()
    )

    if not np.isclose(
        probability_sum,
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Probability sum validation failed: "
            f"{probability_sum}"
        )

    print(
        f"\nProbability sum: "
        f"{probability_sum:.6f}"
    )

    print("\n" + "=" * 72)
    print("M7.2 COMPLETE")
    print("=" * 72)
    print(
        "Production inference engine loaded successfully."
    )


if __name__ == "__main__":
    main()