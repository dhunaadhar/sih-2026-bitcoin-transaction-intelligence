from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    ROOT
    / "models"
    / "production_isolation_forest"
)

MODEL_FILE = MODEL_DIR / "model.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"


class ProductionAnomalyEngine:
    """
    Production inference engine for the temporal-safe
    Isolation Forest anomaly detector.

    The engine returns both:
        - sklearn's native decision_function
        - an inverted anomaly_score

    Semantics:
        higher anomaly_score = more anomalous
        higher decision_function = more normal

    The anomaly score is NOT:
        - a probability
        - an identity attribution
        - proof of illicit activity
        - a classification label
    """

    def __init__(
        self,
        model_file: Path = MODEL_FILE,
        schema_file: Path = SCHEMA_FILE,
    ) -> None:
        self.model_file = Path(model_file)
        self.schema_file = Path(schema_file)

        self._validate_artifacts()

        self.package = joblib.load(
            self.model_file
        )

        with self.schema_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.schema = json.load(f)

        self.imputer = self.package["imputer"]
        self.detector = self.package["detector"]

        self.model_features = list(
            self.package["model_features"]
        )

        self.transformed_feature_names = list(
            self.package[
                "transformed_feature_names"
            ]
        )

        self._validate_loaded_artifacts()

    def _validate_artifacts(self) -> None:
        if not self.model_file.exists():
            raise FileNotFoundError(
                "Production anomaly model not found: "
                f"{self.model_file}"
            )

        if not self.schema_file.exists():
            raise FileNotFoundError(
                "Anomaly feature schema not found: "
                f"{self.schema_file}"
            )

    def _validate_loaded_artifacts(self) -> None:
        schema_features = list(
            self.schema.get(
                "model_features",
                [],
            )
        )

        if schema_features != self.model_features:
            raise ValueError(
                "Model feature list does not match "
                "feature_schema.json."
            )

        expected_original_count = int(
            self.schema.get(
                "original_feature_count",
                len(self.model_features),
            )
        )

        if (
            expected_original_count
            != len(self.model_features)
        ):
            raise ValueError(
                "Original feature count mismatch: "
                f"schema={expected_original_count}, "
                f"model={len(self.model_features)}"
            )

        expected_transformed_count = int(
            self.schema.get(
                "transformed_feature_count",
                len(
                    self.transformed_feature_names
                ),
            )
        )

        actual_transformed_count = len(
            self.transformed_feature_names
        )

        if (
            expected_transformed_count
            != actual_transformed_count
        ):
            raise ValueError(
                "Transformed feature count mismatch: "
                f"schema={expected_transformed_count}, "
                f"model={actual_transformed_count}"
            )

        actual_imputer_names = (
            self.imputer
            .get_feature_names_out(
                self.model_features
            )
            .tolist()
        )

        if (
            actual_imputer_names
            != self.transformed_feature_names
        ):
            raise ValueError(
                "Persisted transformed feature names "
                "do not match the loaded imputer."
            )

        detector_features = getattr(
            self.detector,
            "n_features_in_",
            None,
        )

        if (
            detector_features is not None
            and detector_features
            != actual_transformed_count
        ):
            raise ValueError(
                "Isolation Forest feature count mismatch: "
                f"detector={detector_features}, "
                f"transformed={actual_transformed_count}"
            )

    def _prepare_dataframe(
        self,
        features: (
            Mapping[str, Any]
            | pd.DataFrame
        ),
    ) -> pd.DataFrame:
        if isinstance(
            features,
            pd.DataFrame,
        ):
            df = features.copy()

        elif isinstance(
            features,
            Mapping,
        ):
            df = pd.DataFrame(
                [dict(features)]
            )

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
                "Missing required anomaly features: "
                f"{missing}"
            )

        X = df[
            self.model_features
        ].copy()

        for column in self.model_features:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            )

        values = X.to_numpy(
            dtype=float
        )

        # NaN is expected and handled by the persisted imputer.
        if np.isinf(values).any():
            raise ValueError(
                "Input contains infinite feature values."
            )

        return X

    def score(
        self,
        features: (
            Mapping[str, Any]
            | pd.DataFrame
        ),
    ) -> dict[str, Any]:
        """
        Score one or more transactions.

        Returns:
            {
                "scores": [
                    {
                        "decision_function": ...,
                        "anomaly_score": ...,
                        "anomaly_flag": ...
                    }
                ]
            }

        anomaly_flag:
            -1 = Isolation Forest considers the row anomalous
             1 = Isolation Forest considers the row normal
        """

        X = self._prepare_dataframe(
            features
        )

        X_transformed = (
            self.imputer.transform(X)
        )

        decision_function = (
            self.detector.decision_function(
                X_transformed
            )
        )

        predictions = (
            self.detector.predict(
                X_transformed
            )
        )

        # Invert sklearn's normality convention:
        # higher score = more anomalous.
        anomaly_scores = (
            -decision_function
        )

        results = []

        for native_score, anomaly_score, flag in zip(
            decision_function,
            anomaly_scores,
            predictions,
        ):
            results.append(
                {
                    "decision_function": float(
                        native_score
                    ),
                    "anomaly_score": float(
                        anomaly_score
                    ),
                    "anomaly_flag": int(flag),
                }
            )

        return {
            "scores": results,
            "model": (
                "production_isolation_forest"
            ),
            "feature_count": len(
                self.model_features
            ),
            "transformed_feature_count": len(
                self.transformed_feature_names
            ),
            "score_semantics": (
                "higher anomaly_score = "
                "more anomalous"
            ),
        }

    def score_one(
        self,
        features: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Convenience method for scoring exactly
        one transaction.
        """

        result = self.score(
            features
        )

        if len(result["scores"]) != 1:
            raise RuntimeError(
                "score_one expected exactly "
                "one result."
            )

        return result["scores"][0]


_ENGINE: ProductionAnomalyEngine | None = None


def get_engine() -> ProductionAnomalyEngine:
    """
    Return a process-level singleton anomaly engine.
    """

    global _ENGINE

    if _ENGINE is None:
        _ENGINE = ProductionAnomalyEngine()

    return _ENGINE


def score_transaction(
    features: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Module-level convenience function.
    """

    return get_engine().score_one(
        features
    )


def main() -> None:
    """
    Smoke test using one validation transaction.
    """

    validation_file = (
        ROOT
        / "data"
        / "evaluation"
        / "validation_temporal_safe.parquet"
    )

    if not validation_file.exists():
        raise FileNotFoundError(
            "Validation dataset not found: "
            f"{validation_file}"
        )

    print("=" * 72)
    print("M8.2 PRODUCTION ANOMALY INFERENCE")
    print("=" * 72)

    engine = ProductionAnomalyEngine()

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
            "Validation dataset is missing "
            f"model features: {missing}"
        )

    sample = (
        validation.iloc[0][
            engine.model_features
        ]
        .to_dict()
    )

    txid = validation.iloc[0]["txid"]

    result = engine.score_one(
        sample
    )

    print("\nSmoke-test transaction:")
    print(
        f"  TXID: {txid}"
    )

    print(
        f"  Decision function: "
        f"{result['decision_function']:.6f}"
    )

    print(
        f"  Anomaly score: "
        f"{result['anomaly_score']:.6f}"
    )

    print(
        f"  Anomaly flag: "
        f"{result['anomaly_flag']}"
    )

    if not np.isfinite(
        result["decision_function"]
    ):
        raise ValueError(
            "Decision function is not finite."
        )

    if not np.isfinite(
        result["anomaly_score"]
    ):
        raise ValueError(
            "Anomaly score is not finite."
        )

    if result["anomaly_score"] != (
        -result["decision_function"]
    ):
        raise ValueError(
            "Anomaly score transformation "
            "is inconsistent."
        )

    if result["anomaly_flag"] not in (
        -1,
        1,
    ):
        raise ValueError(
            "Unexpected Isolation Forest flag: "
            f"{result['anomaly_flag']}"
        )

    print("\nScore semantics:")
    print(
        "  Higher anomaly_score = "
        "more anomalous"
    )

    print(
        "  Native decision_function = "
        "higher means more normal"
    )

    print("\n" + "=" * 72)
    print("M8.2 COMPLETE")
    print("=" * 72)

    print(
        "Production anomaly inference engine "
        "loaded successfully."
    )


if __name__ == "__main__":
    main()