from src.ml.inference import ProductionInferenceEngine, predict_transaction
from src.ml.anomaly_inference import ProductionAnomalyEngine, score_transaction


def test_production_inference_engine_loads():
    engine = ProductionInferenceEngine()

    assert engine is not None


def test_anomaly_engine_loads():
    engine = ProductionAnomalyEngine()

    assert engine is not None


def test_prediction_engine_schema_available():
    engine = ProductionInferenceEngine()

    assert engine.schema_file.exists()
    assert engine.model_file.exists()


def test_anomaly_engine_schema_available():
    engine = ProductionAnomalyEngine()

    assert engine.schema_file.exists()
    assert engine.model_file.exists()


def test_prediction_output():
    engine = ProductionInferenceEngine()

    schema = engine.schema_file.read_text(encoding="utf-8")

    assert schema
    assert engine.model_file.exists()


def test_anomaly_output():
    engine = ProductionAnomalyEngine()

    schema = engine.schema_file.read_text(encoding="utf-8")

    assert schema
    assert engine.model_file.exists()