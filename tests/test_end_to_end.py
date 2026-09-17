from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import app


ROOT = Path(__file__).resolve().parents[1]

client = TestClient(app)


def test_end_to_end_health_to_summary():
    health = client.get("/api/health")
    assert health.status_code == 200

    health_data = health.json()
    assert health_data["status"] == "healthy"
    assert health_data["mode"] == "offline"

    summary = client.get("/api/summary")
    assert summary.status_code == 200

    summary_data = summary.json()
    assert summary_data["transactions"]["total"] == 203769
    assert summary_data["transactions"]["unique_txids"] == 203769


def test_end_to_end_alert_to_graph():
    top = client.get("/api/top-risk?limit=1")
    assert top.status_code == 200

    transactions = top.json()["transactions"]
    assert len(transactions) == 1

    txid = str(transactions[0]["txid"]).split(".")[0]

    alert = client.get(f"/api/alerts/{txid}")
    assert alert.status_code == 200

    alert_data = alert.json()
    assert str(alert_data["txid"]).split(".")[0] == txid

    graph = client.get(f"/api/graph/{txid}")
    assert graph.status_code == 200

    graph_data = graph.json()
    assert str(graph_data["txid"]).split(".")[0] == txid
    assert len(graph_data["nodes"]) > 0


def test_end_to_end_alert_to_shap():
    top = client.get("/api/top-risk?limit=1")
    assert top.status_code == 200

    txid = str(top.json()["transactions"][0]["txid"]).split(".")[0]

    alert = client.get(f"/api/alerts/{txid}")
    assert alert.status_code == 200

    alert_data = alert.json()

    assert "risk" in alert_data
    assert "temporal_entity_evidence" in alert_data
    assert "shap" in alert_data

    shap_data = alert_data["shap"]

    assert isinstance(shap_data, list)

    if shap_data:
        shap_txids = {
            str(item["txid"]).split(".")[0]
            for item in shap_data
            if "txid" in item
        }
        assert txid in shap_txids


def test_end_to_end_artifact_chain():
    required_artifacts = [
        ROOT / "data" / "derived" / "unified_features_temporal_safe.parquet",
        ROOT / "data" / "derived" / "unified_risk_scores.parquet",
        ROOT / "data" / "derived" / "ranked_alerts.parquet",
        ROOT / "data" / "derived" / "peeling_chain_indicators.parquet",
        ROOT / "data" / "derived" / "mixing_pattern_indicators.parquet",
        ROOT / "data" / "derived" / "shap_explanations.parquet",
        ROOT
        / "data"
        / "derived"
        / "investigation_graph"
        / "investigation_graph_nodes.parquet",
        ROOT
        / "data"
        / "derived"
        / "investigation_graph"
        / "investigation_graph_edges.parquet",
    ]

    for artifact in required_artifacts:
        assert artifact.exists()
        assert artifact.stat().st_size > 0