from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health():
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["mode"] == "offline"


def test_summary():
    response = client.get("/api/summary")

    assert response.status_code == 200

    data = response.json()

    assert data["transactions"]["total"] == 203769
    assert data["transactions"]["unique_txids"] == 203769


def test_alerts():
    response = client.get(
        "/api/alerts?page=1&page_size=5"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 203769
    assert len(data["alerts"]) == 5


def test_top_risk():
    response = client.get("/api/top-risk?limit=5")

    assert response.status_code == 200

    data = response.json()

    assert data["limit"] == 5
    assert len(data["transactions"]) == 5
    assert data["transactions"][0]["investigation_rank"] == 1
    assert data["transactions"][0]["risk_score"] >= data["transactions"][1]["risk_score"]


def test_alert_detail():
    response = client.get(
        "/api/alerts/126802668"
    )

    assert response.status_code == 200

    data = response.json()

    assert str(data["txid"]) == "126802668"


def test_temporal():
    response = client.get(
        "/api/temporal?start=1&end=49"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["start_time_step"] == 1
    assert data["end_time_step"] == 49
    assert len(data["time_steps"]) == 49


def test_graph():
    response = client.get(
        "/api/graph/126802668"
    )

    assert response.status_code == 200

    data = response.json()

    assert "nodes" in data
    assert "edges" in data


def test_wallet_search():
    response = client.get(
        "/api/wallets/search?query=1"
    )

    assert response.status_code == 200