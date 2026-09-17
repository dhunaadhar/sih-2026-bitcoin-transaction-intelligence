from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def normalize_txids(series):
    return {
        str(value).strip().split(".")[0]
        for value in series.dropna()
    }


def test_risk_alert_feature_join():
    risk = pd.read_parquet(
        ROOT / "data" / "derived" / "unified_risk_scores.parquet"
    )

    alerts = pd.read_parquet(
        ROOT / "data" / "derived" / "ranked_alerts.parquet"
    )

    features = pd.read_parquet(
        ROOT / "data" / "derived" / "unified_features_temporal_safe.parquet"
    )

    assert risk["txid"].nunique() == len(risk)
    assert alerts["txid"].nunique() == len(alerts)
    assert features["txid"].nunique() == len(features)

    risk_txids = normalize_txids(risk["txid"])
    alert_txids = normalize_txids(alerts["txid"])
    feature_txids = normalize_txids(features["txid"])

    assert risk_txids == alert_txids
    assert risk_txids == feature_txids


def test_behavioral_to_risk_coverage():
    risk = pd.read_parquet(
        ROOT / "data" / "derived" / "unified_risk_scores.parquet"
    )

    peeling = pd.read_parquet(
        ROOT / "data" / "derived" / "peeling_chain_indicators.parquet"
    )

    mixing = pd.read_parquet(
        ROOT / "data" / "derived" / "mixing_pattern_indicators.parquet"
    )

    risk_txids = normalize_txids(risk["txid"])

    assert normalize_txids(peeling["txid"]).issubset(risk_txids)
    assert normalize_txids(mixing["txid"]).issubset(risk_txids)


def test_graph_to_alert_integration():
    nodes = pd.read_parquet(
        ROOT
        / "data"
        / "derived"
        / "investigation_graph"
        / "investigation_graph_nodes.parquet"
    )

    alerts = pd.read_parquet(
        ROOT / "data" / "derived" / "ranked_alerts.parquet"
    )

    transaction_nodes = nodes[
        nodes["node_type"].astype(str).str.lower()
        == "transaction"
    ]

    node_txids = {
        value.replace("tx:", "").strip().split(".")[0]
        for value in transaction_nodes["node_id"].astype(str)
    }

    alert_txids = normalize_txids(alerts["txid"])

    assert alert_txids.issubset(node_txids)


def test_shap_alert_integration():
    alerts = pd.read_parquet(
        ROOT / "data" / "derived" / "ranked_alerts.parquet"
    )

    shap = pd.read_parquet(
        ROOT / "data" / "derived" / "shap_explanations.parquet"
    )

    top_alert_txids = normalize_txids(
        alerts.head(1000)["txid"]
    )

    shap_txids = normalize_txids(shap["txid"])

    assert shap_txids == top_alert_txids