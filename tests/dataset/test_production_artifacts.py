from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_temporal_safe_features():
    path = ROOT / "data" / "derived" / "unified_features_temporal_safe.parquet"

    df = pd.read_parquet(path)

    assert len(df) == 203769
    assert df["txid"].nunique() == 203769
    assert df["time_step"].between(1, 49).all()


def test_ranked_alerts():
    path = ROOT / "data" / "derived" / "ranked_alerts.parquet"

    df = pd.read_parquet(path)

    assert len(df) == 203769
    assert df["txid"].nunique() == 203769
    assert df["risk_score"].between(0, 100).all()
    assert df["investigation_rank"].is_monotonic_increasing


def test_behavioral_outputs():
    peeling = pd.read_parquet(
        ROOT / "data" / "derived" / "peeling_chain_indicators.parquet"
    )

    mixing = pd.read_parquet(
      ROOT / "data" / "derived" / "mixing_pattern_indicators.parquet"
    )

    assert peeling["txid"].nunique() == len(peeling)
    assert mixing["txid"].nunique() == len(mixing)


def test_graph_artifacts():
    graph_dir = ROOT / "data" / "derived" / "investigation_graph"

    nodes = pd.read_parquet(
        graph_dir / "investigation_graph_nodes.parquet"
    )

    edges = pd.read_parquet(
        graph_dir / "investigation_graph_edges.parquet"
    )

    assert len(nodes) == 1026715
    assert len(edges) == 1379970

    assert nodes["node_id"].notna().all()
    assert edges["source"].notna().all()
    assert edges["target"].notna().all()


def test_shap_artifact():
    path = ROOT / "data" / "derived" / "shap_explanations.parquet"

    df = pd.read_parquet(path)

    assert len(df) == 1000
    assert df["txid"].nunique() == 1000