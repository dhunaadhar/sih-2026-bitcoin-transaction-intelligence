from src.graph.investigation_queries import InvestigationGraph


TXID = "126802668"
NODE_ID = f"tx:{TXID}"


def test_graph_loads():
    graph = InvestigationGraph()

    assert graph is not None
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0


def test_transaction_node_exists():
    graph = InvestigationGraph()

    assert graph.node_exists(NODE_ID)


def test_transaction_context():
    graph = InvestigationGraph()

    result = graph.transaction_context(TXID)

    assert result is not None
    assert isinstance(result, dict)


def test_transaction_node():
    graph = InvestigationGraph()

    result = graph.transaction_node(TXID)

    assert result is not None


def test_top_risk_transactions():
    graph = InvestigationGraph()

    result = graph.top_risk_transactions(5)

    assert result is not None
    assert isinstance(result, dict)
    assert result["query"] == "top_risk_transactions"
    assert result["top_n"] == 5
    assert isinstance(result["transactions"], list)
    assert 0 < len(result["transactions"]) <= 5


def test_neighborhood():
    graph = InvestigationGraph()

    result = graph.neighborhood(NODE_ID)

    assert result is not None
    assert isinstance(result, dict)


def test_shortest_path_same_node():
    graph = InvestigationGraph()

    result = graph.shortest_path(
        NODE_ID,
        NODE_ID,
    )

    assert result is not None
    assert isinstance(result, dict)


def test_graph_tables_loaded():
    graph = InvestigationGraph()

    assert graph.nodes is not None
    assert graph.edges is not None