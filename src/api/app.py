from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.graph.investigation_queries import (
    InvestigationGraph,
    normalize_txid,
)
from src.ingestion.investigator_import import (
    DATA_TYPES,
    SUPPORTED_FORMATS,
    import_investigator_data,
)


ROOT = Path(__file__).resolve().parents[2]

DERIVED_DIR = ROOT / "data" / "derived"

DASHBOARD_DIR = ROOT / "src" / "dashboard"

INVESTIGATOR_IMPORT_DIR = Path(os.environ.get("SIH_RUNTIME_DATA_DIR", str(DERIVED_DIR))) / "investigator_imports"

RANKED_ALERTS_PATH = (
    DERIVED_DIR / "ranked_alerts.parquet"
)

RISK_SCORES_PATH = (
    DERIVED_DIR / "unified_risk_scores.parquet"
)

TEMPORAL_GRAPH_PATH = (
    DERIVED_DIR
    / "temporal_transaction_entity_evidence.parquet"
)

SHAP_PATH = (
    DERIVED_DIR / "shap_explanations.parquet"
)

GRAPH_NODES_PATH = (
    DERIVED_DIR
    / "investigation_graph"
    / "investigation_graph_nodes.parquet"
)

GRAPH_EDGES_PATH = (
    DERIVED_DIR
    / "investigation_graph"
    / "investigation_graph_edges.parquet"
)


APP_TITLE = (
    "Bitcoin Transaction Intelligence Platform"
)

APP_VERSION = "0.1.0"

MAX_PAGE_SIZE = 100

DEFAULT_PAGE_SIZE = 25

MAX_GRAPH_NEIGHBORS = 500

MAX_WALLET_SEARCH_RESULTS = 50

MAX_IMPORT_SIZE_BYTES = (
    100 * 1024 * 1024
)


# ============================================================================
# API response contracts
# ============================================================================

class RootResponse(BaseModel):
    application: str
    version: str
    mode: str
    status: str
    dashboard: str
    api_docs: str


class HealthResponse(BaseModel):
    status: str
    mode: str
    version: str
    artifacts: dict[str, bool]


class TransactionSummary(BaseModel):
    total: int
    unique_txids: int


class RiskSummary(BaseModel):
    mean: float
    median: float
    min: float
    max: float


class AlertSummary(BaseModel):
    active: int
    high_and_very_high: int
    priority_distribution: dict[str, int]


class EvidenceSummary(BaseModel):
    mean_behavioral_signal: float
    mean_entity_signal: float
    mean_network_signal: float
    mean_anomaly_signal: float
    mean_evidence_channels: float


class TopQueueSummary(BaseModel):
    size: int
    top_1000_mean_risk: float


class SummaryResponse(BaseModel):
    status: str
    transactions: TransactionSummary
    risk: RiskSummary
    alerts: AlertSummary
    evidence: EvidenceSummary
    top_queue: TopQueueSummary


class AlertsResponse(BaseModel):
    page: int
    page_size: int
    total: int
    pages: int
    filters: dict[str, Any]
    alerts: list[dict[str, Any]]
    transactions: list[dict[str, Any]]


class TransactionInvestigationResponse(BaseModel):
    txid: str
    alert: dict[str, Any]
    risk: dict[str, Any]
    temporal_entity_evidence: list[dict[str, Any]]
    shap: list[dict[str, Any]]


class TemporalStep(BaseModel):
    time_step: int
    transaction_count: int
    mean_risk: float
    median_risk: float
    high_alert_count: int
    very_high_alert_count: int
    active_alert_count: int
    top_queue_count: int


class TemporalResponse(BaseModel):
    start_time_step: int
    end_time_step: int
    time_steps: list[TemporalStep]


class GraphInvestigationResponse(BaseModel):
    txid: str
    transaction: dict[str, Any]
    connected_node_count: int
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    limitation: str


class WalletInvestigationResponse(BaseModel):
    address: str
    entity_type: str
    investigation: dict[str, Any]
    offline: bool
    limitation: str


class WalletSearchResponse(BaseModel):
    query: str
    total: int
    wallets: list[dict[str, Any]]


class TopRiskResponse(BaseModel):
    limit: int
    transactions: list[dict[str, Any]]


class ImportResponse(BaseModel):
    status: str
    filename: str
    data_type: str
    result: dict[str, Any]
    offline: bool


app = FastAPI(
    title=APP_TITLE,
    description=(
        "Offline investigation API for Bitcoin "
        "transaction traffic intelligence."
    ),
    version=APP_VERSION,
)


# ============================================================================
# Middleware
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================================
# Dashboard
# ============================================================================

if not DASHBOARD_DIR.exists():
    raise RuntimeError(
        f"Dashboard directory not found: {DASHBOARD_DIR}"
    )


app.mount(
    "/dashboard",
    StaticFiles(
        directory=DASHBOARD_DIR,
        html=True,
    ),
    name="dashboard",
)


# ============================================================================
# Lazy-loaded application state
# ============================================================================

_ranked_alerts: pd.DataFrame | None = None

_risk_scores: pd.DataFrame | None = None

_temporal_graph_features: pd.DataFrame | None = None

_shap_explanations: pd.DataFrame | None = None

_graph: InvestigationGraph | None = None


# ============================================================================
# Utilities
# ============================================================================

def _require_file(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(
            f"Required artifact not found: {path}"
        )


def _normalize_txid(value: Any) -> str:
    normalized = normalize_txid(value)

    if normalized is None:
        raise ValueError(
            "Invalid TXID."
        )

    return normalized


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (
            ValueError,
            TypeError,
        ):
            pass

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    try:
        if pd.isna(value):
            return None
    except (
        TypeError,
        ValueError,
    ):
        pass

    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _json_safe_value(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _json_safe_value(item)
            for item in value
        ]

    return value


def _json_safe_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        str(key): _json_safe_value(value)
        for key, value in record.items()
    }


# ============================================================================
# Artifact loaders
# ============================================================================

def _load_ranked_alerts() -> pd.DataFrame:
    global _ranked_alerts

    if _ranked_alerts is None:
        _require_file(
            RANKED_ALERTS_PATH
        )

        df = pd.read_parquet(
            RANKED_ALERTS_PATH
        ).copy()

        if "txid" not in df.columns:
            raise RuntimeError(
                "ranked_alerts.parquet does not contain txid."
            )

        df["txid_normalized"] = (
            df["txid"]
            .map(_normalize_txid)
        )

        _ranked_alerts = df

    return _ranked_alerts


def _load_risk_scores() -> pd.DataFrame:
    global _risk_scores

    if _risk_scores is None:
        _require_file(
            RISK_SCORES_PATH
        )

        df = pd.read_parquet(
            RISK_SCORES_PATH
        ).copy()

        if "txid" not in df.columns:
            raise RuntimeError(
                "unified_risk_scores.parquet "
                "does not contain txid."
            )

        df["txid_normalized"] = (
            df["txid"]
            .map(_normalize_txid)
        )

        _risk_scores = df

    return _risk_scores


def _load_temporal_graph_features() -> pd.DataFrame:
    global _temporal_graph_features

    if _temporal_graph_features is None:
        _require_file(
            TEMPORAL_GRAPH_PATH
        )

        df = pd.read_parquet(
            TEMPORAL_GRAPH_PATH
        ).copy()

        if "txid" not in df.columns:
            raise RuntimeError(
                "Temporal graph features do not "
                "contain txid."
            )

        df["txid_normalized"] = (
            df["txid"]
            .map(_normalize_txid)
        )

        _temporal_graph_features = df

    return _temporal_graph_features


def _load_shap() -> pd.DataFrame | None:
    global _shap_explanations

    if _shap_explanations is not None:
        return _shap_explanations

    if not SHAP_PATH.exists():
        return None

    df = pd.read_parquet(
        SHAP_PATH
    ).copy()

    if "txid" in df.columns:
        df["txid_normalized"] = (
            df["txid"]
            .map(_normalize_txid)
        )

    _shap_explanations = df

    return _shap_explanations


def _load_graph() -> InvestigationGraph:
    global _graph

    if _graph is None:
        _graph = InvestigationGraph()

    return _graph


# ============================================================================
# Lookup helpers
# ============================================================================

def _find_alert(
    txid: str,
) -> pd.Series:

    normalized = _normalize_txid(
        txid
    )

    alerts_df = _load_ranked_alerts()

    matches = alerts_df[
        alerts_df[
            "txid_normalized"
        ]
        == normalized
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Transaction not found: "
                f"{normalized}"
            ),
        )

    return matches.iloc[0]


def _find_risk(
    txid: str,
) -> pd.Series:

    normalized = _normalize_txid(
        txid
    )

    risk_df = _load_risk_scores()

    matches = risk_df[
        risk_df[
            "txid_normalized"
        ]
        == normalized
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Risk record not found: "
                f"{normalized}"
            ),
        )

    return matches.iloc[0]


# ============================================================================
# Root
# ============================================================================

@app.get(
    "/",
    response_model=RootResponse,
    tags=["system"],
)
def root() -> dict[str, Any]:

    return {
        "application": APP_TITLE,
        "version": APP_VERSION,
        "mode": "offline",
        "status": "running",
        "dashboard": "/dashboard/",
        "api_docs": "/docs",
    }


# ============================================================================
# Health
# ============================================================================

@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["system"],
)
def health() -> dict[str, Any]:

    artifacts = {
        "ranked_alerts": RANKED_ALERTS_PATH,
        "risk_scores": RISK_SCORES_PATH,
        "temporal_graph_features": (
            TEMPORAL_GRAPH_PATH
        ),
        "shap_explanations": SHAP_PATH,
        "investigation_graph_nodes": (
            GRAPH_NODES_PATH
        ),
        "investigation_graph_edges": (
            GRAPH_EDGES_PATH
        ),
        "dashboard": (
            DASHBOARD_DIR / "index.html"
        ),
        "dashboard_styles": (
            DASHBOARD_DIR / "styles.css"
        ),
        "dashboard_javascript": (
            DASHBOARD_DIR / "app.js"
        ),
        "investigator_import_engine": (
            ROOT
            / "src"
            / "ingestion"
            / "investigator_import.py"
        ),
    }

    availability = {
        name: path.exists()
        for name, path in artifacts.items()
    }

    required = [
        "ranked_alerts",
        "risk_scores",
        "temporal_graph_features",
        "dashboard",
        "dashboard_styles",
        "dashboard_javascript",
        "investigator_import_engine",
    ]

    healthy = all(
        availability[name]
        for name in required
    )

    return {
        "status": (
            "healthy"
            if healthy
            else "degraded"
        ),
        "mode": "offline",
        "version": APP_VERSION,
        "artifacts": availability,
    }


# ============================================================================
# Summary
# ============================================================================

@app.get(
    "/api/summary",
    response_model=SummaryResponse,
    tags=["dashboard"],
)
def summary() -> dict[str, Any]:

    alerts_df = _load_ranked_alerts()

    risk_df = _load_risk_scores()

    priority_counts = {
        str(key): int(value)
        for key, value in (
            alerts_df[
                "alert_priority"
            ]
            .value_counts()
            .to_dict()
            .items()
        )
    }

    active_mask = (
        alerts_df[
            "alert_status"
        ]
        .astype(str)
        .str.upper()
        .eq("ACTIVE")
    )

    high_mask = alerts_df[
        "alert_priority"
    ].isin(
        [
            "HIGH",
            "VERY_HIGH",
        ]
    )

    return {
        "status": "ok",
        "transactions": {
            "total": int(
                len(alerts_df)
            ),
            "unique_txids": int(
                alerts_df[
                    "txid_normalized"
                ].nunique()
            ),
        },
        "risk": {
            "mean": float(
                risk_df[
                    "risk_score"
                ].mean()
            ),
            "median": float(
                risk_df[
                    "risk_score"
                ].median()
            ),
            "min": float(
                risk_df[
                    "risk_score"
                ].min()
            ),
            "max": float(
                risk_df[
                    "risk_score"
                ].max()
            ),
        },
        "alerts": {
            "active": int(
                active_mask.sum()
            ),
            "high_and_very_high": int(
                high_mask.sum()
            ),
            "priority_distribution": (
                priority_counts
            ),
        },
        "evidence": {
            "mean_behavioral_signal": float(
                alerts_df[
                    "behavioral_signal"
                ].mean()
            ),
            "mean_entity_signal": float(
                alerts_df[
                    "entity_signal"
                ].mean()
            ),
            "mean_network_signal": float(
                alerts_df[
                    "network_signal"
                ].mean()
            ),
            "mean_anomaly_signal": float(
                alerts_df[
                    "anomaly_signal"
                ].mean()
            ),
            "mean_evidence_channels": float(
                alerts_df[
                    "effective_evidence_channel_count"
                ].mean()
            ),
        },
        "top_queue": {
            "size": int(
                alerts_df[
                    "top_alert_queue"
                ]
                .astype(bool)
                .sum()
            ),
            "top_1000_mean_risk": float(
                alerts_df[
                    alerts_df[
                        "investigation_rank"
                    ]
                    <= 1000
                ][
                    "risk_score"
                ].mean()
            ),
        },
    }


# ============================================================================
# Ranked alerts
# ============================================================================

@app.get(
    "/api/alerts",
    response_model=AlertsResponse,
    tags=["alerts"],
)
def alerts(
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
    ),
    priority: str | None = Query(
        default=None
    ),
    alert_status: str | None = Query(
        default=None
    ),
    time_step: int | None = Query(
        default=None,
        ge=1,
        le=49,
    ),
    min_risk: float | None = Query(
        default=None,
        ge=0.0,
        le=100.0,
    ),
    max_risk: float | None = Query(
        default=None,
        ge=0.0,
        le=100.0,
    ),
) -> dict[str, Any]:

    if (
        min_risk is not None
        and max_risk is not None
        and min_risk > max_risk
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "min_risk cannot be greater "
                "than max_risk."
            ),
        )

    df = _load_ranked_alerts()

    filtered = df

    if priority:
        filtered = filtered[
            filtered[
                "alert_priority"
            ]
            .astype(str)
            .str.upper()
            == priority.upper()
        ]

    if alert_status:
        filtered = filtered[
            filtered[
                "alert_status"
            ]
            .astype(str)
            .str.upper()
            == alert_status.upper()
        ]

    if time_step is not None:
        filtered = filtered[
            filtered[
                "time_step"
            ]
            == time_step
        ]

    if min_risk is not None:
        filtered = filtered[
            filtered[
                "risk_score"
            ]
            >= min_risk
        ]

    if max_risk is not None:
        filtered = filtered[
            filtered[
                "risk_score"
            ]
            <= max_risk
        ]

    total = int(
        len(filtered)
    )

    pages = (
        math.ceil(
            total / page_size
        )
        if total
        else 0
    )

    start = (
        (page - 1)
        * page_size
    )

    end = (
        start
        + page_size
    )

    page_df = filtered.iloc[
        start:end
    ]

    records = []

    for _, row in page_df.iterrows():

        record = {
            "investigation_rank": row[
                "investigation_rank"
            ],
            "active_alert_rank": row[
                "active_alert_rank"
            ],
            "top_alert_queue": row[
                "top_alert_queue"
            ],
            "txid": row[
                "txid"
            ],
            "time_step": row[
                "time_step"
            ],
            "risk_score": row[
                "risk_score"
            ],
            "risk_level": row[
                "risk_level"
            ],
            "alert_priority": row[
                "alert_priority"
            ],
            "alert_status": row[
                "alert_status"
            ],
            "ml_predicted_class": row[
                "ml_predicted_class"
            ],
            "ml_confidence": row[
                "ml_confidence"
            ],
            "anomaly_signal": row[
                "anomaly_signal"
            ],
            "behavioral_signal": row[
                "behavioral_signal"
            ],
            "entity_signal": row[
                "entity_signal"
            ],
            "network_signal": row[
                "network_signal"
            ],
            "effective_evidence_channel_count": (
                row[
                    "effective_evidence_channel_count"
                ]
            ),
            "evidence_agreement": row[
                "evidence_agreement"
            ],
            "evidence_summary": row[
                "evidence_summary"
            ],
            "alert_explanation": row[
                "alert_explanation"
            ],
            "risk_explanation": row[
                "risk_explanation"
            ],
        }

        records.append(
            _json_safe_record(
                record
            )
        )

    response = {
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": pages,
        "filters": {
            "priority": priority,
            "alert_status": alert_status,
            "time_step": time_step,
            "min_risk": min_risk,
            "max_risk": max_risk,
        },
        "alerts": records,
        "transactions": records,
    }

    return response


# ============================================================================
# Transaction investigation
# ============================================================================

@app.get(
    "/api/alerts/{txid}",
    response_model=TransactionInvestigationResponse,
    tags=["investigation"],
)
def transaction_investigation(
    txid: str,
) -> dict[str, Any]:

    alert = _find_alert(
        txid
    )

    risk = _find_risk(
        txid
    )

    temporal_features = (
        _load_temporal_graph_features()
    )

    normalized = _normalize_txid(
        txid
    )

    temporal_match = (
        temporal_features[
            temporal_features[
                "txid_normalized"
            ]
            == normalized
        ]
    )

    result = {
        "txid": normalized,
        "alert": _json_safe_record(
            alert.drop(
                labels=[
                    "txid_normalized"
                ],
                errors="ignore",
            ).to_dict()
        ),
        "risk": _json_safe_record(
            risk.drop(
                labels=[
                    "txid_normalized"
                ],
                errors="ignore",
            ).to_dict()
        ),
        "temporal_entity_evidence": [],
        "shap": [],
    }

    if not temporal_match.empty:

        result[
            "temporal_entity_evidence"
        ] = [
            _json_safe_record(
                row.drop(
                    labels=[
                        "txid_normalized"
                    ],
                    errors="ignore",
                ).to_dict()
            )
            for _, row
            in temporal_match.iterrows()
        ]

    shap = _load_shap()

    if shap is not None:

        if (
            "txid_normalized"
            in shap.columns
        ):

            shap_match = shap[
                shap[
                    "txid_normalized"
                ]
                == normalized
            ]

        elif "txid" in shap.columns:

            shap_txids = (
                shap["txid"]
                .map(_normalize_txid)
            )

            shap_match = shap[
                shap_txids
                == normalized
            ]

        else:

            shap_match = pd.DataFrame()

        if not shap_match.empty:

            result["shap"] = [
                _json_safe_record(
                    row.to_dict()
                )
                for _, row
                in shap_match.iterrows()
            ]

    return result


# ============================================================================
# Temporal analytics
# ============================================================================

@app.get(
    "/api/temporal",
    response_model=TemporalResponse,
    tags=["analytics"],
)
def temporal(
    start_time_step: int = Query(
        default=1,
        ge=1,
        le=49,
    ),
    end_time_step: int = Query(
        default=49,
        ge=1,
        le=49,
    ),
) -> dict[str, Any]:

    if (
        start_time_step
        > end_time_step
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "start_time_step cannot be "
                "greater than end_time_step."
            ),
        )

    alerts_df = _load_ranked_alerts()

    filtered = alerts_df[
        (
            alerts_df[
                "time_step"
            ]
            >= start_time_step
        )
        & (
            alerts_df[
                "time_step"
            ]
            <= end_time_step
        )
    ]

    rows = []

    for step in range(
        start_time_step,
        end_time_step + 1,
    ):

        subset = filtered[
            filtered[
                "time_step"
            ]
            == step
        ]

        if subset.empty:

            rows.append(
                {
                    "time_step": step,
                    "transaction_count": 0,
                    "mean_risk": 0.0,
                    "median_risk": 0.0,
                    "high_alert_count": 0,
                    "very_high_alert_count": 0,
                    "active_alert_count": 0,
                    "top_queue_count": 0,
                }
            )

            continue

        rows.append(
            {
                "time_step": step,
                "transaction_count": int(
                    len(subset)
                ),
                "mean_risk": float(
                    subset[
                        "risk_score"
                    ].mean()
                ),
                "median_risk": float(
                    subset[
                        "risk_score"
                    ].median()
                ),
                "high_alert_count": int(
                    subset[
                        "alert_priority"
                    ]
                    .isin(
                        [
                            "HIGH",
                            "VERY_HIGH",
                        ]
                    )
                    .sum()
                ),
                "very_high_alert_count": int(
                    (
                        subset[
                            "alert_priority"
                        ]
                        == "VERY_HIGH"
                    ).sum()
                ),
                "active_alert_count": int(
                    (
                        subset[
                            "alert_status"
                        ]
                        .astype(str)
                        .str.upper()
                        == "ACTIVE"
                    ).sum()
                ),
                "top_queue_count": int(
                    (
                        subset[
                            "investigation_rank"
                        ]
                        <= 1000
                    ).sum()
                ),
            }
        )

    return {
        "start_time_step": start_time_step,
        "end_time_step": end_time_step,
        "time_steps": rows,
    }


# ============================================================================
# Graph investigation
# ============================================================================

@app.get(
    "/api/graph/{txid}",
    response_model=GraphInvestigationResponse,
    tags=["graph"],
)
def graph_investigation(
    txid: str,
) -> dict[str, Any]:

    normalized = _normalize_txid(
        txid
    )

    graph = _load_graph()

    try:

        context = (
            graph.transaction_context(
                normalized
            )
        )

    except KeyError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    neighborhood = (
        graph.neighborhood(
            f"tx:{normalized}"
        )
    )

    nodes = [
        _json_safe_record(
            node
        )
        for node in neighborhood[
            "neighbors"
        ]
    ]

    edges = [
        _json_safe_record(
            edge
        )
        for edge in neighborhood[
            "relationships"
        ]
    ]

    nodes = nodes[
        :MAX_GRAPH_NEIGHBORS
    ]

    edges = edges[
        :MAX_GRAPH_NEIGHBORS
    ]

    return {
        "txid": normalized,
        "transaction": _json_safe_record(
            context[
                "transaction"
            ]
        ),
        "connected_node_count": int(
            context[
                "connected_node_count"
            ]
        ),
        "nodes": nodes,
        "edges": edges,
        "limitation": (
            "Graph relationships represent "
            "observable structural associations. "
            "They do not establish real-world "
            "identity or ownership."
        ),
    }


# ============================================================================
# Wallet / address investigation
# ============================================================================

@app.get(
    "/api/wallet/{address}",
    response_model=WalletInvestigationResponse,
    tags=["wallet"],
)
def wallet_investigation(
    address: str,
) -> dict[str, Any]:

    normalized_address = str(
        address
    ).strip()

    if not normalized_address:

        raise HTTPException(
            status_code=400,
            detail=(
                "Wallet/address cannot be empty."
            ),
        )

    graph = _load_graph()

    try:

        context = (
            graph.wallet_context(
                normalized_address
            )
        )

    except KeyError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return _json_safe_value(
        {
            "address": normalized_address,
            "entity_type": "wallet_address",
            "investigation": context,
            "offline": True,
            "limitation": (
                "A Bitcoin address is a pseudonymous "
                "identifier. Transaction history, "
                "graph relationships and structural "
                "clustering do not establish "
                "real-world identity or ownership."
            ),
        }
    )


# ============================================================================
# Wallet / address search
# ============================================================================

@app.get(
    "/api/wallets/search",
    response_model=WalletSearchResponse,
    tags=["wallet"],
)
def wallet_search(
    q: str = Query(
        default="",
        max_length=128,
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=MAX_WALLET_SEARCH_RESULTS,
    ),
) -> dict[str, Any]:

    graph = _load_graph()

    wallets = graph.nodes[
        graph.nodes[
            "node_type"
        ]
        == "wallet"
    ]

    if wallets.empty:

        return {
            "query": q,
            "total": 0,
            "wallets": [],
        }

    result = wallets

    query = str(
        q
    ).strip()

    if query:

        if "address" not in result.columns:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Wallet graph nodes do not "
                    "contain an address field."
                ),
            )

        result = result[
            result[
                "address"
            ]
            .astype(str)
            .str.contains(
                query,
                case=False,
                regex=False,
                na=False,
            )
        ]

    result = result.head(
        limit
    )

    records = []

    for _, row in result.iterrows():

        record = {
            "node_id": row.get(
                "node_id"
            ),
            "node_type": row.get(
                "node_type"
            ),
            "node_role": row.get(
                "node_role"
            ),
            "address": row.get(
                "address"
            ),
        }

        records.append(
            _json_safe_record(
                record
            )
        )

    return {
        "query": query,
        "total": int(
            len(result)
        ),
        "wallets": records,
    }


# ============================================================================
# Top-risk transactions
# ============================================================================

@app.get(
    "/api/top-risk",
    response_model=TopRiskResponse,
    tags=["alerts"],
)
def top_risk(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
) -> dict[str, Any]:

    alerts_df = _load_ranked_alerts()

    top = (
        alerts_df
        .sort_values(
            [
                "risk_score",
                "investigation_rank",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(
            limit
        )
    )

    records = []

    for _, row in top.iterrows():

        records.append(
            _json_safe_record(
                {
                    "investigation_rank": row[
                        "investigation_rank"
                    ],
                    "txid": row[
                        "txid"
                    ],
                    "time_step": row[
                        "time_step"
                    ],
                    "risk_score": row[
                        "risk_score"
                    ],
                    "risk_level": row[
                        "risk_level"
                    ],
                    "alert_priority": row[
                        "alert_priority"
                    ],
                    "alert_status": row[
                        "alert_status"
                    ],
                    "ml_predicted_class": row[
                        "ml_predicted_class"
                    ],
                    "ml_confidence": row[
                        "ml_confidence"
                    ],
                    "behavioral_signal": row[
                        "behavioral_signal"
                    ],
                    "entity_signal": row[
                        "entity_signal"
                    ],
                    "network_signal": row[
                        "network_signal"
                    ],
                    "anomaly_signal": row[
                        "anomaly_signal"
                    ],
                    "effective_evidence_channel_count": row[
                        "effective_evidence_channel_count"
                    ],
                }
            )
        )

    return {
        "limit": limit,
        "transactions": records,
    }


# ============================================================================
# Investigator import
# ============================================================================

@app.post(
    "/api/import",
    response_model=ImportResponse,
    tags=["import"],
)
async def investigator_import(
    data_type: str = Query(
        ...
    ),
    file: UploadFile = File(
        ...
    ),
) -> dict[str, Any]:

    normalized_type = (
        str(data_type)
        .strip()
        .upper()
    )

    if normalized_type not in DATA_TYPES:

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Unsupported investigator "
                    "data type."
                ),
                "supported_data_types": sorted(
                    DATA_TYPES
                ),
            },
        )

    filename = Path(
        file.filename
        or "uploaded_file"
    ).name

    suffix = Path(
        filename
    ).suffix.lower()

    if suffix not in SUPPORTED_FORMATS:

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Unsupported file format."
                ),
                "filename": filename,
                "supported_formats": sorted(
                    item.lstrip(".").upper()
                    for item in SUPPORTED_FORMATS
                ),
            },
        )

    content = bytearray()

    while True:

        chunk = await file.read(
            1024 * 1024
        )

        if not chunk:
            break

        content.extend(
            chunk
        )

        if (
            len(content)
            > MAX_IMPORT_SIZE_BYTES
        ):

            raise HTTPException(
                status_code=413,
                detail=(
                    "Uploaded file exceeds "
                    "the maximum allowed size."
                ),
            )

    if not content:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file is empty."
            ),
        )

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="sih_investigator_",
            delete=False,
        ) as temporary:

            temporary.write(
                content
            )

            temporary_path = Path(
                temporary.name
            )

        result = import_investigator_data(
            path=temporary_path,
            data_type=normalized_type,
        )

        import_id = result[
            "import_id"
        ]

        output_directory = (
            INVESTIGATOR_IMPORT_DIR
            / import_id
        )

        result = import_investigator_data(
            path=temporary_path,
            data_type=normalized_type,
            output_directory=output_directory,
        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Investigator import failed."
                ),
                "error": str(exc),
            },
        ) from exc

    finally:

        if temporary_path is not None:

            try:

                temporary_path.unlink(
                    missing_ok=True
                )

            except OSError:
                pass

    return _json_safe_value(
        {
            "status": "ok",
            "filename": filename,
            "data_type": normalized_type,
            "result": result,
            "offline": True,
        }
    )


# ============================================================================
# Startup validation
# ============================================================================

@app.on_event(
    "startup"
)
def startup_validation() -> None:

    required = [
        RANKED_ALERTS_PATH,
        RISK_SCORES_PATH,
        TEMPORAL_GRAPH_PATH,
        DASHBOARD_DIR / "index.html",
        DASHBOARD_DIR / "styles.css",
        DASHBOARD_DIR / "app.js",
        (
            ROOT
            / "src"
            / "ingestion"
            / "investigator_import.py"
        ),
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:

        raise RuntimeError(
            "Missing required application artifacts: "
            + ", ".join(missing)
        )