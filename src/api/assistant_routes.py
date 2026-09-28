from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.analysis.imported_case_intelligence import analyze_import_artifact


ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = ROOT / "data" / "derived"

router = APIRouter(
    prefix="/api/assistant",
    tags=["offline assistant"],
)

IMPORT_ROOT = Path(
    __import__("os").environ.get(
        "SIH_RUNTIME_DATA_DIR",
        str(DERIVED_DIR),
    )
) / "investigator_imports"


class AssistantQuery(BaseModel):
    question: str
    import_id: str | None = None
    txid: str | None = None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _find_records_path(import_id: str) -> Path:
    if not import_id or "/" in import_id or "\\" in import_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid import ID.",
        )

    flat = (
        IMPORT_ROOT
        / f"{import_id}_normalized_records.json"
    )
    nested = (
        IMPORT_ROOT
        / import_id
        / f"{import_id}_normalized_records.json"
    )

    if flat.exists():
        return flat

    if nested.exists():
        return nested

    raise HTTPException(
        status_code=404,
        detail="Imported dataset not found.",
    )


def _load_intelligence(import_id: str) -> dict[str, Any]:
    path = _find_records_path(import_id)

    try:
        return analyze_import_artifact(
            normalized_records_path=path,
            import_id=import_id,
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Imported dataset artifact is invalid JSON.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Offline assistant analysis failed.",
                "error": str(exc),
            },
        ) from exc


def _transaction_rows(
    intelligence: dict[str, Any],
) -> list[dict[str, Any]]:
    for key in (
        "ranked_transactions",
        "transactions",
        "queue",
        "ranked_queue",
        "case_queue",
        "rows",
    ):
        value = intelligence.get(key)
        if isinstance(value, list):
            return [
                item
                for item in value
                if isinstance(item, dict)
            ]

    return []


def _find_transaction(
    rows: list[dict[str, Any]],
    txid: str,
) -> dict[str, Any] | None:
    normalized = _safe_text(txid).lower()

    for row in rows:
        for key in ("txid", "transaction_id", "id"):
            value = _safe_text(row.get(key))
            if value.lower() == normalized:
                return row

    return None


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _risk_summary(row: dict[str, Any]) -> str:
    score = _number(
        row.get("risk_score")
    )
    level = _safe_text(
        row.get("risk_level")
    )

    if score is None and not level:
        return "No risk score is available for this transaction."

    if score is None:
        return f"Risk level: {level}."

    if level:
        return (
            f"Risk score: {score:.2f}/100; "
            f"risk level: {level}."
        )

    return f"Risk score: {score:.2f}/100."


def _channel_summary(row: dict[str, Any]) -> str:
    channels = []

    for name, label in (
        ("ml_signal", "ML"),
        ("anomaly_signal", "anomaly"),
        ("behavioral_signal", "behavioral"),
        ("entity_signal", "entity"),
        ("network_signal", "network"),
    ):
        value = _number(row.get(name))
        if value is not None and value > 0:
            channels.append(
                f"{label} ({value:.3f})"
            )

    if not channels:
        return (
            "No positive evidence channels are available "
            "for this transaction."
        )

    return "Positive evidence channels: " + ", ".join(channels) + "."


def _behavior_summary(row: dict[str, Any]) -> str:
    parts = []

    peeling = _number(
        row.get("peeling_score")
    )
    mixing = _number(
        row.get("mixing_score")
    )

    if peeling is not None and peeling > 0:
        parts.append(
            f"peeling score {peeling:.3f}"
        )

    if mixing is not None and mixing > 0:
        parts.append(
            f"mixing score {mixing:.3f}"
        )

    if not parts:
        return (
            "No positive behavioral indicator is available "
            "for this transaction."
        )

    return "Behavioral indicators: " + ", ".join(parts) + "."


def _explanation(row: dict[str, Any]) -> str:
    for key in (
        "explanation",
        "risk_explanation",
        "alert_explanation",
        "evidence_summary",
    ):
        value = _safe_text(row.get(key))
        if value:
            return value

    return (
        "No additional transaction-level explanation "
        "is available in the imported case intelligence."
    )


def _case_summary(
    intelligence: dict[str, Any],
) -> str:
    scope = _safe_text(
        intelligence.get("scope")
    ) or "imported_case"

    analyzed = intelligence.get(
        "analyzed",
        intelligence.get("transaction_count"),
    )

    evidence_counts = intelligence.get(
        "evidence_channel_counts",
        {},
    )

    if not isinstance(evidence_counts, dict):
        evidence_counts = {}

    if analyzed is None:
        analyzed_text = "the imported case"
    else:
        analyzed_text = f"{analyzed} transactions"

    channels = [
        f"{name}={value}"
        for name, value in evidence_counts.items()
    ]

    channel_text = (
        ", ".join(channels)
        if channels
        else "evidence-channel counts unavailable"
    )

    return (
        f"Case scope: {scope}. "
        f"Analyzed {analyzed_text}. "
        f"Evidence channel counts: {channel_text}."
    )


def _select_intent(question: str) -> str:
    text = question.lower()

    if any(
        token in text
        for token in (
            "summarize",
            "summary",
            "overview",
            "case",
        )
    ):
        return "case"

    if any(
        token in text
        for token in (
            "behavior",
            "behavioral",
            "peeling",
            "mixing",
        )
    ):
        return "behavior"

    if any(
        token in text
        for token in (
            "evidence",
            "channel",
            "contribut",
            "signal",
        )
    ):
        return "evidence"

    if any(
        token in text
        for token in (
            "risk",
            "score",
            "high risk",
            "why",
        )
    ):
        return "risk"

    if any(
        token in text
        for token in (
            "entity",
            "wallet",
            "connected",
            "network",
        )
    ):
        return "context"

    return "general"


def _is_greeting(question: str) -> bool:
    text = question.strip().lower()
    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "hi assistant",
        "hello assistant",
        "hey assistant",
    }
    return text in greetings or text.startswith(("hello ", "hi ", "hey "))


def _next_step(intent: str, transaction: dict[str, Any] | None) -> str:
    if transaction is not None:
        if intent == "risk":
            return (
                "Next investigation step: examine the contributing evidence "
                "channels and behavioral indicators for this transaction."
            )
        if intent == "evidence":
            return (
                "Next investigation step: examine the transaction's risk "
                "explanation and available wallet or network context."
            )
        if intent == "behavior":
            return (
                "Next investigation step: examine the available evidence "
                "channels and transaction context."
            )
        if intent == "context":
            return (
                "Next investigation step: review the transaction's risk "
                "and behavioral evidence."
            )
        return (
            "Next investigation step: ask for the transaction's risk, "
            "evidence, behavior, or wallet/network context."
        )

    if intent == "case":
        return (
            "Next investigation step: identify a transaction for "
            "transaction-level examination, or review the case evidence channels."
        )

    return (
        "Next investigation step: provide a transaction ID or ask for a "
        "case-level risk, evidence, behavioral, or contextual review."
    )


def _assistant_greeting(intelligence: dict[str, Any]) -> str:
    analyzed = intelligence.get("analyzed")
    if analyzed is None:
        analyzed = intelligence.get("records_analyzed")

    if analyzed is not None:
        scope_text = f"{analyzed} transactions are available in the current case."
    else:
        scope_text = "The imported case is available for investigation."

    return (
        "Good morning. I’m the Investigation Assistant. "
        "I’ll keep the analysis evidence-based and limited to the imported case data. "
        f"{scope_text}\n\n"
        "You can ask me to review the case, investigate a specific transaction, "
        "examine risk indicators, inspect behavioral evidence, or review "
        "wallet/entity/network context.\n\n"
        "What would you like to examine first?"
    )


def _answer(
    question: str,
    intelligence: dict[str, Any],
    transaction: dict[str, Any] | None,
) -> tuple[str, str]:
    intent = _select_intent(question)

    if _is_greeting(question):
        return _assistant_greeting(intelligence), "greeting"

    if transaction is not None:
        txid = _safe_text(
            transaction.get("txid")
            or transaction.get("transaction_id")
            or transaction.get("id")
        )

        if intent == "risk":
            answer = (
                f"Transaction {txid}: "
                f"{_risk_summary(transaction)} "
                f"{_explanation(transaction)}"
            )
            return answer + "\n\n" + _next_step(intent, transaction), intent

        if intent == "evidence":
            answer = (
                f"Transaction {txid}: "
                f"{_channel_summary(transaction)} "
                f"{_behavior_summary(transaction)}"
            )
            return answer + "\n\n" + _next_step(intent, transaction), intent

        if intent == "behavior":
            answer = (
                f"Transaction {txid}: "
                f"{_behavior_summary(transaction)}"
            )
            return answer + "\n\n" + _next_step(intent, transaction), intent

        if intent == "context":
            context_parts = []

            for key, label in (
                ("repeated_wallet_count", "repeated wallets"),
                ("participant_count", "participants"),
                ("network_signal", "network signal"),
            ):
                value = transaction.get(key)
                if value is not None:
                    context_parts.append(f"{label}: {value}")

            if not context_parts:
                answer = (
                    f"Transaction {txid}: "
                    "No additional wallet/entity/network context "
                    "is available in the imported case intelligence."
                )
            else:
                answer = (
                    f"Transaction {txid}: "
                    + "; ".join(context_parts)
                    + "."
                )

            return answer + "\n\n" + _next_step(intent, transaction), intent

        answer = (
            f"Transaction {txid}: "
            f"{_risk_summary(transaction)} "
            f"{_channel_summary(transaction)} "
            f"{_behavior_summary(transaction)} "
            f"{_explanation(transaction)}"
        )
        return answer + "\n\n" + _next_step(intent, transaction), intent

    if intent == "case":
        return _case_summary(intelligence) + "\n\n" + _next_step(intent, None), intent

    return (
        _case_summary(intelligence)
        + "\n\n"
        + _next_step(intent, None),
        intent,
    )


@router.post("/query")
def assistant_query(
    request: AssistantQuery,
) -> dict[str, Any]:
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if not request.import_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "import_id is required for the offline "
                "investigation assistant."
            ),
        )

    intelligence = _load_intelligence(
        request.import_id
    )

    rows = _transaction_rows(
        intelligence
    )

    transaction = None

    if request.txid:
        transaction = _find_transaction(
            rows,
            request.txid,
        )

        if transaction is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Transaction not found in imported "
                    "case intelligence."
                ),
            )

    answer, intent = _answer(
        question,
        intelligence,
        transaction,
    )

    return {
        "status": "ok",
        "mode": "offline",
        "assistant": {
            "type": "deterministic_nlp",
            "model": None,
            "local_only": True,
            "cloud_access": False,
        },
        "query": {
            "question": question,
            "intent": intent,
            "import_id": request.import_id,
            "txid": request.txid,
        },
        "answer": answer,
        "limitations": [
            "Responses are generated only from available "
            "imported case intelligence.",
            "The assistant does not infer real-world identity "
            "or ownership from pseudonymous blockchain data.",
            "A local generative model is not required for this "
            "baseline assistant.",
        ],
    }
