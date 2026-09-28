"""
Imported-case intelligence engine.

This module analyzes a normalized investigator import without modifying or
mixing with the canonical production benchmark artifacts.

Design:
- Uses the existing M11.4 evidence weights and risk bands.
- Produces a separate ranked queue for the imported case.
- Derives case-local behavioral, entity, and network evidence only from the
  imported records.
- ML and anomaly channels remain neutral unless their required production
  feature inputs are genuinely present. This avoids fabricating model scores.
- All evidence is explicitly marked as case-derived.

The module is intentionally independent of the API and dashboard so it can be
tested directly before integration.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from src.risk.m11_4_unified_risk_scoring import (
        AGREEMENT_BONUS,
        RISK_BANDS,
        WEIGHTS,
        calculate_scores,
    )
except ImportError:
    # Keep the module importable in lightweight test environments.
    WEIGHTS = {
        "ml": 0.15,
        "anomaly": 0.20,
        "behavioral": 0.30,
        "entity": 0.20,
        "network": 0.15,
    }
    AGREEMENT_BONUS = {
        0: 0.00,
        1: 0.00,
        2: 0.02,
        3: 0.04,
        4: 0.06,
        5: 0.08,
    }
    RISK_BANDS = [
        (80.0, "VERY_HIGH"),
        (60.0, "HIGH"),
        (40.0, "MODERATE"),
        (20.0, "GUARDED"),
        (0.0, "LOW"),
    ]
    calculate_scores = None


SUPPORTED_WALLET_FIELDS = ("input_wallets", "output_wallets")
NETWORK_FIELDS = (
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "geo_country",
    "asn",
)
FINANCIAL_FIELDS = ("input_amount", "output_amount", "fee")


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _wallets(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [text]
    if isinstance(value, (list, tuple, set)):
        return [_text(item) for item in value if _text(item)]
    return []


def _timestamp(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed
    except ValueError:
        return None


def _risk_band(score: float) -> str:
    for threshold, label in RISK_BANDS:
        if score >= threshold:
            return label
    return "LOW"


def _normalize_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for raw in records:
        record = dict(raw)
        txid = _text(record.get("txid"))
        if not txid:
            continue

        record["txid"] = txid
        record["input_wallets"] = _wallets(record.get("input_wallets"))
        record["output_wallets"] = _wallets(record.get("output_wallets"))

        for field in ("input_amount", "output_amount", "fee"):
            number = _optional_float(record.get(field))
            if number is not None:
                record[field] = number

        normalized.append(record)

    return normalized


def _wallet_frequency(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for wallet in set(
            record.get("input_wallets", [])
            + record.get("output_wallets", [])
        ):
            counts[wallet] += 1
    return counts


def _behavioral_evidence(record: dict[str, Any], wallet_counts: Counter[str]) -> dict[str, Any]:
    """
    Case-local approximation of the existing behavioral evidence concepts.

    The canonical M9 implementation requires upstream peeling/mixing artifacts
    that imported investigator files generally do not contain. Therefore this
    function does not claim to reproduce canonical M9 scores. It derives
    auditable case-local structural signals from the imported transaction.

    Peeling-like evidence:
      - repeated output wallets
      - value reduction from input to output

    Mixing-like evidence:
      - fan-in/fan-out
      - participant density
      - multiple input/output participants
    """
    inputs = record.get("input_wallets", [])
    outputs = record.get("output_wallets", [])

    input_count = len(inputs)
    output_count = len(outputs)
    participant_count = len(set(inputs + outputs))

    input_amount = _optional_float(record.get("input_amount"))
    output_amount = _optional_float(record.get("output_amount"))

    repeated_output_count = sum(
        1 for wallet in set(outputs) if wallet_counts.get(wallet, 0) > 1
    )

    continuation_signal = (
        min(1.0, repeated_output_count / max(1, output_count))
        if output_count
        else 0.0
    )

    if input_amount is not None and input_amount > 0 and output_amount is not None:
        reduction = max(0.0, min(1.0, (input_amount - output_amount) / input_amount))
    else:
        reduction = 0.0

    peeling_candidate = continuation_signal > 0.0

    # A simple input/output value reduction is not sufficient evidence of
    # peeling because ordinary transaction fees also create a reduction.
    # Imported-case peeling evidence therefore requires a structural
    # continuation signal from a repeated output wallet.
    peeling_score = (
        min(
            1.0,
            0.60 * reduction + 0.40 * continuation_signal,
        )
        if peeling_candidate
        else 0.0
    )

    fan_in = min(1.0, input_count / 5.0)
    fan_out = min(1.0, output_count / 5.0)
    density = (
        min(1.0, participant_count / 10.0)
        if participant_count
        else 0.0
    )

    if mixing_candidate := (
        input_count > 1
        or output_count > 1
        or participant_count >= 4
    ):
        mixing_score = min(
            1.0,
            0.40 * fan_in + 0.40 * fan_out + 0.20 * density,
        )
    else:
        mixing_score = 0.0

    # Reuse the structural peeling and mixing decisions above.

    # Existing M9 fusion uses equal peeling/mixing weights.
    behavioral_score = 0.50 * peeling_score + 0.50 * mixing_score

    return {
        "peeling_evidence_score": peeling_score,
        "mixing_evidence_score": mixing_score,
        "behavioral_signal_count": int(peeling_candidate)
        + int(mixing_candidate),
        "multiple_behavioral_signals": bool(
            peeling_candidate and mixing_candidate
        ),
        "behavioral_signal_agreement": bool(
            peeling_score > 0 and mixing_score > 0
        ),
        "behavioral_evidence_score": behavioral_score,
        "behavioral_source": "case_local_structural_analysis",
    }


def _entity_evidence(
    record: dict[str, Any],
    wallet_counts: Counter[str],
    wallet_transactions: dict[str, set[str]],
) -> dict[str, Any]:
    """
    Case-local wallet relationship evidence.

    This intentionally does not reuse canonical cosponsor labels because an
    imported wallet has no canonical entity evidence unless it is explicitly
    present in the imported case.
    """
    inputs = record.get("input_wallets", [])
    outputs = record.get("output_wallets", [])
    wallets = list(dict.fromkeys(inputs + outputs))

    repeated_wallets = [
        wallet for wallet in wallets if wallet_counts.get(wallet, 0) > 1
    ]

    shared_transaction_counts = [
        len(wallet_transactions.get(wallet, set()))
        for wallet in repeated_wallets
    ]

    repeated_ratio = (
        len(repeated_wallets) / len(wallets)
        if wallets
        else 0.0
    )

    recurrence_signal = min(1.0, repeated_ratio)

    max_shared = max(shared_transaction_counts, default=0)
    shared_signal = min(1.0, max_shared / 5.0)

    entity_score = min(
        1.0,
        0.70 * recurrence_signal + 0.30 * shared_signal,
    )

    return {
        "entity_signal": entity_score,
        "entity_repeated_wallet_count": len(repeated_wallets),
        "entity_repeated_wallet_ratio": repeated_ratio,
        "entity_max_case_transaction_count": max_shared,
        "entity_source": "case_local_wallet_relationships",
    }


def _network_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """
    Derive network evidence without inventing a risk score.

    M10.5 defines transaction-level network evidence from
    ``temporal_observation_evidence`` on matched network observations.
    Ordinary IP/ASN/country fields are contextual features and must not be
    converted into a risk score merely because they are present.

    Therefore:
    - network context is reported whenever imported network fields exist;
    - the network risk channel is available only when an explicit
      temporal_observation_evidence value exists;
    - otherwise the network signal remains neutral at zero.
    """
    present = [field for field in NETWORK_FIELDS if _text(record.get(field))]

    temporal_value = _optional_float(
        record.get("temporal_observation_evidence")
    )

    if temporal_value is not None:
        network_score = max(0.0, min(1.0, temporal_value))
        evidence_available = True
        source = "imported_temporal_observation_evidence"
    else:
        network_score = 0.0
        evidence_available = False
        source = "network_context_only_no_temporal_evidence"

    return {
        "network_signal": network_score,
        "network_fields_present": present,
        "network_context_available": bool(present),
        "network_evidence_available": evidence_available,
        "network_source": source,
    }


def _case_feature_availability(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "ml": False,
            "anomaly": False,
            "behavioral": False,
            "entity": False,
            "network": False,
        }

    behavioral = any(
        record.get("input_wallets") or record.get("output_wallets")
        for record in records
    )

    entity = behavioral

    network = any(
        _optional_float(record.get("temporal_observation_evidence"))
        is not None
        for record in records
    )

    # Do not claim model availability merely because raw records exist.
    ml = False
    anomaly = False

    return {
        "ml": ml,
        "anomaly": anomaly,
        "behavioral": behavioral,
        "entity": entity,
        "network": network,
    }


def _score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Apply the project's M11.4 fusion formula.

    The canonical implementation uses five normalized [0,1] evidence
    channels, weighted sum, evidence-channel agreement bonus, and a 0-100
    risk score.
    """
    if not rows:
        return []

    result: list[dict[str, Any]] = []

    for row in rows:
        signals = {
            channel: max(
                0.0,
                min(1.0, _finite_float(row.get(f"{channel}_signal")))
            )
            for channel in (
                "ml",
                "anomaly",
                "behavioral",
                "entity",
                "network",
            )
        }

        available = {
            channel: bool(
                row.get(f"{channel}_evidence_available", False)
            )
            for channel in signals
        }

        active_count = sum(available.values())

        weighted = sum(
            signals[channel] * WEIGHTS[channel]
            for channel in signals
        )

        bonus = AGREEMENT_BONUS.get(
            active_count,
            AGREEMENT_BONUS[max(AGREEMENT_BONUS)],
        )

        score = max(
            0.0,
            min(100.0, weighted * (1.0 + bonus) * 100.0),
        )

        output = dict(row)
        output.update(
            {
                "ml_signal": signals["ml"],
                "anomaly_signal": signals["anomaly"],
                "behavioral_signal": signals["behavioral"],
                "entity_signal": signals["entity"],
                "network_signal": signals["network"],
                "ml_evidence_available": available["ml"],
                "anomaly_evidence_available": available["anomaly"],
                "behavioral_evidence_available": available["behavioral"],
                "entity_evidence_available": available["entity"],
                "network_evidence_available": available["network"],
                "effective_evidence_channel_count": active_count,
                "weighted_evidence_score": weighted,
                "evidence_agreement_bonus": bonus,
                "risk_score": score,
                "risk_level": _risk_band(score),
            }
        )
        result.append(output)

    return result


def _explanation(row: dict[str, Any]) -> str:
    contributing: list[str] = []

    for channel, label in (
        ("behavioral", "behavioral"),
        ("entity", "entity/wallet"),
        ("network", "network"),
        ("anomaly", "anomaly"),
        ("ml", "ML"),
    ):
        if row.get(f"{channel}_evidence_available"):
            value = _finite_float(row.get(f"{channel}_signal"))
            if value > 0:
                contributing.append(f"{label}={value:.3f}")

    if not contributing:
        return (
            "No positive evidence channels were available for this imported "
            "transaction; the case score remains neutral."
        )

    return (
        "Case-derived evidence: "
        + ", ".join(contributing)
        + ". Risk score uses the project's M11.4 weighted evidence fusion "
          "and agreement adjustment."
    )


def analyze_imported_case(
    records: Iterable[Mapping[str, Any]],
    import_id: str | None = None,
) -> dict[str, Any]:
    """
    Analyze one imported case and return a separate ranked queue.

    This function does not read or modify canonical benchmark artifacts.
    """
    normalized = _normalize_records(records)

    wallet_counts = _wallet_frequency(normalized)
    wallet_transactions: dict[str, set[str]] = defaultdict(set)

    for record in normalized:
        for wallet in set(
            record.get("input_wallets", [])
            + record.get("output_wallets", [])
        ):
            wallet_transactions[wallet].add(record["txid"])

    availability = _case_feature_availability(normalized)

    rows: list[dict[str, Any]] = []

    for record in normalized:
        behavioral = _behavioral_evidence(record, wallet_counts)
        entity = _entity_evidence(
            record,
            wallet_counts,
            wallet_transactions,
        )
        network = _network_evidence(record)

        row = {
            "txid": record["txid"],
            "ml_signal": 0.0,
            "anomaly_signal": 0.0,
            "behavioral_signal": behavioral["behavioral_evidence_score"],
            "entity_signal": entity["entity_signal"],
            "network_signal": network["network_signal"],
            "ml_evidence_available": availability["ml"],
            "anomaly_evidence_available": availability["anomaly"],
            "behavioral_evidence_available": (
                behavioral["behavioral_evidence_score"] > 0.0
            ),
            "entity_evidence_available": entity["entity_signal"] > 0.0,
            "network_evidence_available": availability["network"]
            and network["network_evidence_available"],
            "input_wallet_count": len(record.get("input_wallets", [])),
            "output_wallet_count": len(record.get("output_wallets", [])),
            "input_amount": _optional_float(record.get("input_amount")),
            "output_amount": _optional_float(record.get("output_amount")),
            "fee": _optional_float(record.get("fee")),
            "timestamp": record.get("timestamp"),
            "src_ip": record.get("src_ip"),
            "dst_ip": record.get("dst_ip"),
            "geo_country": record.get("geo_country"),
            "asn": record.get("asn"),
            **behavioral,
            **entity,
            **network,
        }

        # Preserve the explicit signal values after the evidence detail dicts.
        row["behavioral_signal"] = behavioral["behavioral_evidence_score"]
        row["entity_signal"] = entity["entity_signal"]
        row["network_signal"] = network["network_signal"]

        rows.append(row)

    scored = _score_rows(rows)

    for row in scored:
        row["explanation"] = _explanation(row)

    scored.sort(
        key=lambda row: (
            -_finite_float(row.get("risk_score")),
            -_finite_float(row.get("behavioral_signal")),
            row.get("txid", ""),
        )
    )

    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank

    band_counts = Counter(
        row["risk_level"] for row in scored
    )

    evidence_channel_counts = {
        channel: sum(
            bool(row.get(f"{channel}_evidence_available"))
            for row in scored
        )
        for channel in (
            "ml",
            "anomaly",
            "behavioral",
            "entity",
            "network",
        )
    }

    return {
        "status": "ok",
        "scope": "imported_case",
        "import_id": import_id,
        "records": {
            "analyzed": len(scored),
            "input_records": len(normalized),
        },
        "evidence_availability": availability,
        "evidence_channel_counts": evidence_channel_counts,
        "risk_summary": {
            "very_high": band_counts.get("VERY_HIGH", 0),
            "high": band_counts.get("HIGH", 0),
            "moderate": band_counts.get("MODERATE", 0),
            "guarded": band_counts.get("GUARDED", 0),
            "low": band_counts.get("LOW", 0),
        },
        "risk_methodology": {
            "type": "M11.4_weighted_evidence_fusion",
            "weights": dict(WEIGHTS),
            "agreement_bonus": dict(AGREEMENT_BONUS),
            "bands": [
                {"threshold": threshold, "label": label}
                for threshold, label in RISK_BANDS
            ],
            "canonical_queue_untouched": True,
            "case_specific_queue": True,
        },
        "ranked_transactions": scored,
    }


def analyze_import_artifact(
    normalized_records_path: str | Path,
    import_id: str | None = None,
) -> dict[str, Any]:
    """
    Load a persisted normalized-record JSON artifact and analyze it.
    """
    import json

    path = Path(normalized_records_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        records = payload.get("records", [])
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(
            "Normalized import artifact must contain a list of records "
            "or an object with a 'records' list."
        )

    return analyze_imported_case(
        records=records,
        import_id=import_id,
    )


__all__ = [
    "analyze_imported_case",
    "analyze_import_artifact",
]
