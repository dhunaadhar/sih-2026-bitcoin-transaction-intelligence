from src.detection.peeling_chain_detector import (
    calculate_evidence_score,
    is_valid_reduction,
)
from src.detection.mixing_pattern_detector import (
    calculate_balance_metrics,
    calculate_mixing_evidence,
)


def test_valid_reduction_range():
    assert is_valid_reduction(0.05)
    assert is_valid_reduction(0.50)
    assert is_valid_reduction(0.95)


def test_invalid_reduction_range():
    assert not is_valid_reduction(None)
    assert not is_valid_reduction(0.0)
    assert not is_valid_reduction(0.049)
    assert not is_valid_reduction(0.951)
    assert not is_valid_reduction(-0.1)


def test_peeling_evidence_score_is_bounded():
    score = calculate_evidence_score(
        direct_continuation=True,
        gradual_reduction=True,
        chain_valid=True,
        chain_length=5,
        temporal_span=4,
        continuation_address_count=3,
        chain_gaps=[1, 1, 2],
    )

    assert 0.0 <= score <= 1.0


def test_peeling_evidence_score_changes_with_evidence():
    weak = calculate_evidence_score(
        direct_continuation=False,
        gradual_reduction=False,
        chain_valid=False,
        chain_length=0,
        temporal_span=None,
        continuation_address_count=0,
        chain_gaps=[],
    )

    strong = calculate_evidence_score(
        direct_continuation=True,
        gradual_reduction=True,
        chain_valid=True,
        chain_length=6,
        temporal_span=5,
        continuation_address_count=4,
        chain_gaps=[1, 1, 1, 1],
    )

    assert strong > weak


def test_balance_metrics_zero_inputs():
    result = calculate_balance_metrics(0, 0)

    assert isinstance(result, dict)


def test_balance_metrics_normal_inputs():
    result = calculate_balance_metrics(5, 5)

    assert isinstance(result, dict)
    assert len(result) > 0


def test_mixing_evidence_normal_inputs():
    result = calculate_mixing_evidence(5, 5)

    assert isinstance(result, dict)
    assert len(result) > 0


def test_mixing_evidence_is_bounded():
    result = calculate_mixing_evidence(10, 10)

    numeric_values = [
        value
        for value in result.values()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]

    assert numeric_values
    assert all(value >= 0 for value in numeric_values)
