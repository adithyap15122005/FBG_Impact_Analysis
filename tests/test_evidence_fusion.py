"""
Tests for weighted evidence fusion.
"""

from src.impact_detection.evidence_fusion import (
    assign_evidence,
    compute_evidence_score,
    detector_weights,
)


def test_weights_normalize_to_one():
    weights = detector_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_weights_reflect_arguments():
    weights = detector_weights(
        peak_weight=0.5,
        threshold_weight=0.3,
        derivative_weight=0.15,
        change_point_weight=0.05,
    )

    assert abs(weights["peak"] - 0.5) < 1e-9
    assert abs(weights["change_point"] - 0.05) < 1e-9


def test_negative_weight_raises():
    import pytest

    with pytest.raises(ValueError):
        detector_weights(peak_weight=-0.1)


def test_zero_total_weight_raises():
    import pytest

    with pytest.raises(ValueError):
        detector_weights(
            peak_weight=0,
            threshold_weight=0,
            derivative_weight=0,
            change_point_weight=0,
        )


def test_evidence_score_sums_supporting_weights():
    weights = detector_weights()
    # peak + threshold + derivative = 0.30 + 0.30 + 0.25
    score = compute_evidence_score(
        ["peak", "threshold", "derivative"],
        weights,
    )
    assert abs(score - 0.85) < 1e-9


def test_evidence_score_clamps_to_one():
    weights = detector_weights()
    score = compute_evidence_score(
        ["peak", "threshold", "derivative", "change_point"],
        weights,
    )
    assert score == 1.0


def test_evidence_score_empty_methods_is_zero():
    weights = detector_weights()
    assert compute_evidence_score([], weights) == 0.0


def test_evidence_score_ignores_unknown_methods():
    weights = detector_weights()
    score = compute_evidence_score(["not_a_detector"], weights)
    assert score == 0.0


def test_assign_evidence_sets_score_on_event():
    from src.impact_detection.ensemble_event import ImpactEvent

    weights = detector_weights()

    event = ImpactEvent(
        start_index=0,
        peak_index=5,
        end_index=10,
        start_time=0.0,
        peak_time=0.1,
        end_time=0.2,
        peak_value=0.05,
        duration=0.2,
        detection_methods=["peak", "threshold"],
        channel="FBG1",
    )

    assign_evidence(event, weights)

    assert abs(event.evidence_score - 0.6) < 1e-9
