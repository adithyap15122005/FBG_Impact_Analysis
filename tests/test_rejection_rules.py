"""
Tests for the false-positive rejection rules.
"""

from src.impact_detection.ensemble_event import ImpactEvent
from src.impact_detection.rejection_rules import (
    DEFAULT_RULES,
    apply_rejection_rules,
)

BASELINE_MEAN = 0.0
BASELINE_STD = 0.002


def make_event(**overrides):
    values = {
        "start_index": 100,
        "peak_index": 120,
        "end_index": 150,
        "start_time": 2.0,
        "peak_time": 2.4,
        "end_time": 3.0,
        "peak_value": 0.05,
        "duration": 1.0,
        "detection_methods": ["threshold", "peak"],
        "evidence_score": 0.6,
        "channel": "FBG1",
    }
    values.update(overrides)
    return ImpactEvent(**values)


def test_valid_event_accepted():
    event = make_event()
    accepted, reason = apply_rejection_rules(
        event,
        BASELINE_MEAN,
        BASELINE_STD,
    )
    assert accepted is True
    assert reason is None


def test_duration_too_short_rejected():
    event = make_event(start_index=100, peak_index=101, end_index=101)
    accepted, reason = apply_rejection_rules(
        event,
        BASELINE_MEAN,
        BASELINE_STD,
    )
    assert accepted is False
    assert reason.startswith("duration_below_min_samples")


def test_low_amplitude_rejected():
    # peak deviation = 0.002 -> 1.0 std < 2.5 std.
    event = make_event(peak_value=0.002)
    accepted, reason = apply_rejection_rules(
        event,
        BASELINE_MEAN,
        BASELINE_STD,
    )
    assert accepted is False
    assert reason.startswith("low_amplitude")


def test_insufficient_agreement_rejected():
    event = make_event(
        detection_methods=["peak"],
        evidence_score=0.3,
    )
    accepted, reason = apply_rejection_rules(
        event,
        BASELINE_MEAN,
        BASELINE_STD,
    )
    assert accepted is False
    assert reason.startswith("insufficient_method_agreement")


def test_low_evidence_rejected():
    event = make_event(
        detection_methods=["peak", "threshold", "derivative"],
        evidence_score=0.10,
    )
    accepted, reason = apply_rejection_rules(
        event,
        BASELINE_MEAN,
        BASELINE_STD,
    )
    assert accepted is False
    assert reason.startswith("evidence_below_minimum")


def test_noise_like_event_rejected():
    # Single method, modest amplitude (1.0 std < 3.0 std).
    event = make_event(
        detection_methods=["peak"],
        evidence_score=0.3,
        peak_value=0.002,
    )
    accepted, reason = apply_rejection_rules(
        event,
        BASELINE_MEAN,
        BASELINE_STD,
    )
    assert accepted is False
    assert reason.startswith("noise_like_event")


def test_custom_rules_override_defaults():
    event = make_event(
        detection_methods=["peak"],
        evidence_score=0.3,
    )

    rules = dict(DEFAULT_RULES)
    rules["min_method_agreement"] = 1
    rules["noise_like_max_methods"] = 0

    accepted, reason = apply_rejection_rules(
        event,
        BASELINE_MEAN,
        BASELINE_STD,
        rules=rules,
    )
    assert accepted is True
    assert reason is None


def test_flat_baseline_does_not_reject_strong_peak():
    event = make_event(peak_value=0.05)
    accepted, reason = apply_rejection_rules(
        event,
        BASELINE_MEAN,
        baseline_std=0.0,
    )
    assert accepted is True
    assert reason is None
