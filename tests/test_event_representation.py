"""
Tests for the standardized ImpactEvent representation.
"""

import pytest

from src.impact_detection.ensemble_event import ImpactEvent

REQUIRED_FIELDS = [
    "event_id",
    "dataset",
    "channel",
    "start_index",
    "peak_index",
    "end_index",
    "start_time",
    "peak_time",
    "end_time",
    "duration",
    "peak_value",
    "detection_methods",
    "method_count",
    "evidence_score",
    "accepted",
    "rejection_reason",
]


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
        "event_id": "test-FBG1-001",
        "dataset": "test",
        "channel": "FBG1",
    }
    values.update(overrides)
    return ImpactEvent(**values)


def test_valid_event():
    event = make_event()
    assert event.start_index <= event.peak_index <= event.end_index
    assert event.start_time <= event.peak_time <= event.end_time
    assert event.method_count == 2


def test_start_after_peak_raises():
    with pytest.raises(ValueError):
        make_event(start_index=150, peak_index=100)


def test_peak_after_end_raises():
    with pytest.raises(ValueError):
        make_event(peak_index=160, end_index=150)


def test_evidence_score_out_of_range_raises():
    with pytest.raises(ValueError):
        make_event(evidence_score=1.5)


def test_method_count_matches_methods():
    event = make_event(detection_methods=["threshold", "peak", "derivative"])
    assert event.method_count == 3


def test_add_detection_method_deduplicates():
    event = make_event(detection_methods=["threshold"])
    event.add_detection_method("peak")
    event.add_detection_method("threshold")
    assert event.detection_methods == ["threshold", "peak"]
    assert event.method_count == 2


def test_set_evidence_validates_range():
    event = make_event()
    event.set_evidence(0.9)
    assert event.evidence_score == 0.9

    with pytest.raises(ValueError):
        event.set_evidence(-0.1)

    with pytest.raises(ValueError):
        event.set_evidence(1.1)


def test_confidence_score_backward_compat_alias():
    event = make_event(evidence_score=0.75)
    assert event.confidence_score == 0.75

    event.set_confidence(0.8)
    assert event.evidence_score == 0.8


def test_accept_and_reject():
    event = make_event()
    event.accept()
    assert event.accepted is True
    assert event.rejection_reason is None

    event.reject("some_reason")
    assert event.accepted is False
    assert event.rejection_reason == "some_reason"


def test_to_dict_contains_all_required_fields():
    event = make_event()
    event.accept()
    record = event.to_dict()

    for field in REQUIRED_FIELDS:
        assert field in record, f"missing field: {field}"

    assert record["method_count"] == 2
    assert record["evidence_score"] == 0.6
    assert record["accepted"] is True
