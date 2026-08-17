"""
Tests for Phase 5 impact feature extraction.

Known-value checks follow the specification examples:
- peak shift: baseline 2, peak 12 -> 10
- rise time: start 1.0 s, peak 1.2 s -> 0.2 s
- recovery time: peak 1.2 s, end 1.7 s -> 0.5 s
- residual shift: baseline 2, post-impact median 3 -> 1

Also covers negative/signed peak shift, insufficient post-impact
data (NaN + reason), normal event timing, NaN handling and the
absence of any division-by-zero behaviour.
"""

import numpy as np
import pytest

from src.analysis.phase5_features import (
    MIN_RESIDUAL_SAMPLES,
    RESIDUAL_START_GAP_SAMPLES,
    RESIDUAL_WINDOW_SAMPLES,
    compute_peak_shift,
    compute_recovery_time,
    compute_rise_time,
    estimate_post_impact_level,
    extract_features,
    extract_phase5_dataset,
)
from src.impact_detection.ensemble_event import ImpactEvent


def make_event(
    start_index=50,
    peak_index=70,
    end_index=90,
    start_time=1.0,
    peak_time=1.4,
    end_time=1.8,
    peak_value=12.0,
    event_id="test-FBG2-001",
    dataset="test",
):
    return ImpactEvent(
        start_index=start_index,
        peak_index=peak_index,
        end_index=end_index,
        start_time=start_time,
        peak_time=peak_time,
        end_time=end_time,
        peak_value=peak_value,
        duration=end_time - start_time,
        detection_methods=["peak"],
        event_id=event_id,
        dataset=dataset,
        channel="FBG2",
    )


# ------------------------------------------------------------------
# Peak shift
# ------------------------------------------------------------------

def test_peak_shift_basic():
    assert compute_peak_shift(12.0, 2.0) == pytest.approx(10.0)


def test_peak_shift_negative():
    shift = compute_peak_shift(-3.0, 2.0)
    assert shift == pytest.approx(-5.0)
    assert abs(shift) == pytest.approx(5.0)


def test_peak_shift_zero_peak():
    assert compute_peak_shift(2.0, 2.0) == pytest.approx(0.0)


# ------------------------------------------------------------------
# Rise time / recovery time
# ------------------------------------------------------------------

def test_rise_time_basic():
    assert compute_rise_time(1.0, 1.2) == pytest.approx(0.2)


def test_recovery_time_basic():
    assert compute_recovery_time(1.2, 1.7) == pytest.approx(0.5)


def test_rise_and_recovery_never_negative_for_valid_event():
    event = make_event()
    rise = compute_rise_time(event.start_time, event.peak_time)
    recovery = compute_recovery_time(event.peak_time, event.end_time)
    assert rise >= 0.0
    assert recovery >= 0.0
    assert rise == pytest.approx(0.4)
    assert recovery == pytest.approx(0.4)


# ------------------------------------------------------------------
# Residual shift
# ------------------------------------------------------------------

def test_residual_shift_basic():
    signal = np.full(200, 2.0)
    signal[50:90] = 12.0  # impact
    signal[95:150] = 3.0  # stable post-impact level

    event = make_event()

    features = extract_features(
        event,
        signal,
        np.arange(200) * 0.02,
        pre_impact_baseline=2.0,
    )

    assert features.post_impact_level == pytest.approx(3.0)
    assert features.residual_shift == pytest.approx(1.0)
    assert features.residual_reason == ""


def test_residual_shift_median_not_single_sample():
    signal = np.full(200, 2.0)
    signal[50:90] = 12.0
    signal[95:150] = 3.0
    signal[97] = 100.0  # one noisy spike must not drive the median

    features = extract_features(
        make_event(),
        signal,
        np.arange(200) * 0.02,
        pre_impact_baseline=2.0,
    )

    assert features.post_impact_level == pytest.approx(3.0)
    assert features.residual_shift == pytest.approx(1.0)


def test_residual_insufficient_samples_near_end():
    signal = np.full(200, 2.0)

    event = make_event(
        start_index=150,
        peak_index=170,
        end_index=198,
    )

    features = extract_features(
        event,
        signal,
        np.arange(200) * 0.02,
        pre_impact_baseline=2.0,
    )

    assert np.isnan(features.post_impact_level)
    assert np.isnan(features.residual_shift)
    assert features.residual_n_samples < MIN_RESIDUAL_SAMPLES
    assert features.residual_reason.startswith(
        "insufficient_post_impact_samples"
    )


def test_residual_window_skips_next_impact():
    signal = np.full(300, 2.0)
    signal[50:90] = 12.0   # first impact
    signal[97:250] = 12.0  # second impact starts right after the first

    first = make_event(
        start_index=50,
        peak_index=70,
        end_index=90,
        event_id="test-FBG2-001",
    )
    second = make_event(
        start_index=97,
        peak_index=120,
        end_index=250,
        event_id="test-FBG2-002",
    )

    features = extract_features(
        first,
        signal,
        np.arange(300) * 0.02,
        pre_impact_baseline=2.0,
        other_events=[first, second],
    )

    # The window starts at end(90)+5=95 and the next impact occupies
    # everything from 97 on, so only 2 valid samples remain -> NaN.
    assert features.residual_n_samples == 2
    assert np.isnan(features.post_impact_level)
    assert features.residual_reason.startswith(
        "insufficient_post_impact_samples"
    )


def test_estimate_post_impact_level_excludes_regions():
    signal = np.full(200, 2.0)
    signal[50:90] = 12.0
    signal[95:150] = 3.0

    level, n, reason = estimate_post_impact_level(
        signal,
        end_index=90,
        excluded_regions=[(95, 150)],
    )

    assert np.isnan(level)
    assert reason.startswith("insufficient_post_impact_samples")


# ------------------------------------------------------------------
# Full extraction
# ------------------------------------------------------------------

def test_extract_features_all_fields_consistent():
    signal = np.full(200, 2.0)
    signal[50:90] = 12.0
    signal[95:150] = 3.0
    time = np.arange(200) * 0.02

    event = make_event()
    features = extract_features(
        event,
        signal,
        time,
        pre_impact_baseline=2.0,
    )

    assert features.dataset == "test"
    assert features.fbg == "FBG2"
    assert features.impact_id == "test-FBG2-001"
    assert features.start_time == pytest.approx(1.0)
    assert features.peak_time == pytest.approx(1.4)
    assert features.end_time == pytest.approx(1.8)
    assert features.peak_shift == pytest.approx(10.0)
    assert features.absolute_peak_shift == pytest.approx(10.0)
    assert features.rise_time == pytest.approx(0.4)
    assert features.recovery_time == pytest.approx(0.4)
    assert features.residual_shift == pytest.approx(1.0)


def test_extract_features_nan_does_not_affect_other_features():
    signal = np.full(100, 2.0)
    event = make_event(
        start_index=50,
        peak_index=70,
        end_index=98,
        event_id="test-FBG2-001",
    )

    features = extract_features(
        event,
        signal,
        np.arange(100) * 0.02,
        pre_impact_baseline=2.0,
    )

    assert np.isnan(features.residual_shift)
    assert features.peak_shift == pytest.approx(10.0)
    assert features.rise_time == pytest.approx(
        1.4 - 1.0,
        abs=1e-6,
    )
    assert features.recovery_time == pytest.approx(
        1.8 - 1.4,
        abs=1e-6,
    )


def test_no_division_by_zero_on_flat_signal():
    signal = np.full(200, 2.0)
    time = np.arange(200) * 0.02

    features = extract_features(
        make_event(),
        signal,
        time,
        pre_impact_baseline=2.0,
    )

    assert np.isfinite(features.peak_shift)
    assert np.isfinite(features.rise_time)
    assert np.isfinite(features.recovery_time)
    assert features.peak_shift == pytest.approx(10.0)


def test_extract_phase5_dataset_returns_one_record_per_event():
    signal = np.full(300, 2.0)
    signal[50:90] = 12.0
    signal[150:190] = 11.0
    signal[95:145] = 3.0
    signal[195:245] = 3.5
    time = np.arange(300) * 0.02

    first = make_event(
        start_index=50,
        peak_index=70,
        end_index=90,
        event_id="test-FBG2-001",
    )
    second = make_event(
        start_index=150,
        peak_index=170,
        end_index=190,
        event_id="test-FBG2-002",
    )

    dataset_result = {
        "accepted_events": [first, second],
        "baseline_mean": 2.0,
        "channel": "FBG2",
    }

    features = extract_phase5_dataset(
        dataset_result,
        signal,
        time,
    )

    assert len(features) == 2
    assert [f.impact_id for f in features] == [
        "test-FBG2-001",
        "test-FBG2-002",
    ]

    for feature in features:
        assert feature.absolute_peak_shift == pytest.approx(
            abs(feature.peak_shift)
        )
