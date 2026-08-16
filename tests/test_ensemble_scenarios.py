"""
End-to-end scenario tests for the multi-method ensemble detector.

All tests use deterministic synthetic signals. These tests only
validate the software; they are not real experimental results.
"""

import numpy as np

from src.impact_detection.ensemble import (
    build_candidate_events,
    detect_events_channel,
)
from src.impact_detection.rejection_rules import apply_rejection_rules

from conftest import (
    DT,
    add_gaussian_impact,
    gaussian_noise,
    make_time,
    noise_only,
    one_impact,
)


def accepted_peaks(result):
    return [
        event.peak_index
        for event in result["accepted_events"]
    ]


def test_no_impact_plus_noise_produces_no_events():
    time, signal = noise_only(n_samples=2000, seed=7)

    result = detect_events_channel(
        signal,
        time,
        channel="FBG1",
        dataset="noise",
    )

    assert result["accepted_events"] == []


def test_one_impact_detected():
    time, signal = one_impact(
        n_samples=2000,
        seed=7,
        amplitude=0.05,
        peak_index=1000,
    )

    result = detect_events_channel(
        signal,
        time,
        channel="FBG1",
        dataset="one",
    )

    peaks = accepted_peaks(result)

    assert len(peaks) == 1
    # The peak must be near the true impact location (index 1000).
    assert abs(peaks[0] - 1000) <= 30

    event = result["accepted_events"][0]
    assert event.method_count >= 2
    assert event.accepted is True


def test_multiple_impacts_detected():
    time = make_time(3000)
    signal = gaussian_noise(3000, seed=3)

    for peak_index in (500, 1400, 2400):
        signal = add_gaussian_impact(signal, peak_index, 0.05)

    result = detect_events_channel(
        signal,
        time,
        channel="FBG1",
        dataset="multiple",
    )

    peaks = accepted_peaks(result)

    assert len(peaks) == 3

    for true_peak in (500, 1400, 2400):
        assert any(
            abs(peak - true_peak) <= 30
            for peak in peaks
        )


def test_nearby_impacts_remain_separate():
    time = make_time(3000)
    signal = gaussian_noise(3000, seed=5)

    signal = add_gaussian_impact(signal, 1000, 0.05)
    signal = add_gaussian_impact(signal, 1120, 0.05)

    result = detect_events_channel(
        signal,
        time,
        channel="FBG1",
        dataset="nearby",
    )

    peaks = accepted_peaks(result)

    assert len(peaks) == 2
    assert abs(peaks[0] - 1000) <= 30
    assert abs(peaks[1] - 1120) <= 30


def test_small_impact_detected():
    time, signal = one_impact(
        n_samples=2000,
        seed=5,
        noise_std=0.002,
        amplitude=0.012,
        peak_index=1500,
    )

    result = detect_events_channel(
        signal,
        time,
        channel="FBG1",
        dataset="small",
    )

    peaks = accepted_peaks(result)

    assert len(peaks) >= 1
    assert any(abs(peak - 1500) <= 30 for peak in peaks)


def test_high_amplitude_impact_detected_with_agreement():
    time, signal = one_impact(
        n_samples=2000,
        seed=5,
        amplitude=0.2,
        peak_index=1500,
    )

    result = detect_events_channel(
        signal,
        time,
        channel="FBG1",
        dataset="high",
    )

    peaks = accepted_peaks(result)

    assert len(peaks) == 1
    assert abs(peaks[0] - 1500) <= 30

    event = result["accepted_events"][0]
    # A strong impact should be supported by several detectors.
    assert event.method_count >= 2


def test_noisy_baseline_impact_detected():
    time, signal = one_impact(
        n_samples=2000,
        seed=5,
        noise_std=0.01,
        amplitude=0.08,
        peak_index=1500,
    )

    result = detect_events_channel(
        signal,
        time,
        channel="FBG1",
        dataset="noisy",
    )

    peaks = accepted_peaks(result)

    assert len(peaks) == 1
    assert abs(peaks[0] - 1500) <= 30


def test_detectors_same_impact_slightly_different_times_fuse_into_one():
    """
    peak 100-130, threshold 104-128, derivative 102-134 describe the
    same impact and must become a single fused event.
    """
    time, signal = one_impact(n_samples=2000, seed=7)

    all_regions = {
        "peak": [(100, 130)],
        "threshold": [(104, 128)],
        "derivative": [(102, 134)],
        "change_point": [],
    }

    candidates = build_candidate_events(
        all_regions,
        signal,
        time,
        channel="FBG1",
        dataset="fuse",
    )

    assert len(candidates) == 1

    event = candidates[0]

    assert set(event.detection_methods) == {
        "peak",
        "threshold",
        "derivative",
    }
    assert event.method_count == 3
    assert event.evidence_score > 0.7
    assert event.start_index <= event.peak_index <= event.end_index


def test_isolated_single_method_candidate_rejected():
    """
    A threshold-only region far from the impact must become a
    single-method candidate and be rejected, while the real impact
    (supported by several detectors) is accepted.
    """
    time, signal = one_impact(
        n_samples=2000,
        seed=7,
        amplitude=0.05,
        peak_index=1000,
    )

    baseline_std = np.std(signal[:100])

    all_regions = {
        "threshold": [(960, 1040), (1500, 1510)],
        "peak": [(985, 1015)],
        "derivative": [(970, 1030)],
        "change_point": [],
    }

    candidates = build_candidate_events(
        all_regions,
        signal,
        time,
        channel="FBG1",
        dataset="disagree",
    )

    assert len(candidates) == 2

    for candidate in candidates:
        accepted, reason = apply_rejection_rules(
            candidate,
            baseline_mean=0.0,
            baseline_std=baseline_std,
        )

        if candidate.peak_index > 1400:
            # The isolated threshold-only region must be rejected
            # (as noise-like or by insufficient agreement).
            assert accepted is False
            assert (
                "insufficient_method_agreement" in reason
                or "noise_like_event" in reason
            )
        else:
            # The real impact is accepted.
            assert accepted is True
            assert candidate.method_count == 3


def test_boundary_refinement_tightens_merged_extent():
    time, signal = one_impact(
        n_samples=2000,
        seed=7,
        amplitude=0.05,
        peak_index=1000,
    )

    # Force a deliberately wide merged region around the impact.
    all_regions = {
        "threshold": [(900, 1100)],
        "peak": [(990, 1010)],
        "derivative": [],
        "change_point": [],
    }

    candidates = build_candidate_events(
        all_regions,
        signal,
        time,
        channel="FBG1",
        dataset="refine",
    )

    assert len(candidates) == 1

    event = candidates[0]

    # Refined boundaries must be strictly inside the merged region.
    assert event.start_index >= 900
    assert event.end_index <= 1100
    assert event.end_index - event.start_index < 200
    assert event.start_index <= event.peak_index <= event.end_index
