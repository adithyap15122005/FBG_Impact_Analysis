"""
Tests for temporal event matching (region overlap/merging).
"""

from src.impact_detection.event_matching import (
    calculate_overlap,
    match_candidate_regions,
    regions_overlap_or_adjacent,
    split_group_at_support_gaps,
)


def test_no_overlap_returns_zero():
    assert calculate_overlap((0, 10), (20, 30)) == 0.0


def test_full_overlap():
    # A inside B -> complete overlap relative to smaller region.
    assert calculate_overlap((10, 20), (5, 25)) == 1.0


def test_partial_overlap():
    # A=(10,20), B=(18,30): overlap samples 18,19,20 = 3 samples.
    # Smaller region A has length 11.
    assert abs(calculate_overlap((10, 20), (18, 30)) - 3.0 / 11.0) < 1e-9


def test_regions_adjacent_within_tolerance():
    # A ends at 10, B starts at 12: gap = 12 - 10 - 1 = 1 <= 5.
    assert regions_overlap_or_adjacent((0, 10), (12, 20)) is True


def test_regions_too_far_apart():
    # A ends at 10, B starts at 20: gap = 9 > 5.
    assert regions_overlap_or_adjacent((0, 10), (20, 30)) is False


def test_match_groups_overlapping_regions():
    regions = {
        "threshold": [(100, 130)],
        "peak": [(104, 128)],
        "derivative": [(102, 134)],
    }

    groups = match_candidate_regions(regions, tolerance_samples=5)

    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_match_keeps_separated_events_separate():
    regions = {
        "threshold": [(100, 130), (900, 930)],
        "peak": [(104, 128)],
        "derivative": [(902, 926)],
    }

    groups = match_candidate_regions(regions, tolerance_samples=5)

    # Two groups: one around index 100, one around index 900.
    assert len(groups) == 2


def test_split_group_at_support_gap_bridges():
    """
    A noisy single-detector region bridging two events must split
    the group back into two.
    """
    group = [
        ("threshold", (871, 927)),
        ("peak", (894, 914)),
        ("peak", (926, 1069)),
        ("threshold", (1072, 1127)),
        ("peak", (1080, 1111)),
        ("peak", (1133, 1188)),
    ]

    sub_groups = split_group_at_support_gaps(
        group,
        signal_length=2000,
        min_split_gap_samples=20,
    )

    assert len(sub_groups) == 2

    methods_in_first = {
        method for method, _ in sub_groups[0]
    }
    methods_in_second = {
        method for method, _ in sub_groups[1]
    }

    assert "threshold" in methods_in_first
    assert "threshold" in methods_in_second
    assert "peak" in methods_in_first
    assert "peak" in methods_in_second


def test_split_group_does_not_split_single_event():
    """
    A genuine single impact (one multi-support zone with long
    single-method tails) must not be split.
    """
    group = [
        ("threshold", (974, 1030)),
        ("peak", (987, 1007)),
    ]

    sub_groups = split_group_at_support_gaps(
        group,
        signal_length=2000,
        min_split_gap_samples=20,
    )

    assert len(sub_groups) == 1
