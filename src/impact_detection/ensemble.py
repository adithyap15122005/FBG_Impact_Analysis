"""
Multi-method ensemble impact detection (Phase 4.5).

This module is the orchestration/fusion layer. It runs the four
independent detectors (peak, threshold, derivative, change-point),
matches their detections into candidate events, fuses the evidence,
refines the event boundaries, and applies false-positive rejection
rules.

It reuses the existing detectors rather than reimplementing them:

- peak_detection.detect_peak_events
- threshold_detection.detect_threshold_crossings
- derivative_detection.detect_derivative_events / apply_persistence
- change_point.detect_change_points
- impact_boundaries.clean_regions
- event_matching.match_candidate_regions / create_candidate_event_from_group
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import (
    BASELINE_SAMPLES,
    CHANGE_POINT_GAP_TOLERANCE,
    CHANGE_POINT_THRESHOLD,
    CHANGE_POINT_WINDOW,
    DERIVATIVE_GAP_TOLERANCE,
    DERIVATIVE_MULTIPLIER,
    DERIVATIVE_PERSISTENCE,
    GROUP_SPLIT_MIN_GAP_SAMPLES,
    MATCH_TOLERANCE_SAMPLES,
    MIN_IMPACT_SAMPLES,
    PEAK_GAP_TOLERANCE,
    PEAK_MIN_DISTANCE_SAMPLES,
    PEAK_PROMINENCE_MULTIPLIER,
    PEAK_REGION_AFTER,
    PEAK_REGION_BEFORE,
    REFINE_CONFIRMATION_SAMPLES,
    REFINE_NOISE_STD,
    REFINE_RECOVERY_RATIO,
    THRESHOLD_GAP_TOLERANCE,
    THRESHOLD_MULTIPLIER,
)

from .boundary_refinement import refine_event_boundaries
from .change_point import detect_change_points
from .derivative_detection import (
    apply_persistence,
    detect_derivative_events,
)
from .ensemble_event import ImpactEvent
from .event_matching import (
    create_candidate_event_from_group,
    match_candidate_regions,
    split_group_at_support_gaps,
)
from .evidence_fusion import (
    assign_evidence,
    compute_evidence_score,
    detector_weights,
)
from .impact_boundaries import clean_regions, regions_to_mask
from .peak_detection import detect_peak_events
from .rejection_rules import apply_rejection_rules
from .threshold_detection import detect_threshold_crossings


def default_detector_params() -> Dict[str, float]:
    """
    Default detector/region parameters (mirrors src/config.py).
    """
    return {
        "baseline_samples": BASELINE_SAMPLES,

        "peak_prominence_multiplier": PEAK_PROMINENCE_MULTIPLIER,
        "peak_min_distance_samples": PEAK_MIN_DISTANCE_SAMPLES,
        "peak_region_before": PEAK_REGION_BEFORE,
        "peak_region_after": PEAK_REGION_AFTER,

        "threshold_multiplier": THRESHOLD_MULTIPLIER,

        "derivative_multiplier": DERIVATIVE_MULTIPLIER,
        "derivative_persistence": DERIVATIVE_PERSISTENCE,

        "change_point_window": CHANGE_POINT_WINDOW,
        "change_point_threshold": CHANGE_POINT_THRESHOLD,

        "min_impact_samples": MIN_IMPACT_SAMPLES,
        "peak_gap_tolerance": PEAK_GAP_TOLERANCE,
        "threshold_gap_tolerance": THRESHOLD_GAP_TOLERANCE,
        "derivative_gap_tolerance": DERIVATIVE_GAP_TOLERANCE,
        "change_point_gap_tolerance": CHANGE_POINT_GAP_TOLERANCE,

        "match_tolerance_samples": MATCH_TOLERANCE_SAMPLES,
        "group_split_min_gap_samples": GROUP_SPLIT_MIN_GAP_SAMPLES,

        "refine_noise_std": REFINE_NOISE_STD,
        "refine_recovery_ratio": REFINE_RECOVERY_RATIO,
        "refine_confirmation_samples": REFINE_CONFIRMATION_SAMPLES,
    }


def default_weights() -> Dict[str, float]:
    """
    Default normalized detector weights (from src/config.py).
    """
    from ..config import (
        CHANGE_POINT_WEIGHT,
        DERIVATIVE_WEIGHT,
        PEAK_WEIGHT,
        THRESHOLD_WEIGHT,
    )

    return detector_weights(
        peak_weight=PEAK_WEIGHT,
        threshold_weight=THRESHOLD_WEIGHT,
        derivative_weight=DERIVATIVE_WEIGHT,
        change_point_weight=CHANGE_POINT_WEIGHT,
    )


# ------------------------------------------------------------
# Detector wrappers
# ------------------------------------------------------------

def _peak_regions(
    signal: np.ndarray,
    time: np.ndarray,
    params: Dict[str, float],
) -> List[Tuple[int, int]]:
    """Candidate regions from the independent peak detector."""
    result = detect_peak_events(
        signal,
        time,
        params["baseline_samples"],
        prominence_multiplier=params["peak_prominence_multiplier"],
        minimum_distance_samples=params["peak_min_distance_samples"],
    )

    signal_length = len(signal)
    candidate_regions = []

    for peak in result["peaks"]:
        peak_index = peak["index"]

        start = max(
            0,
            peak_index - params["peak_region_before"],
        )
        end = min(
            signal_length - 1,
            peak_index + params["peak_region_after"],
        )

        candidate_regions.append((start, end))

    if not candidate_regions:
        return []

    mask = regions_to_mask(candidate_regions, signal_length)

    return clean_regions(
        mask,
        minimum_samples=params["min_impact_samples"],
        gap_tolerance=params["peak_gap_tolerance"],
    )


def _threshold_regions(
    signal: np.ndarray,
    time: np.ndarray,
    params: Dict[str, float],
) -> List[Tuple[int, int]]:
    """Candidate regions from the adaptive threshold detector."""
    result = detect_threshold_crossings(
        signal,
        params["baseline_samples"],
        params["threshold_multiplier"],
    )

    return clean_regions(
        result["mask"],
        minimum_samples=params["min_impact_samples"],
        gap_tolerance=params["threshold_gap_tolerance"],
    )


def _derivative_regions(
    signal: np.ndarray,
    time: np.ndarray,
    params: Dict[str, float],
) -> List[Tuple[int, int]]:
    """Candidate regions from the derivative detector."""
    result = detect_derivative_events(
        signal,
        time,
        params["baseline_samples"],
        params["derivative_multiplier"],
    )

    persistent_mask = apply_persistence(
        result["mask"],
        params["derivative_persistence"],
    )

    return clean_regions(
        persistent_mask,
        minimum_samples=params["min_impact_samples"],
        gap_tolerance=params["derivative_gap_tolerance"],
    )


def _change_point_regions(
    signal: np.ndarray,
    time: np.ndarray,
    params: Dict[str, float],
) -> List[Tuple[int, int]]:
    """Candidate regions from the change-point detector."""
    mask = detect_change_points(
        signal,
        window=params["change_point_window"],
        threshold=params["change_point_threshold"],
    )

    return clean_regions(
        mask,
        minimum_samples=params["min_impact_samples"],
        gap_tolerance=params["change_point_gap_tolerance"],
    )


DETECTOR_FUNCTIONS = {
    "peak": _peak_regions,
    "threshold": _threshold_regions,
    "derivative": _derivative_regions,
    "change_point": _change_point_regions,
}


def run_all_detectors(
    signal: np.ndarray,
    time: np.ndarray,
    methods: Optional[List[str]] = None,
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Run the requested independent detectors and return their regions.

    Parameters
    ----------
    signal : array-like
        Filtered wavelength-shift signal.
    time : array-like
        Time values corresponding to the signal.
    methods : list of str, optional
        Detector names to run. Defaults to all four detectors.
    params : dict, optional
        Detector parameter overrides.

    Returns
    -------
    dict
        Mapping: detector name -> list of (start, end) regions.
    """
    if params is None:
        params = default_detector_params()

    if methods is None:
        methods = list(DETECTOR_FUNCTIONS.keys())

    signal = np.asarray(signal, dtype=float)
    time = np.asarray(time, dtype=float)

    regions: Dict[str, List[Tuple[int, int]]] = {}

    for method in methods:
        if method not in DETECTOR_FUNCTIONS:
            raise ValueError(f"Unknown detector: {method}")

        regions[method] = DETECTOR_FUNCTIONS[method](
            signal,
            time,
            params,
        )

    return regions


# ------------------------------------------------------------
# Candidate event construction
# ------------------------------------------------------------

def build_candidate_events(
    all_regions: Dict[str, List[Tuple[int, int]]],
    signal: np.ndarray,
    time: np.ndarray,
    channel: str,
    dataset: str = "",
    weights: Optional[Dict[str, float]] = None,
    match_tolerance_samples: int = MATCH_TOLERANCE_SAMPLES,
    group_split_min_gap_samples: int = GROUP_SPLIT_MIN_GAP_SAMPLES,
    baseline_mean: Optional[float] = None,
    baseline_std: Optional[float] = None,
    refine: bool = True,
    refine_params: Optional[Dict[str, float]] = None,
) -> List[ImpactEvent]:
    """
    Convert per-detector regions into fused candidate events.

    Steps
    -----
    1. Match regions across detectors (temporal matching).
    2. Create a candidate event per matched group.
    3. Compute the weighted evidence score.
    4. Optionally refine start/peak/end boundaries.

    Parameters
    ----------
    all_regions : dict
        Mapping: detector name -> list of (start, end) regions.
    signal : array-like
        Filtered signal.
    time : array-like
        Time values.
    channel : str
        Channel identifier.
    dataset : str
        Dataset name.
    weights : dict, optional
        Normalized detector weights.
    match_tolerance_samples : int
        Tolerance for temporal matching.
    baseline_mean / baseline_std : float, optional
        Baseline statistics. Computed from the first
        baseline_samples when omitted.
    refine : bool
        Whether to refine boundaries.
    refine_params : dict, optional
        Refinement parameters.

    Returns
    -------
    list of ImpactEvent
        Fused candidate events (evidence assigned, boundaries
        refined, not yet accepted/rejected).
    """
    if weights is None:
        weights = default_weights()

    if refine_params is None:
        full = default_detector_params()
        refine_params = {
            "noise_std_multiplier": full["refine_noise_std"],
            "recovery_ratio": full["refine_recovery_ratio"],
            "confirmation_samples": full["refine_confirmation_samples"],
        }

    signal = np.asarray(signal, dtype=float)
    time = np.asarray(time, dtype=float)

    if baseline_mean is None or baseline_std is None:
        baseline_samples = int(
            default_detector_params()["baseline_samples"]
        )
        baseline = signal[:baseline_samples]
        baseline_mean = float(np.mean(baseline))
        baseline_std = float(np.std(baseline))

    matched_groups = match_candidate_regions(
        all_regions,
        tolerance_samples=match_tolerance_samples,
    )

    # Prevent noisy single-detector regions from bridging two
    # distinct impacts into a single event.
    split_groups = []

    for group in matched_groups:
        split_groups.extend(
            split_group_at_support_gaps(
                group,
                len(signal),
                min_split_gap_samples=group_split_min_gap_samples,
            )
        )

    events: List[ImpactEvent] = []

    for group_index, group in enumerate(split_groups):
        event = create_candidate_event_from_group(
            group,
            signal,
            time,
            channel=channel,
        )

        event.dataset = dataset
        event.event_id = (
            f"{dataset}-{channel}-{group_index + 1:03d}"
        )

        assign_evidence(event, weights)

        event.diagnostics["baseline_mean"] = baseline_mean
        event.diagnostics["baseline_std"] = baseline_std

        # Record the peak index each method implies within its own
        # region, used later for timing-consistency analysis.
        method_peak_indices = {}
        for method, region in group:
            method_start, method_end = region
            region_values = signal[method_start:method_end + 1]
            local = int(np.argmax(np.abs(region_values)))
            method_peak_indices[method] = method_start + local
        event.diagnostics["method_peak_indices"] = method_peak_indices

        if refine:
            event = refine_event_boundaries(
                event,
                signal,
                time,
                baseline_mean,
                baseline_std,
                noise_std_multiplier=refine_params["noise_std_multiplier"],
                recovery_ratio=refine_params["recovery_ratio"],
                confirmation_samples=refine_params["confirmation_samples"],
            )

        events.append(event)

    return events


# ------------------------------------------------------------
# Full channel pipeline
# ------------------------------------------------------------

def detect_events_channel(
    signal,
    time,
    channel: str = "FBG1",
    dataset: str = "",
    baseline_samples: int = BASELINE_SAMPLES,
    methods: Optional[List[str]] = None,
    params: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    rejection_rules: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Run the complete ensemble detection on one channel.

    This is the main entry point for a single FBG channel. It runs
    all detectors, matches events, fuses evidence, refines
    boundaries and applies the false-positive rejection rules.

    Parameters
    ----------
    signal : array-like
        Filtered wavelength-shift signal.
    time : array-like
        Time values corresponding to the signal.
    channel : str
        Channel identifier (e.g., "FBG1").
    dataset : str
        Dataset name (e.g., "expert10").
    baseline_samples : int
        Number of initial samples used for the baseline.
    methods : list of str, optional
        Detectors to run.
    params : dict, optional
        Detector parameter overrides.
    weights : dict, optional
        Normalized detector weights.
    rejection_rules : dict, optional
        Overrides for the false-positive rejection thresholds.

    Returns
    -------
    dict
        Contains:
        - "events": all candidate events (accepted and rejected).
        - "accepted_events": only accepted events.
        - "detections": per-detector regions (for plotting).
        - "baseline_mean", "baseline_std": baseline statistics.
        - "weights": normalized detector weights.
    """
    if params is None:
        params = default_detector_params()

    if weights is None:
        weights = default_weights()

    signal = np.asarray(signal, dtype=float)
    time = np.asarray(time, dtype=float)

    baseline = signal[:baseline_samples]
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline))

    signal_mean = float(np.mean(signal))
    signal_std = float(np.std(signal))
    drift_std = (
        float(abs(signal_mean - baseline_mean) / baseline_std)
        if baseline_std > 1e-12
        else 0.0
    )
    excursion_std = (
        float(max(abs(signal.min() - baseline_mean),
                  abs(signal.max() - baseline_mean)) / baseline_std)
        if baseline_std > 1e-12
        else 0.0
    )

    all_regions = run_all_detectors(
        signal,
        time,
        methods=methods,
        params=params,
    )

    candidates = build_candidate_events(
        all_regions,
        signal,
        time,
        channel=channel,
        dataset=dataset,
        weights=weights,
        match_tolerance_samples=params["match_tolerance_samples"],
        group_split_min_gap_samples=params[
            "group_split_min_gap_samples"
        ],
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        refine=True,
    )

    events: List[ImpactEvent] = []
    accepted_events: List[ImpactEvent] = []

    for event in candidates:
        accepted, reason = apply_rejection_rules(
            event,
            baseline_mean,
            baseline_std,
            rejection_rules,
            signal_length=len(signal),
        )

        if accepted:
            event.accept()
            accepted_events.append(event)
        else:
            event.reject(reason)

        events.append(event)

    return {
        "events": events,
        "accepted_events": accepted_events,
        "detections": all_regions,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "signal_mean": signal_mean,
        "signal_std": signal_std,
        "drift_std": drift_std,
        "excursion_std": excursion_std,
        "weights": weights,
    }


__all__ = [
    "default_detector_params",
    "default_weights",
    "DETECTOR_FUNCTIONS",
    "run_all_detectors",
    "build_candidate_events",
    "detect_events_channel",
    "compute_evidence_score",
]
