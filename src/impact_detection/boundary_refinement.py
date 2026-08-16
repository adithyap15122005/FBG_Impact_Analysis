"""
Boundary refinement for fused impact events.

After temporal matching, a fused event's start/end are the outer
extent of all supporting detector regions. These boundaries can be
wider than the physical impact. This module tightens them using
the actual signal behaviour:

- The start is the onset: the first sample where the signal
  deviates beyond the noise tolerance when moving back from the
  peak.
- The end is the recovery: the first sample after the peak where
  the signal stays within the noise tolerance for a number of
  consecutive samples.

The refinement never expands boundaries beyond the merged region
found by event matching.
"""

from typing import Optional

import numpy as np

from .ensemble_event import ImpactEvent


def recovery_tolerance(
    baseline_std: float,
    peak_deviation: float,
    noise_std_multiplier: float = 2.0,
    recovery_ratio: float = 0.20,
) -> float:
    """
    Compute the tolerance used to decide whether the signal has
    returned to the baseline.

    The tolerance is the larger of:

    - noise_std_multiplier x baseline_std
    - recovery_ratio x peak_deviation

    Parameters
    ----------
    baseline_std : float
        Standard deviation of the baseline region.
    peak_deviation : float
        Absolute deviation of the peak from the baseline mean.
    noise_std_multiplier : float
        Noise tolerance in units of baseline std.
    recovery_ratio : float
        Fraction of the peak deviation accepted as recovery.

    Returns
    -------
    float
        Recovery tolerance.
    """
    noise_tolerance = noise_std_multiplier * baseline_std
    peak_tolerance = recovery_ratio * peak_deviation

    return max(noise_tolerance, peak_tolerance)


def refine_start(
    signal: np.ndarray,
    baseline_mean: float,
    peak_index: int,
    merged_start: int,
    tolerance: float,
) -> int:
    """
    Find the onset of the impact.

    Walks backwards from the peak while the signal stays above the
    tolerance. The returned index is the first sample (from the
    left) that is still above the tolerance.

    Parameters
    ----------
    signal : numpy.ndarray
        Filtered signal.
    baseline_mean : float
        Baseline mean.
    peak_index : int
        Index of the event peak.
    merged_start : int
        Lower boundary of the merged region.
    tolerance : float
        Deviation above which the signal is considered active.

    Returns
    -------
    int
        Refined start index, clamped to [merged_start, peak_index].
    """
    refined = peak_index

    index = peak_index

    while index > merged_start:
        previous = signal[index - 1]
        if abs(previous - baseline_mean) <= tolerance:
            break
        index -= 1

    refined = index

    return int(max(merged_start, min(refined, peak_index)))


def refine_end(
    signal: np.ndarray,
    baseline_mean: float,
    peak_index: int,
    merged_end: int,
    tolerance: float,
    confirmation_samples: int = 5,
) -> int:
    """
    Find the recovery/end of the impact.

    Scans forward from the peak and returns the first index where
    the signal stays within the tolerance for
    confirmation_samples consecutive samples.

    Parameters
    ----------
    signal : numpy.ndarray
        Filtered signal.
    baseline_mean : float
        Baseline mean.
    peak_index : int
        Index of the event peak.
    merged_end : int
        Upper boundary of the merged region.
    tolerance : float
        Deviation below which the signal is considered recovered.
    confirmation_samples : int
        Consecutive samples required to confirm recovery.

    Returns
    -------
    int
        Refined end index, clamped to [peak_index, merged_end].
    """
    signal_length = len(signal)

    if confirmation_samples < 1:
        raise ValueError("confirmation_samples must be >= 1")

    last_start = (
        min(merged_end, signal_length - confirmation_samples)
        - peak_index
        + 1
    )

    if last_start < 1:
        return int(merged_end)

    for offset in range(last_start):
        window = signal[
            peak_index + offset:
            peak_index + offset + confirmation_samples
        ]

        if np.all(np.abs(window - baseline_mean) <= tolerance):
            return int(peak_index + offset)

    return int(merged_end)


def refine_event_boundaries(
    event: ImpactEvent,
    signal: np.ndarray,
    time: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    noise_std_multiplier: float = 2.0,
    recovery_ratio: float = 0.20,
    confirmation_samples: int = 5,
) -> ImpactEvent:
    """
    Refine the start/end/peak of a fused candidate event.

    The event's merged boundaries (start_index/end_index) are
    tightened using the signal behaviour around the peak. A new
    ImpactEvent is returned; the input event is left unchanged.

    Parameters
    ----------
    event : ImpactEvent
        Candidate event produced by event matching.
    signal : array-like
        Filtered signal.
    time : array-like
        Time values corresponding to the signal.
    baseline_mean : float
        Baseline mean.
    baseline_std : float
        Baseline standard deviation.
    noise_std_multiplier : float
        Noise tolerance in units of baseline std.
    recovery_ratio : float
        Fraction of the peak deviation accepted as recovery.
    confirmation_samples : int
        Consecutive samples required to confirm the end.

    Returns
    -------
    ImpactEvent
        New event with refined boundaries.
    """
    signal = np.asarray(signal, dtype=float)
    time = np.asarray(time, dtype=float)

    if len(signal) != len(time):
        raise ValueError("Signal and time must have the same length.")

    peak_deviation = abs(
        signal[event.peak_index] - baseline_mean
    )

    tolerance = recovery_tolerance(
        baseline_std,
        peak_deviation,
        noise_std_multiplier,
        recovery_ratio,
    )

    original_start = event.start_index
    original_end = event.end_index

    refined_start = refine_start(
        signal,
        baseline_mean,
        event.peak_index,
        original_start,
        tolerance,
    )

    refined_end = refine_end(
        signal,
        baseline_mean,
        event.peak_index,
        original_end,
        tolerance,
        confirmation_samples,
    )

    # Recompute the peak inside the refined region.
    region = signal[refined_start:refined_end + 1]

    if len(region) == 0:
        refined_start = event.start_index
        refined_end = event.end_index
        region = signal[refined_start:refined_end + 1]

    local_peak = int(np.argmax(np.abs(region)))
    refined_peak = refined_start + local_peak

    refined_event = ImpactEvent(
        start_index=int(refined_start),
        peak_index=int(refined_peak),
        end_index=int(refined_end),
        start_time=float(time[refined_start]),
        peak_time=float(time[refined_peak]),
        end_time=float(time[refined_end]),
        peak_value=float(signal[refined_peak]),
        duration=float(time[refined_end] - time[refined_start]),
        detection_methods=list(event.detection_methods),
        evidence_score=event.evidence_score,
        event_id=event.event_id,
        dataset=event.dataset,
        channel=event.channel,
        accepted=event.accepted,
        rejection_reason=event.rejection_reason,
        diagnostics=dict(event.diagnostics),
    )

    refined_event.diagnostics["original_start_index"] = original_start
    refined_event.diagnostics["original_end_index"] = original_end
    refined_event.diagnostics["recovery_tolerance"] = tolerance

    return refined_event
