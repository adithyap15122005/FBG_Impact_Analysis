"""
Phase 5 - Impact feature extraction.

Phase 5 consumes the ACCEPTED impact events produced by the selected
primary analysis (FBG2 + Savitzky-Golay + Peak Detection) and
extracts four scalar features per event:

1. Peak Shift     : peak_value - pre_impact_baseline
2. Residual Shift : post_impact_level - pre_impact_baseline
3. Rise Time      : peak_time - start_time
4. Recovery Time  : end_time - peak_time

No new detector is introduced and no ensemble fusion is used. All
boundaries (start / peak / recovery / end) come from the existing
selected pipeline (src/pipeline/selected_pipeline.py). The
pre-impact baseline is the same baseline statistic already used by
that pipeline (mean of the first BASELINE_SAMPLES samples of the
filtered signal).

Residual shift is estimated from the MEDIAN of a stable window after
the recovery/end point, so a single noisy sample never drives it.
The post-impact window starts a small gap after the event end,
contains enough samples and excludes any region belonging to another
detected event. When there is not enough valid post-impact data the
residual features are set to NaN and a clear reason is recorded.
Residual shift is only a remaining signal offset relative to the
pre-impact baseline; it is NOT physical damage.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..impact_detection.ensemble_event import ImpactEvent

# ------------------------------------------------------------------
# Post-impact window settings (heuristic, sample units).
# ------------------------------------------------------------------

# Samples skipped immediately after the event end to avoid the
# recovery transient.
RESIDUAL_START_GAP_SAMPLES = 5

# Maximum length (in samples) of the post-impact estimation window.
# At the 50 Hz sampling rate 50 samples = 1.0 s.
RESIDUAL_WINDOW_SAMPLES = 50

# Minimum number of valid post-impact samples required to estimate
# the stable level. Below this the value is not trustworthy and the
# residual features are reported as NaN.
MIN_RESIDUAL_SAMPLES = 5


@dataclass
class Phase5Features:
    """
    Scalar features extracted for one accepted impact event.

    Attributes
    ----------
    dataset : str
        Dataset (experiment) name.
    fbg : str
        FBG channel identifier (the selected channel, FBG2).
    impact_id : str
        Impact event identifier.
    start_time / peak_time / end_time : float
        Event boundaries in seconds.
    pre_impact_baseline : float
        Pre-impact baseline of the filtered signal.
    peak_value : float
        Wavelength shift at the detected peak.
    peak_shift : float
        Signed peak shift: peak_value - pre_impact_baseline.
    absolute_peak_shift : float
        abs(peak_shift).
    rise_time : float
        peak_time - start_time (seconds).
    recovery_time : float
        end_time - peak_time (seconds).
    post_impact_level : float
        Median of the stable post-recovery window (NaN if
        insufficient data).
    residual_shift : float
        post_impact_level - pre_impact_baseline (NaN if
        insufficient data).
    residual_n_samples : int
        Number of valid samples used for the post-impact median.
    residual_reason : str
        Empty when the residual features were computed, otherwise a
        short reason (e.g. "insufficient_post_impact_samples(3<5)").
    """

    dataset: str
    fbg: str
    impact_id: str
    start_time: float
    peak_time: float
    end_time: float
    pre_impact_baseline: float
    peak_value: float
    peak_shift: float
    absolute_peak_shift: float
    rise_time: float
    recovery_time: float
    peak_width: float = float("nan")
    max_slope: float = float("nan")
    rms: float = float("nan")
    signal_energy: float = float("nan")
    peak_to_peak: float = float("nan")
    variance: float = float("nan")
    standard_deviation: float = float("nan")
    entropy: float = float("nan")
    area_under_curve: float = float("nan")
    post_impact_level: float = float("nan")
    residual_shift: float = float("nan")
    residual_n_samples: int = 0
    residual_reason: str = ""

    def to_dict(self) -> Dict:
        """Flat dictionary representation for CSV/JSON export."""
        return {
            "dataset": self.dataset,
            "fbg": self.fbg,
            "impact_id": self.impact_id,
            "start_time": self.start_time,
            "peak_time": self.peak_time,
            "end_time": self.end_time,
            "pre_impact_baseline": self.pre_impact_baseline,
            "peak_value": self.peak_value,
            "peak_shift": self.peak_shift,
            "absolute_peak_shift": self.absolute_peak_shift,
            "post_impact_level": self.post_impact_level,
            "residual_shift": self.residual_shift,
            "rise_time": self.rise_time,
            "recovery_time": self.recovery_time,
            "peak_width": self.peak_width,
            "max_slope": self.max_slope,
            "rms": self.rms,
            "signal_energy": self.signal_energy,
            "peak_to_peak": self.peak_to_peak,
            "variance": self.variance,
            "standard_deviation": self.standard_deviation,
            "entropy": self.entropy,
            "area_under_curve": self.area_under_curve,
            "residual_n_samples": self.residual_n_samples,
            "residual_reason": self.residual_reason,
        }


# ------------------------------------------------------------------
# Pure feature calculators
# ------------------------------------------------------------------

def compute_peak_shift(
    peak_value: float,
    pre_impact_baseline: float,
) -> float:
    """
    Peak shift: difference between the peak wavelength shift and the
    pre-impact baseline.

    Parameters
    ----------
    peak_value : float
        Wavelength shift at the detected peak.
    pre_impact_baseline : float
        Pre-impact baseline of the filtered signal.

    Returns
    -------
    float
        peak_value - pre_impact_baseline (signed).
    """
    return float(peak_value - pre_impact_baseline)


def compute_rise_time(
    start_time: float,
    peak_time: float,
) -> float:
    """
    Rise time: time from the detected impact start to the peak.

    Parameters
    ----------
    start_time : float
        Event start time (seconds).
    peak_time : float
        Event peak time (seconds).

    Returns
    -------
    float
        peak_time - start_time (seconds).
    """
    return float(peak_time - start_time)


def compute_recovery_time(
    peak_time: float,
    end_time: float,
) -> float:
    """
    Recovery time: time from the peak to the recovery/end boundary.

    The end/recovery boundary is the one produced by the existing
    boundary refinement: the first sample after the peak where the
    filtered signal stays within the recovery tolerance for
    confirmation_samples consecutive samples.

    Parameters
    ----------
    peak_time : float
        Event peak time (seconds).
    end_time : float
        Event end/recovery time (seconds).

    Returns
    -------
    float
        end_time - peak_time (seconds).
    """
    return float(end_time - peak_time)


# ------------------------------------------------------------------
# Residual shift
# ------------------------------------------------------------------
def compute_peak_width(
    start_time: float,
    end_time: float,
) -> float:
    """
    Peak width of the event.
    """

    return float(end_time - start_time)


def compute_max_slope(
    signal_window: np.ndarray,
    time_window: np.ndarray,
) -> float:
    """
    Maximum absolute slope during the impact event.
    """

    if len(signal_window) < 2:
        return float("nan")

    slopes = np.gradient(
        signal_window,
        time_window,
    )

    return float(
        np.max(np.abs(slopes))
    )


def compute_rms(
    signal_window: np.ndarray,
) -> float:
    """
    RMS value of the impact window.
    """

    return float(
        np.sqrt(
            np.mean(signal_window ** 2)
        )
    )


def compute_signal_energy(
    signal_window: np.ndarray,
    time_window: np.ndarray,
) -> float:
    """
    Energy contained in the impact response.
    """

    return float(
        np.trapz(
            signal_window ** 2,
            time_window,
        )
    )


def compute_peak_to_peak(
    signal_window: np.ndarray,
) -> float:
    """
    Peak-to-peak amplitude.
    """

    return float(
        np.max(signal_window)
        - np.min(signal_window)
    )
def compute_variance(signal_window):
    return float(np.var(signal_window))


def compute_standard_deviation(signal_window):
    return float(np.std(signal_window))


def compute_area_under_curve(signal_window, time_window):
    return float(np.trapz(np.abs(signal_window), time_window))


def compute_entropy(signal_window):
    hist, _ = np.histogram(
        signal_window,
        bins=20,
        density=True,
    )

    hist = hist[hist > 0]

    return float(
        -np.sum(hist * np.log2(hist))
    )

def estimate_post_impact_level(
    signal: np.ndarray,
    end_index: int,
    excluded_regions: Optional[List[Tuple[int, int]]] = None,
    start_gap_samples: int = RESIDUAL_START_GAP_SAMPLES,
    window_samples: int = RESIDUAL_WINDOW_SAMPLES,
    min_samples: int = MIN_RESIDUAL_SAMPLES,
) -> Tuple[float, int, str]:
    """
    Estimate the stable post-impact level using the median of a
    window after the recovery/end point.

    The window:
    - starts `start_gap_samples` samples after `end_index`;
    - is at most `window_samples` samples long;
    - excludes every sample inside `excluded_regions` (other
      detected impacts);
    - stops at the end of the signal.

    Parameters
    ----------
    signal : array-like
        Filtered wavelength-shift signal.
    end_index : int
        Recovery/end index of the impact.
    excluded_regions : list of (int, int), optional
        (start, end) sample regions to exclude (other impacts).
    start_gap_samples : int
        Samples skipped right after the event end.
    window_samples : int
        Maximum window length in samples.
    min_samples : int
        Minimum valid samples required.

    Returns
    -------
    (level, n_samples, reason)
        level : float
            Median of the valid post-impact samples, or NaN when
            there is not enough valid data.
        n_samples : int
            Number of valid samples used.
        reason : str
            Empty when level was computed, otherwise a short reason.
    """
    signal = np.asarray(signal, dtype=float)

    length = len(signal)

    excluded = np.zeros(length, dtype=bool)

    if excluded_regions:
        for start, end in excluded_regions:
            start = max(0, int(start))
            end = min(length - 1, int(end))

            if start <= end:
                excluded[start:end + 1] = True

    start_index = int(end_index) + start_gap_samples

    if start_index >= length:
        return (
            float("nan"),
            0,
            "insufficient_post_impact_samples(start_past_end)",
        )

    window_end = min(
        start_index + window_samples,
        length,
    )

    window = np.arange(start_index, window_end)
    valid = window[~excluded[window]]

    n_samples = int(len(valid))

    if n_samples < min_samples:
        return (
            float("nan"),
            n_samples,
            f"insufficient_post_impact_samples({n_samples}<{min_samples})",
        )

    level = float(np.median(signal[valid]))

    return level, n_samples, ""


def post_impact_excluded_regions(
    events: List[ImpactEvent],
    current_event: ImpactEvent,
) -> List[Tuple[int, int]]:
    """
    Regions of every event other than the current one, used so the
    post-impact window avoids landing on another detected impact.
    """
    return [
        (event.start_index, event.end_index)
        for event in events
        if event.event_id != current_event.event_id
    ]


# ------------------------------------------------------------------
# Per-event feature extraction
# ------------------------------------------------------------------

def extract_features(
    event: ImpactEvent,
    signal: np.ndarray,
    time: np.ndarray,
    pre_impact_baseline: float,
    other_events: Optional[List[ImpactEvent]] = None,
) -> Phase5Features:
    """
    Extract the four Phase 5 features for one accepted event.

    Parameters
    ----------
    event : ImpactEvent
        Accepted event from the selected pipeline. Its start/peak/end
        boundaries and peak_value are reused as-is.
    signal : array-like
        FBG2 Savitzky-Golay filtered wavelength-shift signal.
    time : array-like
        Time values (seconds) matching the signal.
    pre_impact_baseline : float
        Pre-impact baseline from the selected pipeline.
    other_events : list of ImpactEvent, optional
        Other events of the same dataset, used to exclude other
        impacts from the residual-shift window.

    Returns
    -------
    Phase5Features
        Extracted features for the event.
    """
    signal = np.asarray(signal, dtype=float)
    time = np.asarray(time, dtype=float)
    start_idx = event.start_index
    end_idx = event.end_index

    signal_window = signal[start_idx:end_idx + 1]
    time_window = time[start_idx:end_idx + 1]

    peak_shift_value = compute_peak_shift(
        event.peak_value,
        pre_impact_baseline,
    )

    rise_time_value = compute_rise_time(
        event.start_time,
        event.peak_time,
    )

    recovery_time_value = compute_recovery_time(
        event.peak_time,
        event.end_time,
    )
    peak_width_value = compute_peak_width(
    event.start_time,
    event.end_time,
    )

    max_slope_value = compute_max_slope(
    signal_window,
    time_window,
    )

    rms_value = compute_rms(
    signal_window,
    )

    signal_energy_value = compute_signal_energy(
    signal_window,
    time_window,
    )

    peak_to_peak_value = compute_peak_to_peak(
    signal_window,
    )
    variance_value = compute_variance(
    signal_window,
)

    standard_deviation_value = (
    compute_standard_deviation(
        signal_window,
    )
)

    entropy_value = compute_entropy(
    signal_window,
)

    area_under_curve_value = (
    compute_area_under_curve(
        signal_window,
        time_window,
    )
)

    excluded_regions: List[Tuple[int, int]] = []

    if other_events:
        excluded_regions = post_impact_excluded_regions(
            other_events,
            event,
        )

    level, n_samples, reason = estimate_post_impact_level(
        signal,
        event.end_index,
        excluded_regions=excluded_regions,
    )

    residual_shift_value = (
        level - pre_impact_baseline
        if not np.isnan(level)
        else float("nan")
    )

    return Phase5Features(
        dataset=event.dataset,
        fbg=event.channel,
        impact_id=event.event_id,
        start_time=float(event.start_time),
        peak_time=float(event.peak_time),
        end_time=float(event.end_time),
        pre_impact_baseline=float(pre_impact_baseline),
        peak_value=float(event.peak_value),
        peak_shift=peak_shift_value,
        absolute_peak_shift=float(abs(peak_shift_value)),
        rise_time=rise_time_value,
        recovery_time=recovery_time_value,
        peak_width=peak_width_value,
        max_slope=max_slope_value,
        rms=rms_value,
        signal_energy=signal_energy_value,
        peak_to_peak=peak_to_peak_value,
        variance=variance_value,
        standard_deviation=standard_deviation_value,
        entropy=entropy_value,
        area_under_curve=area_under_curve_value,
        post_impact_level=level,
        residual_shift=residual_shift_value,
        residual_n_samples=n_samples,
        residual_reason=reason,
    )


# ------------------------------------------------------------------
# Dataset-level extraction
# ------------------------------------------------------------------

def extract_phase5_dataset(
    dataset_result: Dict,
    signal: np.ndarray,
    time: np.ndarray,
) -> List[Phase5Features]:
    """
    Extract Phase 5 features for every accepted event of one dataset.

    Parameters
    ----------
    dataset_result : dict
        Result dict from the selected pipeline (must contain
        "accepted_events", "baseline_mean" and "channel").
    signal : array-like
        FBG2 Savitzky-Golay filtered signal.
    time : array-like
        Time values matching the signal.

    Returns
    -------
    list of Phase5Features
        One feature record per accepted event.
    """
    events = dataset_result["accepted_events"]
    baseline_mean = dataset_result["baseline_mean"]

    features: List[Phase5Features] = []

    for event in events:
        features.append(
            extract_features(
                event,
                signal,
                time,
                baseline_mean,
                other_events=events,
            )
        )

    return features


__all__ = [
    "Phase5Features",
    "compute_peak_shift",
    "compute_rise_time",
    "compute_recovery_time",
    "estimate_post_impact_level",
    "compute_peak_width",
    "compute_max_slope",
    "compute_rms",
    "compute_signal_energy",
    "compute_peak_to_peak",
    "post_impact_excluded_regions",
    "extract_features",
    "extract_phase5_dataset",
    "RESIDUAL_START_GAP_SAMPLES",
    "RESIDUAL_WINDOW_SAMPLES",
    "MIN_RESIDUAL_SAMPLES",
]
