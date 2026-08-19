"""
Phase 6 - Multi-Domain Signal Analysis orchestrator.

Consumes the ACCEPTED impact events from the selected pipeline and
extracts time-domain, frequency-domain (FFT), and time-frequency
(STFT) features for each event. Phase 6 is additive and does not
redesign or break previous phases.

The orchestrator coordinates:
1. Sampling frequency estimation from timestamps.
2. Event segment extraction.
3. Time-domain feature extraction.
4. Frequency-domain FFT feature extraction.
5. Time-frequency STFT feature extraction.
6. Combined output with status tracking.

Output
------
- One row per accepted event in the combined CSV.
- Feature status: 'ok' or 'partial' with failure_reason.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..impact_detection.ensemble_event import ImpactEvent
from .frequency_domain import (
    DEFAULT_FFT_WINDOW,
    DEFAULT_MIN_FFT_SAMPLES,
    DEFAULT_REMOVE_DC,
    DEFAULT_ROLLOFF_PERCENTAGE,
    estimate_sampling_frequency,
    extract_frequency_domain_features,
)
from .phase5_features import Phase5Features
from .time_domain_features import extract_time_domain_features
from .time_frequency import (
    DEFAULT_ENABLE_STFT,
    DEFAULT_MIN_STFT_SAMPLES,
    DEFAULT_STFT_OVERLAP_RATIO,
    extract_stft_features,
)


@dataclass
class Phase6Config:
    """
    Configuration for Phase 6 multi-domain analysis.

    All fields have sensible defaults matching the specification.
    """
    # FFT settings
    remove_dc: bool = DEFAULT_REMOVE_DC
    fft_window: str = DEFAULT_FFT_WINDOW
    minimum_fft_samples: int = DEFAULT_MIN_FFT_SAMPLES
    spectral_rolloff_percentage: float = DEFAULT_ROLLOFF_PERCENTAGE

    # STFT settings
    enable_stft: bool = DEFAULT_ENABLE_STFT
    minimum_stft_samples: int = DEFAULT_MIN_STFT_SAMPLES
    stft_overlap_ratio: float = DEFAULT_STFT_OVERLAP_RATIO

    def to_dict(self) -> Dict:
        return {
            "remove_dc": self.remove_dc,
            "fft_window": self.fft_window,
            "minimum_fft_samples": self.minimum_fft_samples,
            "spectral_rolloff_percentage": self.spectral_rolloff_percentage,
            "enable_stft": self.enable_stft,
            "minimum_stft_samples": self.minimum_stft_samples,
            "stft_overlap_ratio": self.stft_overlap_ratio,
        }


@dataclass
class Phase6Features:
    """
    Combined multi-domain features for one accepted impact event.

    Includes Phase 5 context (timing, peak shift, residual shift,
    rise/recovery time) plus new time-domain, frequency-domain, and
    time-frequency features.
    """
    # Identifiers and timing
    dataset: str
    fbg: str
    impact_id: str
    start_time: float
    peak_time: float
    end_time: float
    duration: float

    # Phase 5 context
    peak_shift: float = float("nan")
    residual_shift: float = float("nan")
    rise_time: float = float("nan")
    recovery_time: float = float("nan")

    # Time-domain features
    mean: float = float("nan")
    median: float = float("nan")
    std: float = float("nan")
    variance: float = float("nan")
    rms: float = float("nan")
    minimum: float = float("nan")
    maximum: float = float("nan")
    peak_to_peak: float = float("nan")
    skewness: float = float("nan")
    kurtosis: float = float("nan")
    crest_factor: float = float("nan")

    # Frequency-domain features
    sampling_frequency_hz: float = float("nan")
    num_samples: int = 0
    dominant_frequency_hz: float = float("nan")
    dominant_magnitude: float = float("nan")
    spectral_energy: float = float("nan")
    spectral_entropy: float = float("nan")
    spectral_centroid_hz: float = float("nan")
    spectral_bandwidth_hz: float = float("nan")
    spectral_flatness: float = float("nan")
    spectral_rolloff_hz: float = float("nan")

    # Time-frequency features
    stft_peak_frequency_hz: float = float("nan")
    stft_max_energy: float = float("nan")
    stft_valid: bool = False
    stft_reason: str = ""

    # Status
    feature_status: str = "ok"
    failure_reason: str = ""

    def to_dict(self) -> Dict:
        """Flat dictionary for CSV/JSON export."""
        return {
            "dataset": self.dataset,
            "fbg": self.fbg,
            "impact_id": self.impact_id,
            "start_time": self.start_time,
            "peak_time": self.peak_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "peak_shift": self.peak_shift,
            "residual_shift": self.residual_shift,
            "rise_time": self.rise_time,
            "recovery_time": self.recovery_time,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "variance": self.variance,
            "rms": self.rms,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "peak_to_peak": self.peak_to_peak,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "crest_factor": self.crest_factor,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "num_samples": self.num_samples,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "dominant_magnitude": self.dominant_magnitude,
            "spectral_energy": self.spectral_energy,
            "spectral_entropy": self.spectral_entropy,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "spectral_bandwidth_hz": self.spectral_bandwidth_hz,
            "spectral_flatness": self.spectral_flatness,
            "spectral_rolloff_hz": self.spectral_rolloff_hz,
            "stft_peak_frequency_hz": self.stft_peak_frequency_hz,
            "stft_max_energy": self.stft_max_energy,
            "stft_valid": self.stft_valid,
            "feature_status": self.feature_status,
            "failure_reason": self.failure_reason,
        }


# Column order for CSV export
PHASE6_CSV_COLUMNS = [
    "dataset",
    "fbg",
    "impact_id",
    "start_time",
    "peak_time",
    "end_time",
    "duration",
    "peak_shift",
    "residual_shift",
    "rise_time",
    "recovery_time",
    "mean",
    "median",
    "std",
    "variance",
    "rms",
    "minimum",
    "maximum",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "crest_factor",
    "sampling_frequency_hz",
    "num_samples",
    "dominant_frequency_hz",
    "dominant_magnitude",
    "spectral_energy",
    "spectral_entropy",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_flatness",
    "spectral_rolloff_hz",
    "stft_peak_frequency_hz",
    "stft_max_energy",
    "stft_valid",
    "feature_status",
    "failure_reason",
]


def extract_event_features(
    event: ImpactEvent,
    signal: np.ndarray,
    time: np.ndarray,
    fs: float,
    phase5_feature: Optional[Phase5Features] = None,
    config: Optional[Phase6Config] = None,
) -> Phase6Features:
    """
    Extract multi-domain features for one accepted event.

    Parameters
    ----------
    event : ImpactEvent
        Accepted event with valid boundaries.
    signal : np.ndarray
        Full filtered signal (not just the event segment).
    time : np.ndarray
        Full time array matching signal.
    fs : float
        Estimated sampling frequency (Hz).
    phase5_feature : Phase5Features, optional
        Pre-computed Phase 5 features (for context reuse).
    config : Phase6Config, optional
        Analysis configuration (uses defaults if None).

    Returns
    -------
    Phase6Features
        Combined feature record for the event.
    """
    if config is None:
        config = Phase6Config()

    start_idx = event.start_index
    end_idx = event.end_index

    # Basic context
    features = Phase6Features(
        dataset=event.dataset,
        fbg=event.channel,
        impact_id=event.event_id,
        start_time=float(event.start_time),
        peak_time=float(event.peak_time),
        end_time=float(event.end_time),
        duration=float(event.end_time - event.start_time),
    )

    # Reuse Phase 5 features where available
    if phase5_feature is not None:
        features.peak_shift = phase5_feature.peak_shift
        features.residual_shift = phase5_feature.residual_shift
        features.rise_time = phase5_feature.rise_time
        features.recovery_time = phase5_feature.recovery_time

    # Extract event segment
    signal_window = signal[start_idx:end_idx + 1]
    time_window = time[start_idx:end_idx + 1]

    failure_parts = []

    # --- Time-domain features ---
    td_features, td_reason = extract_time_domain_features(
        signal_window,
    )
    features.mean = td_features["mean"]
    features.median = td_features["median"]
    features.std = td_features["std"]
    features.variance = td_features["variance"]
    features.rms = td_features["rms"]
    features.minimum = td_features["minimum"]
    features.maximum = td_features["maximum"]
    features.peak_to_peak = td_features["peak_to_peak"]
    features.skewness = td_features["skewness"]
    features.kurtosis = td_features["kurtosis"]
    features.crest_factor = td_features["crest_factor"]

    if td_reason:
        failure_parts.append(f"time_domain:{td_reason}")

    # --- Frequency-domain features ---
    fd_features, fd_reason = extract_frequency_domain_features(
        signal_window,
        time_window,
        fs=fs,
        remove_dc=config.remove_dc,
        fft_window=config.fft_window,
        min_fft_samples=config.minimum_fft_samples,
        rolloff_percentage=config.spectral_rolloff_percentage,
    )
    features.sampling_frequency_hz = fd_features["sampling_frequency_hz"]
    features.num_samples = fd_features["num_samples"]
    features.dominant_frequency_hz = fd_features["dominant_frequency_hz"]
    features.dominant_magnitude = fd_features["dominant_magnitude"]
    features.spectral_energy = fd_features["spectral_energy"]
    features.spectral_entropy = fd_features["spectral_entropy"]
    features.spectral_centroid_hz = fd_features["spectral_centroid_hz"]
    features.spectral_bandwidth_hz = fd_features["spectral_bandwidth_hz"]
    features.spectral_flatness = fd_features["spectral_flatness"]
    features.spectral_rolloff_hz = fd_features["spectral_rolloff_hz"]

    if fd_reason:
        failure_parts.append(f"frequency_domain:{fd_reason}")

    # --- Time-frequency features ---
    stft_features, stft_valid, stft_reason = extract_stft_features(
        signal_window,
        fs=fs,
        enable_stft=config.enable_stft,
        min_stft_samples=config.minimum_stft_samples,
        overlap_ratio=config.stft_overlap_ratio,
    )
    features.stft_peak_frequency_hz = stft_features["stft_peak_frequency_hz"]
    features.stft_max_energy = stft_features["stft_max_energy"]
    features.stft_valid = stft_valid
    features.stft_reason = stft_reason

    if stft_reason and stft_reason != "stft_disabled":
        failure_parts.append(f"stft:{stft_reason}")

    # --- Status ---
    if failure_parts:
        features.feature_status = "partial"
        features.failure_reason = "; ".join(failure_parts)
    else:
        features.feature_status = "ok"
        features.failure_reason = ""

    return features


def extract_phase6_dataset(
    dataset_result: Dict,
    signal: np.ndarray,
    time: np.ndarray,
    phase5_features: Optional[List[Phase5Features]] = None,
    config: Optional[Phase6Config] = None,
) -> List[Phase6Features]:
    """
    Extract Phase 6 features for every accepted event of one dataset.

    Parameters
    ----------
    dataset_result : dict
        Result dict from the selected pipeline (must contain
        "accepted_events", "baseline_mean", and "channel").
    signal : np.ndarray
        FBG2 Savitzky-Golay filtered wavelength-shift signal.
    time : np.ndarray
        Time values matching the signal.
    phase5_features : list of Phase5Features, optional
        Pre-computed Phase 5 features for context reuse.
    config : Phase6Config, optional
        Analysis configuration.

    Returns
    -------
    list of Phase6Features
        One feature record per accepted event.
    """
    events = dataset_result["accepted_events"]

    if config is None:
        config = Phase6Config()

    # Estimate sampling frequency from timestamps
    fs, fs_reason = estimate_sampling_frequency(time)

    if not np.isfinite(fs):
        # Cannot compute frequency-domain features at all
        return [
            _failed_event_features(event, f"sampling_freq:{fs_reason}")
            for event in events
        ]

    # Build lookup from event_id to Phase5Features for reuse
    p5_lookup = {}
    if phase5_features:
        for p5 in phase5_features:
            p5_lookup[p5.impact_id] = p5

    features: List[Phase6Features] = []

    for event in events:
        p5 = p5_lookup.get(event.event_id)

        feature = extract_event_features(
            event,
            signal,
            time,
            fs=fs,
            phase5_feature=p5,
            config=config,
        )

        features.append(feature)

    return features


def _failed_event_features(
    event: ImpactEvent,
    reason: str,
) -> Phase6Features:
    """Create a Phase6Features with all-NaN for a failed event."""
    return Phase6Features(
        dataset=event.dataset,
        fbg=event.channel,
        impact_id=event.event_id,
        start_time=float(event.start_time),
        peak_time=float(event.peak_time),
        end_time=float(event.end_time),
        duration=float(event.end_time - event.start_time),
        feature_status="failed",
        failure_reason=reason,
    )


__all__ = [
    "Phase6Config",
    "Phase6Features",
    "PHASE6_CSV_COLUMNS",
    "extract_event_features",
    "extract_phase6_dataset",
]
