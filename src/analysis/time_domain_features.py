"""
Phase 6 - Time-domain signal features for accepted impact events.

Extracts statistical and shape features from the event segment of
the filtered wavelength-shift signal. These complement the Phase 5
features (peak shift, residual shift, rise time, recovery time) by
characterizing the signal shape within the impact window.

Features
--------
- mean, median, std, variance, rms, minimum, maximum, peak_to_peak
- skewness, kurtosis, crest_factor

Definitions
-----------
Mean:  (1/N) * sum(x_i)
Median: middle value of sorted x
Std:   sqrt( (1/N) * sum((x_i - mean)^2) )  [population std]
Var:   (1/N) * sum((x_i - mean)^2)
RMS:   sqrt( (1/N) * sum(x_i^2) )
Peak-to-Peak: max(x) - min(x)
Skewness: (1/N) * sum(((x_i - mean)/std)^3)  [third central moment]
Kurtosis: (1/N) * sum(((x_i - mean)/std)^4)  [excess kurtosis = kurt - 3]
Crest Factor: max(|x|) / RMS  [only when RMS > 0]

All calculations protect against division by zero and constant signals.
When a feature cannot be computed reliably it is set to NaN with a
meaningful failure_reason.
"""

from typing import Tuple

import numpy as np


# ------------------------------------------------------------------
# Pure feature calculators (all take numpy arrays, return scalars)
# ------------------------------------------------------------------

def compute_mean(signal_window: np.ndarray) -> float:
    """Arithmetic mean of the event signal."""
    return float(np.mean(signal_window))


def compute_median(signal_window: np.ndarray) -> float:
    """Median of the event signal."""
    return float(np.median(signal_window))


def compute_std(signal_window: np.ndarray) -> float:
    """Population standard deviation of the event signal."""
    return float(np.std(signal_window))


def compute_variance(signal_window: np.ndarray) -> float:
    """Population variance of the event signal."""
    return float(np.var(signal_window))


def compute_rms(signal_window: np.ndarray) -> float:
    """Root mean square of the event signal.

    RMS = sqrt( (1/N) * sum(x_i^2) )
    """
    return float(np.sqrt(np.mean(signal_window ** 2)))


def compute_minimum(signal_window: np.ndarray) -> float:
    """Minimum value in the event signal."""
    return float(np.min(signal_window))


def compute_maximum(signal_window: np.ndarray) -> float:
    """Maximum value in the event signal."""
    return float(np.max(signal_window))


def compute_peak_to_peak(signal_window: np.ndarray) -> float:
    """Peak-to-peak range: max(x) - min(x)."""
    return float(np.max(signal_window) - np.min(signal_window))


def compute_skewness(signal_window: np.ndarray) -> float:
    """Sample skewness (third standardized moment).

    skewness = (1/N) * sum(((x_i - mean)/std)^3)

    Returns NaN for constant signals (std == 0).
    """
    n = len(signal_window)
    if n < 3:
        return float("nan")

    mean = np.mean(signal_window)
    std = np.std(signal_window)

    if std < 1e-15:
        return float("nan")

    return float(np.mean(((signal_window - mean) / std) ** 3))


def compute_kurtosis(signal_window: np.ndarray) -> float:
    """Excess kurtosis (fourth standardized moment minus 3).

    kurtosis = (1/N) * sum(((x_i - mean)/std)^4) - 3

    A normal distribution has excess kurtosis 0.
    Returns NaN for constant signals (std == 0).
    """
    n = len(signal_window)
    if n < 4:
        return float("nan")

    mean = np.mean(signal_window)
    std = np.std(signal_window)

    if std < 1e-15:
        return float("nan")

    return float(np.mean(((signal_window - mean) / std) ** 4) - 3.0)


def compute_crest_factor(signal_window: np.ndarray) -> float:
    """Crest factor: max(|x|) / RMS.

    Returns NaN when RMS is zero (constant zero signal).
    """
    rms = compute_rms(signal_window)
    if rms < 1e-15:
        return float("nan")

    return float(np.max(np.abs(signal_window)) / rms)


# ------------------------------------------------------------------
# Combined extraction for one event
# ------------------------------------------------------------------

def extract_time_domain_features(
    signal_window: np.ndarray,
) -> Tuple[dict, str]:
    """
    Extract all time-domain features from one event signal segment.

    Parameters
    ----------
    signal_window : np.ndarray
        The filtered signal segment corresponding to the event
        (signal[start_index : end_index + 1]).

    Returns
    -------
    (features_dict, failure_reason)
        features_dict : dict with feature name -> value
        failure_reason : empty string on success, or a short
            description of why features could not be computed.
    """
    signal_window = np.asarray(signal_window, dtype=float)

    if len(signal_window) == 0:
        return _nan_features("empty_event_segment")

    if len(signal_window) == 1:
        return _nan_features("single_sample_event")

    if not np.any(np.isfinite(signal_window)):
        return _nan_features("all_nan_signal")

    features = {
        "mean": compute_mean(signal_window),
        "median": compute_median(signal_window),
        "std": compute_std(signal_window),
        "variance": compute_variance(signal_window),
        "rms": compute_rms(signal_window),
        "minimum": compute_minimum(signal_window),
        "maximum": compute_maximum(signal_window),
        "peak_to_peak": compute_peak_to_peak(signal_window),
        "skewness": compute_skewness(signal_window),
        "kurtosis": compute_kurtosis(signal_window),
        "crest_factor": compute_crest_factor(signal_window),
    }

    return features, ""


def _nan_features(reason: str) -> Tuple[dict, str]:
    """Return a dict of NaN features with the given failure reason."""
    features = {
        "mean": float("nan"),
        "median": float("nan"),
        "std": float("nan"),
        "variance": float("nan"),
        "rms": float("nan"),
        "minimum": float("nan"),
        "maximum": float("nan"),
        "peak_to_peak": float("nan"),
        "skewness": float("nan"),
        "kurtosis": float("nan"),
        "crest_factor": float("nan"),
    }
    return features, reason


__all__ = [
    "compute_mean",
    "compute_median",
    "compute_std",
    "compute_variance",
    "compute_rms",
    "compute_minimum",
    "compute_maximum",
    "compute_peak_to_peak",
    "compute_skewness",
    "compute_kurtosis",
    "compute_crest_factor",
    "extract_time_domain_features",
]
