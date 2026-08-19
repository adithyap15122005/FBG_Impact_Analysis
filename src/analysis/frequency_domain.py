"""
Phase 6 - Frequency-domain (FFT) analysis for accepted impact events.

Computes spectral features from the event segment of the filtered
wavelength-shift signal using a one-sided real FFT.

Sampling Frequency
------------------
Estimated from timestamps using robust positive differences:
    dt = median( positive_diff(time) )
    fs = 1.0 / dt

When timestamps are unavailable or invalid the caller must supply fs.

FFT Processing
--------------
1. Extract event signal segment.
2. Interpolate or skip NaN values.
3. Optionally remove DC bias (mean subtraction).
4. Apply a configurable Hann window.
5. Compute one-sided rfft and correct frequency axis.
6. Exclude DC bin from dominant-frequency search.

Features
--------
- dominant_frequency_hz: frequency with the largest non-DC spectral
  magnitude (Hz).
- dominant_magnitude: magnitude at the dominant frequency.
- spectral_energy: sum of squared FFT magnitudes (Parseval-consistent).
- spectral_entropy: Shannon entropy of the normalized spectral power
  distribution. A concentrated spectrum gives low entropy; a flat/
  broadband spectrum gives high entropy.
- spectral_centroid_hz: energy-weighted mean frequency.
- spectral_bandwidth_hz: energy-weighted spread around the centroid.
- spectral_flatness: geometric mean / arithmetic mean of the power
  spectrum (Wiener entropy). 0 = pure tone, 1 = white noise.
- spectral_rolloff_hz: frequency below which a given percentage
  (default 85%) of total spectral energy is contained.

All calculations guard against division by zero, constant signals,
and invalid inputs. Features that cannot be computed are set to NaN
with a meaningful failure_reason.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.fft import rfft, rfftfreq


# ------------------------------------------------------------------
# Configuration defaults (overridable per-call)
# ------------------------------------------------------------------

DEFAULT_REMOVE_DC = True
DEFAULT_FFT_WINDOW = "hann"
DEFAULT_MIN_FFT_SAMPLES = 8
DEFAULT_ROLLOFF_PERCENTAGE = 85.0


# ------------------------------------------------------------------
# Sampling frequency estimation
# ------------------------------------------------------------------

def estimate_sampling_frequency(
    time: np.ndarray,
) -> Tuple[float, str]:
    """
    Robustly estimate the sampling frequency from a time array.

    Uses the median of positive first differences.

    Parameters
    ----------
    time : np.ndarray
        Time values in seconds (must be monotonically non-decreasing
        for best results, but the median makes it robust to a few
        irregular gaps).

    Returns
    -------
    (fs, reason)
        fs : float
            Estimated sampling frequency in Hz, or NaN on failure.
        reason : str
            Empty on success, otherwise a short failure description.
    """
    time = np.asarray(time, dtype=float)

    if len(time) < 2:
        return float("nan"), "insufficient_timestamps(len<2)"

    diffs = np.diff(time)
    positive_diffs = diffs[diffs > 0]

    if len(positive_diffs) == 0:
        return float("nan"), "no_positive_time_differences"

    dt = float(np.median(positive_diffs))

    if dt <= 0 or not np.isfinite(dt):
        return float("nan"), "invalid_dt(non_positive_or_nan)"

    fs = 1.0 / dt

    if not np.isfinite(fs) or fs <= 0:
        return float("nan"), "invalid_fs(non_positive_or_nan)"

    return fs, ""


# ------------------------------------------------------------------
# FFT feature calculators
# ------------------------------------------------------------------

def _prepare_event_signal(
    signal_window: np.ndarray,
    remove_dc: bool,
    fft_window: str,
) -> Tuple[np.ndarray, str]:
    """
    Prepare the event signal for FFT analysis.

    Steps: validate -> handle NaN -> optional DC removal -> windowing.
    Returns (prepared_signal, failure_reason).
    """
    signal_window = np.asarray(signal_window, dtype=float)

    if len(signal_window) == 0:
        return signal_window, "empty_event_segment"

    if not np.any(np.isfinite(signal_window)):
        return signal_window, "all_nan_signal"

    # Replace NaN with linear interpolation where possible
    if np.any(~np.isfinite(signal_window)):
        valid_mask = np.isfinite(signal_window)
        if np.sum(valid_mask) < 2:
            return signal_window, "insufficient_finite_samples"
        indices = np.arange(len(signal_window))
        signal_window = np.interp(
            indices,
            indices[valid_mask],
            signal_window[valid_mask],
        )

    # Remove DC bias (mean subtraction)
    if remove_dc:
        signal_window = signal_window - np.mean(signal_window)

    # Apply window
    n = len(signal_window)
    if fft_window == "hann" or fft_window == "hanning":
        window = np.hanning(n)
    elif fft_window == "hamming":
        window = np.hamming(n)
    elif fft_window == "blackman":
        window = np.blackman(n)
    elif fft_window == "none" or fft_window == "":
        window = np.ones(n)
    else:
        window = np.hanning(n)

    # Avoid multiplying by zero window
    if np.all(window < 1e-15):
        window = np.ones(n)

    signal_window = signal_window * window

    return signal_window, ""


def compute_dominant_frequency(
    magnitudes: np.ndarray,
    freqs: np.ndarray,
) -> Tuple[float, float]:
    """
    Find the dominant (non-DC) frequency and its magnitude.

    Parameters
    ----------
    magnitudes : np.ndarray
        One-sided FFT magnitudes.
    freqs : np.ndarray
        Corresponding frequency axis (Hz).

    Returns
    -------
    (dominant_freq_hz, dominant_magnitude)
    """
    if len(magnitudes) < 2:
        return float("nan"), float("nan")

    # Exclude DC (index 0)
    mag_no_dc = magnitudes[1:]
    freq_no_dc = freqs[1:]

    if len(mag_no_dc) == 0 or not np.any(np.isfinite(mag_no_dc)):
        return float("nan"), float("nan")

    peak_idx = int(np.argmax(mag_no_dc))

    return float(freq_no_dc[peak_idx]), float(mag_no_dc[peak_idx])


def compute_spectral_energy(magnitudes: np.ndarray) -> float:
    """
    Spectral energy: sum of squared FFT magnitudes.

    By Parseval's theorem this is proportional to the total signal
    energy.  For a real signal the one-sided spectrum doubles the
    energy of all non-DC/non-Nyquist bins, so we use the raw
    one-sided magnitudes without the factor-of-2 correction; the
    resulting value is a consistent relative energy measure.

    spectral_energy = sum(|X_k|^2)

    Returns NaN for empty or all-NaN input.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)

    if len(magnitudes) == 0 or not np.any(np.isfinite(magnitudes)):
        return float("nan")

    return float(np.sum(magnitudes ** 2))


def compute_spectral_entropy(magnitudes: np.ndarray) -> float:
    """
    Shannon spectral entropy of the normalized power spectrum.

    H = -sum( p_k * log2(p_k) )

    where p_k = |X_k|^2 / sum(|X_k|^2) is the normalized power
    distribution. A concentrated spectrum (single tone) gives low
    entropy; a flat/broadband spectrum gives high entropy.

    Returns NaN for empty input, zero total power, or invalid values.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)

    if len(magnitudes) == 0:
        return float("nan")

    power = magnitudes ** 2
    total_power = np.sum(power)

    if total_power < 1e-30 or not np.isfinite(total_power):
        return float("nan")

    p = power / total_power

    # Only consider positive probabilities
    p = p[p > 0]

    if len(p) == 0:
        return float("nan")

    return float(-np.sum(p * np.log2(p)))


def compute_spectral_centroid(
    magnitudes: np.ndarray,
    freqs: np.ndarray,
) -> float:
    """
    Spectral centroid: energy-weighted mean frequency.

    centroid = sum(f_k * |X_k|^2) / sum(|X_k|^2)

    Returns NaN for zero total power or invalid inputs.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    if len(magnitudes) == 0 or not np.any(np.isfinite(magnitudes)):
        return float("nan")

    power = magnitudes ** 2
    total_power = np.sum(power)

    if total_power < 1e-30 or not np.isfinite(total_power):
        return float("nan")

    return float(np.sum(freqs * power) / total_power)


def compute_spectral_bandwidth(
    magnitudes: np.ndarray,
    freqs: np.ndarray,
) -> float:
    """
    Spectral bandwidth: energy-weighted spread around the centroid.

    bandwidth = sqrt( sum((f_k - centroid)^2 * |X_k|^2)
                       / sum(|X_k|^2) )

    Returns NaN for zero total power or invalid inputs.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    if len(magnitudes) == 0 or not np.any(np.isfinite(magnitudes)):
        return float("nan")

    power = magnitudes ** 2
    total_power = np.sum(power)

    if total_power < 1e-30 or not np.isfinite(total_power):
        return float("nan")

    centroid = np.sum(freqs * power) / total_power
    variance = np.sum(((freqs - centroid) ** 2) * power) / total_power

    return float(np.sqrt(max(variance, 0.0)))


def compute_spectral_flatness(magnitudes: np.ndarray) -> float:
    """
    Spectral flatness (Wiener entropy): geometric_mean / arithmetic_mean
    of the power spectrum.

    Values near 0 indicate a concentrated/tonal spectrum.
    Values near 1 indicate a flat/white-noise spectrum.

    Returns NaN for zero arithmetic mean or invalid inputs.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)

    if len(magnitudes) == 0 or not np.any(np.isfinite(magnitudes)):
        return float("nan")

    power = magnitudes ** 2

    # Geometric mean requires positive values
    positive_power = power[power > 0]

    if len(positive_power) == 0:
        return float("nan")

    arithmetic_mean = np.mean(power)

    if arithmetic_mean < 1e-30 or not np.isfinite(arithmetic_mean):
        return float("nan")

    geometric_mean = np.exp(np.mean(np.log(positive_power)))

    return float(geometric_mean / arithmetic_mean)


def compute_spectral_rolloff(
    magnitudes: np.ndarray,
    freqs: np.ndarray,
    percentage: float = DEFAULT_ROLLOFF_PERCENTAGE,
) -> float:
    """
    Spectral roll-off frequency: frequency below which the given
    percentage of total spectral energy is contained.

    Parameters
    ----------
    percentage : float
        Percentage of cumulative energy (0-100). Default 85%.

    Returns NaN for zero total power or invalid inputs.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    if len(magnitudes) == 0 or not np.any(np.isfinite(magnitudes)):
        return float("nan")

    power = magnitudes ** 2
    total_power = np.sum(power)

    if total_power < 1e-30 or not np.isfinite(total_power):
        return float("nan")

    cumulative = np.cumsum(power)
    threshold = (percentage / 100.0) * total_power

    idx = int(np.searchsorted(cumulative, threshold))

    # Clamp to valid range
    idx = min(idx, len(freqs) - 1)

    return float(freqs[idx])


# ------------------------------------------------------------------
# Combined extraction for one event
# ------------------------------------------------------------------

def extract_frequency_domain_features(
    signal_window: np.ndarray,
    time_window: np.ndarray,
    fs: float,
    remove_dc: bool = DEFAULT_REMOVE_DC,
    fft_window: str = DEFAULT_FFT_WINDOW,
    min_fft_samples: int = DEFAULT_MIN_FFT_SAMPLES,
    rolloff_percentage: float = DEFAULT_ROLLOFF_PERCENTAGE,
) -> Tuple[Dict[str, float], str]:
    """
    Extract all frequency-domain features from one event segment.

    Parameters
    ----------
    signal_window : np.ndarray
        The filtered signal segment (signal[start:end+1]).
    time_window : np.ndarray
        Corresponding time values.
    fs : float
        Sampling frequency in Hz.
    remove_dc : bool
        Whether to remove the DC component before FFT.
    fft_window : str
        Window function name ('hann', 'hamming', 'blackman', 'none').
    min_fft_samples : int
        Minimum number of samples required for FFT analysis.
    rolloff_percentage : float
        Cumulative energy percentage for roll-off (default 85%).

    Returns
    -------
    (features_dict, failure_reason)
    """
    signal_window = np.asarray(signal_window, dtype=float)
    time_window = np.asarray(time_window, dtype=float)

    # Validate inputs
    if not np.isfinite(fs) or fs <= 0:
        return _nan_freq_features("invalid_sampling_frequency")

    if len(signal_window) < min_fft_samples:
        return _nan_freq_features(
            f"insufficient_samples({len(signal_window)}<{min_fft_samples})"
        )

    # Prepare signal (handle NaN, DC removal, windowing)
    prepared, reason = _prepare_event_signal(
        signal_window, remove_dc, fft_window,
    )
    if reason:
        return _nan_freq_features(reason)

    n = len(prepared)

    # Compute FFT
    fft_values = rfft(prepared)
    magnitudes = np.abs(fft_values)
    freqs = rfftfreq(n, d=1.0 / fs)

    # Validate FFT output
    if len(magnitudes) < 2:
        return _nan_freq_features("fft_output_too_short")

    if not np.any(np.isfinite(magnitudes)):
        return _nan_freq_features("all_nan_fft_magnitudes")

    # Extract features
    dom_freq, dom_mag = compute_dominant_frequency(magnitudes, freqs)

    features = {
        "sampling_frequency_hz": float(fs),
        "num_samples": int(n),
        "dominant_frequency_hz": dom_freq,
        "dominant_magnitude": dom_mag,
        "spectral_energy": compute_spectral_energy(magnitudes),
        "spectral_entropy": compute_spectral_entropy(magnitudes),
        "spectral_centroid_hz": compute_spectral_centroid(
            magnitudes, freqs,
        ),
        "spectral_bandwidth_hz": compute_spectral_bandwidth(
            magnitudes, freqs,
        ),
        "spectral_flatness": compute_spectral_flatness(magnitudes),
        "spectral_rolloff_hz": compute_spectral_rolloff(
            magnitudes, freqs, rolloff_percentage,
        ),
    }

    return features, ""


def _nan_freq_features(reason: str) -> Tuple[Dict[str, float], str]:
    """Return NaN frequency-domain features with failure reason."""
    features = {
        "sampling_frequency_hz": float("nan"),
        "num_samples": 0,
        "dominant_frequency_hz": float("nan"),
        "dominant_magnitude": float("nan"),
        "spectral_energy": float("nan"),
        "spectral_entropy": float("nan"),
        "spectral_centroid_hz": float("nan"),
        "spectral_bandwidth_hz": float("nan"),
        "spectral_flatness": float("nan"),
        "spectral_rolloff_hz": float("nan"),
    }
    return features, reason


__all__ = [
    "estimate_sampling_frequency",
    "compute_dominant_frequency",
    "compute_spectral_energy",
    "compute_spectral_entropy",
    "compute_spectral_centroid",
    "compute_spectral_bandwidth",
    "compute_spectral_flatness",
    "compute_spectral_rolloff",
    "extract_frequency_domain_features",
    "DEFAULT_REMOVE_DC",
    "DEFAULT_FFT_WINDOW",
    "DEFAULT_MIN_FFT_SAMPLES",
    "DEFAULT_ROLLOFF_PERCENTAGE",
]
