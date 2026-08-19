"""
Phase 6 - Time-frequency analysis (STFT) for accepted impact events.

Implements a lightweight Short-Time Fourier Transform for events
with sufficient length. Parameters are adapted to event length to
avoid fixed-window pitfalls.

STFT Parameters
---------------
- nperseg: segment length, chosen adaptively based on the event
  length. Too short loses frequency resolution; too long loses time
  resolution. Default: min(event_length, max(32, event_length // 4)).
- noverlap: 50% overlap by default.
- The minimum event length for STFT is configurable.

Features
--------
- stft_peak_frequency_hz: the frequency with the highest STFT
  magnitude across all time frames (peak of the spectrogram).
- stft_max_energy: maximum frame energy (sum of squared magnitudes
  in the peak frame).

When STFT is not feasible (event too short, invalid data, etc.),
features are set to NaN and stft_valid is False.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import stft as scipy_stft


# ------------------------------------------------------------------
# Configuration defaults
# ------------------------------------------------------------------

DEFAULT_ENABLE_STFT = True
DEFAULT_MIN_STFT_SAMPLES = 16
DEFAULT_STFT_OVERLAP_RATIO = 0.5


# ------------------------------------------------------------------
# Adaptive STFT parameter selection
# ------------------------------------------------------------------

def _choose_stft_params(
    event_length: int,
    fs: float,
    min_stft_samples: int,
    overlap_ratio: float,
) -> Tuple[int, int, str]:
    """
    Choose STFT nperseg and noverlap adapted to the event length.

    Returns (nperseg, noverlap, skip_reason).
    skip_reason is empty when STFT should proceed.
    """
    if event_length < min_stft_samples:
        return (
            0,
            0,
            f"event_too_short({event_length}<{min_stft_samples})",
        )

    if not np.isfinite(fs) or fs <= 0:
        return 0, 0, "invalid_sampling_frequency"

    # Adaptive nperseg: use at most event_length samples, with a
    # minimum of 8 for meaningful frequency content.
    nperseg = min(event_length, max(32, event_length // 4))

    # Ensure nperseg doesn't exceed event length
    nperseg = min(nperseg, event_length)

    # nperseg must be at least 4 for a meaningful STFT
    if nperseg < 4:
        return (
            0,
            0,
            f"nperseg_too_small({nperseg})",
        )

    noverlap = int(nperseg * overlap_ratio)
    noverlap = min(noverlap, nperseg - 1)

    return nperseg, noverlap, ""


# ------------------------------------------------------------------
# STFT feature extraction
# ------------------------------------------------------------------

def extract_stft_features(
    signal_window: np.ndarray,
    fs: float,
    enable_stft: bool = DEFAULT_ENABLE_STFT,
    min_stft_samples: int = DEFAULT_MIN_STFT_SAMPLES,
    overlap_ratio: float = DEFAULT_STFT_OVERLAP_RATIO,
) -> Tuple[Dict[str, float], bool, str]:
    """
    Extract STFT-based time-frequency features from an event segment.

    Parameters
    ----------
    signal_window : np.ndarray
        The filtered signal segment for one event.
    fs : float
        Sampling frequency in Hz.
    enable_stft : bool
        Master switch. If False, all STFT features are NaN.
    min_stft_samples : int
        Minimum event length (samples) to attempt STFT.
    overlap_ratio : float
        Overlap fraction for STFT (default 0.5 = 50%).

    Returns
    -------
    (features_dict, stft_valid, failure_reason)
        features_dict : dict with stft_peak_frequency_hz,
            stft_max_energy.
        stft_valid : bool
            True if STFT was successfully computed.
        failure_reason : str
            Empty on success, otherwise why STFT was skipped.
    """
    signal_window = np.asarray(signal_window, dtype=float)

    if not enable_stft:
        return _nan_stft_features("stft_disabled")

    # Handle NaN in signal
    if not np.any(np.isfinite(signal_window)):
        return _nan_stft_features("all_nan_signal")

    if np.any(~np.isfinite(signal_window)):
        valid_mask = np.isfinite(signal_window)
        if np.sum(valid_mask) < 2:
            return _nan_stft_features("insufficient_finite_samples")
        indices = np.arange(len(signal_window))
        signal_window = np.interp(
            indices,
            indices[valid_mask],
            signal_window[valid_mask],
        )

    nperseg, noverlap, skip_reason = _choose_stft_params(
        len(signal_window), fs, min_stft_samples, overlap_ratio,
    )

    if skip_reason:
        return _nan_stft_features(skip_reason)

    try:
        freqs, times, Zxx = scipy_stft(
            signal_window,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=True,
        )
    except Exception as exc:
        return _nan_stft_features(f"stft_error({exc})")

    magnitudes = np.abs(Zxx)

    if (
        magnitudes.size == 0
        or not np.any(np.isfinite(magnitudes))
    ):
        return _nan_stft_features("empty_or_nan_stft_output")

    # Find the peak across all time-frequency bins
    peak_idx = np.unravel_index(
        int(np.argmax(magnitudes)),
        magnitudes.shape,
    )

    peak_freq_idx = peak_idx[0]
    peak_time_idx = peak_idx[1]

    stft_peak_freq = float(freqs[peak_freq_idx])
    stft_max_mag = float(magnitudes[peak_freq_idx, peak_time_idx])

    # Frame energy at the peak time frame
    peak_frame = magnitudes[:, peak_time_idx]
    stft_max_energy = float(np.sum(peak_frame ** 2))

    features = {
        "stft_peak_frequency_hz": stft_peak_freq,
        "stft_max_energy": stft_max_energy,
    }

    return features, True, ""


def _nan_stft_features(
    reason: str,
) -> Tuple[Dict[str, float], bool, str]:
    """Return NaN STFT features with failure reason."""
    features = {
        "stft_peak_frequency_hz": float("nan"),
        "stft_max_energy": float("nan"),
    }
    return features, False, reason


__all__ = [
    "extract_stft_features",
    "DEFAULT_ENABLE_STFT",
    "DEFAULT_MIN_STFT_SAMPLES",
    "DEFAULT_STFT_OVERLAP_RATIO",
]
