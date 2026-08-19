"""
Phase 6 - Multi-domain diagnostic plots.

Generates per-event diagnostic plots showing:
1. Time domain: event signal with start/peak/end boundaries.
2. Frequency domain: one-sided FFT magnitude with dominant frequency marked.
3. Summary panel: key feature values (RMS, dominant frequency, spectral
   energy, spectral entropy, spectral centroid).

Also generates aggregate plots:
- Dominant frequency distribution
- Spectral entropy distribution
- Dominant frequency vs peak shift
- Spectral energy vs peak shift

Plots are only generated when there is analytical value. The number
of per-event plots is configurable.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq

from ..analysis.phase6_multidomain import Phase6Features

# Samples shown around the event in the zoomed panel.
PAD_BEFORE_SAMPLES = 20
PAD_AFTER_SAMPLES = 80


def plot_phase6_event(
    features: Phase6Features,
    time: np.ndarray,
    signal: np.ndarray,
    baseline_mean: float,
    output_path,
    title: str = None,
):
    """
    Create a 3-panel Phase 6 diagnostic plot for one event.

    Panel 1 (top): time-domain signal with boundaries.
    Panel 2 (middle): one-sided FFT magnitude with dominant frequency.
    Panel 3 (bottom): summary text box with key features.

    Parameters
    ----------
    features : Phase6Features
        Extracted features for the event.
    time : array-like
        Full time values (seconds).
    signal : array-like
        Full filtered signal.
    baseline_mean : float
        Pre-impact baseline.
    output_path : path-like
        Where to save the PNG.
    title : str, optional
        Plot title.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    # Find event indices
    start_idx = int(np.searchsorted(time, features.start_time))
    end_idx = int(np.searchsorted(time, features.end_time))

    view_start = max(0, start_idx - PAD_BEFORE_SAMPLES)
    view_end = min(len(signal), end_idx + PAD_AFTER_SAMPLES)

    t_view = time[view_start:view_end]
    s_view = signal[view_start:view_end]

    # Event segment for FFT
    sig_seg = signal[start_idx:end_idx + 1]
    n_seg = len(sig_seg)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[3, 3, 2])

    # --- Panel 1: Time domain ---
    ax_time = axes[0]
    ax_time.plot(t_view, s_view, color="#1f77b4", linewidth=1.0, label="Filtered signal")

    ax_time.axhline(baseline_mean, color="gray", linestyle=":", linewidth=1.0,
                    label=f"Baseline = {baseline_mean:.6f}")

    ax_time.axvline(features.start_time, color="#2ca02c", linestyle="--",
                    linewidth=1.2, label="Start")
    ax_time.axvline(features.peak_time, color="#d62728", linestyle="-",
                    linewidth=1.4, label="Peak")
    ax_time.axvline(features.end_time, color="#9467bd", linestyle="--",
                    linewidth=1.2, label="End")

    ax_time.set_ylabel("Wavelength shift (nm)")
    ax_time.set_title(
        title or f"{features.impact_id}  |  Phase 6 Multi-Domain Analysis",
        fontsize=11,
    )
    ax_time.legend(loc="upper right", fontsize="small")
    ax_time.grid(True, linewidth=0.3)

    # --- Panel 2: Frequency domain ---
    ax_freq = axes[1]

    if n_seg >= 8 and np.isfinite(features.sampling_frequency_hz):
        fs = features.sampling_frequency_hz

        # Remove mean and apply Hann window for visualization
        seg = sig_seg - np.mean(sig_seg)
        seg = seg * np.hanning(n_seg)

        fft_vals = rfft(seg)
        mag = np.abs(fft_vals)
        freqs = rfftfreq(n_seg, d=1.0 / fs)

        # Skip DC for display
        freqs_plot = freqs[1:]
        mag_plot = mag[1:]

        if len(mag_plot) > 0:
            ax_freq.plot(freqs_plot, mag_plot, color="#1f77b4", linewidth=0.8)

            # Mark dominant frequency
            if np.isfinite(features.dominant_frequency_hz):
                ax_freq.axvline(
                    features.dominant_frequency_hz,
                    color="#d62728",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Dominant: {features.dominant_frequency_hz:.2f} Hz",
                )

            ax_freq.legend(loc="upper right", fontsize="small")

    ax_freq.set_ylabel("FFT Magnitude")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_title("One-sided FFT Magnitude", fontsize=10)
    ax_freq.grid(True, linewidth=0.3)

    # --- Panel 3: Summary text ---
    ax_text = axes[2]
    ax_text.axis("off")

    summary_lines = [
        f"RMS: {features.rms:.6f}",
        f"Dominant Freq: {features.dominant_frequency_hz:.2f} Hz",
        f"Spectral Energy: {features.spectral_energy:.6e}",
        f"Spectral Entropy: {features.spectral_entropy:.3f}",
        f"Spectral Centroid: {features.spectral_centroid_hz:.2f} Hz",
        f"Spectral Bandwidth: {features.spectral_bandwidth_hz:.2f} Hz",
        f"Crest Factor: {features.crest_factor:.3f}" if np.isfinite(features.crest_factor) else "Crest Factor: NaN",
        f"Skewness: {features.skewness:.3f}" if np.isfinite(features.skewness) else "Skewness: NaN",
        f"Kurtosis: {features.kurtosis:.3f}" if np.isfinite(features.kurtosis) else "Kurtosis: NaN",
    ]

    if features.stft_valid:
        summary_lines.append(
            f"STFT Peak Freq: {features.stft_peak_frequency_hz:.2f} Hz"
        )
    else:
        summary_lines.append("STFT: not available")

    text_block = "\n".join(summary_lines)

    ax_text.text(
        0.05, 0.95, text_block,
        transform=ax_text.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_dominant_frequency_distribution(
    all_features: list,
    output_path,
):
    """
    Histogram of dominant frequencies across all events.
    """
    features_list = [f for f in all_features
                     if np.isfinite(f.dominant_frequency_hz)]

    if not features_list:
        return None

    dom_freqs = [f.dominant_frequency_hz for f in features_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dom_freqs, bins=min(30, max(5, len(dom_freqs) // 3)),
            color="#1f77b4", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Dominant Frequency (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("Dominant Frequency Distribution Across All Events")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_spectral_entropy_distribution(
    all_features: list,
    output_path,
):
    """
    Histogram of spectral entropy across all events.
    """
    features_list = [f for f in all_features
                     if np.isfinite(f.spectral_entropy)]

    if not features_list:
        return None

    entropies = [f.spectral_entropy for f in features_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(entropies, bins=min(30, max(5, len(entropies) // 3)),
            color="#ff7f0e", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Spectral Entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title("Spectral Entropy Distribution Across All Events")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_dominant_frequency_vs_peak_shift(
    all_features: list,
    output_path,
):
    """
    Scatter plot: dominant frequency vs peak shift.
    """
    features_list = [
        f for f in all_features
        if np.isfinite(f.dominant_frequency_hz)
        and np.isfinite(f.peak_shift)
    ]

    if not features_list:
        return None

    dom_freqs = [f.dominant_frequency_hz for f in features_list]
    peak_shifts = [f.peak_shift for f in features_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(dom_freqs, peak_shifts, color="#1f77b4", alpha=0.7,
               edgecolors="white", linewidth=0.5, s=40)
    ax.set_xlabel("Dominant Frequency (Hz)")
    ax.set_ylabel("Peak Shift (nm)")
    ax.set_title("Dominant Frequency vs Peak Shift")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_spectral_energy_vs_peak_shift(
    all_features: list,
    output_path,
):
    """
    Scatter plot: spectral energy vs peak shift.
    """
    features_list = [
        f for f in all_features
        if np.isfinite(f.spectral_energy)
        and np.isfinite(f.peak_shift)
    ]

    if not features_list:
        return None

    energy = [f.spectral_energy for f in features_list]
    peak_shifts = [f.peak_shift for f in features_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(energy, peak_shifts, color="#2ca02c", alpha=0.7,
               edgecolors="white", linewidth=0.5, s=40)
    ax.set_xlabel("Spectral Energy")
    ax.set_ylabel("Peak Shift (nm)")
    ax.set_title("Spectral Energy vs Peak Shift")
    ax.set_xscale("log")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


__all__ = [
    "plot_phase6_event",
    "plot_dominant_frequency_distribution",
    "plot_spectral_entropy_distribution",
    "plot_dominant_frequency_vs_peak_shift",
    "plot_spectral_energy_vs_peak_shift",
]
