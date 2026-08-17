"""
Phase 5 diagnostic plots.

One plot per accepted event showing the FBG2 Savitzky-Golay filtered
signal around the impact with the existing start / peak / recovery
boundaries and the four extracted features annotated (peak shift,
residual shift, rise time, recovery time).

These plots are validation aids only; the feature values come from
src/analysis/phase5_features.py.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

from ..analysis.phase5_features import (
    Phase5Features,
    RESIDUAL_WINDOW_SAMPLES,
)

# Samples shown around the event in the zoomed panel.
PAD_BEFORE_SAMPLES = 20
PAD_AFTER_SAMPLES = 80


def _event_indices(features, time, signal):
    """
    Return the start/peak/end sample indices for the event.
    """
    start_index = int(np.searchsorted(time, features.start_time))
    peak_index = int(np.searchsorted(time, features.peak_time))
    end_index = int(np.searchsorted(time, features.end_time))

    return start_index, peak_index, end_index


def plot_phase5_event(
    features: Phase5Features,
    time: np.ndarray,
    signal: np.ndarray,
    baseline_mean: float,
    output_path,
    title: str = None,
):
    """
    Create a Phase 5 diagnostic plot for one event.

    Parameters
    ----------
    features : Phase5Features
        Extracted features for the event.
    time : array-like
        Time values (seconds).
    signal : array-like
        FBG2 Savitzky-Golay filtered signal.
    baseline_mean : float
        Pre-impact baseline.
    output_path : path-like
        Where to save the PNG.
    title : str, optional
        Plot title (defaults to impact_id).
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    start_index, peak_index, end_index = _event_indices(
        features,
        time,
        signal,
    )

    view_start = max(0, start_index - PAD_BEFORE_SAMPLES)
    view_end = min(
        len(signal),
        end_index + PAD_AFTER_SAMPLES,
    )

    t_view = time[view_start:view_end]
    s_view = signal[view_start:view_end]

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        t_view,
        s_view,
        color="#1f77b4",
        label="FBG2 Savitzky-Golay signal",
        linewidth=1.0,
    )

    ax.axhline(
        baseline_mean,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=f"Pre-impact baseline = {baseline_mean:.6f}",
    )

    # Boundaries
    ax.axvline(
        features.start_time,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.4,
        label="Start",
    )
    ax.axvline(
        features.peak_time,
        color="#d62728",
        linestyle="-",
        linewidth=1.6,
        label="Peak",
    )
    ax.axvline(
        features.end_time,
        color="#9467bd",
        linestyle="--",
        linewidth=1.4,
        label="Recovery/End",
    )

    # Peak shift arrow
    ax.annotate(
        "",
        xy=(features.peak_time, features.peak_value),
        xytext=(features.peak_time, baseline_mean),
        arrowprops=dict(
            arrowstyle="<->",
            color="#ff7f0e",
            linewidth=1.6,
        ),
    )
    ax.text(
        features.peak_time,
        (features.peak_value + baseline_mean) / 2.0,
        f"peak shift\n{features.peak_shift:.6f}",
        color="#ff7f0e",
        fontsize=9,
        ha="left",
        va="center",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    # Rise / recovery time spans
    rise_mid = (features.start_time + features.peak_time) / 2.0
    ax.annotate(
        "",
        xy=(features.peak_time, 0.0),
        xytext=(features.start_time, 0.0),
        arrowprops=dict(
            arrowstyle="<->",
            color="green",
            linewidth=1.4,
        ),
    )
    ax.text(
        rise_mid,
        0.0,
        f"rise {features.rise_time:.2f} s",
        color="green",
        fontsize=9,
        ha="center",
        va="bottom",
    )

    rec_mid = (features.peak_time + features.end_time) / 2.0
    ax.annotate(
        "",
        xy=(features.end_time, 0.0),
        xytext=(features.peak_time, 0.0),
        arrowprops=dict(
            arrowstyle="<->",
            color="#9467bd",
            linewidth=1.4,
        ),
    )
    ax.text(
        rec_mid,
        0.0,
        f"recovery {features.recovery_time:.2f} s",
        color="#9467bd",
        fontsize=9,
        ha="center",
        va="bottom",
    )

    # Residual level
    if np.isfinite(features.post_impact_level):
        residual_time = min(
            features.end_time + RESIDUAL_WINDOW_SAMPLES / 50.0,
            t_view[-1],
        )

        ax.axhline(
            features.post_impact_level,
            color="#e377c2",
            linestyle="--",
            linewidth=1.2,
            label=(
                f"Post-impact level = {features.post_impact_level:.6f} "
                f"(residual {features.residual_shift:+.6f})"
            ),
        )

        ax.annotate(
            "",
            xy=(residual_time, features.post_impact_level),
            xytext=(residual_time, baseline_mean),
            arrowprops=dict(
                arrowstyle="<->",
                color="#e377c2",
                linewidth=1.4,
            ),
        )

    ax.set_title(
        title
        or (
            f"{features.impact_id}  |  peak shift "
            f"{features.peak_shift:.6f}  |  rise "
            f"{features.rise_time:.3f} s  |  recovery "
            f"{features.recovery_time:.3f} s"
        )
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wavelength shift (nm)")

    ax.grid(True, linewidth=0.3)
    ax.legend(loc="upper right", fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
