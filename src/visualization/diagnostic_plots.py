"""
Diagnostic visualization for the Phase 4.5 ensemble detector.

The plot makes it possible to see WHY an event was accepted or
rejected: it overlays the detections of every individual method,
the fused event region, and the start/peak/end boundaries.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

from ..impact_detection.ensemble_event import ImpactEvent

METHOD_COLORS = {
    "threshold": "#d62728",
    "peak": "#2ca02c",
    "derivative": "#ff7f0e",
    "change_point": "#9467bd",
}


def _shade_regions(ax, time, regions, color, label, alpha=0.25):
    """Shade a list of (start, end) regions on an axis."""
    if not regions:
        return

    labeled = False

    for start_index, end_index in regions:
        if not labeled:
            ax.axvspan(
                time[start_index],
                time[end_index],
                color=color,
                alpha=alpha,
                label=label,
            )
            labeled = True
        else:
            ax.axvspan(
                time[start_index],
                time[end_index],
                color=color,
                alpha=alpha,
            )


def plot_ensemble_diagnostic(
    time,
    signal,
    detections,
    event: ImpactEvent,
    baseline_mean,
    baseline_std,
    output_path,
    title=None,
):
    """
    Create a diagnostic plot for one fused event.

    Parameters
    ----------
    time : array-like
        Time values.
    signal : array-like
        Filtered signal.
    detections : dict
        Mapping: detector name -> list of (start, end) regions.
    event : ImpactEvent
        The fused event to highlight (accepted or rejected).
    baseline_mean : float
        Baseline mean.
    baseline_std : float
        Baseline std.
    output_path : path-like
        Where to save the PNG.
    title : str, optional
        Plot title.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    threshold_multiplier = 4.0
    if "threshold" in detections:
        from ..impact_detection.threshold_detection import (
            calculate_threshold,
        )

        try:
            _, _, threshold = calculate_threshold(
                signal,
                100,
                threshold_multiplier,
            )
        except Exception:
            threshold = threshold_multiplier * baseline_std
    else:
        threshold = threshold_multiplier * baseline_std

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_signal = axes[0]
    ax_derivative = axes[1]

    # ----------------------------------------------------------
    # Top panel: signal + detections + event
    # ----------------------------------------------------------
    ax_signal.plot(
        time,
        signal,
        color="#1f77b4",
        label="Filtered signal",
        linewidth=1.0,
    )

    ax_signal.axhline(
        baseline_mean,
        color="gray",
        linestyle=":",
        label="Baseline",
    )

    ax_signal.axhline(
        baseline_mean + threshold,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="Threshold",
    )

    ax_signal.axhline(
        baseline_mean - threshold,
        color="gray",
        linestyle="--",
        linewidth=0.8,
    )

    for method, regions in detections.items():
        if not regions:
            continue

        color = METHOD_COLORS.get(method, "#555555")

        _shade_regions(
            ax_signal,
            time,
            regions,
            color,
            f"{method} detection",
        )

    # Fused event region
    ax_signal.axvspan(
        time[event.start_index],
        time[event.end_index],
        color="#ffd700",
        alpha=0.35,
        label="Fused event",
    )

    ax_signal.axvline(
        time[event.start_index],
        color="#2ca02c",
        linestyle="--",
        label="Event start",
    )

    ax_signal.axvline(
        time[event.peak_index],
        color="#d62728",
        linestyle="-",
        linewidth=2.0,
        label="Event peak",
    )

    ax_signal.axvline(
        time[event.end_index],
        color="#9467bd",
        linestyle="--",
        label="Event end",
    )

    ax_signal.scatter(
        [time[event.peak_index]],
        [signal[event.peak_index]],
        color="black",
        marker="x",
        s=80,
        zorder=5,
        label=f"Peak value = {event.peak_value:.6f}",
    )

    status = (
        "ACCEPTED"
        if event.accepted
        else f"REJECTED ({event.rejection_reason})"
    )

    ax_signal.set_title(
        title
        or (
            f"{event.dataset} {event.channel} - Event "
            f"{event.event_id} [{status}] "
            f"(methods={event.method_count}, "
            f"evidence={event.evidence_score:.2f})"
        )
    )

    ax_signal.set_ylabel("Wavelength shift (nm)")
    ax_signal.legend(
        loc="upper right",
        fontsize="small",
        ncol=2,
    )

    # ----------------------------------------------------------
    # Bottom panel: derivative
    # ----------------------------------------------------------
    derivative = np.gradient(signal, time)

    ax_derivative.plot(
        time,
        derivative,
        color="#333333",
        linewidth=0.8,
        label="d(signal)/dt",
    )

    derivative_baseline = derivative[:100]
    derivative_std = np.std(derivative_baseline)

    for limit in (
        3.0 * derivative_std,
        -3.0 * derivative_std,
    ):
        ax_derivative.axhline(
            limit,
            color="gray",
            linestyle=":",
            linewidth=0.8,
        )

    ax_derivative.axvspan(
        time[event.start_index],
        time[event.end_index],
        color="#ffd700",
        alpha=0.35,
    )

    ax_derivative.set_ylabel("Derivative (nm/s)")
    ax_derivative.set_xlabel("Time (s)")
    ax_derivative.grid(True, linewidth=0.3)

    ax_signal.grid(True, linewidth=0.3)

    fig.tight_layout()

    fig.savefig(output_path, dpi=150)

    plt.close(fig)

    return output_path
