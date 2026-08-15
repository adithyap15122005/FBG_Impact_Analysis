import numpy as np
from scipy.signal import find_peaks


def find_peak(
    signal,
    start_index,
    end_index
):
    """
    Find the strongest absolute peak inside
    a specified region.

    This function is used AFTER a detection
    method has already found a region.
    """

    values = np.asarray(
        signal,
        dtype=float
    )

    region = values[
        start_index:
        end_index + 1
    ]

    if len(region) == 0:
        raise ValueError(
            "Empty signal region."
        )

    local_index = np.argmax(
        np.abs(region)
    )

    peak_index = (
        start_index +
        local_index
    )

    peak_value = values[
        peak_index
    ]

    return (
        peak_index,
        peak_value
    )


def detect_peak_events(
    signal,
    time,
    baseline_samples,
    prominence_multiplier=3.0,
    minimum_distance_samples=5
):
    """
    Independent Peak Detection method.

    Detects both positive and negative peaks.

    A peak is considered significant when
    its prominence is greater than:

        prominence_multiplier × baseline STD

    Parameters
    ----------
    signal : array-like
        Wavelength shift signal.

    time : array-like
        Time values.

    baseline_samples : int
        Number of samples used for baseline.

    prominence_multiplier : float
        Controls peak significance.

    minimum_distance_samples : int
        Minimum distance between peaks.

    Returns
    -------
    dict
        Detected peaks and associated information.
    """

    values = np.asarray(
        signal,
        dtype=float
    )

    time = np.asarray(
        time,
        dtype=float
    )

    # ==================================================
    # BASELINE
    # ==================================================

    baseline = values[
        :baseline_samples
    ]

    baseline_mean = np.mean(
        baseline
    )

    baseline_std = np.std(
        baseline
    )

    # ==================================================
    # REMOVE BASELINE
    # ==================================================

    signal_deviation = (
        values -
        baseline_mean
    )

    # ==================================================
    # PEAK PROMINENCE
    # ==================================================

    prominence = (
        prominence_multiplier *
        baseline_std
    )

    # ==================================================
    # POSITIVE PEAKS
    # ==================================================

    positive_peaks, positive_properties = (
        find_peaks(
            signal_deviation,
            prominence=prominence,
            distance=minimum_distance_samples
        )
    )

    # ==================================================
    # NEGATIVE PEAKS
    # ==================================================

    negative_peaks, negative_properties = (
        find_peaks(
            -signal_deviation,
            prominence=prominence,
            distance=minimum_distance_samples
        )
    )

    # ==================================================
    # COMBINE PEAKS
    # ==================================================

    detected_peaks = []

    # Positive peaks
    for index in positive_peaks:

        detected_peaks.append({

            "index":
                int(index),

            "time":
                float(time[index]),

            "value":
                float(values[index]),

            "type":
                "positive",

            "prominence":
                float(
                    positive_properties[
                        "prominences"
                    ][
                        list(
                            positive_peaks
                        ).index(index)
                    ]
                )
        })

    # Negative peaks
    for index in negative_peaks:

        detected_peaks.append({

            "index":
                int(index),

            "time":
                float(time[index]),

            "value":
                float(values[index]),

            "type":
                "negative",

            "prominence":
                float(
                    negative_properties[
                        "prominences"
                    ][
                        list(
                            negative_peaks
                        ).index(index)
                    ]
                )
        })

    # Sort by time
    detected_peaks.sort(
        key=lambda x: x["index"]
    )

    return {

        "baseline_mean":
            baseline_mean,

        "baseline_std":
            baseline_std,

        "prominence_threshold":
            prominence,

        "positive_peaks":
            positive_peaks,

        "negative_peaks":
            negative_peaks,

        "peaks":
            detected_peaks
    }