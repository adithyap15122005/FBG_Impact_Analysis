import numpy as np


def calculate_noise_std(signal):
    """
    Calculate the standard deviation of a signal.

    Lower value generally means less variation/noise.
    """

    signal = np.asarray(signal)

    return np.std(signal)


def calculate_noise_rms(signal):
    """
    Calculate RMS of a signal.

    RMS gives the overall magnitude of the signal.
    """

    signal = np.asarray(signal)

    return np.sqrt(
        np.mean(signal ** 2)
    )


def calculate_peak(signal):
    """
    Find the maximum absolute value in the signal.
    """

    signal = np.asarray(signal)

    peak_index = np.argmax(
        np.abs(signal)
    )

    peak_value = signal[
        peak_index
    ]

    return peak_index, peak_value


def calculate_peak_time(
    time,
    signal
):
    """
    Find the time at which the
    maximum absolute signal occurs.
    """

    time = np.asarray(time)
    signal = np.asarray(signal)

    peak_index = np.argmax(
        np.abs(signal)
    )

    peak_time = time[
        peak_index
    ]

    peak_value = signal[
        peak_index
    ]

    return peak_time, peak_value


def calculate_peak_preservation(
    raw_signal,
    filtered_signal
):
    """
    Compare filtered peak amplitude
    with raw peak amplitude.

    NOTE:
    This metric is currently a global
    peak metric. We will replace it
    with impact-region peak preservation
    in Phase 4.
    """

    raw_signal = np.asarray(
        raw_signal
    )

    filtered_signal = np.asarray(
        filtered_signal
    )

    raw_peak = np.max(
        np.abs(raw_signal)
    )

    filtered_peak = np.max(
        np.abs(filtered_signal)
    )

    if raw_peak == 0:

        return 0.0

    preservation = (
        filtered_peak /
        raw_peak
    )

    return preservation


def calculate_peak_time_error(
    time,
    raw_signal,
    filtered_signal
):
    """
    Calculate the difference between
    raw and filtered global peak timing.

    NOTE:
    This is currently only a diagnostic
    metric. Phase 4 will calculate timing
    inside the actual impact region.
    """

    time = np.asarray(time)

    raw_signal = np.asarray(
        raw_signal
    )

    filtered_signal = np.asarray(
        filtered_signal
    )

    raw_peak_index = np.argmax(
        np.abs(raw_signal)
    )

    filtered_peak_index = np.argmax(
        np.abs(filtered_signal)
    )

    raw_peak_time = time[
        raw_peak_index
    ]

    filtered_peak_time = time[
        filtered_peak_index
    ]

    error = abs(
        filtered_peak_time -
        raw_peak_time
    )

    return error


def evaluate_noise_reduction(
    raw_signal,
    filtered_signal,
    baseline_samples
):
    """
    Evaluate noise reduction in the
    baseline region.

    This function accepts both Pandas
    Series and NumPy arrays.
    """

    # -------------------------
    # Convert everything to NumPy
    # -------------------------

    raw_signal = np.asarray(
        raw_signal
    )

    filtered_signal = np.asarray(
        filtered_signal
    )

    # -------------------------
    # Extract baseline
    # -------------------------

    raw_baseline = (
        raw_signal[
            :baseline_samples
        ]
    )

    filtered_baseline = (
        filtered_signal[
            :baseline_samples
        ]
    )

    # -------------------------
    # Noise STD
    # -------------------------

    raw_std = calculate_noise_std(
        raw_baseline
    )

    filtered_std = calculate_noise_std(
        filtered_baseline
    )

    # -------------------------
    # Noise RMS
    # -------------------------

    raw_rms = calculate_noise_rms(
        raw_baseline
    )

    filtered_rms = calculate_noise_rms(
        filtered_baseline
    )

    # -------------------------
    # Noise reduction %
    # -------------------------

    if raw_std == 0:

        noise_reduction_percent = 0.0

    else:

        noise_reduction_percent = (
            (
                raw_std -
                filtered_std
            )
            / raw_std
        ) * 100.0

    return {

        "raw_noise_std":
            raw_std,

        "filtered_noise_std":
            filtered_std,

        "raw_noise_rms":
            raw_rms,

        "filtered_noise_rms":
            filtered_rms,

        "noise_reduction_percent":
            noise_reduction_percent
    }


def evaluate_filter(
    raw_signal,
    filtered_signal,
    time,
    baseline_samples
):
    """
    Full filter evaluation.

    This keeps the old evaluate_filter()
    function available so older code
    doesn't break.
    """

    raw_signal = np.asarray(
        raw_signal
    )

    filtered_signal = np.asarray(
        filtered_signal
    )

    time = np.asarray(
        time
    )

    # -------------------------
    # Baseline
    # -------------------------

    raw_baseline = (
        raw_signal[
            :baseline_samples
        ]
    )

    filtered_baseline = (
        filtered_signal[
            :baseline_samples
        ]
    )

    # -------------------------
    # Noise
    # -------------------------

    raw_noise_std = (
        calculate_noise_std(
            raw_baseline
        )
    )

    filtered_noise_std = (
        calculate_noise_std(
            filtered_baseline
        )
    )

    # -------------------------
    # Peak
    # -------------------------

    peak_time, peak_value = (
        calculate_peak_time(
            time,
            filtered_signal
        )
    )

    # -------------------------
    # Peak preservation
    # -------------------------

    peak_preservation = (
        calculate_peak_preservation(
            raw_signal,
            filtered_signal
        )
    )

    # -------------------------
    # Peak timing error
    # -------------------------

    peak_time_error = (
        calculate_peak_time_error(
            time,
            raw_signal,
            filtered_signal
        )
    )

    return {

        "raw_noise_std":
            raw_noise_std,

        "filtered_noise_std":
            filtered_noise_std,

        "peak_value":
            peak_value,

        "peak_time":
            peak_time,

        "peak_preservation":
            peak_preservation,

        "peak_time_error":
            peak_time_error
    }