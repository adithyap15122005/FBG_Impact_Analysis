from .moving_average import moving_average
from .butterworth import butterworth_filter
from .savitzky_golay import savitzky_golay_filter
from .median import median_filter


def apply_all_filters(
    signal,
    sampling_frequency
):
    """
    Apply all four filters to a signal.
    """

    results = {}

    # Moving Average
    results["moving_average"] = moving_average(
        signal,
        window=5
    )

    # Butterworth
    results["butterworth"] = butterworth_filter(
        signal,
        cutoff=5,
        sampling_frequency=sampling_frequency,
        order=4
    )

    # Savitzky-Golay
    results["savitzky_golay"] = savitzky_golay_filter(
        signal,
        window=11,
        polyorder=3
    )

    # Median
    results["median"] = median_filter(
        signal,
        kernel_size=5
    )

    return results