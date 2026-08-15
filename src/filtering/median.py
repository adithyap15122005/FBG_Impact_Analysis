from scipy.signal import medfilt


def median_filter(
    signal,
    kernel_size=5
):
    """
    Apply a median filter.

    Useful for reducing spike-like noise.
    """

    filtered_signal = medfilt(
        signal,
        kernel_size=kernel_size
    )

    return filtered_signal