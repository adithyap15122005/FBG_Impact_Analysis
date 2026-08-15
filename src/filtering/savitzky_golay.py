from scipy.signal import savgol_filter


def savitzky_golay_filter(
    signal,
    window=11,
    polyorder=3
):
    """
    Apply Savitzky-Golay filtering.
    """

    filtered_signal = savgol_filter(
        signal,
        window_length=window,
        polyorder=polyorder
    )

    return filtered_signal