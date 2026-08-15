from scipy.signal import butter, filtfilt


def butterworth_filter(
    signal,
    cutoff,
    sampling_frequency,
    order=4
):
    """
    Apply a low-pass Butterworth filter.

    Parameters
    ----------
    signal : array-like
        Input wavelength-shift signal.

    cutoff : float
        Cutoff frequency in Hz.

    sampling_frequency : float
        Sampling frequency in Hz.

    order : int
        Filter order.
    """

    nyquist_frequency = sampling_frequency / 2

    normalized_cutoff = (
        cutoff / nyquist_frequency
    )

    b, a = butter(
        order,
        normalized_cutoff,
        btype="low"
    )

    filtered_signal = filtfilt(
        b,
        a,
        signal
    )

    return filtered_signal