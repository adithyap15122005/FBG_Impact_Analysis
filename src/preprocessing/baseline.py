def calculate_baseline(signal, baseline_samples):
    """
    Calculate the baseline wavelength.

    The baseline is the average of the first
    baseline_samples.
    """

    baseline = signal.iloc[
        :baseline_samples
    ].mean()

    return baseline


def baseline_correct(signal, baseline_samples):
    """
    Subtract the baseline from the signal.

    Returns:
        corrected_signal
        baseline
    """

    baseline = calculate_baseline(
        signal,
        baseline_samples
    )

    corrected_signal = signal - baseline

    return corrected_signal, baseline