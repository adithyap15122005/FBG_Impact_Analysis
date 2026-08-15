import numpy as np


def calculate_baseline_statistics(
    signal,
    baseline_samples
):
    """
    Calculate baseline mean and standard deviation.
    """

    values = np.asarray(signal)

    baseline = values[
        :baseline_samples
    ]

    baseline_mean = np.mean(
        baseline
    )

    baseline_std = np.std(
        baseline
    )

    return (
        baseline_mean,
        baseline_std
    )


def calculate_threshold(
    signal,
    baseline_samples,
    threshold_multiplier=4
):
    """
    Adaptive threshold based on baseline noise.

    threshold = k × baseline_std
    """

    (
        baseline_mean,
        baseline_std
    ) = calculate_baseline_statistics(
        signal,
        baseline_samples
    )

    threshold = (
        threshold_multiplier *
        baseline_std
    )

    return (
        baseline_mean,
        baseline_std,
        threshold
    )


def detect_threshold_crossings(
    signal,
    baseline_samples,
    threshold_multiplier=4
):
    """
    Identify samples that significantly
    deviate from the baseline.
    """

    values = np.asarray(signal)

    (
        baseline_mean,
        baseline_std,
        threshold
    ) = calculate_threshold(
        values,
        baseline_samples,
        threshold_multiplier
    )

    deviation = np.abs(
        values - baseline_mean
    )

    mask = (
        deviation > threshold
    )

    return {
        "baseline_mean":
            baseline_mean,

        "baseline_std":
            baseline_std,

        "threshold":
            threshold,

        "mask":
            mask
    }