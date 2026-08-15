import numpy as np


def calculate_derivative(
    signal,
    time
):
    """
    Calculate the first derivative of the signal.

    Mathematical form:

        derivative = d(signal) / dt

    Parameters
    ----------
    signal : array-like
        Wavelength-shift signal.

    time : array-like
        Time values corresponding to the signal.

    Returns
    -------
    numpy.ndarray
        First derivative of the signal.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    time = np.asarray(
        time,
        dtype=float
    )

    if len(signal) != len(time):
        raise ValueError(
            "Signal and time must have "
            "the same length."
        )

    if len(signal) < 2:
        raise ValueError(
            "At least two samples are "
            "required to calculate derivative."
        )

    # Check that time is increasing
    if np.any(
        np.diff(time) <= 0
    ):
        raise ValueError(
            "Time values must be strictly increasing."
        )

    derivative = np.gradient(
        signal,
        time
    )

    return derivative


def apply_persistence(
    mask,
    minimum_consecutive_samples=2
):
    """
    Require a detection to remain active
    for a minimum number of consecutive samples.

    This prevents isolated noisy derivative
    spikes from being classified as events.

    Example:

        Input:

        0 0 1 0 0 1 1 0

        persistence = 2

        Output:

        0 0 0 0 0 1 1 0

    Parameters
    ----------
    mask : array-like of bool
        Initial detection mask.

    minimum_consecutive_samples : int
        Minimum number of consecutive True
        samples required.

    Returns
    -------
    numpy.ndarray
        Cleaned Boolean detection mask.
    """

    mask = np.asarray(
        mask,
        dtype=bool
    )

    if minimum_consecutive_samples < 1:
        raise ValueError(
            "minimum_consecutive_samples "
            "must be >= 1."
        )

    result = np.zeros_like(
        mask,
        dtype=bool
    )

    count = 0

    for index, value in enumerate(mask):

        if value:

            count += 1

        else:

            count = 0

        if (
            count >=
            minimum_consecutive_samples
        ):

            start_index = (
                index -
                minimum_consecutive_samples +
                1
            )

            result[
                start_index:
                index + 1
            ] = True

    return result


def detect_derivative_events(
    signal,
    time,
    baseline_samples,
    threshold_multiplier=3.0
):
    """
    Detect sudden changes in wavelength shift
    using the first derivative.

    The derivative during the initial baseline
    region is used to estimate normal derivative
    noise.

    A point is detected when:

        |derivative - baseline_mean|
        >
        threshold_multiplier × baseline_std

    Parameters
    ----------
    signal : array-like
        Wavelength-shift signal.

    time : array-like
        Time values.

    baseline_samples : int
        Number of initial samples representing
        the baseline.

    threshold_multiplier : float
        Number of standard deviations used for
        derivative event detection.

    Returns
    -------
    dict
        Contains derivative, baseline statistics,
        threshold, detection mask and diagnostics.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    time = np.asarray(
        time,
        dtype=float
    )

    # ==================================================
    # VALIDATION
    # ==================================================

    if len(signal) != len(time):
        raise ValueError(
            "Signal and time must have "
            "the same length."
        )

    if baseline_samples <= 1:
        raise ValueError(
            "baseline_samples must be > 1."
        )

    if baseline_samples >= len(signal):
        raise ValueError(
            "baseline_samples must be smaller "
            "than the number of signal samples."
        )

    if threshold_multiplier <= 0:
        raise ValueError(
            "threshold_multiplier must be positive."
        )

    # ==================================================
    # 1. CALCULATE DERIVATIVE
    # ==================================================

    derivative = calculate_derivative(
        signal,
        time
    )

    # ==================================================
    # 2. BASELINE DERIVATIVE
    # ==================================================

    baseline_derivative = (
        derivative[
            :baseline_samples
        ]
    )

    baseline_mean = np.mean(
        baseline_derivative
    )

    baseline_std = np.std(
        baseline_derivative
    )

    # ==================================================
    # 3. DERIVATIVE DEVIATION
    # ==================================================

    derivative_deviation = np.abs(
        derivative -
        baseline_mean
    )

    # ==================================================
    # 4. ADAPTIVE THRESHOLD
    # ==================================================

    threshold = (
        threshold_multiplier *
        baseline_std
    )

    # ==================================================
    # 5. DETECTION MASK
    # ==================================================

    mask = (
        derivative_deviation >
        threshold
    )

    # ==================================================
    # 6. DIAGNOSTIC VALUES
    # ==================================================

    max_derivative = np.max(
        np.abs(
            derivative
        )
    )

    max_derivative_deviation = np.max(
        derivative_deviation
    )

    # Location of maximum derivative
    max_derivative_index = np.argmax(
        np.abs(
            derivative
        )
    )

    max_derivative_time = time[
        max_derivative_index
    ]

    # Number of samples exceeding threshold
    detected_samples = np.sum(
        mask
    )

    return {

        # ----------------------------------
        # Main derivative
        # ----------------------------------

        "derivative":
            derivative,

        # ----------------------------------
        # Baseline statistics
        # ----------------------------------

        "baseline_mean":
            baseline_mean,

        "baseline_std":
            baseline_std,

        # ----------------------------------
        # Threshold
        # ----------------------------------

        "threshold":
            threshold,

        # ----------------------------------
        # Detection
        # ----------------------------------

        "mask":
            mask,

        # ----------------------------------
        # Diagnostics
        # ----------------------------------

        "max_derivative":
            max_derivative,

        "max_derivative_deviation":
            max_derivative_deviation,

        "max_derivative_index":
            max_derivative_index,

        "max_derivative_time":
            max_derivative_time,

        "detected_samples":
            int(detected_samples)
    }