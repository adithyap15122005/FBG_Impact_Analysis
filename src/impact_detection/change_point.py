import numpy as np


def detect_change_points(
    signal,
    window=20,
    threshold=3.0
):
    """
    Simple statistical change-point detector.

    Compares the local mean against a
    preceding rolling window.
    """

    values = np.asarray(signal)

    mask = np.zeros(
        len(values),
        dtype=bool
    )

    for i in range(
        window,
        len(values)
    ):

        previous = values[
            i - window:i
        ]

        current = values[
            i:i + 1
        ]

        previous_mean = np.mean(
            previous
        )

        previous_std = np.std(
            previous
        )

        if previous_std == 0:
            continue

        score = abs(
            current[0] -
            previous_mean
        ) / previous_std

        if score > threshold:
            mask[i] = True

    return mask