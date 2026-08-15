import pandas as pd


def clean_signal(signal):
    """
    Clean one FBG signal.

    Missing values are interpolated.
    Remaining edge values are filled.
    """

    signal = pd.to_numeric(
        signal,
        errors="coerce"
    )

    # Interpolate missing values
    signal = signal.interpolate(
        method="linear"
    )

    # Handle missing values at beginning/end
    signal = signal.bfill()
    signal = signal.ffill()

    return signal