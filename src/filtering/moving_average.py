import pandas as pd


def moving_average(signal, window=5):

    filtered_signal = signal.rolling(
        window=window,
        center=True
    ).mean()

    filtered_signal = (
        filtered_signal
        .bfill()
        .ffill()
    )

    return filtered_signal