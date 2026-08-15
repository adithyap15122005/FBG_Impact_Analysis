from pathlib import Path

import pandas as pd


def save_filtered_signals(
    time,
    raw_signal,
    filtered_signals,
    experiment_name,
    channel
):

    output_directory = Path(
        "data/processed/filtered"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_data = pd.DataFrame({
        "time": time.values,

        "raw_wavelength_shift":
            raw_signal.values,

        "moving_average":
            filtered_signals[
                "moving_average"
            ],

        "butterworth":
            filtered_signals[
                "butterworth"
            ],

        "savitzky_golay":
            filtered_signals[
                "savitzky_golay"
            ],

        "median":
            filtered_signals[
                "median"
            ]
    })

    filename = (
        f"{experiment_name}_{channel}_filters.csv"
    )

    output_path = (
        output_directory / filename
    )

    output_data.to_csv(
        output_path,
        index=False
    )

    return output_path