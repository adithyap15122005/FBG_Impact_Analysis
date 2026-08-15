from pathlib import Path

import pandas as pd

from src.io.data_loader import load_fbg_data

from src.preprocessing.preprocessing_pipeline import (
    preprocess_signal
)

from src.filtering.filter_comparison import (
    apply_all_filters
)

from src.analysis.noise_analysis import (
    evaluate_filter
)

from src.config import (
    FBG_COLUMNS,
    BASELINE_SAMPLES,
    SAMPLING_FREQUENCY
)


def evaluate_experiment(file_path):
    """
    Evaluate all four filters for one experiment.

    Returns a list of metric dictionaries.
    """

    df = load_fbg_data(file_path)

    results = []

    for channel in FBG_COLUMNS:

        # -------------------------
        # Phase 3 preprocessing
        # -------------------------

        preprocessing_result = preprocess_signal(
            df[channel],
            BASELINE_SAMPLES
        )

        wavelength_shift = (
            preprocessing_result[
                "wavelength_shift"
            ]
        )

        # -------------------------
        # Apply all four filters
        # -------------------------

        filtered_signals = apply_all_filters(
            wavelength_shift,
            SAMPLING_FREQUENCY
        )

        # -------------------------
        # Evaluate each filter
        # -------------------------

        for filter_name, filtered_signal in (
            filtered_signals.items()
        ):

            metrics = evaluate_filter(
                wavelength_shift,
                filtered_signal,
                df["time"],
                BASELINE_SAMPLES
            )

            metrics["experiment"] = (
                Path(file_path).stem
            )

            metrics["channel"] = channel

            metrics["filter"] = filter_name

            results.append(metrics)

    return results