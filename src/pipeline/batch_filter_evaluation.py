from pathlib import Path

import pandas as pd

from src.pipeline.filter_evaluation_pipeline import (
    evaluate_experiment
)


def evaluate_all_experiments(
    data_directory
):
    """
    Evaluate all experiment files
    in the raw data directory.
    """

    data_directory = Path(
        data_directory
    )

    all_results = []

    files = sorted(
        data_directory.glob("*.txt")
    )

    print(
        f"Found {len(files)} experiment files."
    )

    for file_path in files:

        print(
            f"\nProcessing: {file_path.name}"
        )

        try:

            results = evaluate_experiment(
                file_path
            )

            all_results.extend(
                results
            )

            print("Completed.")

        except Exception as error:

            print(
                f"ERROR processing "
                f"{file_path.name}: {error}"
            )

    return pd.DataFrame(all_results)