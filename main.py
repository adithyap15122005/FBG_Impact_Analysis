from pathlib import Path
import pandas as pd

from src.io.data_loader import load_fbg_data

from src.io.save_processed_data import (
    save_filtered_signals
)

from src.analysis.noise_analysis import (
    evaluate_noise_reduction
)

from src.preprocessing.preprocessing_pipeline import (
    preprocess_signal
)

from src.filtering.filter_comparison import (
    apply_all_filters
)

from src.config import (
    FBG_COLUMNS,
    BASELINE_SAMPLES,
    SAMPLING_FREQUENCY
)

from src.visualization.plots import (
    plot_wavelength_shift,
    plot_filter_comparison
)


def process_one_experiment(file_path):
    """
    Process one experiment completely.

    If this experiment fails, the error is
    returned to the caller instead of stopping
    the entire batch.
    """

    experiment_name = file_path.stem

    print("\n")
    print("=" * 60)
    print(f"PROCESSING: {experiment_name}")
    print("=" * 60)

    # =========================
    # LOAD DATA
    # =========================

    df = load_fbg_data(file_path)

    print(
        f"Number of samples: {len(df)}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    experiment_results = []

    # =========================
    # PROCESS EACH FBG
    # =========================

    for channel in FBG_COLUMNS:

        print("\n----------------------------------------")
        print(f"Processing {channel}")
        print("----------------------------------------")

        # =========================
        # PREPROCESSING
        # =========================

        result = preprocess_signal(
            df[channel],
            BASELINE_SAMPLES
        )

        baseline = result["baseline"]

        wavelength_shift = (
            result["wavelength_shift"]
        )

        print(
            f"Baseline: "
            f"{baseline:.6f} nm"
        )

        # =========================
        # PLOT WAVELENGTH SHIFT
        # =========================

        plot_wavelength_shift(
            df["time"],
            wavelength_shift,
            f"{experiment_name}_{channel}"
        )

        # =========================
        # APPLY FOUR FILTERS
        # =========================

        filtered_signals = apply_all_filters(
            wavelength_shift,
            SAMPLING_FREQUENCY
        )

        print(
            "\nFilters applied:"
        )

        for filter_name in filtered_signals:
            print(f"  ✓ {filter_name}")

        # =========================
        # SAVE FILTERED DATA
        # =========================

        output_path = save_filtered_signals(
            df["time"],
            wavelength_shift,
            filtered_signals,
            experiment_name,
            channel
        )

        print(
            f"\nFiltered data saved to:"
        )

        print(output_path)

        # =========================
        # NOISE EVALUATION
        # =========================

        print(
            "\n========== NOISE EVALUATION =========="
        )

        for (
            filter_name,
            filtered_signal
        ) in filtered_signals.items():

            metrics = evaluate_noise_reduction(
                wavelength_shift,
                filtered_signal,
                BASELINE_SAMPLES
            )

            print(
                f"\n{filter_name}"
            )

            print(
                f"  Raw noise STD: "
                f"{metrics['raw_noise_std']:.8f}"
            )

            print(
                f"  Filtered noise STD: "
                f"{metrics['filtered_noise_std']:.8f}"
            )

            print(
                f"  Noise reduction: "
                f"{metrics['noise_reduction_percent']:.2f}%"
            )

            # Save result for final CSV
            experiment_results.append({

                "experiment":
                    experiment_name,

                "channel":
                    channel,

                "filter":
                    filter_name,

                "baseline_nm":
                    baseline,

                "raw_noise_std":
                    metrics[
                        "raw_noise_std"
                    ],

                "filtered_noise_std":
                    metrics[
                        "filtered_noise_std"
                    ],

                "raw_noise_rms":
                    metrics[
                        "raw_noise_rms"
                    ],

                "filtered_noise_rms":
                    metrics[
                        "filtered_noise_rms"
                    ],

                "noise_reduction_percent":
                    metrics[
                        "noise_reduction_percent"
                    ]
            })

        # =========================
        # FILTER COMPARISON PLOT
        # =========================

        plot_filter_comparison(
            df["time"],
            wavelength_shift,
            filtered_signals,
            f"{experiment_name}_{channel}"
        )

    return experiment_results


def main():

    # =========================
    # RAW DATA DIRECTORY
    # =========================

    raw_directory = Path(
        "data/raw"
    )

    # =========================
    # RESULT DIRECTORY
    # =========================

    results_directory = Path(
        "results/tables"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================
    # FIND ALL DATASETS
    # =========================

    files = sorted(
        raw_directory.glob("*.txt")
    )

    print("\n")
    print("=" * 60)
    print("PHASE 3 - ALL DATASETS")
    print("=" * 60)

    print(
        f"\nFound {len(files)} dataset(s)."
    )

    # =========================
    # STORAGE
    # =========================

    all_results = []

    successful_files = []

    failed_files = []

    # =========================
    # PROCESS EVERY DATASET
    # =========================

    for file_path in files:

        print("\n")
        print(
            f"Starting: {file_path.name}"
        )

        try:

            results = process_one_experiment(
                file_path
            )

            all_results.extend(
                results
            )

            successful_files.append(
                file_path.name
            )

            print(
                f"\n✓ COMPLETED: "
                f"{file_path.name}"
            )

        except Exception as error:

            # ---------------------------------
            # IMPORTANT:
            # Don't stop the entire pipeline.
            # ---------------------------------

            failed_files.append({

                "file":
                    file_path.name,

                "error":
                    str(error)
            })

            print(
                f"\n✗ FAILED: "
                f"{file_path.name}"
            )

            print(
                f"Reason: {error}"
            )

            print(
                "Skipping this file and "
                "continuing with the next dataset..."
            )

    # =========================
    # SAVE ALL METRICS
    # =========================

    if all_results:

        results_df = pd.DataFrame(
            all_results
        )

        output_file = (
            results_directory /
            "phase3_filter_evaluation.csv"
        )

        results_df.to_csv(
            output_file,
            index=False
        )

        print("\n")
        print(
            "Filter evaluation saved to:"
        )

        print(output_file)

    # =========================
    # SAVE FAILED FILE REPORT
    # =========================

    if failed_files:

        failed_df = pd.DataFrame(
            failed_files
        )

        failed_file = (
            results_directory /
            "phase3_failed_files.csv"
        )

        failed_df.to_csv(
            failed_file,
            index=False
        )

        print("\n")
        print(
            "Failed-file report saved to:"
        )

        print(failed_file)

    # =========================
    # FINAL SUMMARY
    # =========================

    print("\n")
    print("=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)

    print(
        f"\nTotal datasets found: "
        f"{len(files)}"
    )

    print(
        f"Successfully processed: "
        f"{len(successful_files)}"
    )

    print(
        f"Failed/skipped: "
        f"{len(failed_files)}"
    )

    if successful_files:

        print("\nSuccessful datasets:")

        for file_name in successful_files:

            print(
                f"  ✓ {file_name}"
            )

    if failed_files:

        print("\nFailed datasets:")

        for item in failed_files:

            print(
                f"  ✗ {item['file']}"
            )

            print(
                f"    {item['error']}"
            )

    print("\n")
    print("=" * 60)


if __name__ == "__main__":
    main()