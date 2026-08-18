"""
Phase 5 - Impact feature extraction runner.

Consumes the ACCEPTED Peak Detection events produced by the selected
primary analysis (FBG2 + Savitzky-Golay + Peak Detection) and
extracts four features per event:

    1. Peak Shift
    2. Residual Shift
    3. Rise Time
    4. Recovery Time

The signal used is the SAME FBG2 Savitzky-Golay filtered
wavelength-shift signal produced live by the selected pipeline
(src/pipeline/selected_pipeline.py). No old filtered dataset is
loaded. No new detector and no ensemble fusion is used.

Usage:
    python run_phase5.py [--data data/raw] [--out results/phase5]
                         [--max-plots-per-dataset 3] [--no-plots]
"""

import argparse
import sys

from pathlib import Path

import pandas as pd

from src.analysis.phase5_features import (
    extract_phase5_dataset,
)
from src.io.data_loader import load_fbg_data
from src.pipeline.selected_pipeline import (
    detect_selected_events,
    prepare_selected_signal,
)


def process_dataset(
    raw_file,
    max_plots_per_dataset=3,
    plots_directory=None,
):
    """
    Run the selected pipeline on one dataset and extract Phase 5
    features for its accepted events.

    Returns (dataset, features, generated_plots).
    """
    raw_file = Path(raw_file)
    dataset = raw_file.stem

    df = load_fbg_data(raw_file)

    prepared = prepare_selected_signal(df)
    time = prepared["time"]
    signal = prepared["signal"]

    result = detect_selected_events(df, dataset, method="peak")

    features = extract_phase5_dataset(
        result,
        signal,
        time,
    )

    generated = []

    if plots_directory is not None and features:
        from src.visualization.phase5_plots import plot_phase5_event

        plots_directory = Path(plots_directory)
        plots_directory.mkdir(parents=True, exist_ok=True)

        for index, feature in enumerate(
            features[:max_plots_per_dataset]
        ):
            output_path = (
                plots_directory
                / f"{feature.impact_id}_phase5.png"
            )

            plot_phase5_event(
                feature,
                time,
                signal,
                result["baseline_mean"],
                output_path,
            )

            generated.append(output_path)

    return dataset, features, generated


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 impact feature extraction"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/raw",
        help="Directory containing the raw .txt files.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results/phase5",
        help="Output directory for results.",
    )

    parser.add_argument(
        "--max-plots-per-dataset",
        type=int,
        default=3,
        help=(
            "Maximum number of Phase 5 diagnostic plots per dataset. "
            "Use 0 to disable per-dataset plot generation."
        ),
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip diagnostic plot generation.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 5 - IMPACT FEATURE EXTRACTION")
    print("Channel: FBG2 | Filter: Savitzky-Golay | Detector: Peak")
    print("=" * 70)

    print(f"\nData directory : {args.data}")
    print(f"Output         : {args.out}")

    data_directory = Path(args.data)
    output_directory = Path(args.out)

    files = sorted(data_directory.glob("*.txt"))

    output_directory.mkdir(parents=True, exist_ok=True)

    all_records = []
    plot_count = 0
    processed_datasets = 0
    total_accepted = 0

    for raw_file in files:
        print(f"\nProcessing: {raw_file.name}")

        try:
            plots_directory = None

            if not args.no_plots and args.max_plots_per_dataset > 0:
                plots_directory = output_directory / "plots"

            dataset, features, generated = process_dataset(
                raw_file,
                max_plots_per_dataset=args.max_plots_per_dataset,
                plots_directory=plots_directory,
            )

            processed_datasets += 1
            total_accepted += len(features)
            plot_count += len(generated)

            print(
                f"  {dataset}: {len(features)} accepted "
                f"event(s), {len(generated)} plot(s)"
            )

            for feature in features:
                all_records.append(feature.to_dict())
        except Exception as error:
            print(
                f"  ERROR: {raw_file.name}: {error}"
            )

    if not all_records:
        print("\nNo Phase 5 features were extracted.")
        sys.exit(1)

    features_file = (
        output_directory / "phase5_features_all_datasets.csv"
    )

    dataframe = pd.DataFrame(all_records)

    columns = [
    "dataset",
    "fbg",
    "impact_id",
    "start_time",
    "peak_time",
    "end_time",
    "pre_impact_baseline",
    "peak_value",

    "peak_shift",
    "absolute_peak_shift",
    "post_impact_level",
    "residual_shift",

    "rise_time",
    "recovery_time",

    "peak_width",
    "max_slope",
    "rms",
    "signal_energy",
    "peak_to_peak",
    "variance",
    "standard_deviation",
    "entropy",
    "area_under_curve",

    "residual_n_samples",
    "residual_reason",
]

    dataframe = dataframe[columns]

    dataframe.to_csv(features_file, index=False)

    n_residual_nan = int(
        dataframe["residual_shift"].isna().sum()
    )

    print("\n" + "=" * 70)
    print("PHASE 5 SUMMARY")
    print("=" * 70)
    print(f"Datasets processed      : {processed_datasets}")
    print(f"Accepted events         : {total_accepted}")
    print(f"Events with residual    : {total_accepted - n_residual_nan}")
    print(
        f"Events without residual : {n_residual_nan} "
        f"(insufficient post-impact data)"
    )
    print(f"Diagnostic plots        : {plot_count}")
    print(f"\nFeatures saved to: {features_file}")

    print("\nPHASE 5 COMPLETE")


if __name__ == "__main__":
    main()
