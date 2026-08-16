"""
Phase 4.5 runner - multi-method ensemble impact detection.

Usage:
    python run_ensemble.py [--data data/raw] [--filter butterworth]
                           [--out results/ensemble] [--no-plots]

Runs the four-method ensemble detector on every FBG channel of every
dataset, saves structured results (CSV/JSON) and diagnostic plots.
"""

import argparse
import sys

from pathlib import Path

from src.config import DEFAULT_FILTER
from src.pipeline.ensemble_pipeline import (
    generate_diagnostic_plots,
    run_ensemble_datasets,
    save_ensemble_results,
)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4.5 multi-method ensemble impact detection"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/raw",
        help="Directory containing the raw .txt files.",
    )

    parser.add_argument(
        "--filter",
        type=str,
        default=DEFAULT_FILTER,
        choices=[
            "moving_average",
            "butterworth",
            "savitzky_golay",
            "median",
        ],
        help="Filtered signal used for detection.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results/ensemble",
        help="Output directory for results.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip diagnostic plot generation.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 4.5 - MULTI-METHOD ENSEMBLE IMPACT DETECTION")
    print("=" * 70)

    print(f"\nData directory : {args.data}")
    print(f"Filter signal  : {args.filter}")
    print(f"Output         : {args.out}")

    dataset_results = run_ensemble_datasets(
        args.data,
        filter_name=args.filter,
    )

    if not dataset_results:
        print("\nNo datasets were processed successfully.")
        sys.exit(1)

    saved_files = save_ensemble_results(
        dataset_results,
        output_directory=args.out,
    )

    print("\nSaved files:")
    for name, path in sorted(saved_files.items()):
        print(f"  {name}: {path}")

    if not args.no_plots:
        print("\nGenerating diagnostic plots...")

        plots_dir = Path(args.out) / "plots"

        plot_count = 0

        for result in dataset_results:
            generated = generate_diagnostic_plots(
                result,
                plots_directory=plots_dir,
                filter_name=args.filter,
            )
            plot_count += len(generated)

        print(f"  Generated {plot_count} diagnostic plot(s).")

    print("\nPHASE 4.5 COMPLETE")


if __name__ == "__main__":
    main()
