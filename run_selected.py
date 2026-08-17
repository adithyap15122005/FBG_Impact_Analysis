"""
Selected primary analysis runner.

Selected methodology:
    Channel: FBG2
    Filter:  Savitzky-Golay
    Detector: chosen from the data by method selection

The detector is NOT hardcoded: before processing, all four detectors
(peak, threshold, derivative, change_point) are evaluated on the
actual FBG2 + Savitzky-Golay results across datasets, and the best
one is selected by a documented composite criterion (see
src/analysis/method_selection.py). An optional --method flag
overrides the selection.

This runner processes ONLY the selected single detector through the
existing preprocessing, wavelength-shift, Savitzky-Golay filtering,
detection and impact boundary refinement. It does NOT use the
multi-detector ensemble fusion layer, which remains available via
run_ensemble.py.

Usage:
    python run_selected.py [--data data/raw] [--out results/selected]
                           [--method peak] [--no-plots]
"""

import argparse
import sys

from pathlib import Path

from src.analysis.method_selection import select_best_method
from src.pipeline.selected_pipeline import (
    generate_selected_plots,
    run_selected_datasets,
    save_selected_results,
)


def main():
    parser = argparse.ArgumentParser(
        description="Selected primary analysis "
        "(FBG2 + Savitzky-Golay + best single detector)"
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
        default="results/selected",
        help="Output directory for results.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["peak", "threshold", "derivative", "change_point"],
        help="Override the data-driven detector selection.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip diagnostic plot generation.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SELECTED PRIMARY ANALYSIS")
    print("Channel: FBG2 | Filter: Savitzky-Golay")
    print("=" * 70)

    print(f"\nData directory : {args.data}")
    print(f"Output         : {args.out}")

    print("\n--- Detector selection (based on results) ---")

    if args.method:
        selected_method = args.method
        print(f"  Using --method override: {selected_method}")
    else:
        selected_method, comparison = select_best_method(
            args.data,
            filter_name="savitzky_golay",
        )

        print(comparison.round(3).to_string(index=False))
        print(f"\n  Selected method: {selected_method}")

    print(f"\nDetector       : {selected_method}")

    dataset_results = run_selected_datasets(
        args.data,
        method=selected_method,
        output_directory=args.out,
    )

    if not dataset_results:
        print("\nNo datasets were processed successfully.")
        sys.exit(1)

    saved_files = save_selected_results(
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
            generated = generate_selected_plots(
                result,
                plots_directory=plots_dir,
            )
            plot_count += len(generated)

        print(f"  Generated {plot_count} diagnostic plot(s).")

    print("\nSELECTED ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
