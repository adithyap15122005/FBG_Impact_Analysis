"""
Phase 4.5 evaluation - consistency metrics and method comparison.

Usage:
    python evaluate_ensemble.py [--data data/raw] [--filter butterworth]

NOTE
----
No genuine labelled ground truth exists in this repository, so
precision/recall/F1, detection rate and false-positive rate cannot
be computed. This script reports descriptive and consistency
metrics only. Real labels are required for event-level P/R/F1.
"""

import argparse

from pathlib import Path

import pandas as pd

from src.config import DEFAULT_FILTER
from src.evaluation.event_evaluation import (
    compute_consistency_report,
    compare_methods,
)
from src.pipeline.ensemble_pipeline import (
    run_ensemble_datasets,
)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4.5 evaluation"
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
        help="Output directory for evaluation results.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 4.5 EVALUATION")
    print("=" * 70)

    print("\nGround truth: NOT AVAILABLE")
    print(
        "Precision/Recall/F1 cannot be computed. "
        "Reporting descriptive/consistency metrics only."
    )

    dataset_results = run_ensemble_datasets(
        args.data,
        filter_name=args.filter,
    )

    report = compute_consistency_report(dataset_results)

    print("\n" + "=" * 70)
    print("CONSISTENCY REPORT")
    print("=" * 70)

    print(
        f"\nTotal candidate events : "
        f"{report['total_candidate_events']}"
    )

    print(
        f"Total accepted events : "
        f"{report['total_accepted_events']}"
    )

    characteristics = report["event_characteristics"]

    print("\n--- Event characteristics (accepted events) ---")
    for key, value in characteristics.items():
        print(f"  {key}: {value}")

    print("\n--- Detector agreement (accepted events) ---")
    agreement = report["detector_agreement"]
    print(
        f"  method_count_distribution: "
        f"{agreement['method_count_distribution']}"
    )
    print(f"  mean_method_count: {agreement['mean_method_count']}")

    print("\n--- Peak-timing consistency (accepted events) ---")
    timing = report["timing_consistency"]
    for key, value in timing.items():
        print(f"  {key}: {value}")

    print("\n--- Channel consistency ---")
    print(
        report["channel_consistency"].to_string(
            index=False
        )
    )

    print("\n--- Rejection summary ---")
    rejections = report["rejection_summary"]
    if rejections.empty:
        print("  (no rejected events)")
    else:
        print(rejections.to_string(index=False))

    print("\n" + "=" * 70)
    print("METHOD COMPARISON (counts only, no GT available)")
    print("=" * 70)

    comparison = compare_methods(
        dataset_results,
        args.data,
        filter_name=args.filter,
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    output_directory = Path(args.out)

    output_directory.mkdir(parents=True, exist_ok=True)

    consistency_file = (
        output_directory / "consistency_report.csv"
    )

    rows = []

    for key, value in characteristics.items():
        rows.append({
            "metric": f"characteristics_{key}",
            "value": value,
        })

    for key, value in agreement.items():
        rows.append({
            "metric": f"agreement_{key}",
            "value": value,
        })

    for key, value in timing.items():
        rows.append({
            "metric": f"timing_{key}",
            "value": value,
        })

    pd.DataFrame(rows).to_csv(
        consistency_file,
        index=False,
    )

    print(f"\nConsistency report saved to: {consistency_file}")

    comparison_file = (
        output_directory / "method_comparison.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False,
    )

    print(f"Method comparison saved to: {comparison_file}")


if __name__ == "__main__":
    main()
