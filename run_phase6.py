"""
Phase 6 - Multi-Domain Signal Analysis runner.

Consumes the ACCEPTED Peak Detection events produced by the selected
primary analysis (FBG2 + Savitzky-Golay + Peak Detection) and
extracts multi-domain features per event:

    1. Time-domain statistical and shape features
    2. Frequency-domain FFT features
    3. Time-frequency STFT features

Reuses the same FBG2 Savitzky-Golay filtered signal, boundaries
and Phase 5 context from the selected pipeline.

Usage:
    python run_phase6.py [--data data/raw] [--out results/phase6]
                         [--max-plots-per-dataset 3] [--no-plots]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.phase5_features import extract_phase5_dataset
from src.analysis.phase6_multidomain import (
    PHASE6_CSV_COLUMNS,
    Phase6Config,
    extract_phase6_dataset,
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
    config=None,
):
    """
    Run the selected pipeline on one dataset, extract Phase 5 context,
    and extract Phase 6 multi-domain features.

    Returns (dataset, phase6_features, generated_plots).
    """
    raw_file = Path(raw_file)
    dataset = raw_file.stem

    df = load_fbg_data(raw_file)

    prepared = prepare_selected_signal(df)
    time = prepared["time"]
    signal = prepared["signal"]

    result = detect_selected_events(df, dataset, method="peak")

    # Phase 5 context (for reuse in Phase 6)
    phase5_features = extract_phase5_dataset(
        result,
        signal,
        time,
    )

    # Phase 6 multi-domain features
    phase6_features = extract_phase6_dataset(
        result,
        signal,
        time,
        phase5_features=phase5_features,
        config=config,
    )

    generated = []

    if plots_directory is not None and phase6_features:
        from src.visualization.phase6_plots import (
            plot_phase6_event,
        )

        plots_directory = Path(plots_directory)
        plots_directory.mkdir(parents=True, exist_ok=True)

        for index, feature in enumerate(
            phase6_features[:max_plots_per_dataset]
        ):
            output_path = (
                plots_directory
                / f"{feature.impact_id}_phase6.png"
            )

            plot_phase6_event(
                feature,
                time,
                signal,
                result["baseline_mean"],
                output_path,
            )

            generated.append(output_path)

    return dataset, phase6_features, generated


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 - Multi-Domain Signal Analysis"
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
        default="results/phase6",
        help="Output directory for results.",
    )

    parser.add_argument(
        "--max-plots-per-dataset",
        type=int,
        default=3,
        help=(
            "Maximum number of per-event diagnostic plots per dataset. "
            "Use 0 to disable per-event plot generation."
        ),
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip diagnostic plot generation.",
    )

    parser.add_argument(
        "--no-stft",
        action="store_true",
        help="Disable STFT analysis.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 6 - MULTI-DOMAIN SIGNAL ANALYSIS")
    print("Channel: FBG2 | Filter: Savitzky-Golay | Detector: Peak")
    print("=" * 70)

    print(f"\nData directory : {args.data}")
    print(f"Output         : {args.out}")

    data_directory = Path(args.data)
    output_directory = Path(args.out)

    files = sorted(data_directory.glob("*.txt"))

    output_directory.mkdir(parents=True, exist_ok=True)
    plots_directory = output_directory / "plots"

    # Configuration
    config = Phase6Config(enable_stft=not args.no_stft)

    # Save configuration
    config_file = output_directory / "phase6_config.json"
    config_file.write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )

    all_records = []
    plot_count = 0
    processed_datasets = 0
    total_accepted = 0
    total_ok = 0
    total_partial = 0
    total_failed = 0
    stft_valid_count = 0
    all_features = []

    for raw_file in files:
        print(f"\nProcessing: {raw_file.name}")

        try:
            p_plots_dir = None

            if not args.no_plots and args.max_plots_per_dataset > 0:
                p_plots_dir = plots_directory

            dataset, features, generated = process_dataset(
                raw_file,
                max_plots_per_dataset=args.max_plots_per_dataset,
                plots_directory=p_plots_dir,
                config=config,
            )

            processed_datasets += 1
            total_accepted += len(features)
            plot_count += len(generated)

            for f in features:
                if f.feature_status == "ok":
                    total_ok += 1
                elif f.feature_status == "partial":
                    total_partial += 1
                else:
                    total_failed += 1
                if f.stft_valid:
                    stft_valid_count += 1

            all_features.extend(features)

            print(
                f"  {dataset}: {len(features)} accepted "
                f"event(s), {len(generated)} plot(s)"
            )

            for feature in features:
                all_records.append(feature.to_dict())
        except Exception as error:
            print(f"  ERROR: {raw_file.name}: {error}")

    if not all_records:
        print("\nNo Phase 6 features were extracted.")
        sys.exit(1)

    # Save main CSV
    features_file = output_directory / "phase6_multidomain_features.csv"

    dataframe = pd.DataFrame(all_records)
    dataframe = dataframe[PHASE6_CSV_COLUMNS]
    dataframe.to_csv(features_file, index=False)

    # Generate summary
    summary = _build_summary(
        processed_datasets,
        total_accepted,
        total_ok,
        total_partial,
        total_failed,
        stft_valid_count,
        all_features,
    )

    summary_file = output_directory / "phase6_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_file, index=False)

    # Generate aggregate plots
    if not args.no_plots and all_features:
        from src.visualization.phase6_plots import (
            plot_dominant_frequency_distribution,
            plot_dominant_frequency_vs_peak_shift,
            plot_spectral_energy_vs_peak_shift,
            plot_spectral_entropy_distribution,
        )

        plots_directory.mkdir(parents=True, exist_ok=True)

        plot_dominant_frequency_distribution(
            all_features,
            plots_directory / "phase6_dominant_frequency_distribution.png",
        )
        plot_spectral_entropy_distribution(
            all_features,
            plots_directory / "phase6_spectral_entropy_distribution.png",
        )
        plot_dominant_frequency_vs_peak_shift(
            all_features,
            plots_directory / "phase6_dominant_freq_vs_peak_shift.png",
        )
        plot_spectral_energy_vs_peak_shift(
            all_features,
            plots_directory / "phase6_spectral_energy_vs_peak_shift.png",
        )

    _print_summary(summary, features_file)


def _build_summary(
    processed_datasets,
    total_accepted,
    total_ok,
    total_partial,
    total_failed,
    stft_valid_count,
    all_features,
):
    """Build the summary statistics dictionary."""
    dom_freqs = [
        f.dominant_frequency_hz for f in all_features
        if np.isfinite(f.dominant_frequency_hz)
    ]
    spectral_energies = [
        f.spectral_energy for f in all_features
        if np.isfinite(f.spectral_energy)
    ]
    spectral_entropies = [
        f.spectral_entropy for f in all_features
        if np.isfinite(f.spectral_entropy)
    ]

    summary = {
        "total_accepted_events": total_accepted,
        "successfully_analyzed": total_ok,
        "partial_analysis": total_partial,
        "failed_analysis": total_failed,
        "valid_stft_count": stft_valid_count,
        "dominant_frequency_mean_hz": (
            float(np.mean(dom_freqs)) if dom_freqs else float("nan")
        ),
        "dominant_frequency_median_hz": (
            float(np.median(dom_freqs)) if dom_freqs else float("nan")
        ),
        "dominant_frequency_std_hz": (
            float(np.std(dom_freqs)) if dom_freqs else float("nan")
        ),
        "spectral_energy_mean": (
            float(np.mean(spectral_energies))
            if spectral_energies
            else float("nan")
        ),
        "spectral_energy_median": (
            float(np.median(spectral_energies))
            if spectral_energies
            else float("nan")
        ),
        "spectral_entropy_mean": (
            float(np.mean(spectral_entropies))
            if spectral_entropies
            else float("nan")
        ),
        "spectral_entropy_median": (
            float(np.median(spectral_entropies))
            if spectral_entropies
            else float("nan")
        ),
    }

    return summary


def _print_summary(summary, features_file):
    """Print a concise terminal summary."""
    print("\n" + "=" * 70)
    print("PHASE 6 SUMMARY")
    print("=" * 70)
    print(
        f"Accepted events         : {summary['total_accepted_events']}"
    )
    print(
        f"Successfully analyzed   : {summary['successfully_analyzed']}"
    )
    print(
        f"Partial analysis        : {summary['partial_analysis']}"
    )
    print(
        f"Failed analysis         : {summary['failed_analysis']}"
    )
    print(
        f"Valid STFT              : {summary['valid_stft_count']}"
    )
    print()
    print("Dominant Frequency (Hz):")
    print(
        f"  mean={summary['dominant_frequency_mean_hz']:.3f}  "
        f"median={summary['dominant_frequency_median_hz']:.3f}  "
        f"std={summary['dominant_frequency_std_hz']:.3f}"
    )
    print("Spectral Energy:")
    print(
        f"  mean={summary['spectral_energy_mean']:.6e}  "
        f"median={summary['spectral_energy_median']:.6e}"
    )
    print("Spectral Entropy (bits):")
    print(
        f"  mean={summary['spectral_entropy_mean']:.3f}  "
        f"median={summary['spectral_entropy_median']:.3f}"
    )
    print(f"\nFeatures saved to: {features_file}")
    print("\nPHASE 6 COMPLETE")


if __name__ == "__main__":
    main()
