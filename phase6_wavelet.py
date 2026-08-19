"""
Standalone Wavelet Analysis for FBG-2 Impact Signals.

Uses the existing selected pipeline:
    FBG2
    -> preprocessing
    -> wavelength shift
    -> Savitzky-Golay filtering
    -> peak detection
    -> accepted impact events

Then performs Wavelet analysis on each accepted impact event.
"""

from pathlib import Path

import pandas as pd

from src.io.data_loader import load_fbg_data
from src.pipeline.selected_pipeline import (
    prepare_selected_signal,
    detect_selected_events,
)
from src.analysis.wavelet_analysis import extract_wavelet_features


DATA_DIRECTORY = Path("data/raw")
OUTPUT_DIRECTORY = Path("results/wavelet")


def process_dataset(raw_file):
    """Process one dataset and extract Wavelet features."""

    raw_file = Path(raw_file)
    dataset = raw_file.stem

    # Load raw FBG data
    df = load_fbg_data(raw_file)

    # Prepare the same FBG2 filtered signal used by Phase 6
    prepared = prepare_selected_signal(df)

    time = prepared["time"]
    signal = prepared["signal"]

    # Use the existing selected impact-detection pipeline
    result = detect_selected_events(
        df,
        dataset,
        method="peak",
    )

    records = []

    # Analyze only accepted impact events
    for event in result["accepted_events"]:

        start_idx = event.start_index
        end_idx = event.end_index

        # Extract exactly the accepted impact segment
        signal_window = signal[start_idx:end_idx + 1]
        time_window = time[start_idx:end_idx + 1]

        # Wavelet features
        wavelet_features = extract_wavelet_features(
            signal_window
        )

        records.append(
            {
                "dataset": dataset,
                "fbg": event.channel,
                "impact_id": event.event_id,
                "start_time": float(event.start_time),
                "peak_time": float(event.peak_time),
                "end_time": float(event.end_time),
                "duration": float(event.end_time - event.start_time),
                "num_samples": len(signal_window),
                "sampling_frequency_hz": (
                    1.0 / (time_window[1] - time_window[0])
                    if len(time_window) > 1
                    else float("nan")
                ),
                "wavelet": "db4",
                "decomposition_level": 3,
                "wavelet_energy": wavelet_features[
                    "wavelet_energy"
                ],
                "approximation_energy": wavelet_features[
                    "approximation_energy"
                ],
                "detail_energy": wavelet_features[
                    "detail_energy"
                ],
                "wavelet_entropy": wavelet_features[
                    "wavelet_entropy"
                ],
            }
        )

    return records


def main():

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_records = []

    files = sorted(
        DATA_DIRECTORY.glob("*.txt")
    )

    print("=" * 70)
    print("WAVELET ANALYSIS - FBG2")
    print("Pipeline: FBG2 + Savitzky-Golay + Peak Detection")
    print("=" * 70)

    for raw_file in files:

        print(f"\nProcessing: {raw_file.name}")

        try:

            records = process_dataset(raw_file)

            all_records.extend(records)

            print(
                f"  Accepted events analyzed: {len(records)}"
            )

        except Exception as error:

            print(
                f"  ERROR: {raw_file.name}: {error}"
            )

    if not all_records:

        print("\nNo accepted events were analyzed.")
        return

    dataframe = pd.DataFrame(all_records)

    output_file = (
        OUTPUT_DIRECTORY
        / "wavelet_features.csv"
    )

    dataframe.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("WAVELET ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Total events analyzed: {len(dataframe)}"
    )

    print(
        f"Results saved to: {output_file}"
    )

    print("\nFeatures:")
    print("  - Wavelet Energy")
    print("  - Approximation Energy")
    print("  - Detail Energy")
    print("  - Wavelet Entropy")


if __name__ == "__main__":
    main()