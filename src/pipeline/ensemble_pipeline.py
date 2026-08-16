"""
Multi-channel ensemble pipeline and results storage (Phase 4.5).

Runs the multi-method ensemble detector independently on every FBG
channel of every dataset, preserving channel identity, and stores
the structured results (CSV/JSON).

Multi-channel behaviour
-----------------------
- Each channel is processed independently.
- Events keep their channel identity.
- A channel that detects no events does not cause events on other
  channels to be rejected (no cross-channel gating).
- Impact localization is NOT implemented in this phase.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import (
    BASELINE_SAMPLES,
    DEFAULT_FILTER,
    FBG_COLUMNS,
    SAMPLING_FREQUENCY,
)
from ..filtering.filter_comparison import apply_all_filters
from ..io.data_loader import load_fbg_data
from ..preprocessing.preprocessing_pipeline import preprocess_signal
from ..visualization.diagnostic_plots import plot_ensemble_diagnostic
from ..impact_detection.ensemble import detect_events_channel
from ..impact_detection.ensemble_event import ImpactEvent


def events_to_dataframe(events: List[ImpactEvent]) -> pd.DataFrame:
    """
    Convert a list of ImpactEvent objects into a flat DataFrame.

    Both accepted and rejected events are included so that the
    rejection reasons are preserved.
    """
    records = [event.to_dict() for event in events]

    if not records:
        return pd.DataFrame(
            columns=[
                "event_id",
                "dataset",
                "channel",
                "start_index",
                "peak_index",
                "end_index",
                "start_time",
                "peak_time",
                "end_time",
                "peak_value",
                "duration",
                "detection_methods",
                "method_count",
                "evidence_score",
                "accepted",
                "rejection_reason",
            ]
        )

    return pd.DataFrame(records)


def prepare_channel_signal(
    df: pd.DataFrame,
    channel: str,
    filter_name: str = DEFAULT_FILTER,
) -> tuple:
    """
    Run the Phase 3 preprocessing and filtering for one channel.

    Returns (time, filtered_signal).
    """
    result = preprocess_signal(
        df[channel],
        BASELINE_SAMPLES,
    )

    wavelength_shift = result["wavelength_shift"]

    filtered_signals = apply_all_filters(
        wavelength_shift,
        SAMPLING_FREQUENCY,
    )

    if filter_name not in filtered_signals:
        raise ValueError(
            f"Unknown filter '{filter_name}'. "
            f"Available: {list(filtered_signals.keys())}"
        )

    time = np.asarray(df["time"], dtype=float)
    signal = np.asarray(filtered_signals[filter_name], dtype=float)

    return time, signal


def run_ensemble_dataset(
    raw_file,
    filter_name: str = DEFAULT_FILTER,
    channels: Optional[List[str]] = None,
) -> Dict:
    """
    Run the ensemble detector on every channel of one dataset.

    Parameters
    ----------
    raw_file : path-like
        Path to the raw interrogator .txt file.
    filter_name : str
        Filtered signal used for detection.
    channels : list of str, optional
        Channels to process. Defaults to all FBG_COLUMNS.

    Returns
    -------
    dict
        Contains:
        - "dataset": dataset name.
        - "file": path as string.
        - "channel_results": dict channel -> result from
          detect_events_channel.
        - "events": all events across channels (list of ImpactEvent).
        - "accepted_events": accepted events across channels.
    """
    raw_file = Path(raw_file)

    if channels is None:
        channels = list(FBG_COLUMNS)

    df = load_fbg_data(raw_file)

    dataset = raw_file.stem

    channel_results: Dict[str, Dict] = {}
    all_events: List[ImpactEvent] = []
    accepted_events: List[ImpactEvent] = []

    for channel in channels:
        time, signal = prepare_channel_signal(
            df,
            channel,
            filter_name,
        )

        result = detect_events_channel(
            signal,
            time,
            channel=channel,
            dataset=dataset,
            baseline_samples=BASELINE_SAMPLES,
        )

        channel_results[channel] = result

        all_events.extend(result["events"])
        accepted_events.extend(result["accepted_events"])

    return {
        "dataset": dataset,
        "file": str(raw_file),
        "channel_results": channel_results,
        "events": all_events,
        "accepted_events": accepted_events,
    }


def run_ensemble_datasets(
    data_directory,
    filter_name: str = DEFAULT_FILTER,
    channels: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Run the ensemble detector on all datasets in a directory.

    A failed dataset is reported but does not stop the batch.

    Parameters
    ----------
    data_directory : path-like
        Directory containing the raw .txt files.
    filter_name : str
        Filtered signal used for detection.
    channels : list of str, optional
        Channels to process.

    Returns
    -------
    list of dict
        Per-dataset results (see run_ensemble_dataset).
    """
    data_directory = Path(data_directory)

    files = sorted(data_directory.glob("*.txt"))

    dataset_results = []

    for raw_file in files:
        print(f"Processing: {raw_file.name}")
        try:
            result = run_ensemble_dataset(
                raw_file,
                filter_name=filter_name,
                channels=channels,
            )
            dataset_results.append(result)
            print(
                f"  Detected: "
                f"{len(result['accepted_events'])} accepted event(s)"
            )
        except Exception as error:
            print(
                f"  ERROR: {raw_file.name}: {error}"
            )

    return dataset_results


def save_ensemble_results(
    dataset_results: List[Dict],
    output_directory="results/ensemble",
) -> Dict[str, Path]:
    """
    Save the ensemble results to CSV/JSON files.

    Outputs
    -------
    - events_all_datasets.csv  (accepted + rejected, all datasets)
    - accepted_events_all_datasets.csv
    - ensemble_summary.csv     (per dataset/channel counts)
    - events_<dataset>.json    (per-dataset structured records)
    - plots/<dataset>_<channel>_ensemble_diagnostic.png

    Returns a mapping of name -> output path.
    """
    output_directory = Path(output_directory)

    plots_directory = output_directory / "plots"

    plots_directory.mkdir(parents=True, exist_ok=True)

    all_events: List[ImpactEvent] = []
    accepted_events: List[ImpactEvent] = []

    summary_records = []

    saved_files = {}

    for result in dataset_results:
        all_events.extend(result["events"])
        accepted_events.extend(result["accepted_events"])

        for channel, channel_result in (
            result["channel_results"].items()
        ):
            summary_records.append({
                "dataset": result["dataset"],
                "channel": channel,
                "baseline_mean": channel_result["baseline_mean"],
                "baseline_std": channel_result["baseline_std"],
                "signal_mean": channel_result.get("signal_mean"),
                "signal_std": channel_result.get("signal_std"),
                "drift_std": channel_result.get("drift_std"),
                "excursion_std": channel_result.get("excursion_std"),
                "candidate_events": len(channel_result["events"]),
                "accepted_events": len(
                    channel_result["accepted_events"]
                ),
                "rejected_events": (
                    len(channel_result["events"])
                    - len(channel_result["accepted_events"])
                ),
            })

    # ------------------------------------------------
    # Per-dataset JSON
    # ------------------------------------------------
    import json

    for result in dataset_results:
        json_file = output_directory / (
            f"events_{result['dataset']}.json"
        )
        records = [event.to_dict() for event in result["events"]]
        json_file.write_text(
            json.dumps(records, indent=2),
            encoding="utf-8",
        )
        saved_files[f"json_{result['dataset']}"] = json_file

    # ------------------------------------------------
    # CSV files
    # ------------------------------------------------
    events_file = output_directory / "events_all_datasets.csv"
    events_to_dataframe(all_events).to_csv(events_file, index=False)
    saved_files["events_csv"] = events_file

    accepted_file = (
        output_directory / "accepted_events_all_datasets.csv"
    )
    events_to_dataframe(accepted_events).to_csv(
        accepted_file,
        index=False,
    )
    saved_files["accepted_events_csv"] = accepted_file

    summary_file = output_directory / "ensemble_summary.csv"
    pd.DataFrame(summary_records).to_csv(
        summary_file,
        index=False,
    )
    saved_files["summary_csv"] = summary_file

    return saved_files


def generate_diagnostic_plots(
    dataset_result: Dict,
    plots_directory="results/ensemble/plots",
    filter_name: str = DEFAULT_FILTER,
    include_rejected: bool = False,
) -> List[Path]:
    """
    Generate one diagnostic plot per event for one dataset.

    For each event the plot shows the filtered signal, the regions
    of every individual detector, and the fused event boundaries
    (start/peak/end), making the accept/reject decision auditable.

    Returns the list of generated plot paths.
    """
    plots_directory = Path(plots_directory)

    plots_directory.mkdir(parents=True, exist_ok=True)

    raw_file = Path(dataset_result["file"])

    df = load_fbg_data(raw_file)

    generated = []

    for channel, channel_result in (
        dataset_result["channel_results"].items()
    ):
        time, signal = prepare_channel_signal(
            df,
            channel,
            filter_name,
        )

        detections = channel_result["detections"]

        events = (
            channel_result["events"]
            if include_rejected
            else channel_result["accepted_events"]
        )

        for event in events:
            output_path = (
                plots_directory
                / (
                    f"{event.dataset}_{event.channel}_"
                    f"event_{event.event_id.split('-')[-1]}_"
                    f"{'accepted' if event.accepted else 'rejected'}.png"
                )
            )

            plot_ensemble_diagnostic(
                time,
                signal,
                detections,
                event,
                channel_result["baseline_mean"],
                channel_result["baseline_std"],
                output_path,
            )

            generated.append(output_path)

    return generated
