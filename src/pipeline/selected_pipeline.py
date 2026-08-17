"""
Selected primary analysis pipeline (FBG2 + Savitzky-Golay + one detector).

This module implements the single-method path selected for the current
controlled analysis. It reuses the existing preprocessing, wavelength
shift, Savitzky-Golay filter, the chosen detector, impact-boundary
cleaning and boundary refinement. It does NOT use multi-detector
ensemble fusion; the ensemble framework remains available in
src/impact_detection/ for the experimental multi-method path
(run_ensemble.py).

The single detector is chosen from the data by
src.analysis.method_selection.select_best_method, which scores the
four detectors on the actual FBG2 results (coverage x plausible
duration fraction x median peak deviation). See that module for the
criterion and the auditable comparison table.

Selected path
-------------
Raw FBG data
    -> FBG2 column
    -> existing preprocessing/cleaning
    -> existing baseline correction
    -> existing wavelength-shift calculation
    -> Savitzky-Golay filter
    -> best single detector (peak/threshold/derivative/change_point)
    -> existing impact-boundary refinement
    -> single-method quality gates (amplitude, duration, recovery)
    -> impact event output
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import (
    BASELINE_SAMPLES,
    FBG_COLUMNS,
    PEAK_GAP_TOLERANCE,
    PEAK_MIN_DISTANCE_SAMPLES,
    PEAK_PROMINENCE_MULTIPLIER,
    PEAK_REGION_AFTER,
    PEAK_REGION_BEFORE,
    REFINE_CONFIRMATION_SAMPLES,
    REFINE_NOISE_STD,
    REFINE_RECOVERY_RATIO,
    SAMPLING_FREQUENCY,
)
from ..filtering.savitzky_golay import savitzky_golay_filter
from ..impact_detection.boundary_refinement import refine_event_boundaries
from ..impact_detection.ensemble import (
    default_detector_params,
    run_all_detectors,
)
from ..impact_detection.ensemble_event import ImpactEvent
from ..impact_detection.peak_detection import find_peak
from ..io.data_loader import load_fbg_data
from ..preprocessing.preprocessing_pipeline import preprocess_signal

# Channel used by the selected primary analysis.
SELECTED_CHANNEL = "FBG2"

# Savitzky-Golay settings used elsewhere in the repository
# (see src/filtering/filter_comparison.py). Window 11 at the 50 Hz
# sampling rate corresponds to a ~0.22 s window.
SAVITZKY_GOLAY_WINDOW = 11
SAVITZKY_GOLAY_POLYORDER = 3


def prepare_selected_signal(df: pd.DataFrame):
    """
    Prepare the FBG2 signal for the selected pipeline.

    Applies the existing preprocessing (cleaning -> baseline ->
    wavelength shift) and the existing Savitzky-Golay filter.

    Parameters
    ----------
    df : pd.DataFrame
        Raw FBG dataframe from load_fbg_data.

    Returns
    -------
    dict with:
        - "time": time array
        - "wavelength_shift": raw wavelength shift array
        - "signal": Savitzky-Golay filtered wavelength shift array
        - "baseline_mean": baseline mean
        - "baseline_std": baseline std
        - "drift_std": overall mean shift in baseline-std units
        - "excursion_std": max excursion in baseline-std units
    """
    result = preprocess_signal(
        df[SELECTED_CHANNEL],
        BASELINE_SAMPLES,
    )

    wavelength_shift = result["wavelength_shift"]
    time = np.asarray(df["time"], dtype=float)
    signal = savitzky_golay_filter(
        wavelength_shift,
        window=SAVITZKY_GOLAY_WINDOW,
        polyorder=SAVITZKY_GOLAY_POLYORDER,
    )

    signal = np.asarray(signal, dtype=float)
    baseline = signal[:BASELINE_SAMPLES]
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline))

    drift_std = (
        float(abs(np.mean(signal) - baseline_mean) / baseline_std)
        if baseline_std > 1e-12
        else 0.0
    )
    excursion_std = (
        float(max(abs(signal.min() - baseline_mean),
                  abs(signal.max() - baseline_mean)) / baseline_std)
        if baseline_std > 1e-12
        else 0.0
    )

    return {
        "time": time,
        "wavelength_shift": np.asarray(wavelength_shift, dtype=float),
        "signal": signal,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "drift_std": drift_std,
        "excursion_std": excursion_std,
    }


def _method_regions_from_signal(
    signal: np.ndarray,
    time: np.ndarray,
    method: str,
):
    """
    Candidate impact regions from ONE detector.

    Reuses the existing detector and impact-boundary cleaning.
    """
    params = default_detector_params()

    regions_by_method = run_all_detectors(
        signal,
        time,
        methods=[method],
        params=params,
    )

    return regions_by_method[method]


def detect_selected_events(
    df: pd.DataFrame,
    dataset: str,
    method: str = "peak",
) -> Dict:
    """
    Run the selected FBG2 + Savitzky-Golay + single-detector pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw FBG dataframe from load_fbg_data.
    dataset : str
        Dataset name used in event ids.
    method : str
        Single detector to use. Defaults to the method selected by
        src.analysis.method_selection.select_best_method.

    Returns
    -------
    dict with:
        - "dataset": dataset name
        - "channel": SELECTED_CHANNEL
        - "method": detector used
        - "events": list of ImpactEvent (candidates, classified)
        - "accepted_events": list of ImpactEvent (passing gates)
        - "detections": {method: [(start, end), ...]}
        - "baseline_mean", "baseline_std": baseline statistics
        - "drift_std", "excursion_std": drift diagnostics
    """
    prepared = prepare_selected_signal(df)
    time = prepared["time"]
    signal = prepared["signal"]

    regions = _method_regions_from_signal(signal, time, method)

    events: List[ImpactEvent] = []

    for region_index, (start_index, end_index) in enumerate(regions):
        peak_index, peak_value = find_peak(
            signal,
            start_index,
            end_index,
        )

        event = ImpactEvent(
            start_index=int(start_index),
            peak_index=int(peak_index),
            end_index=int(end_index),
            start_time=float(time[start_index]),
            peak_time=float(time[peak_index]),
            end_time=float(time[end_index]),
            peak_value=float(peak_value),
            duration=float(time[end_index] - time[start_index]),
            detection_methods=[method],
            event_id=f"{dataset}-{SELECTED_CHANNEL}-{region_index + 1:03d}",
            dataset=dataset,
            channel=SELECTED_CHANNEL,
        )

        event = refine_event_boundaries(
            event,
            signal,
            time,
            prepared["baseline_mean"],
            prepared["baseline_std"],
            noise_std_multiplier=REFINE_NOISE_STD,
            recovery_ratio=REFINE_RECOVERY_RATIO,
            confirmation_samples=REFINE_CONFIRMATION_SAMPLES,
        )

        event.diagnostics["baseline_mean"] = prepared["baseline_mean"]
        event.diagnostics["baseline_std"] = prepared["baseline_std"]

        events.append(event)

    events = _classify_selected_events(
        events,
        prepared["baseline_mean"],
        prepared["baseline_std"],
        len(signal),
    )

    accepted_events = [
        event for event in events if event.accepted
    ]

    return {
        "dataset": dataset,
        "channel": SELECTED_CHANNEL,
        "method": method,
        "events": events,
        "accepted_events": accepted_events,
        "detections": {method: regions},
        "baseline_mean": prepared["baseline_mean"],
        "baseline_std": prepared["baseline_std"],
        "drift_std": prepared["drift_std"],
        "excursion_std": prepared["excursion_std"],
    }


def _classify_selected_events(
    events: List[ImpactEvent],
    baseline_mean: float,
    baseline_std: float,
    signal_length: int,
) -> List[ImpactEvent]:
    """
    Assign evidence and apply the single-method quality gates.

    Uses the same permissive agreement settings as the method
    selection study: only the agreement/evidence rules are relaxed,
    while amplitude, duration and no-confirmed-recovery gates stay
    active. Events passing every gate are accepted.
    """
    from ..analysis.method_selection import PERMISSIVE_RULES
    from ..impact_detection.ensemble import default_weights
    from ..impact_detection.evidence_fusion import assign_evidence
    from ..impact_detection.rejection_rules import apply_rejection_rules

    weights = default_weights()

    for event in events:
        assign_evidence(event, weights)

        accepted, reason = apply_rejection_rules(
            event,
            baseline_mean,
            baseline_std,
            rules=PERMISSIVE_RULES,
            signal_length=signal_length,
        )

        if accepted:
            event.accept()
        else:
            event.reject(reason)

    return events


def run_selected_dataset(
    raw_file,
    method: str = "peak",
    output_directory="results/selected",
) -> Dict:
    """
    Run the selected pipeline on one dataset.

    Returns the per-dataset result dict (see detect_selected_events).
    """
    raw_file = Path(raw_file)
    df = load_fbg_data(raw_file)
    dataset = raw_file.stem

    result = detect_selected_events(df, dataset, method=method)
    result["file"] = str(raw_file)

    return result


def run_selected_datasets(
    data_directory,
    method: str = "peak",
    output_directory="results/selected",
) -> List[Dict]:
    """
    Run the selected pipeline on every dataset in a directory.

    A failed dataset is reported but does not stop the batch.
    """
    data_directory = Path(data_directory)
    files = sorted(data_directory.glob("*.txt"))

    dataset_results: List[Dict] = []

    for raw_file in files:
        print(f"Processing: {raw_file.name}")
        try:
            result = run_selected_dataset(
                raw_file,
                method=method,
                output_directory=output_directory,
            )
            dataset_results.append(result)
            print(
                f"  {result['channel']} ({result['method']}): "
                f"{len(result['events'])} candidate(s), "
                f"{len(result['accepted_events'])} accepted"
            )
        except Exception as error:
            print(
                f"  ERROR: {raw_file.name}: {error}"
            )

    return dataset_results


def save_selected_results(
    dataset_results: List[Dict],
    output_directory="results/selected",
) -> Dict[str, Path]:
    """
    Save the selected-pipeline results to CSV/JSON.

    Outputs
    -------
    - selected_events_all_datasets.csv
    - selected_summary.csv
    - events_<dataset>.json
    """
    from ..pipeline.ensemble_pipeline import events_to_dataframe

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    all_events: List[ImpactEvent] = []
    summary_records = []

    for result in dataset_results:
        all_events.extend(result["events"])
        summary_records.append({
            "dataset": result["dataset"],
            "channel": result["channel"],
            "method": result["method"],
            "baseline_mean": result["baseline_mean"],
            "baseline_std": result["baseline_std"],
            "drift_std": result["drift_std"],
            "excursion_std": result["excursion_std"],
            "candidate_events": len(result["events"]),
            "accepted_events": len(result["accepted_events"]),
        })

    import json

    saved_files: Dict[str, Path] = {}

    for result in dataset_results:
        json_file = output_directory / (
            f"selected_events_{result['dataset']}.json"
        )
        records = [event.to_dict() for event in result["events"]]
        json_file.write_text(
            json.dumps(records, indent=2),
            encoding="utf-8",
        )
        saved_files[f"json_{result['dataset']}"] = json_file

    events_file = (
        output_directory / "selected_events_all_datasets.csv"
    )
    events_to_dataframe(all_events).to_csv(events_file, index=False)
    saved_files["events_csv"] = events_file

    accepted_events = [
        event
        for event in all_events
        if event.accepted
    ]

    accepted_file = (
        output_directory / "selected_accepted_all_datasets.csv"
    )
    events_to_dataframe(accepted_events).to_csv(
        accepted_file,
        index=False,
    )
    saved_files["accepted_events_csv"] = accepted_file

    summary_file = output_directory / "selected_summary.csv"
    pd.DataFrame(summary_records).to_csv(summary_file, index=False)
    saved_files["summary_csv"] = summary_file

    return saved_files


def generate_selected_plots(
    dataset_result: Dict,
    plots_directory="results/selected/plots",
) -> List[Path]:
    """
    Generate one diagnostic plot per peak event for one dataset.

    Reuses the existing ensemble diagnostic plot function with only
    the peak detector's regions.
    """
    from ..visualization.diagnostic_plots import plot_ensemble_diagnostic

    plots_directory = Path(plots_directory)
    plots_directory.mkdir(parents=True, exist_ok=True)

    raw_file = Path(dataset_result["file"])

    df = load_fbg_data(raw_file)
    prepared = prepare_selected_signal(df)

    generated: List[Path] = []

    method = dataset_result.get("method", "peak")

    for event in dataset_result["events"]:
        output_path = (
            plots_directory
            / (
                f"{event.dataset}_{event.channel}_"
                f"event_{event.event_id.split('-')[-1]}_{method}.png"
            )
        )

        plot_ensemble_diagnostic(
            prepared["time"],
            prepared["signal"],
            dataset_result["detections"],
            event,
            dataset_result["baseline_mean"],
            dataset_result["baseline_std"],
            output_path,
        )

        generated.append(output_path)

    return generated
