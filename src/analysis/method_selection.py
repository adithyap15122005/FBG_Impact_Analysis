"""
Data-driven selection of the single best detection method.

No labelled ground truth exists in this repository, so the "best"
method cannot be chosen by precision/recall/F1. Instead this module
evaluates the four detectors on the selected channel (FBG2) and
filter (Savitzky-Golay) across all datasets and scores each method
on a transparent, documented composite criterion based on the
detected events themselves.

Selection criterion
-------------------
For every dataset, each detector is run through the normal ensemble
path with ONLY the method-agreement rules relaxed (a single method
can never satisfy ">= 2 detectors agreed" by construction). The
amplitude, duration and no-confirmed-recovery gates still apply, so
an accepted event is a strong, short, recovered signal excursion.

Each method is then scored as:

    score = coverage x plausible_fraction x median_dev_std

where:
- coverage = fraction of datasets with at least one accepted event
  (higher is better: the method works on all experiments).
- plausible_fraction = fraction of accepted events whose duration
  falls in [DURATION_LOWER_S, DURATION_UPPER_S]. Events far below
  the lower bound are glitch/edge artifacts (single-sample jumps);
  events far above the upper bound are typically drift or multiple
  merged impacts. 1.0 is ideal.
- median_dev_std = median |peak - baseline| / baseline_std. Higher
  means the method reports stronger, more clearly separated impacts.

The method with the highest score is selected. The full comparison
table is returned so the choice is auditable.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import (
    BASELINE_SAMPLES,
    DEFAULT_FILTER,
)
from ..impact_detection.ensemble import detect_events_channel
from ..io.data_loader import load_fbg_data
from ..pipeline.ensemble_pipeline import prepare_channel_signal

# Detectors evaluated for selection.
METHODS = ["peak", "threshold", "derivative", "change_point"]

# Channel the selected primary analysis operates on.
SELECTION_CHANNEL = "FBG2"

# Plausible impact-duration window (seconds).
DURATION_LOWER_S = 0.1
DURATION_UPPER_S = 3.0

# Single-method evaluation: relax only the agreement/evidence rules.
PERMISSIVE_RULES = {
    "min_method_agreement": 1,
    "min_evidence_score": 0.0,
    "noise_like_max_methods": 0,
}


def run_method_on_dataset(
    raw_file,
    method: str,
    filter_name: str = DEFAULT_FILTER,
    channel: str = SELECTION_CHANNEL,
) -> List[Dict]:
    """
    Run ONE detector on the selected channel of one dataset.

    Uses the normal ensemble path with permissive agreement rules
    so that single-method events can be accepted. Returns the list
    of accepted ImpactEvent-compatible dicts.

    Parameters
    ----------
    raw_file : path-like
        Path to the raw .txt file.
    method : str
        Detector name (peak/threshold/derivative/change_point).
    filter_name : str
        Filter used for the signal.
    channel : str
        Channel to process.

    Returns
    -------
    list of dict
        Accepted events (each with peak_value, duration, peak_time
        and diagnostics.baseline_mean/baseline_std).
    """
    df = load_fbg_data(raw_file)

    time, signal = prepare_channel_signal(
        df,
        channel,
        filter_name,
    )

    result = detect_events_channel(
        signal,
        time,
        channel=channel,
        dataset=Path(raw_file).stem,
        baseline_samples=BASELINE_SAMPLES,
        methods=[method],
        rejection_rules=PERMISSIVE_RULES,
    )

    return result["accepted_events"]


def _event_metrics(
    events: List,
) -> Dict:
    """
    Aggregate dev-std and duration lists for accepted events.

    Returns the per-event peak deviation (in baseline-std units) and
    duration in seconds, skipping events without a valid baseline_std.
    """
    dev_stds = []
    durations = []

    for event in events:
        baseline_std = event.diagnostics.get("baseline_std", np.nan)
        baseline_mean = event.diagnostics.get("baseline_mean", 0.0)

        if baseline_std and baseline_std > 0:
            dev_stds.append(
                abs(event.peak_value - baseline_mean) / baseline_std
            )
        else:
            dev_stds.append(np.nan)

        durations.append(float(event.duration))

    return dev_stds, durations


def evaluate_methods(
    data_directory,
    filter_name: str = DEFAULT_FILTER,
    methods: Optional[List[str]] = None,
    channel: str = SELECTION_CHANNEL,
) -> pd.DataFrame:
    """
    Evaluate each detector across all datasets.

    Returns one row per method with:

    - accepted_events: total accepted events across datasets.
    - datasets_covered: number of datasets with >= 1 event.
    - coverage: datasets_covered / total datasets.
    - median_dev_std: median |peak-baseline| in baseline-std units.
    - plausible_fraction: fraction of events with duration inside
      [DURATION_LOWER_S, DURATION_UPPER_S].
    - score: coverage x plausible_fraction x median_dev_std.
    """
    if methods is None:
        methods = list(METHODS)

    files = sorted(Path(data_directory).glob("*.txt"))

    loaded = []

    for raw_file in files:
        try:
            df = load_fbg_data(raw_file)
            loaded.append((raw_file, df))
        except Exception:
            continue

    total_datasets = len(loaded)

    records = []

    for method in methods:
        all_dev_stds = []
        all_durations = []
        datasets_covered = 0

        for raw_file, df in loaded:
            time, signal = prepare_channel_signal(
                df,
                channel,
                filter_name,
            )

            result = detect_events_channel(
                signal,
                time,
                channel=channel,
                dataset=raw_file.stem,
                baseline_samples=BASELINE_SAMPLES,
                methods=[method],
                rejection_rules=PERMISSIVE_RULES,
            )

            events = result["accepted_events"]

            dev_stds, durations = _event_metrics(events)
            all_dev_stds.extend(dev_stds)
            all_durations.extend(durations)

            if events:
                datasets_covered += 1

        n_events = len(all_durations)
        coverage = (
            datasets_covered / total_datasets
            if total_datasets > 0
            else 0.0
        )

        median_dev_std = float(
            np.nanmedian(all_dev_stds)
            if all_dev_stds
            else 0.0
        )

        if n_events > 0:
            plausible_fraction = float(
                np.mean([
                    1.0
                    if DURATION_LOWER_S <= duration <= DURATION_UPPER_S
                    else 0.0
                    for duration in all_durations
                ])
            )
        else:
            plausible_fraction = 0.0

        score = coverage * plausible_fraction * median_dev_std

        records.append({
            "method": method,
            "accepted_events": n_events,
            "datasets_covered": datasets_covered,
            "coverage": coverage,
            "median_dev_std": median_dev_std,
            "plausible_fraction": plausible_fraction,
            "score": score,
        })

    return pd.DataFrame(records)


def select_best_method(
    data_directory,
    filter_name: str = DEFAULT_FILTER,
    methods: Optional[List[str]] = None,
    channel: str = SELECTION_CHANNEL,
) -> tuple:
    """
    Select the best detector from the results.

    Parameters
    ----------
    data_directory : path-like
        Directory containing the raw .txt files.
    filter_name : str
        Filter used for the signal.
    methods : list of str, optional
        Detectors to evaluate.
    channel : str
        Channel to evaluate on.

    Returns
    -------
    (best_method, comparison_df)
        best_method : str
            Name of the highest-scoring detector.
        comparison_df : pd.DataFrame
            Full per-method metric table (already sorted by score).
    """
    comparison = evaluate_methods(
        data_directory,
        filter_name=filter_name,
        methods=methods,
        channel=channel,
    )

    if comparison.empty:
        raise RuntimeError("No methods could be evaluated.")

    comparison = comparison.sort_values(
        "score",
        ascending=False,
    ).reset_index(drop=True)

    best_method = comparison.iloc[0]["method"]

    return best_method, comparison


__all__ = [
    "METHODS",
    "SELECTION_CHANNEL",
    "DURATION_LOWER_S",
    "DURATION_UPPER_S",
    "PERMISSIVE_RULES",
    "evaluate_methods",
    "select_best_method",
    "run_method_on_dataset",
]
