"""
Evaluation of ensemble impact detection WITHOUT labelled ground truth.

Ground-truth status
-------------------
No genuine labelled ground truth exists in this repository (no
expert annotations, no impact timesheets, no label files). We do
NOT invent one. Consequently precision/recall/F1, detection rate,
false-positive rate and boundary timing errors CANNOT be computed.

Instead this module reports descriptive and consistency metrics:

- number of detected events per dataset/channel
- event duration and peak-amplitude statistics
- detector agreement (methods supporting each event)
- timing consistency (spread of the per-method peak estimates
  inside a fused event)
- channel consistency (do channels see events at similar times)
- rejected events by rejection reason

Method comparison (single detectors vs ensemble) reports event
counts and consistency metrics only, for the same reason.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import CHANNEL_CONSISTENCY_TOLERANCE_S
from ..impact_detection.ensemble import (
    detect_events_channel,
)
from ..impact_detection.ensemble_event import ImpactEvent
from ..pipeline.ensemble_pipeline import (
    prepare_channel_signal,
)
from ..io.data_loader import load_fbg_data
from ..config import (
    BASELINE_SAMPLES,
    DEFAULT_FILTER,
    FBG_COLUMNS,
)


# ------------------------------------------------------------
# Descriptive metrics
# ------------------------------------------------------------

def summarize_events_per_channel(
    dataset_results: List[Dict],
) -> pd.DataFrame:
    """
    Per-dataset/per-channel event counts.
    """
    records = []

    for result in dataset_results:
        for channel, channel_result in (
            result["channel_results"].items()
        ):
            records.append({
                "dataset": result["dataset"],
                "channel": channel,
                "candidate_events": len(channel_result["events"]),
                "accepted_events": len(
                    channel_result["accepted_events"]
                ),
                "rejected_events": (
                    len(channel_result["events"])
                    - len(channel_result["accepted_events"])
                ),
            })

    return pd.DataFrame(records)


def event_characteristics(
    accepted_events: List[ImpactEvent],
) -> Dict:
    """
    Summary statistics of the accepted events.

    Reports duration, peak amplitude (both raw value and in units
    of baseline std) and peak time. Used as a descriptive metric
    because no ground truth exists for comparison.
    """
    if not accepted_events:
        return {
            "n_events": 0,
            "duration_mean_s": np.nan,
            "duration_std_s": np.nan,
            "peak_value_mean": np.nan,
            "peak_value_max": np.nan,
            "peak_dev_std_mean": np.nan,
            "evidence_mean": np.nan,
            "method_count_mean": np.nan,
        }

    durations = np.array([e.duration for e in accepted_events])
    peak_values = np.array([e.peak_value for e in accepted_events])

    dev_stds = []
    evidence = []
    method_counts = []

    for event in accepted_events:
        dev_std = event.diagnostics.get("baseline_std", np.nan)
        baseline_mean = event.diagnostics.get("baseline_mean", 0.0)
        if dev_std and dev_std == dev_std and dev_std > 0:
            dev_stds.append(
                abs(event.peak_value - baseline_mean) / dev_std
            )
        else:
            dev_stds.append(np.nan)

        evidence.append(event.evidence_score)
        method_counts.append(event.method_count)

    return {
        "n_events": len(accepted_events),
        "duration_mean_s": float(np.mean(durations)),
        "duration_std_s": float(np.std(durations)),
        "peak_value_mean": float(np.mean(peak_values)),
        "peak_value_max": float(np.max(np.abs(peak_values))),
        "peak_dev_std_mean": float(np.nanmean(dev_stds)),
        "evidence_mean": float(np.mean(evidence)),
        "method_count_mean": float(np.mean(method_counts)),
    }


def detector_agreement(
    accepted_events: List[ImpactEvent],
) -> Dict:
    """
    Distribution of how many detectors support each accepted event.
    """
    if not accepted_events:
        return {
            "n_events": 0,
            "method_count_distribution": {},
            "mean_method_count": np.nan,
        }

    counts = [event.method_count for event in accepted_events]

    distribution = {}
    for count in sorted(set(counts)):
        distribution[int(count)] = counts.count(count)

    return {
        "n_events": len(accepted_events),
        "method_count_distribution": distribution,
        "mean_method_count": float(np.mean(counts)),
    }


def timing_consistency(
    accepted_events: List[ImpactEvent],
    tolerance_samples: int = 10,
) -> Dict:
    """
    Peak-timing spread of the methods inside each fused event.

    For each event, every supporting method implies a peak position
    (the strongest absolute value inside its own region). The spread
    of these positions, relative to the fused peak, is a measure of
    how consistent the detectors are about WHEN the impact peaked.

    Returns the mean/median spread in samples and seconds.
    """
    if not accepted_events:
        return {
            "n_events": 0,
            "mean_spread_samples": np.nan,
            "median_spread_samples": np.nan,
            "events_within_tolerance": np.nan,
        }

    spreads_samples = []
    within_tolerance = 0

    for event in accepted_events:
        method_peaks = event.diagnostics.get(
            "method_peak_indices",
            {},
        )

        if len(method_peaks) < 2:
            continue

        peaks = np.array(list(method_peaks.values()), dtype=float)
        spread = np.std(peaks)

        spreads_samples.append(spread)

        if spread <= tolerance_samples:
            within_tolerance += 1

    if not spreads_samples:
        return {
            "n_events": len(accepted_events),
            "mean_spread_samples": np.nan,
            "median_spread_samples": np.nan,
            "events_within_tolerance": np.nan,
        }

    return {
        "n_events": len(accepted_events),
        "mean_spread_samples": float(np.mean(spreads_samples)),
        "median_spread_samples": float(np.median(spreads_samples)),
        "events_within_tolerance": within_tolerance,
    }


def channel_consistency(
    dataset_result: Dict,
    tolerance_seconds: float = CHANNEL_CONSISTENCY_TOLERANCE_S,
) -> Dict:
    """
    Cross-channel temporal consistency within one dataset.

    For each accepted event on one channel, this counts whether any
    accepted event on another channel has a peak time within
    tolerance_seconds. This is descriptive only: a physical impact
    is expected to be seen nearly simultaneously on all channels.

    NOTE: this does NOT imply localization or that events on one
    channel validate events on another.
    """
    channels = list(dataset_result["channel_results"].keys())

    channel_peak_times = {}

    for channel in channels:
        channel_result = dataset_result["channel_results"][channel]
        channel_peak_times[channel] = [
            event.peak_time
            for event in channel_result["accepted_events"]
        ]

    if len(channels) < 2:
        return {
            "n_channels": len(channels),
            "n_events": 0,
            "events_with_cross_channel_support": 0,
            "fraction_with_cross_channel_support": np.nan,
        }

    total_events = 0
    supported_events = 0

    for channel in channels:
        other_channels = [
            other
            for other in channels
            if other != channel
        ]

        for peak_time in channel_peak_times[channel]:
            total_events += 1

            has_support = False

            for other in other_channels:
                if any(
                    abs(other_time - peak_time)
                    <= tolerance_seconds
                    for other_time in channel_peak_times[other]
                ):
                    has_support = True
                    break

            if has_support:
                supported_events += 1

    fraction = (
        supported_events / total_events
        if total_events > 0
        else np.nan
    )

    return {
        "n_channels": len(channels),
        "n_events": total_events,
        "events_with_cross_channel_support": supported_events,
        "fraction_with_cross_channel_support": fraction,
    }


def rejection_summary(events: List[ImpactEvent]) -> pd.DataFrame:
    """
    Count rejected events by rejection reason.
    """
    reasons = {}

    for event in events:
        if event.accepted:
            continue

        reason = event.rejection_reason or "unknown"

        if reason.startswith("duration_below_min_samples"):
            reason = "duration_below_min_samples"

        if reason.startswith("low_amplitude"):
            reason = "low_amplitude"

        if reason.startswith("insufficient_method_agreement"):
            reason = "insufficient_method_agreement"

        if reason.startswith("evidence_below_minimum"):
            reason = "evidence_below_minimum"

        if reason.startswith("noise_like_event"):
            reason = "noise_like_event"

        reasons[reason] = reasons.get(reason, 0) + 1

    return pd.DataFrame(
        [
            {"rejection_reason": reason, "count": count}
            for reason, count in sorted(
                reasons.items(),
                key=lambda item: -item[1],
            )
        ]
    )


# ------------------------------------------------------------
# Method comparison (without ground truth)
# ------------------------------------------------------------

def run_single_method_dataset(
    raw_file,
    method: str,
    filter_name: str = DEFAULT_FILTER,
    channels: Optional[List[str]] = None,
) -> Dict:
    """
    Run ONE detector on all channels of a dataset using the same
    ensemble path, but without the agreement/evidence rules (a
    single method can never satisfy those by construction).

    This is used only for the method comparison study.
    """
    if channels is None:
        channels = list(FBG_COLUMNS)

    df = load_fbg_data(raw_file)
    dataset = Path(raw_file).stem

    accepted_events = []
    all_events = []

    permissive_rules = {
        "min_method_agreement": 1,
        "min_evidence_score": 0.0,
        "noise_like_max_methods": 0,
    }

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
            methods=[method],
            rejection_rules=permissive_rules,
        )

        all_events.extend(result["events"])
        accepted_events.extend(result["accepted_events"])

    return {
        "dataset": dataset,
        "method": method,
        "events": all_events,
        "accepted_events": accepted_events,
    }


def compare_methods(
    dataset_results: List[Dict],
    data_directory: str,
    methods: Optional[List[str]] = None,
    filter_name: str = DEFAULT_FILTER,
) -> pd.DataFrame:
    """
    Compare individual detectors against the ensemble.

    Because no ground truth exists, the comparison is limited to:

    - number of detected events
    - event duration statistics
    - peak amplitude statistics

    Precision/recall/F1 cannot be computed without labels. No claim
    is made that the ensemble is better.
    """
    if methods is None:
        methods = ["threshold", "peak", "derivative", "change_point"]

    records = []

    for result in dataset_results:
        dataset = result["dataset"]
        raw_file = result["file"]

        # Ensemble row (accepted events from the full pipeline).
        ensemble_events = result["accepted_events"]
        stats = event_characteristics(ensemble_events)
        records.append({
            "dataset": dataset,
            "method": "ensemble",
            "accepted_events": stats["n_events"],
            "mean_duration_s": stats["duration_mean_s"],
            "mean_peak_dev_std": stats["peak_dev_std_mean"],
            "mean_method_count": stats["method_count_mean"],
        })

        # Single-method rows.
        for method in methods:
            single_result = run_single_method_dataset(
                raw_file,
                method,
                filter_name=filter_name,
            )

            single_stats = event_characteristics(
                single_result["accepted_events"]
            )

            records.append({
                "dataset": dataset,
                "method": method,
                "accepted_events": single_stats["n_events"],
                "mean_duration_s": single_stats["duration_mean_s"],
                "mean_peak_dev_std": single_stats[
                    "peak_dev_std_mean"
                ],
                "mean_method_count": 1,
            })

    return pd.DataFrame(records)


def compute_consistency_report(
    dataset_results: List[Dict],
) -> Dict:
    """
    Aggregate the descriptive/consistency metrics across datasets.
    """
    all_accepted: List[ImpactEvent] = []
    all_events: List[ImpactEvent] = []

    for result in dataset_results:
        all_accepted.extend(result["accepted_events"])
        all_events.extend(result["events"])

    agreement = detector_agreement(all_accepted)
    timing = timing_consistency(all_accepted)
    characteristics = event_characteristics(all_accepted)
    rejections = rejection_summary(all_events)

    channel_rows = []

    for result in dataset_results:
        consistency = channel_consistency(result)
        channel_rows.append({
            "dataset": result["dataset"],
            "n_channels": consistency["n_channels"],
            "n_events": consistency["n_events"],
            "events_with_cross_channel_support": consistency[
                "events_with_cross_channel_support"
            ],
            "fraction_with_cross_channel_support": consistency[
                "fraction_with_cross_channel_support"
            ],
        })

    return {
        "total_accepted_events": len(all_accepted),
        "total_candidate_events": len(all_events),
        "event_characteristics": characteristics,
        "detector_agreement": agreement,
        "timing_consistency": timing,
        "channel_consistency": pd.DataFrame(channel_rows),
        "rejection_summary": rejections,
    }
