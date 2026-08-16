"""Evaluation utilities for Phase 4.5."""

from .event_evaluation import (
    channel_consistency,
    compare_methods,
    detector_agreement,
    event_characteristics,
    rejection_summary,
    run_single_method_dataset,
    summarize_events_per_channel,
    timing_consistency,
)

__all__ = [
    "channel_consistency",
    "compare_methods",
    "detector_agreement",
    "event_characteristics",
    "rejection_summary",
    "run_single_method_dataset",
    "summarize_events_per_channel",
    "timing_consistency",
]
