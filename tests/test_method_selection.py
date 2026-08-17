"""
Tests for the data-driven method selection and the selected
single-detector pipeline (FBG2 + Savitzky-Golay + best detector).

Synthetic signals are generated with a fixed seed so the tests are
deterministic. These are software-validation checks only; they must
never be presented as real experimental results.
"""

from pathlib import Path

import numpy as np
import pytest

from src.analysis.method_selection import (
    METHODS,
    PERMISSIVE_RULES,
    evaluate_methods,
    select_best_method,
)
from src.config import FBG_COLUMNS
from src.pipeline.selected_pipeline import (
    detect_selected_events,
    run_selected_dataset,
)

FS = 50.0


def write_synthetic_raw_file(
    directory,
    name="expert_test.txt",
    n_samples=4000,
    seed=3,
    amplitudes=(0.05, 0.06, 0.045),
):
    """
    Write a synthetic FBG raw file in the interrogator format.

    The FBG2 channel contains gaussian_noise plus three smooth
    Gaussian impacts. Metadata columns are filled with plausible
    values (they are not used by the pipeline).
    """
    rng = np.random.default_rng(seed)

    time = np.arange(n_samples) / FS

    signal = rng.normal(0.0, 0.002, n_samples)

    for peak_index, amplitude in zip(
        (1000, 2000, 3000),
        amplitudes,
    ):
        indices = np.arange(n_samples)
        signal = signal + amplitude * np.exp(
            -((indices - peak_index) ** 2) / (2.0 * 20 ** 2)
        )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    file_path = directory / name

    with open(file_path, "w", encoding="utf-8") as handle:
        for index in range(n_samples):
            handle.write(
                f"{time[index]:.6f}\t1\t1\t0\t0\t"
                f"{signal[index]:.8f}\t{signal[index]:.8f}\t"
                f"{signal[index]:.8f}\n"
            )

    return file_path


def test_method_selection_returns_valid_comparison(tmp_path):
    write_synthetic_raw_file(tmp_path)

    best, comparison = select_best_method(
        tmp_path,
        filter_name="savitzky_golay",
    )

    assert best in METHODS
    assert len(comparison) == len(METHODS)
    assert set(comparison["method"]) == set(METHODS)
    assert "score" in comparison.columns
    assert comparison.iloc[0]["method"] == best


def test_selected_pipeline_detects_synthetic_impacts(tmp_path):
    raw_file = write_synthetic_raw_file(tmp_path)

    result = run_selected_dataset(raw_file)

    assert result["channel"] == "FBG2"
    assert result["method"] in METHODS

    events = result["events"]
    accepted = result["accepted_events"]

    assert len(events) >= 3
    assert len(accepted) >= 3

    for event in accepted:
        assert event.accepted is True
        assert event.evidence_score > 0.0


def test_selected_pipeline_evidence_and_classification(tmp_path):
    raw_file = write_synthetic_raw_file(tmp_path)

    result = run_selected_dataset(raw_file)

    for event in result["events"]:
        assert event.evidence_score == pytest.approx(0.3)
        if event.accepted:
            assert event.rejection_reason is None
        else:
            assert event.rejection_reason is not None


def test_permissive_rules_accept_strong_single_method_event():
    from src.impact_detection.ensemble_event import ImpactEvent
    from src.impact_detection.rejection_rules import apply_rejection_rules

    event = ImpactEvent(
        start_index=100,
        peak_index=120,
        end_index=150,
        start_time=2.0,
        peak_time=2.4,
        end_time=3.0,
        peak_value=0.05,
        duration=1.0,
        detection_methods=["peak"],
        channel="FBG2",
    )

    accepted, reason = apply_rejection_rules(
        event,
        baseline_mean=0.0,
        baseline_std=0.002,
        rules=PERMISSIVE_RULES,
    )

    assert accepted is True
    assert reason is None


def test_detections_use_only_selected_method(tmp_path):
    from src.io.data_loader import load_fbg_data

    raw_file = write_synthetic_raw_file(tmp_path)
    df = load_fbg_data(raw_file)

    result = detect_selected_events(df, "expert_test", method="peak")

    assert list(result["detections"].keys()) == ["peak"]
    assert result["method"] == "peak"
