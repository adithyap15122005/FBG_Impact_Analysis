"""
Multi-channel tests: detection runs independently per channel and
channel identity is preserved. No cross-channel gating.
"""

import numpy as np

from src.pipeline.ensemble_pipeline import (
    events_to_dataframe,
    run_ensemble_dataset,
)

from conftest import (
    add_gaussian_impact,
    gaussian_noise,
    make_time,
)


def write_synthetic_raw_file(
    path,
    signals,
    base_wavelengths=(1546.0, 1524.0, 1529.0),
):
    """
    Write a synthetic raw interrogator file (tab-separated, 8
    columns) matching load_fbg_data's expected format.
    """
    n_samples = len(signals[0])

    time = make_time(n_samples)

    rows = []

    for index in range(n_samples):
        rows.append(
            f"{time[index]:.6f}\t1\t1\t1\t0"
            f"\t{base_wavelengths[0] + signals[0][index]:.8f}"
            f"\t{base_wavelengths[1] + signals[1][index]:.8f}"
            f"\t{base_wavelengths[2] + signals[2][index]:.8f}"
        )

    path.write_text("\n".join(rows), encoding="utf-8")


def test_multi_channel_independent_detection(tmp_path):
    """
    FBG1 and FBG2 contain one impact each; FBG3 is pure noise.

    FBG1/FBG2 must each yield one accepted event with their own
    channel identity. FBG3 yields none, and its lack of events must
    not cause the other channels' events to be rejected.
    """
    n_samples = 2000
    noise_std = 0.002

    fbg1 = gaussian_noise(n_samples, seed=10, std=noise_std)
    fbg1 = add_gaussian_impact(fbg1, 1000, 0.05)

    fbg2 = gaussian_noise(n_samples, seed=11, std=noise_std)
    fbg2 = add_gaussian_impact(fbg2, 1050, 0.04)

    fbg3 = gaussian_noise(n_samples, seed=12, std=noise_std)

    raw_file = tmp_path / "synthetic_raw.txt"

    write_synthetic_raw_file(
        raw_file,
        signals=[fbg1, fbg2, fbg3],
    )

    result = run_ensemble_dataset(raw_file)

    df = events_to_dataframe(result["events"])

    accepted = df[df["accepted"] == True]  # noqa: E712

    accepted_by_channel = (
        accepted.groupby("channel")["event_id"].count().to_dict()
    )

    # FBG1 and FBG2 each see one accepted event.
    assert accepted_by_channel.get("FBG1", 0) == 1
    assert accepted_by_channel.get("FBG2", 0) == 1
    # FBG3 (pure noise) sees none.
    assert accepted_by_channel.get("FBG3", 0) == 0

    # Channel identity is preserved on every record.
    assert set(df["channel"].unique()) == {"FBG1", "FBG2", "FBG3"}

    # The FBG1 event is at the true impact location.
    fbg1_events = df[
        (df["channel"] == "FBG1") & (df["accepted"] == True)
    ]  # noqa: E712

    assert abs(fbg1_events.iloc[0]["peak_index"] - 1000) <= 30


def test_channels_keep_identity_in_results():
    """
    The events_to_dataframe representation must preserve the
    channel and dataset identity fields.
    """
    from src.impact_detection.ensemble_event import ImpactEvent

    event = ImpactEvent(
        start_index=10,
        peak_index=20,
        end_index=30,
        start_time=0.2,
        peak_time=0.4,
        end_time=0.6,
        peak_value=0.05,
        duration=0.4,
        detection_methods=["peak", "threshold"],
        event_id="expert1-FBG2-001",
        dataset="expert1",
        channel="FBG2",
    )

    df = events_to_dataframe([event])

    record = df.iloc[0]

    assert record["event_id"] == "expert1-FBG2-001"
    assert record["dataset"] == "expert1"
    assert record["channel"] == "FBG2"
    assert record["method_count"] == 2
    assert record["evidence_score"] == 0.0
