from .threshold_detection import (
    detect_threshold_crossings
)

from .impact_boundaries import (
    find_contiguous_regions
)

from .peak_detection import (
    find_peak
)


def detect_impact(
    signal,
    time,
    baseline_samples,
    threshold_multiplier=4
):
    """
    Detect impact events using
    adaptive threshold detection.
    """

    # =========================
    # 1. THRESHOLD
    # =========================

    detection = (
        detect_threshold_crossings(
            signal,
            baseline_samples,
            threshold_multiplier
        )
    )

    mask = detection["mask"]

    # =========================
    # 2. IMPACT REGIONS
    # =========================

    regions = (
        find_contiguous_regions(
            mask
        )
    )

    impacts = []

    # =========================
    # 3. PEAK
    # =========================

    for (
        start_index,
        end_index
    ) in regions:

        (
            peak_index,
            peak_value
        ) = find_peak(
            signal,
            start_index,
            end_index
        )

        impacts.append({

            "start_index":
                start_index,

            "start_time":
                time.iloc[start_index],

            "peak_index":
                peak_index,

            "peak_time":
                time.iloc[peak_index],

            "peak_value":
                peak_value,

            "end_index":
                end_index,

            "end_time":
                time.iloc[end_index]
        })

    return {

        "baseline_mean":
            detection[
                "baseline_mean"
            ],

        "baseline_std":
            detection[
                "baseline_std"
            ],

        "threshold":
            detection[
                "threshold"
            ],

        "impacts":
            impacts
    }