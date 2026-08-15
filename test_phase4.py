import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# IMPORTS
# ============================================================

from src.config import BASELINE_SAMPLES

from src.impact_detection.peak_detection import (
    detect_peak_events,
    find_peak
)

from src.impact_detection.threshold_detection import (
    detect_threshold_crossings
)

from src.impact_detection.derivative_detection import (
    detect_derivative_events,
    apply_persistence
)

from src.impact_detection.change_point import (
    detect_change_points
)

from src.impact_detection.impact_boundaries import (
    clean_regions
)


# ============================================================
# INPUT FILE
# ============================================================

DATA_FILE = (
    "data/processed/filtered/"
    "expert10_FBG1_filters.csv"
)


# ============================================================
# FILTERS
# ============================================================

FILTER_COLUMNS = [
    "moving_average",
    "butterworth",
    "savitzky_golay",
    "median"
]


# ============================================================
# PEAK DETECTION PARAMETERS
# ============================================================

PEAK_PROMINENCE_MULTIPLIER = 3.0

PEAK_MIN_DISTANCE_SAMPLES = 5

PEAK_REGION_BEFORE = 10

PEAK_REGION_AFTER = 10


# ============================================================
# THRESHOLD DETECTION PARAMETERS
# ============================================================

THRESHOLD_MULTIPLIER = 3.0


# ============================================================
# DERIVATIVE DETECTION PARAMETERS
# ============================================================

DERIVATIVE_MULTIPLIER = 3.0

DERIVATIVE_PERSISTENCE = 2


# ============================================================
# CHANGE POINT PARAMETERS
# ============================================================

CHANGE_POINT_WINDOW = 20

CHANGE_POINT_THRESHOLD = 3.0


# ============================================================
# REGION PARAMETERS
# ============================================================

MIN_IMPACT_SAMPLES = 3

PEAK_GAP_TOLERANCE = 5

THRESHOLD_GAP_TOLERANCE = 2

DERIVATIVE_GAP_TOLERANCE = 2

CHANGE_POINT_GAP_TOLERANCE = 5


# ============================================================
# RECOVERY PARAMETERS
# ============================================================

RECOVERY_RATIO = 0.20

END_CONFIRMATION_SAMPLES = 5


# ============================================================
# FIND RECOVERY
# ============================================================

def find_recovery(
    signal,
    baseline_mean,
    peak_index,
    end_index,
    baseline_std,
    recovery_ratio=0.20
):
    """
    Find the point after the peak where the
    signal returns sufficiently close to baseline.
    """

    values = np.asarray(
        signal,
        dtype=float
    )

    peak_deviation = abs(
        values[peak_index] -
        baseline_mean
    )

    noise_tolerance = (
        2.0 *
        baseline_std
    )

    peak_tolerance = (
        recovery_ratio *
        peak_deviation
    )

    recovery_tolerance = max(
        noise_tolerance,
        peak_tolerance
    )

    for index in range(
        peak_index + 1,
        end_index + 1
    ):

        deviation = abs(
            values[index] -
            baseline_mean
        )

        if deviation <= recovery_tolerance:

            return index

    return None


# ============================================================
# FIND IMPACT END
# ============================================================

def find_impact_end(
    signal,
    baseline_mean,
    recovery_index,
    signal_length,
    baseline_std,
    confirmation_samples=5
):
    """
    Confirm that the signal stays close to
    baseline for several consecutive samples.
    """

    values = np.asarray(
        signal,
        dtype=float
    )

    tolerance = (
        2.0 *
        baseline_std
    )

    last_possible_index = (
        signal_length -
        confirmation_samples
    )

    for index in range(
        recovery_index,
        last_possible_index + 1
    ):

        future_values = values[
            index:
            index + confirmation_samples
        ]

        deviations = np.abs(
            future_values -
            baseline_mean
        )

        if np.all(
            deviations <= tolerance
        ):

            return index

    return recovery_index


# ============================================================
# CONVERT REGIONS TO MASK
# ============================================================

def regions_to_mask(
    regions,
    signal_length
):
    """
    Convert a list of:

        [(start, end), ...]

    into a Boolean mask.
    """

    mask = np.zeros(
        signal_length,
        dtype=bool
    )

    for start_index, end_index in regions:

        start_index = max(
            0,
            start_index
        )

        end_index = min(
            signal_length - 1,
            end_index
        )

        mask[
            start_index:
            end_index + 1
        ] = True

    return mask


# ============================================================
# ANALYZE DETECTED REGIONS
# ============================================================

def analyze_regions(
    signal,
    time,
    regions,
    baseline_mean,
    baseline_std,
    method,
    filter_name
):
    """
    Convert detection regions into:

        Start
        Peak
        Recovery
        End
        Duration
    """

    results = []

    values = np.asarray(
        signal,
        dtype=float
    )

    for impact_number, (
        start_index,
        end_index
    ) in enumerate(
        regions,
        start=1
    ):

        # ----------------------------------------------------
        # Find strongest peak in the region
        # ----------------------------------------------------

        peak_index, peak_value = find_peak(
            signal,
            start_index,
            end_index
        )

        # ----------------------------------------------------
        # Find recovery
        # ----------------------------------------------------

        recovery_index = find_recovery(
            signal,
            baseline_mean,
            peak_index,
            end_index,
            baseline_std,
            RECOVERY_RATIO
        )

        if recovery_index is None:

            recovery_index = end_index

        # ----------------------------------------------------
        # Find final end
        # ----------------------------------------------------

        final_end_index = find_impact_end(
            signal,
            baseline_mean,
            recovery_index,
            len(values),
            baseline_std,
            END_CONFIRMATION_SAMPLES
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "filter":
                filter_name,

            "method":
                method,

            "impact_number":
                impact_number,

            "start_index":
                int(start_index),

            "start_time":
                float(
                    time.iloc[start_index]
                ),

            "peak_index":
                int(peak_index),

            "peak_time":
                float(
                    time.iloc[peak_index]
                ),

            "peak_value":
                float(
                    peak_value
                ),

            "recovery_index":
                int(recovery_index),

            "recovery_time":
                float(
                    time.iloc[recovery_index]
                ),

            "end_index":
                int(final_end_index),

            "end_time":
                float(
                    time.iloc[final_end_index]
                ),

            "duration":
                float(
                    time.iloc[final_end_index]
                    -
                    time.iloc[start_index]
                )
        })

    return results


# ============================================================
# PEAK DETECTION
# ============================================================

def run_peak_detection(
    signal,
    time
):
    """
    Independent Peak Detection method.

    Detects both positive and negative peaks.
    """

    result = detect_peak_events(
        signal,
        time,
        BASELINE_SAMPLES,
        prominence_multiplier=(
            PEAK_PROMINENCE_MULTIPLIER
        ),
        minimum_distance_samples=(
            PEAK_MIN_DISTANCE_SAMPLES
        )
    )

    peaks = result["peaks"]

    values = np.asarray(
        signal,
        dtype=float
    )

    candidate_regions = []

    # --------------------------------------------------------
    # Create candidate region around each peak
    # --------------------------------------------------------

    for peak in peaks:

        peak_index = peak["index"]

        start_index = max(
            0,
            peak_index -
            PEAK_REGION_BEFORE
        )

        end_index = min(
            len(values) - 1,
            peak_index +
            PEAK_REGION_AFTER
        )

        candidate_regions.append(
            (
                start_index,
                end_index
            )
        )

    # --------------------------------------------------------
    # Clean regions
    # --------------------------------------------------------

    if candidate_regions:

        mask = regions_to_mask(
            candidate_regions,
            len(values)
        )

        regions = clean_regions(
            mask,
            minimum_samples=(
                MIN_IMPACT_SAMPLES
            ),
            gap_tolerance=(
                PEAK_GAP_TOLERANCE
            )
        )

    else:

        regions = []

    return (
        result["baseline_mean"],
        result["baseline_std"],
        result["prominence_threshold"],
        peaks,
        regions
    )


# ============================================================
# THRESHOLD DETECTION
# ============================================================

def run_threshold_detection(
    signal,
    time
):
    """
    Independent threshold detection.
    """

    result = detect_threshold_crossings(
        signal,
        BASELINE_SAMPLES,
        THRESHOLD_MULTIPLIER
    )

    regions = clean_regions(
        result["mask"],
        minimum_samples=(
            MIN_IMPACT_SAMPLES
        ),
        gap_tolerance=(
            THRESHOLD_GAP_TOLERANCE
        )
    )

    return (
        result["baseline_mean"],
        result["baseline_std"],
        result["threshold"],
        regions
    )


# ============================================================
# DERIVATIVE DETECTION
# ============================================================

def run_derivative_detection(
    signal,
    time
):
    """
    Independent derivative detection.
    """

    result = detect_derivative_events(
        signal,
        time,
        BASELINE_SAMPLES,
        DERIVATIVE_MULTIPLIER
    )

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    persistent_mask = apply_persistence(
        result["mask"],
        DERIVATIVE_PERSISTENCE
    )

    # --------------------------------------------------------
    # Clean regions
    # --------------------------------------------------------

    regions = clean_regions(
        persistent_mask,
        minimum_samples=(
            MIN_IMPACT_SAMPLES
        ),
        gap_tolerance=(
            DERIVATIVE_GAP_TOLERANCE
        )
    )

    return (
        result["baseline_mean"],
        result["baseline_std"],
        result["threshold"],
        result["derivative"],
        result["max_derivative"],
        result["max_derivative_deviation"],
        result["detected_samples"],
        regions
    )


# ============================================================
# CHANGE POINT DETECTION
# ============================================================

def run_change_point_detection(
    signal,
    time
):
    """
    Independent change-point detection.
    """

    mask = detect_change_points(
        signal,
        window=CHANGE_POINT_WINDOW,
        threshold=CHANGE_POINT_THRESHOLD
    )

    mask = np.asarray(
        mask,
        dtype=bool
    )

    regions = clean_regions(
        mask,
        minimum_samples=(
            MIN_IMPACT_SAMPLES
        ),
        gap_tolerance=(
            CHANGE_POINT_GAP_TOLERANCE
        )
    )

    values = np.asarray(
        signal,
        dtype=float
    )

    baseline = values[
        :BASELINE_SAMPLES
    ]

    baseline_mean = np.mean(
        baseline
    )

    baseline_std = np.std(
        baseline
    )

    return (
        baseline_mean,
        baseline_std,
        regions,
        mask
    )


# ============================================================
# PRINT IMPACT RESULTS
# ============================================================

def print_impact_results(
    results
):
    """
    Print detected impact information.
    """

    for impact in results:

        print()

        print(
            f"[{impact['method']}] "
            f"Impact "
            f"{impact['impact_number']}"
        )

        print(
            f"  Start: "
            f"{impact['start_time']:.4f} s"
        )

        print(
            f"  Peak: "
            f"{impact['peak_time']:.4f} s"
        )

        print(
            f"  Peak value: "
            f"{impact['peak_value']:.8f}"
        )

        print(
            f"  Recovery: "
            f"{impact['recovery_time']:.4f} s"
        )

        print(
            f"  End: "
            f"{impact['end_time']:.4f} s"
        )

        print(
            f"  Duration: "
            f"{impact['duration']:.4f} s"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "PHASE 4 - AUTOMATIC IMPACT DETECTION"
    )
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print("Input file:")
    print(DATA_FILE)

    df = pd.read_csv(
        DATA_FILE
    )

    # --------------------------------------------------------
    # Validate time
    # --------------------------------------------------------

    if "time" not in df.columns:

        raise ValueError(
            "CSV must contain a 'time' column."
        )

    time = df["time"]

    print()
    print(
        f"Number of samples: "
        f"{len(df)}"
    )

    print(
        f"Time range: "
        f"{time.iloc[0]:.4f}"
        f" → "
        f"{time.iloc[-1]:.4f} s"
    )

    # ========================================================
    # FIND AVAILABLE FILTERS
    # ========================================================

    print()
    print("=" * 70)
    print("FILTERS")
    print("=" * 70)

    available_filters = []

    for filter_name in FILTER_COLUMNS:

        if filter_name in df.columns:

            available_filters.append(
                filter_name
            )

            print(
                f"✓ {filter_name}"
            )

        else:

            print(
                f"✗ {filter_name} NOT FOUND"
            )

    if not available_filters:

        raise ValueError(
            "No filtered signal columns found."
        )

    # ========================================================
    # RESULT STORAGE
    # ========================================================

    all_results = []

    summary_results = []

    # ========================================================
    # PROCESS EACH FILTER
    # ========================================================

    for filter_name in available_filters:

        print()
        print("=" * 70)
        print(
            f"FILTER: {filter_name}"
        )
        print("=" * 70)

        signal = df[
            filter_name
        ]

        # ====================================================
        # 1. PEAK DETECTION
        # ====================================================

        print()
        print(
            "--- PEAK DETECTION ---"
        )

        (
            baseline_mean,
            baseline_std,
            prominence_threshold,
            detected_peaks,
            peak_regions
        ) = run_peak_detection(
            signal,
            time
        )

        positive_peak_count = sum(
            1
            for peak in detected_peaks
            if peak["type"] == "positive"
        )

        negative_peak_count = sum(
            1
            for peak in detected_peaks
            if peak["type"] == "negative"
        )

        print(
            f"Baseline mean: "
            f"{baseline_mean:.8f}"
        )

        print(
            f"Baseline STD: "
            f"{baseline_std:.8f}"
        )

        print(
            f"Prominence multiplier: "
            f"{PEAK_PROMINENCE_MULTIPLIER}σ"
        )

        print(
            f"Prominence threshold: "
            f"{prominence_threshold:.8f}"
        )

        print(
            f"Positive peaks: "
            f"{positive_peak_count}"
        )

        print(
            f"Negative peaks: "
            f"{negative_peak_count}"
        )

        print(
            f"Detected regions: "
            f"{len(peak_regions)}"
        )

        for peak_number, peak in enumerate(
            detected_peaks,
            start=1
        ):

            print()
            print(
                f"  Peak {peak_number}"
            )

            print(
                f"    Time: "
                f"{peak['time']:.4f} s"
            )

            print(
                f"    Value: "
                f"{peak['value']:.8f}"
            )

            print(
                f"    Type: "
                f"{peak['type']}"
            )

            print(
                f"    Prominence: "
                f"{peak['prominence']:.8f}"
            )

        peak_results = analyze_regions(
            signal,
            time,
            peak_regions,
            baseline_mean,
            baseline_std,
            "peak",
            filter_name
        )

        print_impact_results(
            peak_results
        )

        all_results.extend(
            peak_results
        )

        summary_results.append({

            "filter":
                filter_name,

            "method":
                "peak",

            "detected_regions":
                len(peak_regions),

            "baseline_mean":
                baseline_mean,

            "baseline_std":
                baseline_std,

            "threshold":
                prominence_threshold,

            "parameter":
                PEAK_PROMINENCE_MULTIPLIER
        })

        # ====================================================
        # 2. THRESHOLD DETECTION
        # ====================================================

        print()
        print(
            "--- THRESHOLD DETECTION ---"
        )

        (
            baseline_mean,
            baseline_std,
            threshold,
            threshold_regions
        ) = run_threshold_detection(
            signal,
            time
        )

        print(
            f"Baseline mean: "
            f"{baseline_mean:.8f}"
        )

        print(
            f"Baseline STD: "
            f"{baseline_std:.8f}"
        )

        print(
            f"Threshold multiplier: "
            f"{THRESHOLD_MULTIPLIER}σ"
        )

        print(
            f"Threshold: "
            f"{threshold:.8f}"
        )

        print(
            f"Detected regions: "
            f"{len(threshold_regions)}"
        )

        threshold_results = analyze_regions(
            signal,
            time,
            threshold_regions,
            baseline_mean,
            baseline_std,
            "threshold",
            filter_name
        )

        print_impact_results(
            threshold_results
        )

        all_results.extend(
            threshold_results
        )

        summary_results.append({

            "filter":
                filter_name,

            "method":
                "threshold",

            "detected_regions":
                len(threshold_regions),

            "baseline_mean":
                baseline_mean,

            "baseline_std":
                baseline_std,

            "threshold":
                threshold,

            "parameter":
                THRESHOLD_MULTIPLIER
        })

        # ====================================================
        # 3. DERIVATIVE DETECTION
        # ====================================================

        print()
        print(
            "--- DERIVATIVE DETECTION ---"
        )

        (
            baseline_mean,
            baseline_std,
            derivative_threshold,
            derivative,
            max_derivative,
            max_derivative_deviation,
            detected_derivative_samples,
            derivative_regions
        ) = run_derivative_detection(
            signal,
            time
        )

        print(
            f"Baseline mean: "
            f"{baseline_mean:.8f}"
        )

        print(
            f"Baseline STD: "
            f"{baseline_std:.8f}"
        )

        print(
            f"Derivative multiplier: "
            f"{DERIVATIVE_MULTIPLIER}σ"
        )

        print(
            f"Derivative threshold: "
            f"{derivative_threshold:.8f}"
        )

        print(
            f"Maximum absolute derivative: "
            f"{max_derivative:.8f}"
        )

        print(
            f"Maximum derivative deviation: "
            f"{max_derivative_deviation:.8f}"
        )

        print(
            f"Derivative samples above threshold: "
            f"{detected_derivative_samples}"
        )

        print(
            f"Persistence: "
            f"{DERIVATIVE_PERSISTENCE} samples"
        )

        print(
            f"Detected regions: "
            f"{len(derivative_regions)}"
        )

        derivative_results = analyze_regions(
            signal,
            time,
            derivative_regions,
            baseline_mean,
            baseline_std,
            "derivative",
            filter_name
        )

        print_impact_results(
            derivative_results
        )

        all_results.extend(
            derivative_results
        )

        summary_results.append({

            "filter":
                filter_name,

            "method":
                "derivative",

            "detected_regions":
                len(derivative_regions),

            "baseline_mean":
                baseline_mean,

            "baseline_std":
                baseline_std,

            "threshold":
                derivative_threshold,

            "parameter":
                DERIVATIVE_MULTIPLIER,

            "max_derivative":
                max_derivative,

            "max_derivative_deviation":
                max_derivative_deviation,

            "detected_samples":
                detected_derivative_samples
        })

        # ====================================================
        # 4. CHANGE POINT DETECTION
        # ====================================================

        print()
        print(
            "--- CHANGE POINT DETECTION ---"
        )

        (
            baseline_mean,
            baseline_std,
            change_regions,
            change_mask
        ) = run_change_point_detection(
            signal,
            time
        )

        print(
            f"Baseline mean: "
            f"{baseline_mean:.8f}"
        )

        print(
            f"Baseline STD: "
            f"{baseline_std:.8f}"
        )

        print(
            f"Window: "
            f"{CHANGE_POINT_WINDOW}"
        )

        print(
            f"Threshold: "
            f"{CHANGE_POINT_THRESHOLD}σ"
        )

        print(
            f"Detected regions: "
            f"{len(change_regions)}"
        )

        change_results = analyze_regions(
            signal,
            time,
            change_regions,
            baseline_mean,
            baseline_std,
            "change_point",
            filter_name
        )

        print_impact_results(
            change_results
        )

        all_results.extend(
            change_results
        )

        summary_results.append({

            "filter":
                filter_name,

            "method":
                "change_point",

            "detected_regions":
                len(change_regions),

            "baseline_mean":
                baseline_mean,

            "baseline_std":
                baseline_std,

            "threshold":
                CHANGE_POINT_THRESHOLD,

            "parameter":
                CHANGE_POINT_WINDOW
        })

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    output_directory = Path(
        "results/impact_detection"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    detailed_file = (
        output_directory /
        "phase4_impact_results.csv"
    )

    if all_results:

        detailed_df = pd.DataFrame(
            all_results
        )

        detailed_df.to_csv(
            detailed_file,
            index=False
        )

        print()
        print(
            "Detailed impact results saved:"
        )

        print(
            detailed_file
        )

    else:

        print()
        print(
            "No impact events detected."
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_file = (
        output_directory /
        "phase4_detection_summary.csv"
    )

    summary_df = pd.DataFrame(
        summary_results
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print()
    print(
        "Detection summary saved:"
    )

    print(
        summary_file
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "PHASE 4 TEST COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Total impact records: "
        f"{len(all_results)}"
    )

    print()
    print(
        "Detection summary:"
    )

    print(
        summary_df[
            [
                "filter",
                "method",
                "detected_regions"
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # GENERATE PLOTS
    # ========================================================

    print()
    print(
        "Generating comparison plots..."
    )

    for filter_name in available_filters:

        signal = df[
            filter_name
        ]

        values = np.asarray(
            signal,
            dtype=float
        )

        baseline = values[
            :BASELINE_SAMPLES
        ]

        baseline_mean = np.mean(
            baseline
        )

        baseline_std = np.std(
            baseline
        )

        threshold = (
            THRESHOLD_MULTIPLIER *
            baseline_std
        )

        plt.figure(
            figsize=(15, 6)
        )

        # ----------------------------------------------------
        # Signal
        # ----------------------------------------------------

        plt.plot(
            time,
            signal,
            label=filter_name
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        plt.axhline(
            baseline_mean,
            linestyle=":",
            label="Baseline"
        )

        # ----------------------------------------------------
        # Upper / lower threshold
        # ----------------------------------------------------

        plt.axhline(
            baseline_mean +
            threshold,
            linestyle="--",
            label="Upper threshold"
        )

        plt.axhline(
            baseline_mean -
            threshold,
            linestyle="--",
            label="Lower threshold"
        )

        # ----------------------------------------------------
        # Peak detections
        # ----------------------------------------------------

        peak_impacts = [

            result

            for result in all_results

            if (
                result["filter"]
                == filter_name
                and
                result["method"]
                == "peak"
            )
        ]

        for index, impact in enumerate(
            peak_impacts
        ):

            if index == 0:

                plt.scatter(
                    impact["peak_time"],
                    impact["peak_value"],
                    s=60,
                    zorder=5,
                    label="Peak detection"
                )

            else:

                plt.scatter(
                    impact["peak_time"],
                    impact["peak_value"],
                    s=60,
                    zorder=5
                )

        # ----------------------------------------------------
        # Threshold detections
        # ----------------------------------------------------

        threshold_impacts = [

            result

            for result in all_results

            if (
                result["filter"]
                == filter_name
                and
                result["method"]
                == "threshold"
            )
        ]

        for index, impact in enumerate(
            threshold_impacts
        ):

            if index == 0:

                plt.axvline(
                    impact["start_time"],
                    linestyle="--",
                    label="Threshold start"
                )

            else:

                plt.axvline(
                    impact["start_time"],
                    linestyle="--"
                )

        # ----------------------------------------------------
        # Plot labels
        # ----------------------------------------------------

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            "Wavelength Shift (nm)"
        )

        plt.title(
            "Phase 4 Impact Detection - "
            f"{filter_name}"
        )

        plt.grid(
            True
        )

        plt.legend()

        plt.tight_layout()

        plot_file = (
            output_directory /
            f"{filter_name}_phase4.png"
        )

        plt.savefig(
            plot_file,
            dpi=150
        )

        plt.close()

        print(
            f"Saved plot: "
            f"{plot_file}"
        )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()