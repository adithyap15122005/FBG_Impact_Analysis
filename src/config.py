# Sampling information
SAMPLING_FREQUENCY = 50.0
SAMPLING_INTERVAL = 0.02


# Columns in the raw interrogator data
TIME_COLUMN = "time"

FBG_COLUMNS = [
    "FBG1",
    "FBG2",
    "FBG3"
]


# Original columns that are useful
RAW_COLUMN_INDICES = {
    "time": 0,
    "FBG1": 5,
    "FBG2": 6,
    "FBG3": 7
}


# Columns we don't need for signal processing
IGNORED_COLUMNS = [
    "channel_id",
    "peak_count",
    "scan_status",
    "error_flag"
]


# Baseline
BASELINE_SAMPLES = 100


# ============================================================
# PHASE 4.5 - MULTI-METHOD ENSEMBLE IMPACT DETECTION
# ============================================================

# Primary filtered signal used for ensemble detection.
# One of: "moving_average", "butterworth", "savitzky_golay", "median"
DEFAULT_FILTER = "butterworth"


# ------------------------------------------------------------
# Detector parameters (shared with Phase 4 detectors)
# ------------------------------------------------------------

# Peak detector
PEAK_PROMINENCE_MULTIPLIER = 3.0
PEAK_MIN_DISTANCE_SAMPLES = 5

# Samples expanded around a detected peak to form a candidate
# region. These are heuristic values, not physical limits.
PEAK_REGION_BEFORE = 10
PEAK_REGION_AFTER = 10

# Threshold detector
THRESHOLD_MULTIPLIER = 4.0

# Derivative detector
DERIVATIVE_MULTIPLIER = 3.0
DERIVATIVE_PERSISTENCE = 2

# Change-point detector
CHANGE_POINT_WINDOW = 20
CHANGE_POINT_THRESHOLD = 3.0

# Region cleaning (see impact_boundaries.clean_regions)
MIN_IMPACT_SAMPLES = 3
PEAK_GAP_TOLERANCE = 5
THRESHOLD_GAP_TOLERANCE = 2
DERIVATIVE_GAP_TOLERANCE = 2
CHANGE_POINT_GAP_TOLERANCE = 5


# ------------------------------------------------------------
# Event matching
# ------------------------------------------------------------

# Maximum gap (in samples) between two detector regions before
# they are considered part of the same physical impact.
# Derived from the configured sampling frequency:
# tolerance_seconds = MATCH_TOLERANCE_SAMPLES / SAMPLING_FREQUENCY
MATCH_TOLERANCE_SAMPLES = 5

# When a matched group contains two or more "multi-support zones"
# (regions where at least two detectors overlap) separated by a gap
# larger than this, the group is split into separate events. This
# prevents noisy single-detector regions from bridging two distinct
# impacts. A genuine single impact normally contains exactly one
# multi-support zone and is never split.
GROUP_SPLIT_MIN_GAP_SAMPLES = 20


# ------------------------------------------------------------
# Evidence fusion weights (HEURISTIC, NOT calibrated/optimized)
# ------------------------------------------------------------

# Each weight represents the assumed reliability of a detector.
# The weights are normalized so that they sum to 1.0 and the
# evidence score of an event is the sum of the weights of the
# methods that support it.
# Do NOT claim these values are scientifically optimal.
PEAK_WEIGHT = 0.30
THRESHOLD_WEIGHT = 0.30
DERIVATIVE_WEIGHT = 0.25
CHANGE_POINT_WEIGHT = 0.15


# ------------------------------------------------------------
# Boundary refinement
# ------------------------------------------------------------

# Noise tolerance multiplier applied to the baseline std when
# deciding whether the signal has returned to the baseline.
REFINE_NOISE_STD = 2.0

# Fraction of the peak deviation used as the recovery tolerance.
REFINE_RECOVERY_RATIO = 0.20

# Number of consecutive samples that must stay close to the
# baseline before the impact end is confirmed.
REFINE_CONFIRMATION_SAMPLES = 5


# ------------------------------------------------------------
# False-positive rejection rules (heuristic, configurable)
# ------------------------------------------------------------

# Minimum event duration in samples.
MIN_EVENT_DURATION_SAMPLES = 3

# Minimum number of independent detectors that must agree
# on an event for it to be accepted.
MIN_METHOD_AGREEMENT = 2

# Minimum weighted evidence score in [0, 1].
MIN_EVIDENCE_SCORE = 0.30

# Minimum peak deviation expressed as multiples of the baseline
# standard deviation. Events whose peak amplitude is not clearly
# above the baseline noise are rejected. On filtered signals the
# baseline std is small, so genuine impacts in this dataset reach
# 10-70 sigma while pure-noise fluctuations rarely exceed 5 sigma.
# This is a heuristic threshold, not a calibrated or physical limit.
MIN_PEAK_DEVIATION_STD = 5.0

# Guard value used to avoid division by zero when the baseline
# standard deviation is essentially zero.
MIN_BASELINE_STD_EPS = 1e-12

# An event whose refined end falls within this many samples of the
# end of the recording is rejected as "no_confirmed_recovery": the
# impact never demonstrably returned to baseline before the
# recording stopped, so the event cannot be confirmed.
MAX_NO_RECOVERY_END_SAMPLES = 10


# ------------------------------------------------------------
# Multi-channel / consistency evaluation (no ground truth)
# ------------------------------------------------------------

# Tolerance (seconds) used when checking whether events seen on
# different channels occur at approximately the same time.
CHANNEL_CONSISTENCY_TOLERANCE_S = 0.5

# Tolerance (samples) used when computing the peak-timing spread
# of the methods supporting a fused event.
TIMING_SPREAD_TOLERANCE_SAMPLES = 10