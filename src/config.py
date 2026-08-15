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