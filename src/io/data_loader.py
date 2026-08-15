import pandas as pd


def load_fbg_data(file_path):
    """
    Load raw FBG interrogator data.

    The raw files are tab-separated and contain
    8 columns.
    """

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None
    )

    # Give meaningful names to the columns
    df.columns = [
        "time",
        "channel_id",
        "peak_count",
        "scan_status",
        "error_flag",
        "FBG1",
        "FBG2",
        "FBG3"
    ]

    return df