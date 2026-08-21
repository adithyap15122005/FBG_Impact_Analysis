from pathlib import Path

import pandas as pd

from src.analysis.phase7_validation import validate_features


INPUT_FILE = (
    "results/phase6/phase6_multidomain_features.csv"
)

OUTPUT_DIR = (
    Path("results/phase7")
)

OUTPUT_FILE = (
    OUTPUT_DIR / "phase7_statistical_validation.csv"
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_FILE)

    

    feature_columns = [
    "peak_shift",
    "residual_shift",
    "rise_time",
    "recovery_time",

    "mean",
    "median",
    "std",
    "variance",
    "rms",
    "minimum",
    "maximum",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "crest_factor",

    "dominant_frequency_hz",
    "dominant_magnitude",
    "spectral_energy",
    "spectral_entropy",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_flatness",
    "spectral_rolloff_hz",

    "stft_peak_frequency_hz",
    "stft_max_energy",

    ]

    results = validate_features(
        df,
        feature_columns,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 60)
    print("PHASE 7 COMPLETE")
    print("=" * 60)
    print(results)
    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()