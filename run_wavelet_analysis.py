from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = Path("results/wavelet/wavelet_features.csv")
OUTPUT_DIR = Path("results/wavelet/plots")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    # Remove rows where Wavelet features could not be calculated
    df = df.dropna(
        subset=[
            "wavelet_energy",
            "approximation_energy",
            "detail_energy",
            "wavelet_entropy",
        ]
    )

    print("=" * 70)
    print("WAVELET RESULT ANALYSIS")
    print("=" * 70)

    print(f"Valid events used: {len(df)}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")

    features = [
        ("wavelet_energy", "Wavelet Energy"),
        ("approximation_energy", "Approximation Energy"),
        ("detail_energy", "Detail Energy"),
        ("wavelet_entropy", "Wavelet Entropy"),
    ]

    for column, title in features:

        grouped = (
            df.groupby("dataset")[column]
            .mean()
            .sort_index()
        )

        plt.figure(figsize=(10, 6))

        grouped.plot(
            kind="bar",
            edgecolor="black",
        )

        plt.title(f"{title} - FBG2")
        plt.xlabel("Dataset")
        plt.ylabel(title)
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_file = (
            OUTPUT_DIR
            / f"{column}_comparison.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Created: {output_file}")

    print("\nWavelet plots generated successfully.")


if __name__ == "__main__":
    main()