from pathlib import Path

from src.pipeline.batch_filter_evaluation import (
    evaluate_all_experiments
)


def main():

    results = evaluate_all_experiments(
        "data/raw"
    )

    print("\n")
    print("=" * 60)
    print("ALL EXPERIMENT RESULTS")
    print("=" * 60)

    print(results.to_string(index=False))

    # -------------------------
    # Create output directory
    # -------------------------

    output_directory = Path(
        "results/tables"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------
    # Save results
    # -------------------------

    output_file = (
        output_directory /
        "filter_evaluation_all.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print("\nResults saved to:")
    print(output_file)


if __name__ == "__main__":
    main()