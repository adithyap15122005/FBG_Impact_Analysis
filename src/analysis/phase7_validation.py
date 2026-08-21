import numpy as np
import pandas as pd


def compute_mean(values):
    return float(np.mean(values))


def compute_std(values):
    return float(np.std(values, ddof=1))


def compute_cv(values):
    mean = abs(np.mean(values))
    if mean == 0:
     return np.nan
    return (np.std(values, ddof=1) / mean) * 100


def compute_confidence_interval(values, confidence=0.95):
    n = len(values)

    if n < 2:
        return np.nan, np.nan

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    z = 1.96  # 95% confidence interval

    margin = z * (std / np.sqrt(n))

    return (
        float(mean - margin),
        float(mean + margin),
    )
def validate_features(df, feature_columns):

    results = []

    for feature in feature_columns:

        values = df[feature].dropna().values

        if len(values) == 0:
            continue

        ci_lower, ci_upper = compute_confidence_interval(values)

        results.append(
            {
                "feature": feature,
                "mean": compute_mean(values),
                "standard_deviation": compute_std(values),
                "coefficient_of_variation": compute_cv(values),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_samples": len(values),
            }
        )

    return pd.DataFrame(results)