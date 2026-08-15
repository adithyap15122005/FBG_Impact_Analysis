from .cleaning import clean_signal
from .baseline import baseline_correct
from .wavelength import calculate_wavelength_shift


def preprocess_signal(
    signal,
    baseline_samples
):
    """
    Complete preprocessing pipeline
    for one FBG signal.

    Steps:
        1. Cleaning
        2. Baseline correction
        3. Wavelength shift
    """

    # -------------------------
    # Step 1: Cleaning
    # -------------------------

    cleaned_signal = clean_signal(signal)

    # -------------------------
    # Step 2: Baseline
    # -------------------------

    baseline_corrected, baseline = baseline_correct(
        cleaned_signal,
        baseline_samples
    )

    # -------------------------
    # Step 3: Wavelength shift
    # -------------------------

    wavelength_shift = calculate_wavelength_shift(
        cleaned_signal,
        baseline
    )

    return {
        "cleaned": cleaned_signal,
        "baseline": baseline,
        "wavelength_shift": wavelength_shift
    }