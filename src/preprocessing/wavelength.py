def calculate_wavelength_shift(
    signal,
    baseline
):
    """
    Calculate wavelength shift.

    Δλ = λ - λ₀
    """

    wavelength_shift = signal - baseline

    return wavelength_shift