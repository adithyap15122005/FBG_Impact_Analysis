"""
Wavelet analysis for FBG-2 impact signals.

Calculates:
- Wavelet Energy
- Approximation Energy
- Detail Energy
- Wavelet Entropy
"""

import numpy as np
import pywt


DEFAULT_WAVELET = "db4"
DEFAULT_LEVEL = 3


def extract_wavelet_features(
    signal: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
) -> dict:
    """
    Extract wavelet-domain features from one FBG impact segment.

    Parameters
    ----------
    signal : np.ndarray
        FBG-2 filtered impact signal.
    wavelet : str
        Wavelet family used for decomposition.
    level : int
        Decomposition level.

    Returns
    -------
    dict
        Wavelet Energy, Approximation Energy,
        Detail Energy and Wavelet Entropy.
    """

    signal = np.asarray(signal, dtype=float)

    # Remove invalid values
    signal = signal[np.isfinite(signal)]

    if signal.size < 4:
        return {
            "wavelet_energy": np.nan,
            "approximation_energy": np.nan,
            "detail_energy": np.nan,
            "wavelet_entropy": np.nan,
        }

    # Make sure requested level is possible
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(
        data_len=len(signal),
        filter_len=wavelet_obj.dec_len,
    )

    if max_level < 1:
        return {
            "wavelet_energy": np.nan,
            "approximation_energy": np.nan,
            "detail_energy": np.nan,
            "wavelet_entropy": np.nan,
        }

    level = min(level, max_level)

    # Wavelet decomposition
    coefficients = pywt.wavedec(
        signal,
        wavelet=wavelet,
        level=level,
    )

    # coefficients:
    # [Approximation, Detail_level_N, ..., Detail_level_1]
    approximation = coefficients[0]
    details = coefficients[1:]

    # Energy of approximation
    approximation_energy = float(np.sum(approximation ** 2))

    # Total energy of all detail coefficients
    detail_energies = [
        float(np.sum(detail ** 2))
        for detail in details
    ]

    detail_energy = float(np.sum(detail_energies))

    # Total wavelet-domain energy
    wavelet_energy = float(
        approximation_energy + detail_energy
    )

    # Energy distribution for entropy
    component_energies = [
        approximation_energy,
        *detail_energies,
    ]

    total_energy = float(np.sum(component_energies))

    if total_energy > 0:
        probabilities = (
            np.asarray(component_energies) / total_energy
        )

        probabilities = probabilities[probabilities > 0]

        wavelet_entropy = float(
            -np.sum(probabilities * np.log2(probabilities))
        )
    else:
        wavelet_entropy = np.nan

    return {
        "wavelet_energy": wavelet_energy,
        "approximation_energy": approximation_energy,
        "detail_energy": detail_energy,
        "wavelet_entropy": wavelet_entropy,
    }