"""
Shared fixtures and synthetic-signal helpers for the Phase 4.5 tests.

All synthetic signals are generated with a fixed seed so the tests
are deterministic. Synthetic tests are only software-validation
checks; they must never be presented as real experimental results.
"""

import numpy as np
import pytest

FS = 50.0
DT = 1.0 / FS


@pytest.fixture
def fs():
    return FS


def make_time(n_samples, fs=FS):
    return np.arange(n_samples) * (1.0 / fs)


def gaussian_noise(n_samples, seed, std=0.002):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, std, n_samples)


def add_gaussian_impact(signal, peak_index, amplitude, sigma_samples=15):
    """
    Add a smooth Gaussian impact centred at peak_index.
    """
    indices = np.arange(len(signal))
    signal = signal + amplitude * np.exp(
        -((indices - peak_index) ** 2) / (2.0 * sigma_samples**2)
    )
    return signal


def noise_only(n_samples=2000, seed=42, std=0.002):
    """Pure noise signal (no impacts)."""
    return make_time(n_samples), gaussian_noise(n_samples, seed, std)


def one_impact(
    n_samples=2000,
    seed=7,
    noise_std=0.002,
    amplitude=0.05,
    peak_index=1000,
    sigma_samples=15,
):
    """Noise plus exactly one Gaussian impact."""
    signal = gaussian_noise(n_samples, seed, noise_std)
    signal = add_gaussian_impact(
        signal,
        peak_index,
        amplitude,
        sigma_samples,
    )
    return make_time(n_samples), signal
