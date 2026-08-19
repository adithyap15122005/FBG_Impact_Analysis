"""
Tests for Phase 6 multi-domain signal analysis.

Tests cover:
- Dominant frequency detection (known sine wave).
- Spectral entropy (concentrated vs broadband).
- Spectral centroid (known frequency mixture).
- Spectral energy (Parseval consistency).
- Time-domain features (skewness, kurtosis, crest factor).
- STFT features (adaptive parameter selection).
- Sampling frequency estimation.
- Edge cases: empty, single sample, short, constant, NaN, zero power,
  invalid timestamps.

Synthetic signals are deterministic (fixed seed). They are software-
validation checks only; never presented as real experimental results.
"""

import numpy as np
import pytest

from src.analysis.frequency_domain import (
    compute_dominant_frequency,
    compute_spectral_bandwidth,
    compute_spectral_centroid,
    compute_spectral_energy,
    compute_spectral_entropy,
    compute_spectral_flatness,
    compute_spectral_rolloff,
    estimate_sampling_frequency,
    extract_frequency_domain_features,
)
from src.analysis.phase6_multidomain import (
    Phase6Config,
    Phase6Features,
    extract_event_features,
)
from src.analysis.time_domain_features import (
    compute_crest_factor,
    compute_kurtosis,
    compute_rms,
    compute_skewness,
    extract_time_domain_features,
)
from src.analysis.time_frequency import (
    extract_stft_features,
)
from src.impact_detection.ensemble_event import ImpactEvent


FS = 50.0
DT = 1.0 / FS


def make_time(n_samples, fs=FS):
    return np.arange(n_samples) * DT


def make_event(
    start_index=50,
    peak_index=70,
    end_index=90,
    start_time=1.0,
    peak_time=1.4,
    end_time=1.8,
    peak_value=12.0,
    event_id="test-FBG2-001",
    dataset="test",
):
    return ImpactEvent(
        start_index=start_index,
        peak_index=peak_index,
        end_index=end_index,
        start_time=start_time,
        peak_time=peak_time,
        end_time=end_time,
        peak_value=peak_value,
        duration=end_time - start_time,
        detection_methods=["peak"],
        event_id=event_id,
        dataset=dataset,
        channel="FBG2",
    )


# ==================================================================
# Dominant frequency
# ==================================================================

class TestDominantFrequency:
    def test_known_sine_wave(self):
        """A 5 Hz sine sampled at 50 Hz should be detected at ~5 Hz."""
        n = 200
        t = make_time(n)
        freq = 5.0
        signal = np.sin(2.0 * np.pi * freq * t)

        fs, reason = estimate_sampling_frequency(t)
        assert reason == ""
        assert fs == pytest.approx(FS, rel=0.01)

        features, reason = extract_frequency_domain_features(
            signal, t, fs=fs,
        )
        assert reason == ""
        assert features["dominant_frequency_hz"] == pytest.approx(
            freq, abs=0.5
        )

    def test_different_frequency(self):
        """A 12 Hz sine should be detected near 12 Hz."""
        n = 200
        t = make_time(n)
        freq = 12.0
        signal = np.sin(2.0 * np.pi * freq * t)

        fs, _ = estimate_sampling_frequency(t)
        features, _ = extract_frequency_domain_features(signal, t, fs=fs)

        assert features["dominant_frequency_hz"] == pytest.approx(
            freq, abs=0.5
        )


# ==================================================================
# Spectral entropy
# ==================================================================

class TestSpectralEntropy:
    def test_concentrated_spectrum_low_entropy(self):
        """A pure sine has low spectral entropy (concentrated energy)."""
        n = 256
        t = make_time(n)
        signal = np.sin(2.0 * np.pi * 5.0 * t)

        fs, _ = estimate_sampling_frequency(t)
        features, _ = extract_frequency_domain_features(signal, t, fs=fs)

        # Pure sine should have low entropy
        assert features["spectral_entropy"] < 2.0

    def test_broadband_noise_high_entropy(self):
        """White noise should have high spectral entropy."""
        rng = np.random.default_rng(42)
        n = 256
        t = make_time(n)
        signal = rng.normal(0, 1, n)

        fs, _ = estimate_sampling_frequency(t)
        features, _ = extract_frequency_domain_features(signal, t, fs=fs)

        # White noise should have relatively high entropy
        # (close to log2(nfft/2))
        assert features["spectral_entropy"] > 3.0


# ==================================================================
# Spectral centroid
# ==================================================================

class TestSpectralCentroid:
    def test_mixture_two_tones(self):
        """Centroid of a 3 Hz + 9 Hz mixture should be between them."""
        n = 256
        t = make_time(n)
        sig3 = np.sin(2.0 * np.pi * 3.0 * t)
        sig9 = np.sin(2.0 * np.pi * 9.0 * t)
        signal = sig3 + sig9  # equal amplitude

        fs, _ = estimate_sampling_frequency(t)
        features, _ = extract_frequency_domain_features(signal, t, fs=fs)

        centroid = features["spectral_centroid_hz"]
        assert np.isfinite(centroid)
        # Centroid should be between 3 and 9 Hz (roughly near 6)
        assert 2.0 < centroid < 10.0

    def test_low_frequency_dominates(self):
        """Stronger low frequency should pull centroid lower."""
        n = 256
        t = make_time(n)
        sig_low = 10.0 * np.sin(2.0 * np.pi * 2.0 * t)
        sig_high = 1.0 * np.sin(2.0 * np.pi * 15.0 * t)
        signal = sig_low + sig_high

        fs, _ = estimate_sampling_frequency(t)
        features, _ = extract_frequency_domain_features(signal, t, fs=fs)

        centroid = features["spectral_centroid_hz"]
        assert np.isfinite(centroid)
        # Should be much closer to 2 Hz than 15 Hz
        assert centroid < 5.0


# ==================================================================
# Spectral energy
# ==================================================================

class TestSpectralEnergy:
    def test_energy_proportional_to_amplitude(self):
        """Doubling amplitude should 4x the spectral energy."""
        n = 128
        t = make_time(n)
        signal1 = np.sin(2.0 * np.pi * 5.0 * t)
        signal2 = 2.0 * signal1

        fs, _ = estimate_sampling_frequency(t)
        f1, _ = extract_frequency_domain_features(signal1, t, fs=fs)
        f2, _ = extract_frequency_domain_features(signal2, t, fs=fs)

        ratio = f2["spectral_energy"] / f1["spectral_energy"]
        assert ratio == pytest.approx(4.0, rel=0.05)

    def test_zero_signal_zero_energy(self):
        """A zero signal should have zero spectral energy."""
        n = 128
        t = make_time(n)
        signal = np.zeros(n)

        fs, _ = estimate_sampling_frequency(t)
        f, _ = extract_frequency_domain_features(signal, t, fs=fs)

        assert f["spectral_energy"] == pytest.approx(0.0, abs=1e-15)


# ==================================================================
# Time-domain features
# ==================================================================

class TestTimeDomainFeatures:
    def test_crest_factor_sine(self):
        """Crest factor of a pure sine is sqrt(2) ~ 1.414.

        With a finite window the peak may not align exactly with a
        sample, so a slightly wider tolerance is used.
        """
        n = 200
        signal = np.sin(2.0 * np.pi * 5.0 * make_time(n))

        features, reason = extract_time_domain_features(signal)
        assert reason == ""
        assert features["crest_factor"] == pytest.approx(
            np.sqrt(2.0), rel=0.05
        )

    def test_skewness_symmetric(self):
        """Symmetric signal should have near-zero skewness."""
        n = 200
        signal = np.sin(2.0 * np.pi * 5.0 * make_time(n))

        features, _ = extract_time_domain_features(signal)
        assert abs(features["skewness"]) < 0.2

    def test_kurtosis_gaussian(self):
        """Gaussian noise should have excess kurtosis near 0."""
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 1000)

        features, _ = extract_time_domain_features(signal)
        # With 1000 samples, kurtosis should be close to 0
        assert abs(features["kurtosis"]) < 0.5

    def test_crest_factor_constant_zero(self):
        """Constant zero signal should give NaN crest factor."""
        signal = np.zeros(100)
        features, _ = extract_time_domain_features(signal)
        assert np.isnan(features["crest_factor"])

    def test_skewness_constant_signal(self):
        """Constant signal should give NaN skewness."""
        signal = np.full(100, 5.0)
        features, _ = extract_time_domain_features(signal)
        assert np.isnan(features["skewness"])

    def test_kurtosis_short_signal(self):
        """Very short signal should give NaN kurtosis."""
        signal = np.array([1.0, 2.0, 3.0])
        features, _ = extract_time_domain_features(signal)
        assert np.isnan(features["kurtosis"])


# ==================================================================
# Sampling frequency estimation
# ==================================================================

class TestSamplingFrequencyEstimation:
    def test_regular_timestamps(self):
        """Regular 50 Hz timestamps should give fs ~ 50."""
        t = np.arange(100) * 0.02
        fs, reason = estimate_sampling_frequency(t)
        assert reason == ""
        assert fs == pytest.approx(50.0)

    def test_insufficient_timestamps(self):
        """Single timestamp should fail."""
        t = np.array([1.0])
        fs, reason = estimate_sampling_frequency(t)
        assert np.isnan(fs)
        assert "insufficient_timestamps" in reason

    def test_constant_timestamps(self):
        """All-zero timestamps should fail."""
        t = np.zeros(10)
        fs, reason = estimate_sampling_frequency(t)
        assert np.isnan(fs)
        assert "no_positive" in reason or "invalid" in reason


# ==================================================================
# STFT features
# ==================================================================

class TestSTFTFeatures:
    def test_sufficient_event(self):
        """Long enough event should produce valid STFT features."""
        n = 100
        t = make_time(n)
        signal = np.sin(2.0 * np.pi * 5.0 * t)

        features, valid, reason = extract_stft_features(
            signal, fs=FS,
        )
        assert valid is True
        assert reason == ""
        assert np.isfinite(features["stft_peak_frequency_hz"])
        assert np.isfinite(features["stft_max_energy"])

    def test_short_event_skipped(self):
        """Very short event should be skipped."""
        signal = np.array([1.0, 2.0])

        features, valid, reason = extract_stft_features(
            signal, fs=FS,
            min_stft_samples=16,
        )
        assert valid is False
        assert "too_short" in reason

    def test_disabled_stft(self):
        """When STFT is disabled, features should be NaN."""
        signal = np.sin(2.0 * np.pi * 5.0 * make_time(100))

        features, valid, reason = extract_stft_features(
            signal, fs=FS, enable_stft=False,
        )
        assert valid is False
        assert "disabled" in reason
        assert np.isnan(features["stft_peak_frequency_hz"])

    def test_nan_signal(self):
        """All-NaN signal should give NaN STFT features."""
        signal = np.full(50, np.nan)

        features, valid, reason = extract_stft_features(
            signal, fs=FS,
        )
        assert valid is False
        assert "all_nan" in reason


# ==================================================================
# Edge cases
# ==================================================================

class TestEdgeCases:
    def test_empty_signal(self):
        """Empty event segment should produce NaN features."""
        signal = np.array([])
        time = np.array([])

        features, reason = extract_time_domain_features(signal)
        assert reason == "empty_event_segment"
        assert all(np.isnan(v) for v in features.values())

    def test_single_sample(self):
        """Single-sample event should produce NaN features."""
        signal = np.array([5.0])
        features, reason = extract_time_domain_features(signal)
        assert reason == "single_sample_event"

    def test_constant_signal(self):
        """Constant (non-zero) signal should give NaN skewness/kurtosis.

        Crest factor = max(|x|)/RMS = 1.0 for a constant signal.
        """
        signal = np.full(50, 3.0)
        features, reason = extract_time_domain_features(signal)
        assert reason == ""
        assert np.isnan(features["skewness"])
        assert np.isnan(features["kurtosis"])
        assert features["crest_factor"] == pytest.approx(1.0)

    def test_nan_containing_signal(self):
        """Signal with some NaN: time-domain features propagate NaN,
        frequency-domain interpolates NaN for FFT."""
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 100)
        signal[10] = np.nan
        signal[50] = np.nan

        time = make_time(100)
        fs, _ = estimate_sampling_frequency(time)

        # Time domain: NaN propagates through np.mean
        td, reason = extract_time_domain_features(signal)
        assert not np.isfinite(td["mean"])

        # Frequency domain: NaN interpolated before FFT
        fd, fd_reason = extract_frequency_domain_features(
            signal, time, fs=fs,
        )
        assert fd_reason == ""
        assert np.isfinite(fd["dominant_frequency_hz"])

    def test_invalid_timestamps(self):
        """Negative timestamps with constant spacing still give valid fs.

        dt = median(diff([-5, -3, -1])) = 2.0, fs = 0.5 Hz.
        """
        t = np.array([-5.0, -3.0, -1.0])
        fs, reason = estimate_sampling_frequency(t)
        assert reason == ""
        assert fs == pytest.approx(0.5)

    def test_zero_spectral_power(self):
        """Zero signal should give NaN spectral entropy."""
        n = 128
        t = make_time(n)
        signal = np.zeros(n)

        fs, _ = estimate_sampling_frequency(t)
        features, _ = extract_frequency_domain_features(signal, t, fs=fs)

        assert np.isnan(features["spectral_entropy"])
        assert np.isnan(features["spectral_centroid_hz"])


# ==================================================================
# Spectral flatness and roll-off
# ==================================================================

class TestSpectralFlatness:
    def test_tone_low_flatness(self):
        """A pure tone should have low spectral flatness."""
        n = 256
        t = make_time(n)
        signal = np.sin(2.0 * np.pi * 5.0 * t)

        fs, _ = estimate_sampling_frequency(t)
        features, _ = extract_frequency_domain_features(signal, t, fs=fs)

        assert features["spectral_flatness"] < 0.1

    def test_noise_higher_flatness(self):
        """White noise should have higher spectral flatness than a tone."""
        rng = np.random.default_rng(42)
        n = 256
        t = make_time(n)
        noise = rng.normal(0, 1, n)
        tone = np.sin(2.0 * np.pi * 5.0 * t)

        fs, _ = estimate_sampling_frequency(t)
        f_noise, _ = extract_frequency_domain_features(noise, t, fs=fs)
        f_tone, _ = extract_frequency_domain_features(tone, t, fs=fs)

        assert f_noise["spectral_flatness"] > f_tone["spectral_flatness"]


class TestSpectralRolloff:
    def test_rolloff_below_nyquist(self):
        """Roll-off frequency should be below Nyquist."""
        n = 256
        t = make_time(n)
        signal = np.sin(2.0 * np.pi * 5.0 * t)

        fs, _ = estimate_sampling_frequency(t)
        features, _ = extract_frequency_domain_features(signal, t, fs=fs)

        nyquist = fs / 2.0
        assert features["spectral_rolloff_hz"] <= nyquist


# ==================================================================
# Full extraction (Phase 6 orchestrator)
# ==================================================================

class TestPhase6Extraction:
    def test_full_extraction(self):
        """Full extraction should produce valid features for a good event."""
        n = 200
        t = make_time(n)
        signal = np.sin(2.0 * np.pi * 5.0 * t) + 0.01 * np.random.default_rng(42).normal(0, 1, n)

        event = make_event(
            start_index=20,
            peak_index=50,
            end_index=100,
            start_time=t[20],
            peak_time=t[50],
            end_time=t[100],
        )

        fs, _ = estimate_sampling_frequency(t)
        config = Phase6Config()

        features = extract_event_features(
            event, signal, t, fs=fs, config=config,
        )

        assert features.dataset == "test"
        assert features.fbg == "FBG2"
        assert features.feature_status == "ok"
        assert features.failure_reason == ""
        assert np.isfinite(features.rms)
        assert np.isfinite(features.dominant_frequency_hz)
        assert np.isfinite(features.spectral_energy)
        assert np.isfinite(features.spectral_entropy)

    def test_stft_valid_for_long_event(self):
        """STFT should be valid for a sufficiently long event."""
        n = 200
        t = make_time(n)
        signal = np.sin(2.0 * np.pi * 5.0 * t)

        event = make_event(
            start_index=20,
            peak_index=50,
            end_index=150,
            start_time=t[20],
            peak_time=t[50],
            end_time=t[150],
        )

        fs, _ = estimate_sampling_frequency(t)
        features = extract_event_features(event, signal, t, fs=fs)

        assert features.stft_valid is True

    def test_stft_skipped_for_short_event(self):
        """STFT should be skipped for very short events."""
        n = 200
        t = make_time(n)
        signal = np.sin(2.0 * np.pi * 5.0 * t)

        event = make_event(
            start_index=50,
            peak_index=52,
            end_index=53,
            start_time=t[50],
            peak_time=t[52],
            end_time=t[53],
        )

        fs, _ = estimate_sampling_frequency(t)
        config = Phase6Config(minimum_stft_samples=16)
        features = extract_event_features(
            event, signal, t, fs=fs, config=config,
        )

        assert features.stft_valid is False

    def test_to_dict_columns(self):
        """to_dict should contain expected column names."""
        features = Phase6Features(
            dataset="test",
            fbg="FBG2",
            impact_id="test-FBG2-001",
            start_time=1.0,
            peak_time=1.4,
            end_time=1.8,
            duration=0.8,
        )

        d = features.to_dict()
        assert "dataset" in d
        assert "dominant_frequency_hz" in d
        assert "spectral_entropy" in d
        assert "stft_valid" in d
        assert "feature_status" in d
        assert "failure_reason" in d
