"""
Configurable false-positive rejection rules for fused impact events.

These rules remove obvious noise artifacts. All thresholds are
relative quantities (sample counts, multiples of baseline noise,
weighted agreement) so that no arbitrary physical units or
unexplained physical limits are introduced.

Each rule produces a human-readable rejection reason. The first
failing rule determines the event's rejection_reason.

NOTES
-----
- These are heuristic defaults, not calibrated parameters.
- Every threshold is configurable and documented in src/config.py.
"""

from typing import Dict, Optional, Tuple

from .ensemble_event import ImpactEvent

DEFAULT_RULES = {
    # Minimum event length in samples.
    "min_duration_samples": 3,
    # Minimum |peak - baseline| expressed in units of baseline std.
    "min_peak_deviation_std": 5.0,
    # Minimum number of independent detectors that must agree.
    "min_method_agreement": 2,
    # Minimum weighted evidence score in [0, 1].
    "min_evidence_score": 0.30,
    # Noise-like event definition: at most this many methods AND
    # peak deviation at most this many baseline stds.
    "noise_like_max_methods": 1,
    "noise_like_max_deviation_std": 5.0,
    # Guard to avoid division by zero for (almost) zero baseline std.
    "baseline_std_eps": 1e-12,
    # An event ending within this many samples of the recording end
    # never showed a confirmed recovery.
    "max_no_recovery_end_samples": 10,
}


def peak_deviation_std(
    event: ImpactEvent,
    baseline_mean: float,
    baseline_std: float,
    baseline_std_eps: float = 1e-12,
) -> float:
    """
    Peak deviation expressed in units of baseline standard deviation.

    Returns the absolute deviation of the event peak from the
    baseline mean, divided by the baseline std. When the baseline
    std is essentially zero, a large sentinel value is returned so
    that the amplitude rules do not reject strong impacts on a flat
    baseline.
    """
    if baseline_std <= baseline_std_eps:
        return float("inf")

    return abs(event.peak_value - baseline_mean) / baseline_std


def apply_rejection_rules(
    event: ImpactEvent,
    baseline_mean: float,
    baseline_std: float,
    rules: Optional[Dict[str, float]] = None,
    signal_length: Optional[int] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Apply all false-positive rejection rules to an event.

    Parameters
    ----------
    event : ImpactEvent
        Candidate event to validate.
    baseline_mean : float
        Baseline mean of the channel signal.
    baseline_std : float
        Baseline standard deviation of the channel signal.
    rules : dict, optional
        Overrides for the rejection thresholds. Missing keys fall
        back to DEFAULT_RULES.
    signal_length : int, optional
        Total length of the signal. Required for the
        no-confirmed-recovery rule.

    Returns
    -------
    (bool, Optional[str])
        (accepted, rejection_reason).
        accepted is True only when every rule passes.
    """
    if rules is None:
        rules = {}

    config = dict(DEFAULT_RULES)
    config.update(rules)

    duration_samples = event.end_index - event.start_index + 1
    dev_std = peak_deviation_std(
        event,
        baseline_mean,
        baseline_std,
        config["baseline_std_eps"],
    )

    # --------------------------------------------------------
    # Rule 1: extremely short events.
    # --------------------------------------------------------
    if duration_samples < config["min_duration_samples"]:
        return (
            False,
            "duration_below_min_samples"
            f" ({duration_samples} < "
            f"{config['min_duration_samples']})",
        )

    # --------------------------------------------------------
    # Rule 2: no confirmed recovery before the recording ended.
    # --------------------------------------------------------
    if signal_length is not None:
        no_recovery_margin = int(
            config["max_no_recovery_end_samples"]
        )

        if event.end_index >= signal_length - no_recovery_margin:
            return (
                False,
                "no_confirmed_recovery"
                f" (end_index={event.end_index}, "
                f"signal_length={signal_length})",
            )

    # --------------------------------------------------------
    # Rule 3: noise-like events.
    # --------------------------------------------------------
    # Rule 1: extremely short events.
    # --------------------------------------------------------
    if duration_samples < config["min_duration_samples"]:
        return (
            False,
            "duration_below_min_samples"
            f" ({duration_samples} < "
            f"{config['min_duration_samples']})",
        )

    # --------------------------------------------------------
    # Rule 2: noise-like events.
    # Low detector agreement combined with modest amplitude.
    # --------------------------------------------------------
    if (
        event.method_count <= config["noise_like_max_methods"]
        and dev_std <= config["noise_like_max_deviation_std"]
    ):
        return (
            False,
            "noise_like_event"
            f" (methods={event.method_count}, "
            f"dev_std={dev_std:.2f})",
        )

    # --------------------------------------------------------
    # Rule 3: very low amplitude relative to baseline noise.
    # --------------------------------------------------------
    if dev_std < config["min_peak_deviation_std"]:
        return (
            False,
            f"low_amplitude"
            f" (dev_std={dev_std:.2f} < "
            f"{config['min_peak_deviation_std']})",
        )

    # --------------------------------------------------------
    # Rule 4: insufficient detector agreement.
    # --------------------------------------------------------
    if event.method_count < config["min_method_agreement"]:
        return (
            False,
            "insufficient_method_agreement"
            f" ({event.method_count} < "
            f"{config['min_method_agreement']})",
        )

    # --------------------------------------------------------
    # Rule 5: low evidence score.
    # --------------------------------------------------------
    if event.evidence_score < config["min_evidence_score"]:
        return (
            False,
            f"evidence_below_minimum"
            f" ({event.evidence_score:.3f} < "
            f"{config['min_evidence_score']})",
        )

    return True, None
