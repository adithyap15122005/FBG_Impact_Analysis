"""
Evidence fusion for multi-method impact detection.

Different detectors observe the same physical impact from
different perspectives. Evidence fusion combines their individual
votes into a single weighted evidence score.

IMPORTANT
---------
The evidence score is a heuristic confidence measure derived from
the weighted agreement of independent detectors. It is NOT a
calibrated probability. No statistical calibration has been
performed, and the weights are manually chosen defaults, not
optimized values.
"""

from typing import Dict, List

from .ensemble_event import ImpactEvent

# Canonical detector names used across the ensemble framework.
DETECTOR_NAMES = [
    "peak",
    "threshold",
    "derivative",
    "change_point",
]

DEFAULT_WEIGHTS = {
    "peak": 0.30,
    "threshold": 0.30,
    "derivative": 0.25,
    "change_point": 0.15,
}


def detector_weights(
    peak_weight: float = 0.30,
    threshold_weight: float = 0.30,
    derivative_weight: float = 0.25,
    change_point_weight: float = 0.15,
) -> Dict[str, float]:
    """
    Build the normalized detector weight mapping.

    Weights are normalized so they sum to 1.0, which guarantees
    that the weighted evidence score of any event lies in [0, 1].

    Parameters
    ----------
    peak_weight : float
        Weight assigned to the peak detector.
    threshold_weight : float
        Weight assigned to the threshold detector.
    derivative_weight : float
        Weight assigned to the derivative detector.
    change_point_weight : float
        Weight assigned to the change-point detector.

    Returns
    -------
    dict
        Mapping: detector name -> normalized weight.
    """
    raw_weights = {
        "peak": peak_weight,
        "threshold": threshold_weight,
        "derivative": derivative_weight,
        "change_point": change_point_weight,
    }

    for name, weight in raw_weights.items():
        if weight < 0.0:
            raise ValueError(
                f"Weight for '{name}' must be non-negative, got {weight}"
            )

    total = sum(raw_weights.values())

    if total <= 0.0:
        raise ValueError("Sum of detector weights must be positive.")

    return {
        name: weight / total
        for name, weight in raw_weights.items()
    }


def compute_evidence_score(
    methods: List[str],
    weights: Dict[str, float],
) -> float:
    """
    Compute the weighted evidence score for an event.

    The score is the sum of the weights of the detection methods
    that support the event. With normalized weights the result is
    always in [0, 1].

    Parameters
    ----------
    methods : list of str
        Names of the detection methods supporting the event.
        Example: ["threshold", "peak", "derivative"]
    weights : dict
        Mapping: detector name -> normalized weight.

    Returns
    -------
    float
        Weighted evidence score in [0, 1].
    """
    if not methods:
        return 0.0

    score = sum(weights.get(name, 0.0) for name in methods)

    return min(1.0, max(0.0, score))


def assign_evidence(
    event: ImpactEvent,
    weights: Dict[str, float],
) -> ImpactEvent:
    """
    Compute and store the evidence score on an event.

    The event is modified in place and also returned for
    convenience.

    Parameters
    ----------
    event : ImpactEvent
        Candidate event with detection_methods populated.
    weights : dict
        Mapping: detector name -> normalized weight.

    Returns
    -------
    ImpactEvent
        The same event with evidence_score set.
    """
    score = compute_evidence_score(
        event.detection_methods,
        weights,
    )

    event.set_evidence(score)

    return event
