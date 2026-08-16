"""
Standardized impact event representation for ensemble detection.

This module defines the core data structure used throughout the
Phase 4.5 multi-method ensemble framework.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ImpactEvent:
    """
    Standardized internal representation of a detected impact event.
    
    This structure is used consistently across all detection methods
    and serves as the interface for evidence fusion and evaluation.
    
    Attributes
    ----------
    start_index : int
        Sample index at impact start.
    
    peak_index : int
        Sample index at peak (maximum absolute deviation).
    
    end_index : int
        Sample index at impact end.
    
    start_time : float
        Time value (seconds) at impact start.
    
    peak_time : float
        Time value (seconds) at peak.
    
    end_time : float
        Time value (seconds) at impact end.
    
    peak_value : float
        Wavelength shift value (nm) at peak.
    
    duration : float
        Impact duration in seconds: end_time - start_time.
    
    detection_methods : List[str]
        Names of detection methods that identified this event.
        Examples: ["threshold", "peak", "derivative"]
    
    method_count : int
        Number of detection methods supporting this event.
        Automatically set based on detection_methods length.
    
    evidence_score : float
        Evidence score in range [0.0, 1.0].
        NOT a calibrated probability; represents weighted evidence.
    
    event_id : str
        Unique identifier for this event within the dataset.
        Example: "expert10-FBG1-001"
    
    dataset : str
        Dataset (experiment) name. Example: "expert10"
    
    channel : str
        FBG channel identifier. Example: "FBG1"
        Preserved for multi-channel analysis.
    
    accepted : bool
        Final decision: True if event passes all validation checks.
        Default False until validated.
    
    rejection_reason : Optional[str]
        If rejected, explanation of why.
        Examples: "duration < min_samples", "isolated_detection"
    
    diagnostics : dict
        Additional analysis data for interpretation.
        Examples:
        - {"threshold_deviation": 4.2}
        - {"peak_prominence": 0.0083}
        - {"derivative_max": 0.15}
        - {"changepoint_score": 2.8}
    """
    
    # Timing and location
    start_index: int
    peak_index: int
    end_index: int
    start_time: float
    peak_time: float
    end_time: float
    
    # Signal characteristics
    peak_value: float
    duration: float
    
    # Detection information
    detection_methods: List[str] = field(default_factory=list)
    evidence_score: float = 0.0
    
    # Identification
    event_id: str = ""
    dataset: str = ""
    
    # Channel and status
    channel: str = "FBG1"
    accepted: bool = False
    rejection_reason: Optional[str] = None
    
    # Diagnostics
    diagnostics: dict = field(default_factory=dict)
    
    # Backward-compatible alias for the evidence score.
    # Older code that referenced `confidence_score` keeps working.
    @property
    def confidence_score(self) -> float:
        """Deprecated alias for :attr:`evidence_score`."""
        return self.evidence_score
    
    @property
    def method_count(self) -> int:
        """Number of detection methods supporting this event."""
        return len(self.detection_methods)
    
    def __post_init__(self):
        """Validate basic invariants."""
        if self.start_index > self.peak_index:
            raise ValueError(
                f"start_index ({self.start_index}) must be "
                f"<= peak_index ({self.peak_index})"
            )
        
        if self.peak_index > self.end_index:
            raise ValueError(
                f"peak_index ({self.peak_index}) must be "
                f"<= end_index ({self.end_index})"
            )
        
        if self.start_time > self.peak_time:
            raise ValueError(
                f"start_time ({self.start_time}) must be "
                f"<= peak_time ({self.peak_time})"
            )
        
        if self.peak_time > self.end_time:
            raise ValueError(
                f"peak_time ({self.peak_time}) must be "
                f"<= end_time ({self.end_time})"
            )
        
        if self.evidence_score < 0.0 or self.evidence_score > 1.0:
            raise ValueError(
                f"evidence_score must be in [0.0, 1.0], "
                f"got {self.evidence_score}"
            )
    
    def add_detection_method(self, method_name: str) -> None:
        """
        Record that a detector identified this event.
        
        Parameters
        ----------
        method_name : str
            Name of detection method. Examples: "threshold", "peak"
        """
        if method_name not in self.detection_methods:
            self.detection_methods.append(method_name)
    
    def set_evidence(self, score: float) -> None:
        """
        Set the evidence/confidence score.
        
        Parameters
        ----------
        score : float
            Score in [0.0, 1.0]. Not a calibrated probability.
        
        Raises
        ------
        ValueError
            If score is outside valid range.
        """
        if not (0.0 <= score <= 1.0):
            raise ValueError(
                f"Evidence score must be in [0.0, 1.0], got {score}"
            )
        self.evidence_score = score
    
    # Backward-compatible alias for set_evidence
    def set_confidence(self, score: float) -> None:
        """Deprecated alias for :meth:`set_evidence`."""
        self.set_evidence(score)
    
    def reject(self, reason: str) -> None:
        """
        Mark this event as rejected with explanation.
        
        Parameters
        ----------
        reason : str
            Explanation for rejection.
        """
        self.accepted = False
        self.rejection_reason = reason
    
    def accept(self) -> None:
        """Mark this event as accepted."""
        self.accepted = True
        self.rejection_reason = None
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for CSV/JSON export.
        
        Returns
        -------
        dict
            Flat dictionary representation of event.
        """
        return {
            "event_id": self.event_id,
            "dataset": self.dataset,
            "channel": self.channel,
            "start_index": self.start_index,
            "peak_index": self.peak_index,
            "end_index": self.end_index,
            "start_time": self.start_time,
            "peak_time": self.peak_time,
            "end_time": self.end_time,
            "peak_value": self.peak_value,
            "duration": self.duration,
            "detection_methods": "|".join(self.detection_methods),
            "method_count": self.method_count,
            "evidence_score": self.evidence_score,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason or "",
        }
