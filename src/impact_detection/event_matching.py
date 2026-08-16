"""
Temporal matching and merging of candidate events from multiple detectors.

This module handles the critical task of recognizing that different detectors
may identify the same physical impact at slightly different times, and merging
such detections into unified candidate events.
"""

from typing import List, Tuple
import numpy as np

from .ensemble_event import ImpactEvent


def calculate_overlap(
    region_a: Tuple[int, int],
    region_b: Tuple[int, int]
) -> float:
    """
    Calculate the overlap between two (start, end) regions.
    
    Returns the fraction of overlap relative to the smaller region.
    
    Parameters
    ----------
    region_a : tuple of (int, int)
        (start_index, end_index) for region A.
    
    region_b : tuple of (int, int)
        (start_index, end_index) for region B.
    
    Returns
    -------
    float
        Overlap fraction in [0, 1].
        0 = no overlap
        1 = complete overlap (one region inside another)
    """
    start_a, end_a = region_a
    start_b, end_b = region_b
    
    # Calculate intersection
    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)
    
    if overlap_start > overlap_end:
        return 0.0
    
    overlap_length = overlap_end - overlap_start + 1
    
    # Relative to smaller region
    region_a_length = end_a - start_a + 1
    region_b_length = end_b - start_b + 1
    smaller_length = min(region_a_length, region_b_length)
    
    if smaller_length == 0:
        return 0.0
    
    return overlap_length / smaller_length


def regions_overlap_or_adjacent(
    region_a: Tuple[int, int],
    region_b: Tuple[int, int],
    tolerance_samples: int = 5
) -> bool:
    """
    Check if two regions overlap or are within tolerance distance.
    
    Regions are considered related if they overlap or have a gap
    smaller than tolerance_samples between them.
    
    Parameters
    ----------
    region_a : tuple of (int, int)
        (start_index, end_index) for region A.
    
    region_b : tuple of (int, int)
        (start_index, end_index) for region B.
    
    tolerance_samples : int
        Maximum gap (samples) to consider regions related.
        Default: 5 (0.1 seconds at 50 Hz)
    
    Returns
    -------
    bool
        True if regions overlap or are close enough to merge.
    """
    start_a, end_a = region_a
    start_b, end_b = region_b
    
    # Ensure a comes before b
    if start_a > start_b:
        start_a, end_a, start_b, end_b = start_b, end_b, start_a, end_a
    
    # Check if b starts within tolerance of a's end
    gap = start_b - end_a - 1
    
    return gap <= tolerance_samples


def merge_regions(
    region_a: Tuple[int, int],
    region_b: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Merge two regions into their combined extent.
    
    Parameters
    ----------
    region_a : tuple of (int, int)
        (start_index, end_index)
    
    region_b : tuple of (int, int)
        (start_index, end_index)
    
    Returns
    -------
    tuple of (int, int)
        (min_start, max_end)
    """
    start_a, end_a = region_a
    start_b, end_b = region_b
    
    return (
        min(start_a, start_b),
        max(end_a, end_b)
    )


def match_candidate_regions(
    all_regions: dict,
    tolerance_samples: int = 5
) -> List[List[Tuple[str, Tuple[int, int]]]]:
    """
    Match regions from different detection methods.
    
    Groups detection regions from multiple methods if they overlap or
    are close enough to represent the same physical event.
    
    Parameters
    ----------
    all_regions : dict
        Mapping: method_name → list of (start, end) tuples.
        Example: {
            "threshold": [(10, 50), (200, 250)],
            "peak": [(12, 48), (198, 252)],
            "derivative": [(15, 45)],
        }
    
    tolerance_samples : int
        Maximum gap between regions to merge. Default: 5
    
    Returns
    -------
    list of lists
        Groups of matched regions.
        Each group is a list of (method_name, (start, end)) tuples.
        Example: [
            [("threshold", (10, 50)), ("peak", (12, 48)), ("derivative", (15, 45))],
            [("threshold", (200, 250)), ("peak", (198, 252))],
        ]
    """
    # Flatten: (method, region_index, start, end)
    all_candidates = []
    for method_name, regions in all_regions.items():
        for region_idx, region in enumerate(regions):
            all_candidates.append({
                "method": method_name,
                "region_idx": region_idx,
                "region": region,
                "matched": False
            })
    
    matched_groups = []
    
    for candidate in all_candidates:
        
        if candidate["matched"]:
            continue
        
        # Start a new group
        current_group = [
            (candidate["method"], candidate["region"])
        ]
        candidate["matched"] = True
        
        current_merged_region = candidate["region"]
        
        # Find other candidates that overlap with this one
        changed = True
        while changed:
            changed = False
            
            for other in all_candidates:
                
                if other["matched"]:
                    continue
                
                if regions_overlap_or_adjacent(
                    current_merged_region,
                    other["region"],
                    tolerance_samples
                ):
                    current_group.append(
                        (other["method"], other["region"])
                    )
                    other["matched"] = True
                    current_merged_region = merge_regions(
                        current_merged_region,
                        other["region"]
                    )
                    changed = True
        
        matched_groups.append(current_group)
    
    return matched_groups


def create_candidate_event_from_group(
    group: List[Tuple[str, Tuple[int, int]]],
    signal: np.ndarray,
    time: np.ndarray,
    channel: str = "FBG1"
) -> ImpactEvent:
    """
    Create a candidate event from a group of matched detections.
    
    Combines spatial information from all detectors in the group.
    The peak is found as the strongest absolute value in the merged region.
    
    Parameters
    ----------
    group : list of tuples
        List of (method_name, (start, end)) tuples.
    
    signal : array-like
        Wavelength shift signal.
    
    time : array-like
        Time values corresponding to signal.
    
    channel : str
        Channel name (e.g., "FBG1")
    
    Returns
    -------
    ImpactEvent
        Candidate event with detection methods recorded.
    """
    signal = np.asarray(signal, dtype=float)
    time = np.asarray(time, dtype=float)
    
    # Collect all start/end indices
    all_starts = []
    all_ends = []
    methods = []
    
    for method_name, (start, end) in group:
        all_starts.append(start)
        all_ends.append(end)
        methods.append(method_name)
    
    # Merged region
    merged_start = min(all_starts)
    merged_end = max(all_ends)
    
    # Find peak (strongest absolute value)
    region_signal = signal[merged_start:merged_end + 1]
    local_peak_idx = np.argmax(np.abs(region_signal))
    peak_index = merged_start + local_peak_idx
    peak_value = signal[peak_index]
    
    # Create event
    event = ImpactEvent(
        start_index=int(merged_start),
        peak_index=int(peak_index),
        end_index=int(merged_end),
        start_time=float(time[merged_start]),
        peak_time=float(time[peak_index]),
        end_time=float(time[merged_end]),
        peak_value=float(peak_value),
        duration=float(time[merged_end] - time[merged_start]),
        detection_methods=sorted(list(set(methods))),
        channel=channel
    )
    
    return event
