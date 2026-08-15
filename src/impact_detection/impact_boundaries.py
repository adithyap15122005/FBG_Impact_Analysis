import numpy as np


def find_contiguous_regions(mask):
    """
    Convert a Boolean detection mask into
    continuous (start, end) regions.

    Example:

        [False, True, True, False]

    becomes:

        [(1, 2)]
    """

    mask = np.asarray(
        mask,
        dtype=bool
    )

    regions = []

    start = None

    for index, value in enumerate(mask):

        # Start of a region
        if value and start is None:

            start = index

        # End of a region
        elif not value and start is not None:

            regions.append(
                (
                    start,
                    index - 1
                )
            )

            start = None

    # Region continues until final sample
    if start is not None:

        regions.append(
            (
                start,
                len(mask) - 1
            )
        )

    return regions


def filter_short_regions(
    regions,
    minimum_samples=3
):
    """
    Remove very short regions.

    This prevents isolated noise detections
    from becoming impact events.
    """

    valid_regions = []

    for start_index, end_index in regions:

        length = (
            end_index -
            start_index +
            1
        )

        if length >= minimum_samples:

            valid_regions.append(
                (
                    start_index,
                    end_index
                )
            )

    return valid_regions


def merge_overlapping_regions(
    regions,
    gap_tolerance=2
):
    """
    Merge overlapping or very closely spaced
    regions.

    Example:

        (100, 200)
        (180, 250)

    becomes:

        (100, 250)

    With gap_tolerance:

        (100, 200)
        (202, 250)

    can also become:

        (100, 250)
    """

    if not regions:

        return []

    # Sort regions by start position
    regions = sorted(
        regions,
        key=lambda x: x[0]
    )

    merged = [
        regions[0]
    ]

    for current_start, current_end in regions[1:]:

        previous_start, previous_end = (
            merged[-1]
        )

        # Check for overlap or small gap
        if current_start <= (
            previous_end +
            gap_tolerance +
            1
        ):

            merged[-1] = (
                previous_start,
                max(
                    previous_end,
                    current_end
                )
            )

        else:

            merged.append(
                (
                    current_start,
                    current_end
                )
            )

    return merged


def clean_regions(
    mask,
    minimum_samples=3,
    gap_tolerance=2
):
    """
    Complete region-cleaning pipeline.

    Steps:

        Boolean detection mask
                 ↓
        contiguous regions
                 ↓
        remove short regions
                 ↓
        merge overlapping/nearby regions
                 ↓
        final impact regions
    """

    # ----------------------------------
    # Step 1: Find regions
    # ----------------------------------

    regions = find_contiguous_regions(
        mask
    )

    # ----------------------------------
    # Step 2: Remove tiny regions
    # ----------------------------------

    regions = filter_short_regions(
        regions,
        minimum_samples
    )

    # ----------------------------------
    # Step 3: Merge nearby regions
    # ----------------------------------

    regions = merge_overlapping_regions(
        regions,
        gap_tolerance
    )

    return regions