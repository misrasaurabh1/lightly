""" Utility method for comparing versions of libraries """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved


def version_compare(v0: str, v1: str) -> int:
    """Returns 1 if version of v0 is larger than v1 and -1 otherwise

    Use this method to compare Python package versions and see which one is
    newer.

    Examples:

        >>> # compare two versions
        >>> version_compare('1.2.0', '1.1.2')
        >>> 1
    """

    # Fast path: parse at most 3 segments, avoid list comprehensions and slice reversals
    def parse_version(s):
        parts = s.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Length of version strings is not 3 (expected pattern `x.y.z`) but is "
                f"{parts}."
            )
        # Convert to integers directly
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except Exception:
            raise ValueError(
                f"Version string '{s}' could not be parsed to three integers."
            )

    v0_major, v0_minor, v0_patch = parse_version(v0)
    v1_major, v1_minor, v1_patch = parse_version(v1)

    # Compare each part directly, no zipping and reversal
    if v0_major != v1_major:
        return 1 if v0_major > v1_major else -1
    if v0_minor != v1_minor:
        return 1 if v0_minor > v1_minor else -1
    if v0_patch != v1_patch:
        return 1 if v0_patch > v1_patch else -1
    return 0
