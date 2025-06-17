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

    # Highly optimized: no inner functions, parse & compare as soon as possible
    v0_parts = v0.split(".")
    v1_parts = v1.split(".")
    if len(v0_parts) != 3 or len(v1_parts) != 3:
        raise ValueError(
            f"Length of version strings is not 3 (expected pattern `x.y.z`) but is "
            f"{v0_parts if len(v0_parts) != 3 else v1_parts}."
        )
    try:
        for i in range(3):
            n0 = int(v0_parts[i])
            n1 = int(v1_parts[i])
            if n0 != n1:
                return 1 if n0 > n1 else -1
        return 0
    except Exception:
        raise ValueError(
            f"Version string '{v0}' or '{v1}' could not be parsed to three integers."
        )
