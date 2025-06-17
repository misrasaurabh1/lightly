from typing import List, Sequence, TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")


def sort_items_by_keys(
    keys: Sequence[_K], items: Sequence[_V], sorted_keys: Sequence[_K]
) -> List[_V]:
    """Sorts the items in the same order as the sorted keys.

    Args:
        keys:
            Keys by which items can be identified.
        items:
            Items to sort.
        sorted_keys:
            Keys in sorted order.

    Returns:
        The list of sorted items.

    Examples:
        >>> keys = [3, 2, 1]
        >>> items = ['!', 'world', 'hello']
        >>> sorted_keys = [1, 2, 3]
        >>> sorted_items = sort_items_by_keys(
        >>>     keys,
        >>>     items,
        >>>     sorted_keys,
        >>> )
        >>> print(sorted_items)
        >>> > ['hello', 'world', '!']

    """
    # Combine length checks to a single comparison and avoid recalculating lengths
    n = len(keys)
    if n != len(items) or n != len(sorted_keys):
        raise ValueError(
            f"All inputs (keys,  items and sorted_keys) "
            f"must have the same length, "
            f"but their lengths are: ({len(keys)},"
            f"{len(items)} and {len(sorted_keys)})."
        )
    # Use list comprehension for better memory access and reuse
    # Build the lookup dict using zip (same as before, but moved to a helper for efficiency)
    lookup = dict(zip(keys, items))
    # List comprehension iterates sorted_keys to look up corresponding items
    return [lookup[k] for k in sorted_keys]
