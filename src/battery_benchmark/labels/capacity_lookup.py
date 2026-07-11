"""Fixed nominal capacity lookup for known datasets."""

from __future__ import annotations

NOMINAL_CAPACITY_AH: dict[str, float] = {
    "panasonic_18650pf": 2.9,
    "lg_hg2": 3.0,
    "samsung_48g": 4.8,
    "samsung_30t": 3.0,
    "synthetic": 2.5,
}


def get_nominal_capacity(dataset_name: str) -> float:
    """Return the nominal capacity in Ah for *dataset_name*.

    Raises
    ------
    KeyError
        If the dataset is not in the lookup table.
    """
    if dataset_name not in NOMINAL_CAPACITY_AH:
        raise KeyError(
            f"No nominal capacity registered for '{dataset_name}'. "
            f"Available: {sorted(NOMINAL_CAPACITY_AH.keys())}. "
            f"Add it to labels/capacity_lookup.py."
        )
    return NOMINAL_CAPACITY_AH[dataset_name]
