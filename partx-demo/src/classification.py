"""Simplified reclassification utilities adapted from original PartX implementation.

This module provides a small helper to reclassify regions based on confidence intervals.
"""
from typing import Tuple


def classify_region(ci_lower: float, ci_upper: float, current_type: str) -> Tuple[str, str]:
    """Classify or reclassify a region based on the confidence interval bounds.

    Parameters
    - ci_lower, ci_upper: confidence interval for falsification probability (lower, upper)
    - current_type: existing label ('+', '-', 'r')

    Returns (new_type, reclass_tag)
    - new_type: one of '+', '-', 'r'
    - reclass_tag: None or a string like 'r+' or 'r-' indicating reclassification event
    """
    current_type = str(current_type).lower()
    reclass_tag = None

    if current_type == "+":
        if ci_lower > 0:
            new_type = "+"
        else:
            new_type = "r"
            reclass_tag = "r+"

    elif current_type == "-":
        if ci_upper < 0:
            new_type = "-"
        else:
            new_type = "r"
            reclass_tag = "r-"

    else:
        # active/remaining: determine from CI
        if ci_lower > 0:
            new_type = "+"
        elif ci_upper < 0:
            new_type = "-"
        else:
            new_type = "r"

    return new_type, reclass_tag
