"""Simple probability estimators for the demo.

Implements binomial proportion estimates and confidence intervals (Wilson interval).
"""
from typing import Iterable
import numpy as np


def compute_confidence_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute a (1-alpha) confidence interval for a binomial proportion using the Wilson score interval.

    Returns (lower, upper).
    """
    if n == 0:
        return 0.0, 1.0
    from math import sqrt
    from scipy.stats import norm
    z = abs(norm.ppf(alpha / 2.0))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + (z * z) / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + (z * z) / (4 * n * n)) ** 0.5) / denom
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return lower, upper


def estimate_region_falsification_probability(region) -> tuple[float, tuple[float, float]]:
    """Estimate the falsification probability p_hat and a confidence interval for a single region.

    Uses sample counts within the region: p_hat = (#values<0)/n. Returns (p_hat, (low, high)).
    """
    if region.values is None or len(region.values) == 0:
        return 0.0, (0.0, 1.0)
    vals = region.values
    n = len(vals)
    k = int((vals < 0.0).sum())
    ci = compute_confidence_interval(k, n)
    return float(k / n), ci


def estimate_global_falsification_probability(regions: Iterable) -> tuple[float, dict]:
    """Estimate the global falsification probability by volume-weighted averaging of region estimates.

    Returns (p_global, dict_of_region_estimates)
    """
    total = 0.0
    estimates = {}
    for r in regions:
        vol = r.volume()
        p_hat, ci = estimate_region_falsification_probability(r)
        estimates[r.region_id] = {"p_hat": p_hat, "ci": ci, "vol": vol}
        total += vol * p_hat
    # domain volume assumed 1.0
    return float(total), estimates
