"""Sampling helpers for regions."""
from typing import Iterable, List
import numpy as np
from .benchmark import synthetic_robustness_function
from .region import Region


def uniform_sample_region(region: Region, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample `n` points uniformly from `region` and return the points.

    Does not attach values to the region. Use `sample_regions` to update regions.
    """
    return region.sample_uniform(n, rng)


def sample_regions(regions: Iterable[Region], n_per_region: int, rng: np.random.Generator, eval_fn=None) -> None:
    """Sample each region `n_per_region` points, evaluate using `eval_fn`, and append to region.samples/values.

    If `eval_fn` is None, uses the built-in synthetic_robustness_function.
    """
    if eval_fn is None:
        eval_fn = synthetic_robustness_function

    for r in regions:
        pts = r.sample_uniform(n_per_region, rng)
        vals = eval_fn(pts)
        if r.samples is None:
            r.samples = pts
            r.values = vals
        else:
            r.samples = np.vstack([r.samples, pts])
            r.values = np.concatenate([r.values, vals])
