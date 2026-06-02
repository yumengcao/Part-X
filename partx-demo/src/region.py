"""Region abstraction for rectangular partitions in 2D."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Region:
    """Represents an axis-aligned rectangular region in 2D.

    Attributes
    - region_id: unique identifier
    - bounds: ((x1_min,x1_max),(x2_min,x2_max))
    - parent_id: parent region id or None
    - depth: tree depth (root=0)
    - samples: ndarray of sampled points shape (n,2)
    - values: ndarray of robustness values shape (n,)
    """
    region_id: str
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]
    parent_id: Optional[str] = None
    depth: int = 0
    samples: Optional[np.ndarray] = field(default=None)
    values: Optional[np.ndarray] = field(default=None)
    # label: '+' = safe/positive, '-' = falsifying/negative, 'r' = remaining/uncertain
    label: str = 'r'

    def center(self) -> Tuple[float, float]:
        (x0, x1), (y0, y1) = self.bounds
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def volume(self) -> float:
        (x0, x1), (y0, y1) = self.bounds
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return boolean mask of which points are inside the region (inclusive).

        `points` is (n,2) array.
        """
        pts = np.asarray(points)
        x0, x1 = self.bounds[0]
        y0, y1 = self.bounds[1]
        return (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)

    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniformly sample `n` points from inside the region."""
        (x0, x1), (y0, y1) = self.bounds
        xs = rng.uniform(x0, x1, size=n)
        ys = rng.uniform(y0, y1, size=n)
        pts = np.column_stack([xs, ys])
        return pts

    def split(self) -> List["Region"]:
        """Split the region into four equal rectangular children.

        Returns a list of 4 Region objects (order: SW, SE, NW, NE).
        """
        (x0, x1), (y0, y1) = self.bounds
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        children = []
        b = [((x0, xm), (y0, ym)), ((xm, x1), (y0, ym)), ((x0, xm), (ym, y1)), ((xm, x1), (ym, y1))]
        for i, bound in enumerate(b):
            rid = f"{self.region_id}.{i}"
            child = Region(region_id=rid, bounds=bound, parent_id=self.region_id, depth=self.depth + 1)
            children.append(child)
        return children

    def falsification_rate(self) -> float:
        """Return fraction of samples with robustness < 0. If no samples, return 0.0."""
        if self.values is None or len(self.values) == 0:
            return 0.0
        return float(np.mean(self.values < 0.0))

    def min_robustness(self) -> float:
        """Return the minimum robustness observed in this region, or +inf if none."""
        if self.values is None or len(self.values) == 0:
            return float("inf")
        return float(np.min(self.values))


def create_root_region() -> Region:
    """Create the root region covering [0,1]^2."""
    return Region(region_id="root", bounds=((0.0, 1.0), (0.0, 1.0)), parent_id=None, depth=0)


def split_regions_once(regions: List[Region]) -> List[Region]:
    """Split every region in `regions` once and return new list of children.

    This utility is primarily for tests and simple experiments.
    """
    children: List[Region] = []
    for r in regions:
        children.extend(r.split())
    return children
