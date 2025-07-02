import copy
from typing import Dict, List, Tuple


class Partitioning:
    def __init__(self, subregions: Dict[str, List[Tuple[float, float]]], dim_index: int, dim: int,
                 iteration: int, region_vol: float, part_number: int):
        """
        Partitioning algorithm for subdividing regions along one dimension.

        Parameters:
            subregions (dict): Subregions to be partitioned, in format {id: [(low1, up1), (low2, up2), ...]}
            dim_index (int): Dimension index to split
            dim (int): Total number of dimensions
            iteration (int): Current iteration number
            region_vol (float): Volume of entire region
            part_number (int): Number of partitions to create per region
        """
        assert isinstance(subregions, dict), "subregions must be a dictionary"
        assert all(len(bounds) == dim and all(isinstance(b, tuple) and len(b) == 2 for b in bounds)
                   for bounds in subregions.values()), \
            "Each subregion must be a list of (low, high) tuples with correct dimension"

        self.subregions = subregions
        self.dim_index = dim_index
        self.dim = dim
        self.iteration = iteration
        self.region_vol = region_vol
        self.part_number = part_number
        self.re_num = 0  # Counter for assigning new region IDs

    def _validate_bounds(self, region_bounds: List[Tuple[float, float]]):
        """Ensure region bounds are valid and match dimension."""
        assert len(region_bounds) == self.dim, "Region must contain bounds for each dimension"
        for i, (low, high) in enumerate(region_bounds):
            if not low < high:
                raise ValueError(f"Invalid bounds in dimension {i}: lower={low}, upper={high}")

    def _split_interval(self, low: float, high: float, num_parts: int) -> List[Tuple[float, float]]:
        """Split [low, high] into equal intervals."""
        width = (high - low) / num_parts
        return [(low + i * width, low + (i + 1) * width) for i in range(num_parts)]

    def partitioning_algorithm(self) -> Tuple[Dict[str, List[Tuple[float, float]]], int]:
        """
        Partition subregions along dim_index.

        Returns:
            part_sub (dict): New subregions after partitioning
            re_num (int): Total number of generated subregions
        """
        part_sub = {}

        for sub_index, bounds in self.subregions.items():
            self._validate_bounds(bounds)

            if self.part_number == 1:
                part_sub[str(self.re_num)] = bounds
                self.re_num += 1
                continue

            low = bounds[self.dim_index][0]
            high = bounds[self.dim_index][1]

            try:
                intervals = self._split_interval(low, high, self.part_number)
            except Exception as e:
                raise RuntimeError(f"Failed to split subregion {sub_index} along dimension {self.dim_index}: {e}")

            for interval in intervals:
                new_bounds = copy.deepcopy(bounds)
                new_bounds[self.dim_index] = interval
                part_sub[str(self.re_num)] = new_bounds
                self.re_num += 1

        return part_sub, self.re_num
