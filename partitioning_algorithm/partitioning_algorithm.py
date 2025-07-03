import copy
from typing import Dict, List, Tuple

class Partitioning:
    def __init__(self, 
                 subregions: Dict[str, List[Tuple[float, float]]], 
                 dim_index: Dict[str, int], 
                 part_number: Dict[str, int], 
                 dim: int):
        """
        Partitioning algorithm for subdividing regions.

        Parameters:
            subregions (dict): {region_id: [(low1, up1), (low2, up2), ...]}
            dim_index (dict): {region_id: index of dimension to split}
            part_number (dict): {region_id: number of partitions to create}
            dim (int): Total number of dimensions
        """
        assert isinstance(subregions, dict), "subregions must be a dictionary"
        self.subregions = subregions
        self.dim_index = dim_index
        self.part_number = part_number
        self.dim = dim

    def _validate_bounds(self, region_bounds: List[Tuple[float, float]]):
        assert len(region_bounds) == self.dim, "Bounds length must match dimensions"
        for i, (low, high) in enumerate(region_bounds):
            if not low < high:
                raise ValueError(f"Invalid bounds in dim {i}: lower={low}, upper={high}")

    def _split_interval(self, low: float, high: float, num_parts: int) -> List[Tuple[float, float]]:
        width = (high - low) / num_parts
        return [(low + i * width, low + (i + 1) * width) for i in range(num_parts)]

    def partitioning_algorithm(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Partition each subregion along specified dimension.

        Returns:
            dict: {new_region_id: region_bounds}
        """
        part_sub = {}
        new_id = 0

        for region_id, bounds in self.subregions.items():
            self._validate_bounds(bounds)
            d = self.dim_index[region_id]
            num_parts = self.part_number[region_id]

            if num_parts == 1:
                part_sub[str(new_id)] = bounds
                new_id += 1
                continue

            intervals = self._split_interval(bounds[d][0], bounds[d][1], num_parts)

            for interval in intervals:
                new_bounds = copy.deepcopy(bounds)
                new_bounds[d] = interval
                part_sub[str(new_id)] = new_bounds
                new_id += 1

        return part_sub