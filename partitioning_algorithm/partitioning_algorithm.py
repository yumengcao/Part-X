import copy
from typing import Dict, List, Tuple


class Partitioning:
    def __init__(self,
                 subregions: Dict[str, List[Tuple[float, float]]],
                 dim_index: Dict[str, int],
                 dim: int,
                 part_number: Dict[str, int],
                 mother_allocation: Dict[str, int]):
        """
        Parameters:
            subregions: dict {region_id: [(low1, up1), (low2, up2), ...]}
            dim_index: dict {region_id: dimension_to_split}
            dim: total number of dimensions
            part_number: dict {region_id: how many parts to split}
            mother_allocation: dict {region_id: number of samples for this region}
        """
        self.subregions = subregions
        self.dim_index = dim_index
        self.dim = dim
        self.part_number = part_number
        self.mother_allocation = mother_allocation

    def _validate_bounds(self, region_bounds: List[Tuple[float, float]]):
        """Ensure region bounds are valid."""
        assert len(region_bounds) == self.dim, "Region must have bounds for each dimension"
        for i, (low, high) in enumerate(region_bounds):
            if not low < high:
                raise ValueError(f"Invalid bounds in dim {i}: low={low}, high={high}")

    def _split_interval(self, low: float, high: float, num_parts: int) -> List[Tuple[float, float]]:
        """Split [low, high] into equal intervals."""
        width = (high - low) / num_parts
        return [(low + i * width, low + (i + 1) * width) for i in range(num_parts)]

    def _allocate_child_samples(self, total: int, n_children: int) -> List[int]:
        """Evenly split samples from parent to children."""
        base = total // n_children
        rem = total % n_children
        alloc = [base] * n_children
        for i in range(rem):
            alloc[i] += 1
        return alloc

    def partitioning_algorithm(self) -> Tuple[Dict[str, List[Tuple[float, float]]],
                                              Dict[str, int],
                                              Dict[str, int]]:
        """
        Returns:
            part_sub: dict of new subregions {new_id: bounds}
            child_allocation: dict {new_id: samples allocated}
            dim_index_new: dict {new_id: dimension to split next}
        """
        part_sub = {}
        child_allocation = {}
        dim_index_new = {}
        region_counter = 0

        for region_id, bounds in self.subregions.items():
            self._validate_bounds(bounds)
            d = self.dim_index[region_id]
            num_parts = self.part_number[region_id]
            total_samples = self.mother_allocation[region_id]

            if num_parts == 1:
                # No split
                new_id = str(region_counter)
                part_sub[new_id] = bounds
                child_allocation[new_id] = total_samples
                dim_index_new[new_id] = d
                region_counter += 1
                continue

            low, high = bounds[d]
            intervals = self._split_interval(low, high, num_parts)
            sample_alloc = self._allocate_child_samples(total_samples, num_parts)

            for i, interval in enumerate(intervals):
                new_bounds = copy.deepcopy(bounds)
                new_bounds[d] = interval
                new_id = str(region_counter)
                part_sub[new_id] = new_bounds
                child_allocation[new_id] = sample_alloc[i]
                dim_index_new[new_id] = (d + 1) % self.dim
                region_counter += 1

        return part_sub, child_allocation, dim_index_new